"""AttnDirectGenerator: StyleConv-free attention-centric 2.5D synthesis generator.

Architecture:
  QK  mode: conditioner(S3) → weighted_ref [B,1,h,w] → upsample → cat(src, wref) → UNet
  QKV mode: conditioner(S3) → alpha [B,K,h,w], V-proj(ref) → weighted_feat [B,C,H,W]
            → cat(src, wfeat) → UNet
  *_ncc variants: same + NCC teacher signal guides alpha (selector_target_mode='ncc')

Key difference vs original PSSF_S3:
  - No StyleConv: attended ref is concatenated to src, not injected via gamma/beta modulation
  - DS (downsample) reduced to 1-4 for richer Q/K features at near-full resolution
  - V projection (QKV) gives content-rich attendance output vs raw pixel (QK)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_proposed_synthesis import PatchwiseSliceFusionConditioner25D


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class SimpleUNet(nn.Module):
    """Lightweight UNet with skip connections.

    PatchNCE feature layers (index convention for nce_layers config):
      0 → enc0  [B, ngf,   H,   W]
      1 → enc1  [B, ngf*2, H/2, W/2]
      2 → enc2  [B, ngf*4, H/4, W/4]
      3 → bottleneck [B, ngf*4, H/4, W/4]
    """

    def __init__(self, input_nc=2, output_nc=1, ngf=64, n_blocks=4):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, ngf, 3, bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4), nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(*[ResBlock(ngf * 4) for _ in range(n_blocks)])
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 + ngf * 4, ngf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 + ngf * 2, ngf, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf + ngf, output_nc, 3),
            nn.Tanh(),
        )

    def forward(self, x, layers=[], encode_only=False):
        feats = []

        e0 = self.enc0(x)
        if 0 in layers:
            feats.append(e0)
        if encode_only and len(feats) == len(layers):
            return feats

        e1 = self.enc1(e0)
        if 1 in layers:
            feats.append(e1)
        if encode_only and len(feats) == len(layers):
            return feats

        e2 = self.enc2(e1)
        if 2 in layers:
            feats.append(e2)
        if encode_only and len(feats) == len(layers):
            return feats

        bot = self.bottleneck(e2)
        if 3 in layers:
            feats.append(bot)
        if encode_only:
            return feats

        d2 = self.dec2(torch.cat([bot, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        out = self.final(torch.cat([d1, e0], dim=1))

        if layers:
            return feats
        return out


# ─────────────────────────────────────────────────────────────────────────────
# AttnDirectGenerator
# ─────────────────────────────────────────────────────────────────────────────

class AttnDirectGenerator(nn.Module):
    """StyleConv-free generator that uses S3 patch-slice attention directly.

    The conditioner produces weighted_ref (or weighted_feat for QKV mode) which
    is concatenated to src and fed into a plain UNet — no StyleConv modulation.

    Expects merged_input [B, 1+K, H, W]:
      channel 0   : source (src)
      channels 1..K : ref stack (K slices)
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ngf=64,
        n_blocks=4,
        ref_stack_size=3,
        # Attention mode
        attn_mode='qk',            # 'qk' | 'qkv' | 'qk_ncc' | 'qkv_ncc'
        # Conditioner
        downsample=2,              # DS factor: 1 | 2 | 4
        window=3,
        dim=32,                    # Q/K projection dim (larger → richer)
        use_spatial_value_fusion=True,   # S3: attend over window spatial positions
        use_qk_norm=True,
        init_temperature=10.0,
        fixed_temperature=False,
        coarse=False,
        slice_score_pool='logsumexp',
        spatial_tau=0.5,
        direct_alpha=1.0,
        use_confidence_gate=True,
        confidence_mode='entropy',
        confidence_min=0.1,
        qk_input_mode='image',
        selector_target_tau=0.2,
        selector_target_detach=True,
        # QKV: V-projection feature dim
        v_feat_dim=16,
        **kwargs,
    ):
        super().__init__()
        self.ref_stack_size = ref_stack_size
        self.attn_mode = attn_mode
        self.downsample = downsample
        self.v_feat_dim = v_feat_dim

        # NCC hint: activate selector_target_mode='ncc' for QK/QKV variants
        selector_target_mode = 'ncc' if 'ncc' in attn_mode else 'none'

        self.conditioner = PatchwiseSliceFusionConditioner25D(
            dim=dim,
            coarse=coarse,
            window=window,
            slice_score_pool=slice_score_pool,
            spatial_tau=spatial_tau,
            use_spatial_value_fusion=use_spatial_value_fusion,
            use_confidence_gate=use_confidence_gate,
            confidence_mode=confidence_mode,
            confidence_min=confidence_min,
            direct_alpha=direct_alpha,
            use_qk_norm=use_qk_norm,
            init_temperature=init_temperature,
            fixed_temperature=fixed_temperature,
            qk_input_mode=qk_input_mode,
            selector_target_mode=selector_target_mode,
            selector_target_tau=selector_target_tau,
            selector_target_detach=selector_target_detach,
            shared_qk=False,
        )

        if 'qkv' in attn_mode:
            # V projection: map raw ref pixel → feature vector
            self.v_proj = nn.Sequential(
                nn.Conv2d(1, v_feat_dim, 1),
                nn.ReLU(inplace=True),
            )
            unet_input_nc = input_nc + v_feat_dim
        else:
            self.v_proj = None
            unet_input_nc = input_nc + 1   # src + weighted_ref [B,1,H,W]

        self.unet = SimpleUNet(
            input_nc=unet_input_nc,
            output_nc=output_nc,
            ngf=ngf,
            n_blocks=n_blocks,
        )

        # Module-level stats (logging)
        self._last_ref_condition_stats = {}
        self._last_ref_condition_aux_losses = {}

    def _out_size(self, H, W):
        h = max(1, H // self.downsample)
        w = max(1, W // self.downsample)
        return h, w

    def forward(self, merged_input, layers=[], encode_only=False):
        B, _C, H, W = merged_input.shape
        src = merged_input[:, :1, ...]           # [B, 1, H, W]
        ref_stack = merged_input[:, 1:1 + self.ref_stack_size, ...]   # [B, K, H, W]
        K = ref_stack.shape[1]

        h, w = self._out_size(H, W)

        # ── S3 Attention ─────────────────────────────────────────────────────
        cond_out = self.conditioner(source=src, ref_stack=ref_stack, out_size=(h, w))
        # cond_out = (style, stats, aux_losses, extra)
        # extra = {'alpha': [B,K,h,w], 'weighted_ref': [B,1,h,w]}
        if len(cond_out) == 4:
            style, stats, aux_losses, extra = cond_out
        else:
            style, stats, aux_losses = cond_out
            extra = {}

        if not encode_only:
            self._last_ref_condition_stats = stats
            self._last_ref_condition_aux_losses = aux_losses

        # ── Build UNet input ──────────────────────────────────────────────────
        if 'qkv' in self.attn_mode and extra.get('alpha') is not None:
            alpha = extra['alpha']   # [B, K, h, w]

            # V-project ref at downsampled resolution
            ref_low = F.interpolate(ref_stack, size=(h, w), mode='bilinear', align_corners=False)
            ref_flat = ref_low.reshape(B * K, 1, h, w)
            v_feat_flat = self.v_proj(ref_flat)                          # [B*K, C, h, w]
            C_v = v_feat_flat.shape[1]
            v_feat = v_feat_flat.view(B, K, C_v, h, w)                  # [B, K, C, h, w]

            # Weighted sum over K: [B, C, h, w]
            alpha_exp = alpha.unsqueeze(2)                               # [B, K, 1, h, w]
            weighted_feat = (alpha_exp * v_feat).sum(dim=1)              # [B, C, h, w]

            weighted_feat_up = F.interpolate(
                weighted_feat, size=(H, W), mode='bilinear', align_corners=False
            )                                                             # [B, C, H, W]
            net_input = torch.cat([src, weighted_feat_up], dim=1)        # [B, 1+C, H, W]
        else:
            # QK mode: use style ≈ weighted_ref (when direct_alpha=1, confidence_min=1 → style=weighted_ref)
            # style [B,1,h,w] already encodes the attended ref pixels
            style_up = F.interpolate(style, size=(H, W), mode='bilinear', align_corners=False)
            net_input = torch.cat([src, style_up], dim=1)                # [B, 2, H, W]

        return self.unet(net_input, layers=layers, encode_only=encode_only)

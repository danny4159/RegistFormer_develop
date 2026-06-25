"""AttnDirectGenerator: StyleConv-free S3 attention generator.

Root cause of uniform beta (spatial attention):
  beta = softmax(score_kw / tau) over win² positions
  weighted_ref = Σ(alpha_beta × ref_pixel_unfold)
  → d(weighted_ref)/d(beta) ≈ 0 because raw MRI pixels in 3×3 window
    are nearly identical (smooth MRI) → gradient to Q/K dies → beta stays 1/win²

Fix: use learned REF FEATURES (RefFeatureExtractor, multi-layer CNN) as V values.
  Features vary spatially even in smooth MRI → d(weighted_feat)/d(beta) ≠ 0
  → proper gradient flows to Q/K → beta becomes non-uniform

Architecture:
  1. Conditioner (S3): Q/K attention → alpha [B,K,h,w], alpha_beta [B,K,win²,h,w]
  2. RefEncoder: ref_low → V_feat [B,K,C,h,w]  (learned, spatially diverse)
  3. V_unfold: [B,K,C,win²,h,w]
  4. weighted_feat = Σ(alpha_beta × V_unfold) [B,C,h,w]
  5. net_input = cat(src, upsample(weighted_feat)) → SimpleUNet
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_proposed_synthesis import PatchwiseSliceFusionConditioner25D


# ─────────────────────────────────────────────────────────────────────────────
# Shared primitives
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


class RefFeatureExtractor(nn.Module):
    """Multi-layer CNN that maps ref slices to spatially-diverse feature maps.

    Critical: a 1x1 conv would still produce feature vectors colinear to the
    pixel value → neighboring positions produce nearly identical features in smooth
    MRI → d(weighted_feat)/d(beta) ≈ 0. Using 3x3 convs captures local context,
    producing spatially diverse features even in homogeneous regions.
    """

    def __init__(self, in_ch=1, feat_dim=32):
        super().__init__()
        mid = feat_dim // 2
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, feat_dim, 3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.enc(x)   # [B, feat_dim, h, w]


class SimpleUNet(nn.Module):
    """Lightweight UNet with skip connections.

    PatchNCE layer index convention (nce_layers config):
      0 → enc0 [B, ngf,   H,   W]
      1 → enc1 [B, ngf*2, H/2, W/2]
      2 → enc2 [B, ngf*4, H/4, W/4]
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
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, output_nc, 3),
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

def _unfold_same(x, win):
    """Unfold with same-padding: [B,C,h,w] → [B, C*win², h, w]."""
    return F.unfold(x, kernel_size=win, padding=win // 2)


class AttnDirectGenerator(nn.Module):
    """StyleConv-free S3 attention generator with feature-level V.

    The conditioner computes alpha/alpha_beta via Q/K dot-product attention.
    A learned RefFeatureExtractor produces spatially-diverse V features.
    Weighted features are then concatenated to src and fed into a plain UNet.

    This ensures d(weighted_feat)/d(beta) ≠ 0, so spatial attention gets
    meaningful gradient and can learn non-uniform beta.

    Expects merged_input [B, 1+K, H, W]:
      channel 0    : source (src)
      channels 1..K: ref stack (K slices)
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ngf=64,
        n_blocks=4,
        ref_stack_size=3,
        # Attention mode: 'feat' uses RefEncoder V; 'feat_ncc' adds NCC teacher
        attn_mode='feat',
        # Conditioner settings
        downsample=2,
        window=5,
        dim=32,
        use_spatial_value_fusion=True,
        use_qk_norm=True,
        init_temperature=10.0,
        fixed_temperature=False,
        coarse=False,
        slice_score_pool='logsumexp',
        spatial_tau=0.5,
        direct_alpha=1.0,
        use_confidence_gate=False,
        confidence_mode='entropy',
        confidence_min=0.0,
        qk_input_mode='image',
        selector_target_tau=0.2,
        selector_target_detach=True,
        # V feature dim (RefFeatureExtractor output channels)
        v_feat_dim=32,
        # Feature-level beta: use RefEncoder cosine similarity instead of Q/K dot product
        # Fixes uniform-beta local minimum — RefEncoder 3x3 CNN has spatial diversity
        use_feat_beta=True,
        feat_beta_tau=0.1,
        **kwargs,
    ):
        super().__init__()
        self.ref_stack_size = ref_stack_size
        self.attn_mode = attn_mode
        self.downsample = downsample
        self.v_feat_dim = v_feat_dim
        self.window = window
        self.use_feat_beta = use_feat_beta
        self.feat_beta_tau = feat_beta_tau

        selector_target_mode = 'ncc' if 'ncc' in attn_mode else 'none'

        # S3 conditioner: Q/K attention → alpha, alpha_beta
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

        # V: learned feature extractor — multi-layer 3x3 conv ensures spatial diversity
        self.ref_encoder = RefFeatureExtractor(in_ch=1, feat_dim=v_feat_dim)

        # UNet input: src [B,1,H,W] + weighted_feat_upsampled [B,v_feat_dim,H,W]
        unet_input_nc = input_nc + v_feat_dim
        self.unet = SimpleUNet(
            input_nc=unet_input_nc,
            output_nc=output_nc,
            ngf=ngf,
            n_blocks=n_blocks,
        )

        self._last_ref_condition_stats = {}
        self._last_ref_condition_aux_losses = {}

    def _out_size(self, H, W):
        h = max(1, H // self.downsample)
        w = max(1, W // self.downsample)
        return h, w

    def forward(self, merged_input, layers=[], encode_only=False):
        B, _C, H, W = merged_input.shape
        src = merged_input[:, :1, ...]
        ref_stack = merged_input[:, 1:1 + self.ref_stack_size, ...]   # [B, K, H, W]
        K = ref_stack.shape[1]

        h, w = self._out_size(H, W)
        win = self.window
        win2 = win * win

        # ── S3 Conditioner: Q/K attention ────────────────────────────────────
        cond_out = self.conditioner(source=src, ref_stack=ref_stack, out_size=(h, w))
        if len(cond_out) == 4:
            _, stats, aux_losses, extra = cond_out
        else:
            _, stats, aux_losses = cond_out
            extra = {}

        if not encode_only:
            self._last_ref_condition_stats = stats
            self._last_ref_condition_aux_losses = aux_losses
            self._last_use_feat_beta = self.use_feat_beta

        alpha_beta = extra.get('alpha_beta')   # [B, K, win2, h, w] or None
        alpha = extra.get('alpha')             # [B, K, h, w]

        # ── V: learned ref features ───────────────────────────────────────────
        ref_low = F.interpolate(ref_stack, size=(h, w), mode='bilinear', align_corners=False)
        ref_flat = ref_low.reshape(B * K, 1, h, w)
        v_feat_flat = self.ref_encoder(ref_flat)               # [B*K, C, h, w]
        C_v = v_feat_flat.shape[1]
        v_feat = v_feat_flat.view(B, K, C_v, h, w)            # [B, K, C, h, w]

        # ── Weighted feature: S3 spatial+slice attention over V ───────────────
        if alpha_beta is not None or (self.use_feat_beta and alpha is not None):
            # Unfold V features: [B*K, C, h, w] → [B*K, C*win2, h, w]
            v_unfold_flat = _unfold_same(
                v_feat.reshape(B * K, C_v, h, w), win
            )                                                  # [B*K, C*win2, h, w]
            v_unfold = v_unfold_flat.view(
                B, K, C_v, win2, h, w
            )                                                  # [B, K, C, win2, h, w]

            if self.use_feat_beta and alpha is not None:
                # Feature cosine similarity beta — fixes uniform-beta local minimum.
                # Q/K trained on smooth T1/T2 pixels can't distinguish win² positions.
                # RefEncoder 3x3 CNN produces spatially diverse features that CAN
                # distinguish neighboring positions even in smooth MRI regions.
                #
                # feat_beta is computed with no_grad + detached features to avoid
                # storing large backward activations (saves ~500 MB peak memory).
                # alpha and v_unfold still carry gradients: encoder trains through
                # weighted_feat, and slice selection trains through alpha.
                with torch.no_grad():
                    # src_feat has no downstream gradient path (feat_beta is detached),
                    # so compute entirely inside no_grad to avoid storing activations.
                    src_low = F.interpolate(
                        src, size=(h, w), mode='bilinear', align_corners=False
                    )
                    src_feat = self.ref_encoder(src_low)        # [B, C_v, h, w]
                    src_feat_n = F.normalize(src_feat, dim=1)
                    v_unfold_n = F.normalize(v_unfold.detach(), dim=2)  # [B,K,Cv,win2,h,w]
                    src_exp = src_feat_n.unsqueeze(1).unsqueeze(3)      # [B,1,Cv,1,h,w]
                    feat_score = (src_exp * v_unfold_n).sum(dim=2)      # [B,K,win2,h,w]
                    feat_beta = F.softmax(feat_score / self.feat_beta_tau, dim=2)
                ab = (alpha.unsqueeze(2) * feat_beta).unsqueeze(2)  # [B, K, 1, win2, h, w]

                if not encode_only:
                    with torch.no_grad():
                        fb_ent = -(feat_beta * feat_beta.clamp_min(1e-8).log()).sum(dim=2)
                        fb_eff_k = fb_ent.exp().mean()
                        fb_max = feat_beta.max(dim=2).values.mean()
                        center_idx = win2 // 2
                        fb_center = feat_beta[:, :, center_idx, :, :].mean()
                    if self._last_ref_condition_stats is None:
                        self._last_ref_condition_stats = {}
                    self._last_ref_condition_stats['beta_spatial_eff_k'] = fb_eff_k
                    self._last_ref_condition_stats['beta_max'] = fb_max
                    self._last_ref_condition_stats['beta_center_weight'] = fb_center
            else:
                # alpha_beta from conditioner Q/K
                ab = alpha_beta.unsqueeze(2)                   # [B, K, 1, win2, h, w]

            weighted_feat = (ab * v_unfold).sum(dim=(1, 3))   # [B, C, h, w]
        elif alpha is not None:
            # Fallback: K-direction only (no spatial window)
            alpha_exp = alpha.unsqueeze(2)                     # [B, K, 1, h, w]
            weighted_feat = (alpha_exp * v_feat).sum(dim=1)   # [B, C, h, w]
        else:
            # No attention weights available (e.g. K=1)
            weighted_feat = v_feat[:, K // 2]                 # [B, C, h, w]

        # ── Upsample and concat with src ─────────────────────────────────────
        weighted_feat_up = F.interpolate(
            weighted_feat, size=(H, W), mode='bilinear', align_corners=False
        )                                                       # [B, C, H, W]
        net_input = torch.cat([src, weighted_feat_up], dim=1)  # [B, 1+C, H, W]

        return self.unet(net_input, layers=layers, encode_only=encode_only)

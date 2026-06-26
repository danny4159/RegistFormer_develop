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


class StridedRefFeatureExtractor(nn.Module):
    """Full-resolution input → stride conv downsampling → feature map.

    Hypothesis: bilinear downsampling before RefFeatureExtractor smooths out
    spatial detail before features are extracted, making win² neighbor features
    nearly identical → uniform beta. Using stride conv on the full-res image
    preserves edge/detail information through the downsampling step itself,
    potentially producing more spatially diverse features within the win² window.

    DS=2: one stride-2 conv layer (H,W) → (H/2, W/2)
    DS=4: two stride-2 conv layers  (H,W) → (H/4, W/4)
    """

    def __init__(self, in_ch=1, feat_dim=32, downsample=2):
        super().__init__()
        mid = feat_dim // 2
        assert downsample in (1, 2, 4), "StridedRefFeatureExtractor supports DS=1,2,4"

        if downsample == 1:
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, feat_dim, 3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
            )
        elif downsample == 2:
            # stride-2 on first conv: (H,W) → (H/2, W/2) while extracting features
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, feat_dim, 3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
            )
        else:  # downsample == 4
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, feat_dim, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.enc(x)   # [B, feat_dim, H/DS, W/DS]


class MultiScaleStridedRefFeatureExtractor(nn.Module):
    """Two-stage stride conv: full-res → feat_s1 (DS=2) → feat_s2 (DS=4).

    Each stage downsamples by 2× via stride conv while extracting features,
    preserving spatial detail lost by bilinear pre-downsampling.
    Returns features at both scales for multi-scale attention.

    Stage 1: (H,W)     → (H/2, W/2)  [feat_dim ch]
    Stage 2: (H/2,W/2) → (H/4, W/4)  [feat_dim ch]
    """

    def __init__(self, in_ch=1, feat_dim=32):
        super().__init__()
        # Stage 1: full-res → DS=2 feature
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
        )
        # Stage 2: DS=2 → DS=4 feature
        self.stage2 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f1 = self.stage1(x)   # [B, feat_dim, H/2, W/2]
        f2 = self.stage2(f1)  # [B, feat_dim, H/4, W/4]
        return f1, f2


# ─────────────────────────────────────────────────────────────────────────────
# v8 Unified Attention Encoders (K×h×w×25 single attention)
# ─────────────────────────────────────────────────────────────────────────────

class Ref2DEncoder(nn.Module):
    """2D per-slice encoder: each slice through same 2D conv independently.

    Input: ref_flat [B*K, 1, H, W] (flattened K slices)
    Output: [B*K, C, h, w] (not reshaped to K dimension — caller handles reshape)
    """
    def __init__(self, in_ch=1, feat_dim=32, downsample=2):
        super().__init__()
        mid = feat_dim // 2
        assert downsample in (1, 2, 4)

        if downsample == 1:
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, feat_dim, 3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
            )
        elif downsample == 2:
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, feat_dim, 3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
            )
        else:  # downsample == 4
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, feat_dim, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(feat_dim), nn.ReLU(inplace=True),
            )

    def forward(self, ref_flat):
        # Input: [B*K, 1, H, W]
        # Output: [B*K, C, h, w]
        return self.enc(ref_flat)


class Ref2D3DEncoder(nn.Module):
    """2D encoding + 3D projection: each slice 2D encoded, then Conv3D on K dimension.

    Input: ref_flat [B*K, 1, H, W]
    1. 2D encode each slice → [B*K, C, h, w]
    2. Reshape to [B, C, K, h, w] for Conv3D
    3. Apply Conv3D projection (kernel_size=(3,1,1) on K-axis)
    4. Reshape back to [B*K, C, h, w]
    """
    def __init__(self, in_ch=1, feat_dim=32, downsample=2):
        super().__init__()
        self.ref_2d = Ref2DEncoder(in_ch, feat_dim, downsample)
        self.proj_3d = nn.Sequential(
            nn.Conv3d(feat_dim, feat_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.InstanceNorm3d(feat_dim), nn.ReLU(inplace=True),
        )
        self.downsample = downsample

    def forward(self, ref_flat):
        # Input: [B*K, 1, H, W]
        feat = self.ref_2d(ref_flat)  # [B*K, C, h, w]

        # Infer B, K from input shape
        B_mult_K, C, h, w = feat.shape
        # Note: we don't know K here directly, so this encoder is tricky
        # For unified attention usage, caller should provide K information or use differently
        # For now, return feat as-is and handle in caller
        return feat


class Ref3DEncoder(nn.Module):
    """Conv3D encoder: spatial stride=downsample, depth stride=1 (preserve K).

    Input: ref_stack [B, K, H, W]
    1. Reshape to [B, 1, K, H, W] for Conv3D
    2. Apply Conv3D: spatial stride=downsample, depth stride=1
    3. Output: [B, C, K, h, w], permute to [B, K, C, h, w]
    """
    def __init__(self, in_ch=1, feat_dim=32, downsample=2):
        super().__init__()
        mid = feat_dim // 2
        assert downsample in (1, 2, 4)

        if downsample == 1:
            self.enc = nn.Sequential(
                nn.Conv3d(in_ch, mid, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
                nn.InstanceNorm3d(mid), nn.ReLU(inplace=True),
                nn.Conv3d(mid, feat_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
                nn.InstanceNorm3d(feat_dim), nn.ReLU(inplace=True),
            )
        elif downsample == 2:
            self.enc = nn.Sequential(
                nn.Conv3d(in_ch, mid, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.InstanceNorm3d(mid), nn.ReLU(inplace=True),
                nn.Conv3d(mid, feat_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
                nn.InstanceNorm3d(feat_dim), nn.ReLU(inplace=True),
            )
        else:  # downsample == 4
            self.enc = nn.Sequential(
                nn.Conv3d(in_ch, mid, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.InstanceNorm3d(mid), nn.ReLU(inplace=True),
                nn.Conv3d(mid, feat_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.InstanceNorm3d(feat_dim), nn.ReLU(inplace=True),
            )

    def forward(self, ref_stack):
        B, K, H, W = ref_stack.shape
        ref = ref_stack.unsqueeze(1)  # [B, 1, K, H, W]
        feat = self.enc(ref)  # [B, C, K, h, w]
        B, C, K, h, w = feat.shape
        return feat.permute(0, 2, 1, 3, 4)  # [B, K, C, h, w]


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
        # v6 fixes for uniform-beta (shared encoder alignment problem):
        #   use_separate_src_encoder: separate SrcEncoder so src/ref feature spaces
        #     can specialize independently instead of aligning → cosine_sim stays meaningful
        #   use_pos_bias: learnable win² position bias added to feat_score, forcing
        #     the model to prefer specific offsets even if cosine_sim is near-uniform
        use_separate_src_encoder=False,
        use_pos_bias=False,
        # v7: stride conv encoder — extract features from full-res image while
        # downsampling via stride, preserving spatial detail lost by bilinear pre-downsample.
        use_stride_encoder=False,
        # v7-ms: multi-scale stride — attend at DS=2 and DS=4, sum weighted feats
        use_multiscale_stride=False,
        # v7-sc: stride conv downsampling for S3 conditioner Q/K input (slice selection)
        use_stride_conditioner=False,
        # v8: unified K×h×w×25 attention encoders
        ref_encoder_type='stride',  # 'stride' (v7), '2d', '2d3d', '3d'
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
        self.use_separate_src_encoder = use_separate_src_encoder
        self.use_pos_bias = use_pos_bias
        self.use_stride_encoder = use_stride_encoder
        self.use_multiscale_stride = use_multiscale_stride
        self.use_stride_conditioner = use_stride_conditioner
        self.ref_encoder_type = ref_encoder_type

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

        # Stride conv for conditioner Q/K input: replaces bilinear pre-downsample
        # inside PatchwiseSliceFusionConditioner25D. We downsample src/ref externally
        # with stride conv and pass already-downsampled tensors; conditioner's internal
        # interpolate becomes a no-op (same size in == out).
        if use_stride_conditioner:
            assert downsample in (2, 4)
            if downsample == 2:
                self.cond_ds_src = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)
                self.cond_ds_ref = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)
            else:  # 4
                self.cond_ds_src = nn.Sequential(
                    nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False),
                    nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False),
                )
                self.cond_ds_ref = nn.Sequential(
                    nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False),
                    nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False),
                )

        # V: learned feature extractor — multi-layer 3x3 conv ensures spatial diversity.
        # v7: use_multiscale_stride / use_stride_encoder
        # v8: ref_encoder_type selects unified attention encoder ('2d', '2d3d', '3d')

        if ref_encoder_type in ('2d', '2d3d', '3d'):
            # v8 unified attention: separate encoders for src and ref
            if ref_encoder_type == '2d':
                self.ref_encoder = Ref2DEncoder(in_ch=1, feat_dim=v_feat_dim, downsample=downsample)
                self.src_encoder = Ref2DEncoder(in_ch=1, feat_dim=v_feat_dim, downsample=downsample)
            elif ref_encoder_type == '2d3d':
                self.ref_encoder = Ref2D3DEncoder(in_ch=1, feat_dim=v_feat_dim, downsample=downsample)
                # src uses 2D (src is single-slice, no K dimension)
                self.src_encoder = Ref2DEncoder(in_ch=1, feat_dim=v_feat_dim, downsample=downsample)
            else:  # '3d'
                self.ref_encoder = Ref3DEncoder(in_ch=1, feat_dim=v_feat_dim, downsample=downsample)
                # src: need to adapt to 3D? For now, use 2D and squeeze K dimension
                self.src_encoder = Ref2DEncoder(in_ch=1, feat_dim=v_feat_dim, downsample=downsample)
            # v8 always uses separate encoders
            self.use_unified_attention = True
        else:
            # v7 and earlier: original alpha/beta split attention
            self.use_unified_attention = False
            if use_multiscale_stride:
                self.ref_encoder = MultiScaleStridedRefFeatureExtractor(in_ch=1, feat_dim=v_feat_dim)
                if use_separate_src_encoder:
                    self.src_encoder = MultiScaleStridedRefFeatureExtractor(in_ch=1, feat_dim=v_feat_dim)
            else:
                enc_cls = StridedRefFeatureExtractor if use_stride_encoder else RefFeatureExtractor
                enc_kwargs = dict(in_ch=1, feat_dim=v_feat_dim)
                if use_stride_encoder:
                    enc_kwargs['downsample'] = downsample
                self.ref_encoder = enc_cls(**enc_kwargs)
                if use_separate_src_encoder:
                    self.src_encoder = enc_cls(**enc_kwargs)
        # Learnable position bias over win² shifts — forces model to prefer specific offsets
        # even when cosine_sim is near-uniform (fallback for encoder alignment issue).
        if use_pos_bias:
            win2 = window * window
            self.pos_bias = nn.Parameter(torch.zeros(1, 1, win2, 1, 1))

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

    def _unified_attention(self, ref_feat_stacked, src_feat_2d, win, tau):
        """Unified K×h×w×25 attention (v8).

        ref_feat_stacked: [B, K, C, h, w] — ref features for K slices
        src_feat_2d:      [B, C, h, w]    — src feature (single)
        returns:          weighted_feat [B, C, h, w]
        """
        B, K, C_v, h, w = ref_feat_stacked.shape
        win2 = win * win

        # Unfold ref across K and spatial: [B, K, C, win2, h, w]
        ref_flat = ref_feat_stacked.reshape(B*K, C_v, h, w)
        ref_unfold = _unfold_same(ref_flat, win).view(B, K, C_v, win2, h, w)

        with torch.no_grad():
            ref_unfold_n = F.normalize(ref_unfold.detach(), dim=2)  # [B,K,C,win2,h,w]
        src_feat_n = F.normalize(src_feat_2d, dim=1)  # [B,C,h,w]

        # Cosine similarity: src [B,C,h,w] vs ref_unfold [B,K,C,win2,h,w]
        # → [B,K,win2,h,w]
        src_exp = src_feat_n.unsqueeze(1).unsqueeze(3)  # [B,1,C,1,h,w]
        attn_score = (src_exp * ref_unfold_n.detach()).sum(dim=2)  # [B,K,win2,h,w]

        # Softmax over K and win2 dimensions (75-way attention).
        attn = F.softmax(
            (attn_score / tau).reshape(B, K * win2, h, w),
            dim=1,
        ).view(B, K, win2, h, w)

        # Weighted sum
        ab = attn.unsqueeze(2)  # [B,K,1,win2,h,w]
        weighted_feat = (ab * ref_unfold).sum(dim=(1, 3))  # [B,C,h,w]

        return weighted_feat, attn

    def _feat_beta_weighted(self, v_feat, src_feat, alpha, win, tau):
        """Single-scale feat_beta attention.

        v_feat  : [B, K, C, h, w]
        src_feat: [B, C, h, w]
        alpha   : [B, K, h, w]
        returns : weighted_feat [B, C, h, w], feat_beta [B, K, win², h, w]
        """
        B, K, C_v, h, w = v_feat.shape
        win2 = win * win

        v_unfold = _unfold_same(
            v_feat.reshape(B * K, C_v, h, w), win
        ).view(B, K, C_v, win2, h, w)

        with torch.no_grad():
            v_unfold_n = F.normalize(v_unfold.detach(), dim=2)
        src_feat_n = F.normalize(src_feat, dim=1)
        src_exp = src_feat_n.unsqueeze(1).unsqueeze(3)              # [B,1,C,1,h,w]
        feat_score = (src_exp * v_unfold_n.detach()).sum(dim=2)     # [B,K,win2,h,w]

        if self.use_pos_bias and hasattr(self, 'pos_bias'):
            feat_score = feat_score + self.pos_bias

        feat_beta = F.softmax(feat_score / tau, dim=2)
        ab = (alpha.unsqueeze(2) * feat_beta).unsqueeze(2)         # [B,K,1,win2,h,w]
        weighted_feat = (ab * v_unfold).sum(dim=(1, 3))            # [B,C,h,w]
        return weighted_feat, feat_beta

    def _log_beta_stats(self, feat_beta, src_feat, win2):
        """Compute beta stats with brain-masked eff_k to avoid background bias.

        val/beta_spatial_eff_k (full)  — includes background pixels → misleadingly high
        val/beta_eff_k_brain           — foreground only, accurately reflects attention quality

        Brain mask: src feature norm > per-image mean (simple, no external segmentation).
        Background pixels have near-uniform feat_beta (cosine_sim is meaningless there),
        inflating the global eff_k average toward win² even when brain regions are selective.
        """
        B, K, _, h, w = feat_beta.shape
        with torch.no_grad():
            fb_ent = -(feat_beta * feat_beta.clamp_min(1e-8).log()).sum(dim=2)  # [B,K,h,w]
            eff_k_map = fb_ent.exp()

            fb_eff_k_full = eff_k_map.mean()
            fb_max = feat_beta.max(dim=2).values.mean()
            fb_center = feat_beta[:, :, win2 // 2].mean()

            # brain mask per image (src feature norm > per-image mean)
            src_norm = src_feat.norm(dim=1)                                      # [B,h,w]
            thresh = src_norm.mean(dim=(-2, -1), keepdim=True)                  # [B,1,1]
            brain_mask = (src_norm > thresh).unsqueeze(1).expand(B, K, h, w)    # [B,K,h,w]
            fb_eff_k_brain = eff_k_map[brain_mask].mean() if brain_mask.any() else fb_eff_k_full

        return {
            'beta_spatial_eff_k': fb_eff_k_full,    # 기존 지표 (배경 포함, backward compat)
            'beta_eff_k_brain': fb_eff_k_brain,      # 뇌 영역만 — 실제 attention 품질
            'beta_max': fb_max,
            'beta_center_weight': fb_center,
        }

    def forward(self, merged_input, layers=[], encode_only=False):
        B, _C, H, W = merged_input.shape
        src = merged_input[:, :1, ...]
        ref_stack = merged_input[:, 1:1 + self.ref_stack_size, ...]   # [B, K, H, W]
        K = ref_stack.shape[1]

        h, w = self._out_size(H, W)
        win = self.window
        win2 = win * win

        # ── S3 Conditioner: Q/K attention ────────────────────────────────────
        if self.use_stride_conditioner:
            # Pre-downsample with stride conv → conditioner's internal bilinear becomes no-op
            src_cond = self.cond_ds_src(src)                              # [B,1,h,w]
            ref_cond = self.cond_ds_ref(
                ref_stack.reshape(B * K, 1, H, W)
            ).reshape(B, K, h, w)                                         # [B,K,h,w]
            cond_out = self.conditioner(source=src_cond, ref_stack=ref_cond, out_size=(h, w))
        else:
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
        # v8 unified attention: skip conditioner, use combined K×h×w×25 attention
        if self.use_unified_attention:
            if self.ref_encoder_type == '2d':
                # Ref2DEncoder: [B*K, 1, H, W] → [B*K, C, h, w]
                ref_flat = ref_stack.reshape(B * K, 1, H, W)
                ref_feat_flat = self.ref_encoder(ref_flat)  # [B*K, C, h, w]
                C_v = ref_feat_flat.shape[1]
                ref_feat = ref_feat_flat.view(B, K, C_v, h, w)
            elif self.ref_encoder_type == '2d3d':
                # Ref2D3DEncoder: [B*K, 1, H, W] → 2D encode → Conv3D on K
                ref_flat = ref_stack.reshape(B * K, 1, H, W)
                ref_2d_flat = self.ref_encoder.ref_2d(ref_flat)  # [B*K, C, h, w]
                C_v = ref_2d_flat.shape[1]
                ref_2d = ref_2d_flat.view(B, K, C_v, h, w)  # [B, K, C, h, w]
                ref_2d_for_3d = ref_2d.permute(0, 2, 1, 3, 4)  # [B, C, K, h, w]
                ref_3d = self.ref_encoder.proj_3d(ref_2d_for_3d)  # [B, C, K, h, w]
                ref_feat = ref_3d.permute(0, 2, 1, 3, 4)  # [B, K, C, h, w]
            elif self.ref_encoder_type == '3d':
                # Ref3DEncoder: [B, K, H, W] → [B, 1, K, H, W] → Conv3D → [B, K, C, h, w]
                ref_feat = self.ref_encoder(ref_stack)  # [B, K, C, h, w]
            else:
                # Fallback: stride encoder (shouldn't reach here if use_unified_attention is set correctly)
                ref_flat = ref_stack.reshape(B * K, 1, H, W)
                ref_feat_flat = self.ref_encoder(ref_flat)
                C_v = ref_feat_flat.shape[1]
                ref_feat = ref_feat_flat.view(B, K, C_v, h, w)

            # src: always 2D [B, C, h, w]
            src_flat = src.reshape(B, 1, H, W)
            src_feat_flat = self.src_encoder(src_flat)  # [B, C, h, w]

            # Unified attention: K×h×w×25
            weighted_feat, attn_unified = self._unified_attention(
                ref_feat, src_feat_flat, win, self.feat_beta_tau
            )

            if not encode_only:
                with torch.no_grad():
                    attn_ent = -(attn_unified * attn_unified.clamp_min(1e-8).log()).sum(dim=(1,2))
                    attn_eff_k = attn_ent.exp().mean()
                self._last_ref_condition_stats['unified_attn_eff_k'] = attn_eff_k

        elif self.use_multiscale_stride:
            # MultiScaleStridedRefFeatureExtractor: full-res → (feat_s1, feat_s2)
            ref_flat_full = ref_stack.reshape(B * K, 1, H, W)
            v_f1_flat, v_f2_flat = self.ref_encoder(ref_flat_full)  # DS=2, DS=4
            C_v = v_f1_flat.shape[1]
            h2, w2 = v_f2_flat.shape[-2], v_f2_flat.shape[-1]       # H/4, W/4
            v_feat1 = v_f1_flat.view(B, K, C_v, h, w)              # [B,K,C,H/2,W/2]
            v_feat2 = v_f2_flat.view(B, K, C_v, h2, w2)            # [B,K,C,H/4,W/4]

            if self.use_separate_src_encoder:
                sf1, sf2 = self.src_encoder(src)
            else:
                with torch.no_grad():
                    sf1, sf2 = self.ref_encoder(src)

            # alpha at DS=2 (from conditioner) and DS=4 (downsampled)
            alpha_s1 = alpha                                         # [B,K,h,w]
            alpha_s2 = F.interpolate(alpha, size=(h2, w2), mode='bilinear', align_corners=False)

            wf1, fb1 = self._feat_beta_weighted(v_feat1, sf1, alpha_s1, win, self.feat_beta_tau)
            wf2, fb2 = self._feat_beta_weighted(v_feat2, sf2, alpha_s2, win, self.feat_beta_tau)

            # Sum contributions from both scales (DS=4 upsampled to DS=2 resolution)
            wf2_up = F.interpolate(wf2, size=(h, w), mode='bilinear', align_corners=False)
            weighted_feat = wf1 + wf2_up                            # [B,C,h,w]

            if not encode_only:
                with torch.no_grad():
                    fb_all = (fb1 + F.interpolate(
                        fb2.reshape(B * K, win2, h2, w2), size=(h, w)
                    ).reshape(B, K, win2, h, w)) * 0.5
                self._last_ref_condition_stats.update(
                    self._log_beta_stats(fb_all, sf1, win2)
                )

        else:
            if self.use_stride_encoder:
                ref_flat = ref_stack.reshape(B * K, 1, H, W)
            else:
                ref_low = F.interpolate(ref_stack, size=(h, w), mode='bilinear', align_corners=False)
                ref_flat = ref_low.reshape(B * K, 1, h, w)
            v_feat_flat = self.ref_encoder(ref_flat)               # [B*K, C, h, w]
            C_v = v_feat_flat.shape[1]
            v_feat = v_feat_flat.view(B, K, C_v, h, w)

            # ── Weighted feature: S3 spatial+slice attention over V ───────────────
            if self.use_feat_beta and alpha is not None:
                if self.use_stride_encoder:
                    src_enc_input = src
                else:
                    src_enc_input = F.interpolate(src, size=(h, w), mode='bilinear', align_corners=False)
                if self.use_separate_src_encoder:
                    src_feat = self.src_encoder(src_enc_input)
                else:
                    with torch.no_grad():
                        src_feat = self.ref_encoder(src_enc_input)

                weighted_feat, feat_beta = self._feat_beta_weighted(
                    v_feat, src_feat, alpha, win, self.feat_beta_tau
                )

                if not encode_only:
                    self._last_ref_condition_stats.update(
                        self._log_beta_stats(feat_beta, src_feat, win2)
                    )

            elif alpha_beta is not None:
                v_unfold = _unfold_same(
                    v_feat.reshape(B * K, C_v, h, w), win
                ).view(B, K, C_v, win2, h, w)
                ab = alpha_beta.unsqueeze(2)
                weighted_feat = (ab * v_unfold).sum(dim=(1, 3))
            elif alpha is not None:
                alpha_exp = alpha.unsqueeze(2)
                weighted_feat = (alpha_exp * v_feat).sum(dim=1)
            else:
                weighted_feat = v_feat[:, K // 2]

        # ── Upsample and concat with src ─────────────────────────────────────
        weighted_feat_up = F.interpolate(
            weighted_feat, size=(H, W), mode='bilinear', align_corners=False
        )                                                       # [B, C, H, W]
        net_input = torch.cat([src, weighted_feat_up], dim=1)  # [B, 1+C, H, W]

        return self.unet(net_input, layers=layers, encode_only=encode_only)

"""MIGS (Moving-Image-Guided Synthesis) generator network.

Paper <-> code term mapping used throughout this file:
  fixed slice x^k                    -> fixed_slice
  moving stack y_adj^k (K slices)    -> moving_stack / moving_inputs
  synthesized output y_hat^k         -> synthesized_slice
  Slice-Window Attention (SWA)       -> SliceWindowAttention
  Contrast Guidance Map (CGM)        -> cgm
  Moving-Image-Guided Convolution    -> MIGConv
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.migs_local_attention import MIGLocalTransformerStage2D
from src.models.components.migs_pure_transformer import MIGSPureTransformerBackbone


def get_layer_by_dim(is_3d):
    dim = 3 if is_3d else 2
    Conv = getattr(nn, f'Conv{dim}d')
    Norm = getattr(nn, f'InstanceNorm{dim}d')
    return Conv, Norm, dim


def _zero_init_last_conv(seq):
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            break


def _is_sequence(value):
    """True for list-likes but not strings. Covers omegaconf ListConfig, which is
    iterable but not a list/tuple instance."""
    return hasattr(value, '__iter__') and not isinstance(value, (str, bytes))


def _safe_entropy(prob, dim, norm_base):
    ent = -(prob * prob.clamp_min(1e-8).log()).sum(dim=dim)
    return ent.mean() / max(math.log(norm_base), 1e-8)


class LocalWindowAttention2DLegacy(nn.Module):
    """Legacy 2D-only local-window attention conditioner (used only when
    use_25d_style=False, i.e. a single reference slice with no moving stack).
    Not the reported SWA path -- see SliceWindowAttention below for that.
    """

    def __init__(self, dim=16, window=3, residual_scale=0.1):
        super().__init__()
        assert window % 2 == 1
        self.dim = dim
        self.window = window
        self.residual_scale = residual_scale

        self.q_proj = nn.Conv2d(1, dim, 1)
        self.k_proj = nn.Conv2d(1, dim, 1)
        self.v_proj = nn.Conv2d(1, dim, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, 1, 3, padding=1),
        )
        _zero_init_last_conv(self.out_proj)

    def _local_attention(self, q, k, v):
        B, d, h, w = q.shape
        win = self.window
        pad = win // 2
        win2 = win * win

        k_unfold = F.unfold(k, kernel_size=win, padding=pad).view(B, d, win2, h, w)
        v_unfold = F.unfold(v, kernel_size=win, padding=pad).view(B, d, win2, h, w)

        score = (q.unsqueeze(2) * k_unfold).sum(dim=1) / math.sqrt(d)  # [B, win2, h, w]
        attn = torch.softmax(score, dim=1)
        ctx = (attn.unsqueeze(1) * v_unfold).sum(dim=2)  # [B, d, h, w]
        return ctx, attn

    def forward(self, fixed_slice, moving_ref, cgm_size):
        fixed_slice_down = F.interpolate(fixed_slice, size=cgm_size, mode='bilinear', align_corners=False)
        moving_ref_down = F.interpolate(moving_ref, size=cgm_size, mode='bilinear', align_corners=False)
        moving_base = moving_ref_down

        q = self.q_proj(fixed_slice_down)
        k = self.k_proj(moving_base)
        v = self.v_proj(moving_base)

        ctx, attn = self._local_attention(q, k, v)
        delta = self.out_proj(ctx)
        cgm = moving_base + self.residual_scale * delta

        center_idx = (self.window * self.window) // 2
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=1).mean()
        entropy_norm = entropy / math.log(self.window * self.window)

        stats = {
            "base_std": moving_ref_down.std().detach(),
            "ref_base_std": moving_base.std().detach(),
            "style_std": cgm.std().detach(),
            "style_base_delta": ((cgm - moving_ref_down).abs().mean() / (moving_ref_down.abs().mean() + 1e-8)).detach(),
            "attn_entropy": entropy_norm.detach(),
            "attn_max": attn.max(dim=1).values.mean().detach(),
            "attn_center_weight": attn[:, center_idx:center_idx + 1].mean().detach(),
        }
        return cgm, stats


class SliceWindowAttention(nn.Module):
    """Slice-Window Attention (SWA), corresponding to Eqs. (1)-(3) in the paper.

    The downsampled fixed slice provides the query. The K-slice moving stack
    provides shared-projection keys; with use_raw_moving_values=True (the
    reported setting), attention weights are applied directly to the raw
    moving intensities (no learned value projection). A single softmax over
    K * window_size^2 local candidates produces the single-channel Contrast
    Guidance Map (CGM), blended with the center moving slice via cgm_blend_alpha.
    """

    def __init__(self, qk_channels=16, window_size=3, residual_scale=0.1, center_slice_bias=0.2,
                 use_raw_moving_values=False, use_cosine_similarity=False, temperature_init=10.0,
                 use_uniform_attention=False, cgm_blend_alpha=0.5, learnable_temperature=True):
        super().__init__()
        assert window_size % 2 == 1, f"window_size must be odd, got {window_size}"
        self.dim = qk_channels
        self.window = window_size
        self.residual_scale = residual_scale
        self.center_slice_bias = center_slice_bias
        # use_raw_moving_values: attention weights directly mix raw moving intensities;
        # no learned V projection / output projection shortcut
        self.use_direct_attn = bool(use_raw_moving_values)
        self.direct_alpha = float(cgm_blend_alpha)  # cgm = center_moving + alpha*(weighted_moving - center_moving)
        # use_cosine_similarity: cosine-normalized Q/K + learnable softmax temperature;
        # prevents Q,K -> 0 collapse, small init temperature amplifies tiny cosine differences
        self.use_qk_norm = bool(use_cosine_similarity)
        self.temperature_learnable = bool(learnable_temperature)
        if self.use_qk_norm:
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(float(temperature_init))),
                requires_grad=self.temperature_learnable,
            )
        # differentiable attention entropy from the last forward (for optional entropy regularization)
        self.last_attn_entropy = None
        self.use_uniform_attn = bool(use_uniform_attention)

        # Manhattan distance for each position in the local window (used for diagnostic logging)
        r = window_size // 2
        dist = [abs(dy) + abs(dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        self.register_buffer(
            "spatial_rel_dist",
            torch.tensor(dist, dtype=torch.float32).view(1, 1, window_size * window_size, 1, 1),
        )

        self.q_proj = nn.Conv2d(1, qk_channels, 1)
        self.k_proj = nn.Conv2d(1, qk_channels, 1)
        if not self.use_direct_attn:
            self.v_proj = nn.Conv2d(1, qk_channels, 1)
            self.out_proj = nn.Sequential(
                nn.Conv2d(qk_channels, qk_channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(qk_channels, 1, 3, padding=1),
            )
            _zero_init_last_conv(self.out_proj)

    def _extract_local_neighborhoods(self, x, win):
        """Unfold preserving spatial size (odd window, symmetric padding)."""
        return F.unfold(x, kernel_size=win, padding=win // 2)

    def forward(self, fixed_slice, moving_stack, cgm_size):
        B, K, _, _ = moving_stack.shape
        cgm_h, cgm_w = cgm_size
        center_idx = K // 2
        win = self.window
        win2 = win * win

        fixed_slice_down = F.interpolate(fixed_slice, size=(cgm_h, cgm_w), mode='bilinear', align_corners=False)
        moving_stack_down = F.interpolate(moving_stack, size=(cgm_h, cgm_w), mode='bilinear', align_corners=False)
        moving_base = moving_stack_down

        query = self.q_proj(fixed_slice_down)  # [B,dim,h,w]
        keys = self.k_proj(moving_base.reshape(B * K, 1, cgm_h, cgm_w)).view(B, K, self.dim, cgm_h, cgm_w)

        # QK normalization: cosine similarity, prevents Q,K -> 0 collapse
        if self.use_qk_norm:
            query = F.normalize(query, dim=1, eps=1e-8)  # unit norm per spatial position
            keys = F.normalize(keys, dim=2, eps=1e-8)

        # local_keys: keys unfolded into the local window (always needed for score computation)
        local_keys_flat = self._extract_local_neighborhoods(keys.reshape(B * K, self.dim, cgm_h, cgm_w), win)
        local_keys = local_keys_flat.view(B, K, self.dim, win2, cgm_h, cgm_w)

        # ── attention logits ──────────────────────────────────────────────
        if self.use_qk_norm:
            temperature = self.log_temperature.exp()
            logits_nobias = (query.unsqueeze(1).unsqueeze(3) * local_keys).sum(dim=2) * temperature
        else:
            logits_nobias = (query.unsqueeze(1).unsqueeze(3) * local_keys).sum(dim=2) / math.sqrt(self.dim)
        attention_logits = logits_nobias.clone()
        attention_logits[:, center_idx:center_idx + 1] += self.center_slice_bias
        attention_weights = torch.softmax(
            attention_logits.view(B, K * win2, cgm_h, cgm_w), dim=1
        ).view(B, K, win2, cgm_h, cgm_w)
        attn_for_log = attention_weights.unsqueeze(1)  # [B,1,K,win2,h,w]

        center_moving_down = moving_stack_down[:, center_idx:center_idx + 1]
        center_moving_base = moving_base[:, center_idx:center_idx + 1]

        # Uniform attention baseline: bypass learned attention with 1/(K*win^2) weights
        if self.use_uniform_attn:
            attn_for_log = torch.ones_like(attn_for_log) / (K * win2)
            attention_weights = attn_for_log.squeeze(1)

        # ── CGM (Contrast Guidance Map) generation ─────────────────────────
        # attn_mean: [B,K,win2,h,w] (single head, so this equals attn_for_log squeezed)
        attn_mean = attn_for_log.mean(dim=1)

        if self.use_direct_attn:
            moving_value_candidates = self._extract_local_neighborhoods(
                moving_base.reshape(B * K, 1, cgm_h, cgm_w), win
            ).view(B, K, win2, cgm_h, cgm_w)
            weighted_moving = (attn_mean * moving_value_candidates).sum(dim=(1, 2)).unsqueeze(1)  # [B,1,h,w]
            attn_style = weighted_moving
            blend_anchor = center_moving_down
            blend_alpha = self.direct_alpha
            cgm = blend_anchor + blend_alpha * (attn_style - blend_anchor)
        else:
            # V_proj -> out_proj -> delta, residual onto the center moving slice
            v = self.v_proj(moving_base.reshape(B * K, 1, cgm_h, cgm_w)).view(B, K, self.dim, cgm_h, cgm_w)
            v_unfold = self._extract_local_neighborhoods(v.reshape(B * K, self.dim, cgm_h, cgm_w), win)
            v_unfold = v_unfold.view(B, K, self.dim, win2, cgm_h, cgm_w)
            ctx = (attention_weights.unsqueeze(2) * v_unfold).sum(dim=1).sum(dim=2)
            delta = self.out_proj(ctx)
            attn_style = center_moving_base + self.residual_scale * delta
            blend_anchor, blend_alpha, cgm = attn_style, 1.0, attn_style

        # ── unified diagnostic logging ──────────────────────────────────────
        slice_prob = attn_mean.sum(dim=2)                         # [B,K,h,w]
        spatial_center_prob = attn_mean[:, :, win2 // 2].sum(dim=1)  # [B,h,w]

        attn_flat = attn_for_log.view(B, 1, K * win2, cgm_h, cgm_w)
        attn_entropy = _safe_entropy(attn_flat, dim=2, norm_base=K * win2)
        # keep a differentiable copy (mean normalized entropy) for optional entropy regularization
        self.last_attn_entropy = attn_entropy.mean()
        attn_max = attn_flat.max(dim=2).values.mean()

        with torch.no_grad():
            slice_offsets = torch.arange(K, device=attn_mean.device, dtype=attn_mean.dtype) - float(center_idx)
            expected_abs_slice_offset = (slice_prob * slice_offsets.abs().view(1, K, 1, 1)).sum(dim=1).mean()

            sp_dist = self.spatial_rel_dist.to(attn_mean.device).view(1, 1, win2, 1, 1)
            expected_spatial_l1 = (attn_mean * sp_dist).sum(dim=(1, 2)).mean()

            near_slice_mask = (slice_offsets.abs() <= 1).to(attn_mean.dtype).view(1, K, 1, 1)
            near_slice_weight = (slice_prob * near_slice_mask).sum(dim=1).mean()
            near_spatial_mask = (sp_dist <= 1).to(attn_mean.dtype)
            near_spatial_weight = (attn_mean * near_spatial_mask).sum(dim=(1, 2)).mean()

        # QK diagnostics: if qk_score_std_nobias ~ 0 -> Q/K not contributing
        with torch.no_grad():
            qk_score_std_nobias = logits_nobias.std().detach()
            qk_score_std = attention_logits.std().detach()
            q_abs = query.abs().mean().detach()
            k_abs = keys.abs().mean().detach()
            temperature_val = (self.log_temperature.exp().detach()
                               if self.use_qk_norm else torch.zeros(1, device=cgm.device))

        stats = {
            "base_std": moving_stack_down.std().detach(),
            "ref_base_std": moving_base.std().detach(),
            "style_std": cgm.std().detach(),
            "style_center_delta": ((cgm - center_moving_down).abs().mean() / (center_moving_down.abs().mean() + 1e-8)).detach(),
            "style_base_delta": ((cgm - center_moving_base).abs().mean() / (center_moving_base.abs().mean() + 1e-8)).detach(),
            "blend_to_attn_delta": ((cgm - attn_style).abs().mean() / (attn_style.abs().mean() + 1e-8)).detach(),
            "blend_anchor_delta": ((cgm - blend_anchor).abs().mean() / (blend_anchor.abs().mean() + 1e-8)).detach(),
            "blend_alpha": torch.as_tensor(blend_alpha, device=cgm.device).detach(),
            "direct_attn_enabled": torch.as_tensor(float(self.use_direct_attn), device=cgm.device).detach(),
            "qk_norm_enabled": torch.as_tensor(float(self.use_qk_norm), device=cgm.device).detach(),
            "qk_temperature": temperature_val.squeeze().detach(),
            "qk_score_std_nobias": qk_score_std_nobias,
            "qk_score_std": qk_score_std,
            "q_abs": q_abs,
            "k_abs": k_abs,
            "attn_entropy": attn_entropy.mean().detach(),
            "attn_max": attn_max.detach(),
            "slice_entropy": _safe_entropy(slice_prob, dim=1, norm_base=K).detach(),
            "center_slice_weight": slice_prob[:, center_idx].mean().detach(),
            "spatial_center_weight": spatial_center_prob.mean().detach(),
            "expected_abs_slice_offset": expected_abs_slice_offset.detach(),
            "expected_spatial_l1": expected_spatial_l1.detach(),
            "near_slice_weight": near_slice_weight.detach(),
            "near_spatial_weight": near_spatial_weight.detach(),
        }
        return cgm, stats


class ConvGuidanceAblation(nn.Module):
    """Convolution-based guidance map — ABLATION baseline for Slice-Window Attention.

    Same interface / output as SliceWindowAttention (forward(fixed_slice, moving_stack,
    cgm_size) -> (guidance_map [B,1,h,w], stats)) so it drops into the exact same
    generator, but the cross-slice fusion is a plain local convolution instead of
    data-dependent query-key attention.

    Design (fair ablation):
      - Receives the SAME inputs as SWA: downsampled fixed slice (query) + K moving slices.
      - Concatenates them on the channel axis and applies a small local 3x3 conv stack, so the
        guidance map is produced by fixed learned filters rather than input-dependent attention.
      - ref_stack_size=1  -> 2D conv   (fixed + 1 center moving slice; input 2ch)
      - ref_stack_size=3  -> 2.5D conv (fixed + 3 moving slices;       input 4ch)
      Everything downstream (MIGConv modulation, losses) is identical to the SWA model.
    """

    def __init__(self, ref_stack_size=3, hidden=32, depth=3):
        super().__init__()
        self.ref_stack_size = ref_stack_size
        in_ch = 1 + ref_stack_size  # fixed slice (1) + K moving slices
        layers = [nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                       nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(hidden, 1, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*layers)

    def _downsample_for_guidance(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

    def forward(self, fixed_slice, moving_stack, cgm_size):
        h, w = cgm_size
        fixed_slice_down = self._downsample_for_guidance(fixed_slice, h, w)      # [B,1,h,w]
        moving_stack_down = self._downsample_for_guidance(moving_stack, h, w)   # [B,K,h,w]
        x = torch.cat([fixed_slice_down, moving_stack_down], dim=1)  # [B,1+K,h,w]
        guidance_map = self.net(x)                                    # [B,1,h,w]

        K = moving_stack.shape[1]
        center_moving_down = moving_stack_down[:, K // 2:K // 2 + 1]
        stats = {
            "base_std": moving_stack_down.std().detach(),
            "ref_base_std": moving_stack_down.std().detach(),
            "style_std": guidance_map.std().detach(),
            "style_center_delta": ((guidance_map - center_moving_down).abs().mean()
                                   / (center_moving_down.abs().mean() + 1e-8)).detach(),
            "conv_conditioner_enabled": torch.as_tensor(1.0, device=guidance_map.device),
            "ref_stack_size": torch.as_tensor(float(K), device=guidance_map.device),
        }
        return guidance_map, stats


class MIGSGenerator(nn.Module):
    """MIGS synthesis network G.

    Pipeline: fixed_slice + moving_stack -> SliceWindowAttention -> CGM
              -> backbone -> synthesized_slice.

    Two backbones are selectable via ``backbone_type``:

    ``cnn`` (default, the reported model)
        12 MIGConv blocks arranged as a StyleConv U-Net.

    ``local_attention`` (CNN + Transformer hybrid)
        MIGConv is kept only where the resolution changes (stem / 2 downsamples /
        2 upsamples); the same-resolution blocks become CGM-conditioned local
        Transformer stages (Swin windows or NATTEN neighborhoods). Feature
        resolutions, channel counts, skip additions and the ``encode_only``
        feature indices are identical to the CNN backbone, so PatchNCE
        (``nce_layers``, ``netF_A.input_nc``) needs no change.

    The SWA that builds the CGM is orthogonal to this choice: SWA is
    fixed-vs-moving cross-attention producing the guidance map, while the
    Transformer stages are self-attention inside the generator features.
    """

    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.input_nc = kwargs['input_nc']
            self.feat_ch = kwargs['feat_ch']
            self.output_nc = kwargs['output_nc']
            self.use_multiple_outputs = kwargs.get('use_multiple_outputs', None)
            self.use_triple_outputs = kwargs.get('use_triple_outputs', False)
            self.is_3d = kwargs.get('is_3d', False)
            self.use_separate_style_layers = kwargs.get('use_separate_style_layers', False)
            self.noise_independent = kwargs.get('noise_independent', False)
            self.use_25d_style = kwargs.get('use_25d_style', False)
            self.ref_stack_size = kwargs.get('ref_stack_size', 3)

            # guidance_mode: 'swa' (reported) | 'conv_ablation' | 'local_2d_legacy'(auto when
            # use_25d_style=False). Falls back to the legacy boolean ref_condition_use_conv.
            _legacy_use_conv = kwargs.get('ref_condition_use_conv', False)
            self.guidance_mode = kwargs.get(
                'guidance_mode', 'conv_ablation' if _legacy_use_conv else 'swa'
            )

            def _kw(new_key, old_key, default):
                return kwargs.get(new_key, kwargs.get(old_key, default))

            self.swa_window_size = _kw('swa_window_size', 'ref_condition_window', 3)
            self.swa_qk_channels = _kw('swa_qk_channels', 'ref_condition_dim', 16)
            self.swa_center_slice_bias = _kw('swa_center_slice_bias', 'ref_condition_center_bias', 0.2)
            self.swa_use_raw_values = _kw('swa_use_raw_values', 'ref_condition_direct', False)
            self.swa_use_cosine_similarity = _kw('swa_use_cosine_similarity', 'ref_condition_qk_norm', False)
            self.swa_temperature_init = _kw('swa_temperature_init', 'ref_condition_init_temperature', 10.0)
            self.swa_downsampling_factor = _kw('swa_downsampling_factor', 'ref_condition_downsample', 16)
            self.swa_use_uniform_attention = _kw('swa_use_uniform_attention', 'ref_condition_uniform_attn', False)
            self.cgm_blend_alpha = _kw('cgm_blend_alpha', 'ref_condition_direct_alpha', 0.5)
            self.swa_learnable_temperature = _kw(
                'swa_learnable_temperature', 'ref_condition_temperature_learnable', True
            )

            # ── backbone selection ────────────────────────────────────────────
            # 'cnn' (default) = the original 12-MIGConv U-Net, unchanged.
            # 'local_attention' = CNN(resolution changes) + Transformer(same resolution).
            self.backbone_type = kwargs.get('backbone_type', 'cnn')
            self.local_attention_type = kwargs.get('local_attention_type', 'swin')
            self.local_num_heads = int(kwargs.get('local_num_heads', 8))
            self.local_mlp_ratio = float(kwargs.get('local_mlp_ratio', 2.0))
            self.local_cgm_hidden = int(kwargs.get('local_cgm_hidden', 64))
            self.local_drop = float(kwargs.get('local_drop', 0.0))
            self.local_attn_drop = float(kwargs.get('local_attn_drop', 0.0))
            self.local_drop_path = float(kwargs.get('local_drop_path', 0.05))
            _layer_scale = kwargs.get('local_layer_scale_init', 1e-4)
            self.local_layer_scale_init = 0.0 if _layer_scale is None else float(_layer_scale)
            # one entry per stage: [enc1, enc2, bottleneck, dec1, dec2, refine]
            self.local_stage_depths = [
                int(d) for d in kwargs.get('local_stage_depths', [2, 2, 4, 2, 2, 2])
            ]
            self.swin_window_size = int(kwargs.get('swin_window_size', 8))
            self.natten_kernel_size = int(kwargs.get('natten_kernel_size', 7))
            # per stage: an int (all blocks) or a per-block sequence, e.g. [1, 1, [1,2,1,2], 1, 1, 1].
            # Hydra hands these over as ListConfig, so normalize to plain python here.
            self.natten_stage_dilations = [
                [int(d) for d in entry] if _is_sequence(entry) else int(entry)
                for entry in kwargs.get('natten_stage_dilations', [1, 1, 1, 1, 1, 1])
            ]

            # ── pure-transformer backbone (no MIGConv at all) ─────────────────
            self.pure_patch_size = int(kwargs.get('pure_patch_size', 2))
            self.pure_embed_dim = int(kwargs.get('pure_embed_dim', 96))
            self.pure_stage_depths = [
                int(d) for d in kwargs.get('pure_stage_depths', [2, 2, 6, 2, 2, 2])
            ]
            self.pure_num_heads = [
                int(h) for h in kwargs.get('pure_num_heads', [3, 6, 12, 6, 3, 3])
            ]
            # taps have per-stage channel counts, but PatchSampleF shares one MLP
            # across layers -> project every tap to this width (== netF_A.input_nc)
            self.pure_nce_proj_dim = int(kwargs.get('pure_nce_proj_dim', 264))
            self.pure_mlp_ratio = float(kwargs.get('pure_mlp_ratio', 4.0))
            # LayerScale has its own default here, independent of the hybrid's.
            # With no MIGConv to carry the signal, a small LayerScale would make
            # every residual branch ~0 and the whole backbone a near-identity at
            # init. Standard Swin / Swin-Unet use no LayerScale, so default off.
            _pure_ls = kwargs.get('pure_layer_scale_init', 0.0)
            self.pure_layer_scale_init = 0.0 if _pure_ls is None else float(_pure_ls)

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        if self.swa_window_size % 2 == 0:
            raise ValueError(f"swa_window_size must be odd, got {self.swa_window_size}")
        _use_conv = self.guidance_mode == 'conv_ablation'
        if self.is_3d and _use_conv:
            raise NotImplementedError("guidance_mode='conv_ablation' is 2D-only.")
        if (self.use_multiple_outputs or self.use_triple_outputs) and _use_conv:
            raise NotImplementedError("guidance_mode='conv_ablation' is only supported for single-output mode.")

        if self.backbone_type not in ('cnn', 'local_attention', 'pure_transformer'):
            raise ValueError(
                "backbone_type must be 'cnn' (MIGConv U-Net), 'local_attention' "
                f"(CNN+Transformer hybrid) or 'pure_transformer', got '{self.backbone_type}'"
            )
        self.use_local_attention_backbone = self.backbone_type == 'local_attention'
        self.use_pure_transformer_backbone = self.backbone_type == 'pure_transformer'
        _transformer_backbone = self.use_local_attention_backbone or self.use_pure_transformer_backbone

        if _transformer_backbone:
            if self.is_3d:
                raise NotImplementedError(f"backbone_type='{self.backbone_type}' is 2D-only.")
            if self.use_multiple_outputs or self.use_triple_outputs:
                raise NotImplementedError(
                    f"backbone_type='{self.backbone_type}' is only supported for single-output mode."
                )
            if len(self.natten_stage_dilations) != 6:
                raise ValueError(
                    "natten_stage_dilations needs 6 entries "
                    f"[enc1, enc2, bottleneck, dec1, dec2, refine], got {self.natten_stage_dilations}"
                )
        if self.use_pure_transformer_backbone:
            if len(self.pure_stage_depths) != 6:
                raise ValueError(
                    "pure_stage_depths needs 6 entries "
                    f"[enc1, enc2, bottleneck, dec1, dec2, refine], got {self.pure_stage_depths}"
                )
            if len(self.pure_num_heads) != 6:
                raise ValueError(
                    "pure_num_heads needs 6 entries "
                    f"[enc1, enc2, bottleneck, dec1, dec2, refine], got {self.pure_num_heads}"
                )
        if self.use_local_attention_backbone:
            if len(self.local_stage_depths) != 6:
                raise ValueError(
                    "local_stage_depths needs 6 entries "
                    f"[enc1, enc2, bottleneck, dec1, dec2, refine], got {self.local_stage_depths}"
                )

        Conv, _, _ = get_layer_by_dim(self.is_3d)

        # Determine channel multiplier based on output mode
        if self.use_triple_outputs:
            ch = 3
            self.num_style_streams = 3
        elif self.use_multiple_outputs:
            ch = 2
            self.num_style_streams = 2
        else:
            ch = 1
            self.num_style_streams = 1

        # 2.5D: compress moving stack [B, K, H, W] -> [B, 1, H, W] per style stream
        # (used by the is_3d path and by multi/triple-output guidance aggregation --
        #  legacy compatibility paths, not used by the reported single-output 2.5D MIGS model)
        if self.use_25d_style:
            hidden = max(8, self.feat_ch // 8)

            def _make_z_agg():
                return nn.Sequential(
                    nn.Conv2d(self.ref_stack_size, hidden, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
                )
            self.z_aggs = nn.ModuleList([_make_z_agg() for _ in range(self.num_style_streams)])

        # SWA / conv-ablation conditioners: only used for single-output 2D generation
        # (multi/triple-output and 3D always use the z_agg aggregation above)
        self.ref_conditioner_2d = None
        self.ref_conditioner_25d = None
        self.ref_conditioner_conv = None
        self._use_ref_conditioner = not (self.use_multiple_outputs or self.use_triple_outputs)
        if self._use_ref_conditioner:
            if _use_conv:
                self.ref_conditioner_conv = ConvGuidanceAblation(
                    ref_stack_size=self.ref_stack_size, hidden=32, depth=3,
                )
            else:
                self.ref_conditioner_2d = LocalWindowAttention2DLegacy(
                    dim=self.swa_qk_channels, window=self.swa_window_size, residual_scale=0.1,
                )
                if self.use_25d_style and self.ref_stack_size >= 1:
                    self.ref_conditioner_25d = SliceWindowAttention(
                        qk_channels=self.swa_qk_channels, window_size=self.swa_window_size,
                        residual_scale=0.1,
                        center_slice_bias=self.swa_center_slice_bias,
                        use_raw_moving_values=self.swa_use_raw_values,
                        use_cosine_similarity=self.swa_use_cosine_similarity,
                        temperature_init=self.swa_temperature_init,
                        use_uniform_attention=self.swa_use_uniform_attention,
                        cgm_blend_alpha=self.cgm_blend_alpha,
                        learnable_temperature=self.swa_learnable_temperature,
                    )

        self._last_ref_condition_stats: dict = {}
        self._last_attn_entropy_for_loss = None  # differentiable, set on main forward for entropy reg

        # Legacy unused module retained for checkpoint compatibility.
        self.guide_net = nn.Sequential(
            nn.Conv2d(self.input_nc, int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # MIGConv blocks. The stem and the four resolution-changing blocks
        # (conv11/conv21 down, conv41/conv51 up) exist for both backbones; the
        # same-resolution blocks (conv12/22/31/32/42/52/6) exist only for the CNN
        # backbone, where they complete the 12-MIGConv U-Net.
        # NOTE: the CNN branch must keep its original definition order -- resuming a
        # checkpoint restores Adam state by parameter index, not by name.
        _cnn = self.backbone_type == 'cnn'

        if self.use_pure_transformer_backbone:
            self._build_pure_transformer_backbone(Conv)
            return

        self.conv0 = MIGConv(self.input_nc, self.feat_ch * ch, kernel_size=3,
                                                 activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv11 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        if _cnn:
            self.conv12 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=False, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv21 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        if _cnn:
            self.conv22 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=False, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
            self.conv31 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=False, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
            self.conv32 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=False, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv41 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, #feat_ch *4 stays unchanged
                                upsample=True, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        if _cnn:
            self.conv42 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    upsample=False, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv51 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=True, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        if _cnn:
            self.conv52 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    upsample=False, activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
            self.conv6 = MIGConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    activate=True, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        else:
            def _make_local_stage(stage_index):
                return MIGLocalTransformerStage2D(
                    dim=self.feat_ch * ch,
                    depth=self.local_stage_depths[stage_index],
                    num_heads=self.local_num_heads,
                    attention_type=self.local_attention_type,
                    window_size=self.swin_window_size,
                    natten_kernel_size=self.natten_kernel_size,
                    natten_dilation=self.natten_stage_dilations[stage_index],
                    mlp_ratio=self.local_mlp_ratio,
                    cgm_channels=1,
                    cgm_hidden=self.local_cgm_hidden,
                    drop=self.local_drop,
                    attn_drop=self.local_attn_drop,
                    drop_path=self.local_drop_path,
                    layer_scale_init=self.local_layer_scale_init,
                )

            self.enc1_stage = _make_local_stage(0)
            self.enc2_stage = _make_local_stage(1)
            self.bottleneck_stage = _make_local_stage(2)
            self.dec1_stage = _make_local_stage(3)
            self.dec2_stage = _make_local_stage(4)
            self.refine_stage = _make_local_stage(5)

        # -------------------------------------------------------------------------
        # Legacy multi-output / triple-output compatibility paths
        # Not used by the reported single-output 2.5D MIGS model.
        # -------------------------------------------------------------------------
        if self.use_separate_style_layers and self.use_triple_outputs:
            self.conv7_1 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv7_2 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv7_3 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv8_1 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv8_2 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv8_3 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv_final_1 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_3 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
        elif self.use_separate_style_layers and self.use_multiple_outputs:
            self.conv7_1 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv7_2 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv8_1 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv8_2 = MIGConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, ch=1, is_3d=self.is_3d, )
            self.conv_final_1 = Conv(self.feat_ch, self.output_nc // 2, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, self.output_nc // 2, kernel_size=3, padding=1)

        self.conv_final = Conv(self.feat_ch * ch, self.output_nc, kernel_size=3, padding=1)

    def _build_pure_transformer_backbone(self, Conv):
        """Convolution-free backbone: no MIGConv, no conv up/downsampling.

        Only two convolutions remain and neither is part of the backbone: the
        patch-embedding projection (a strided non-overlapping patch projection,
        mathematically a per-patch linear map) and the final output head.
        """
        self.pure_backbone = MIGSPureTransformerBackbone(
            in_nc=self.input_nc,
            embed_dim=self.pure_embed_dim,
            patch_size=self.pure_patch_size,
            stage_depths=self.pure_stage_depths,
            stage_num_heads=self.pure_num_heads,
            attention_type=self.local_attention_type,
            window_size=self.swin_window_size,
            natten_kernel_size=self.natten_kernel_size,
            natten_stage_dilations=self.natten_stage_dilations,
            mlp_ratio=self.pure_mlp_ratio,
            cgm_channels=1,
            cgm_hidden=self.local_cgm_hidden,
            drop=self.local_drop,
            attn_drop=self.local_attn_drop,
            drop_path=self.local_drop_path,
            layer_scale_init=self.pure_layer_scale_init,
        )

        # PatchSampleF shares a single Linear across layers, so every tap has to
        # arrive with the same channel count. 1x1 projections, encode_only path only.
        self.nce_proj = nn.ModuleList([
            nn.Conv2d(dim, self.pure_nce_proj_dim, kernel_size=1)
            for dim in self.pure_backbone.tap_dims
        ])

        # Output head, fed by the full-resolution refine stage. 1x1 (per-pixel
        # linear, as in Swin-Unet) rather than the CNN backbone's 3x3, so no
        # spatial kernel is reintroduced after the Transformer stages.
        self.conv_final = Conv(self.pure_backbone.stage_dims[-1], self.output_nc,
                               kernel_size=1, padding=0)

    def post_init_weights(self):
        """Re-apply init that the generic `init_net` pass would otherwise clobber.

        `networks_define.init_net` re-initializes every Conv/Linear with
        normal(0, 0.02); the local-attention blocks need their relative-position
        bias trunc-normal and their CGM modulation head zeroed (identity start).
        """
        for module in self.modules():
            if module is not self and hasattr(module, 'reset_special_parameters'):
                module.reset_special_parameters()

    def _aggregate_moving_stack_legacy(self, moving_inputs):
        """2.5D: [B, num_streams*K, H, W] -> [B, num_streams, H, W]. Used by the is_3d
        path and by legacy multi/triple-output aggregation (not the reported SWA path)."""
        if not self.use_25d_style:
            return moving_inputs
        chunks = torch.split(moving_inputs, self.ref_stack_size, dim=1)
        style_maps = [z_agg(chunk) for chunk, z_agg in zip(chunks, self.z_aggs)]
        return torch.cat(style_maps, dim=1)

    def _get_cgm_size(self, fixed_slice):
        H, W = fixed_slice.shape[-2:]
        ds = self.swa_downsampling_factor
        return max(1, H // ds), max(1, W // ds)

    def _select_moving_stack(self, moving_inputs):
        """Return the moving tensor for the conditioners.
        2D  : [B,1,H,W]
        2.5D: [B,K,H,W]
        """
        if self.use_25d_style and self.ref_stack_size >= 1:
            return moving_inputs[:, :self.ref_stack_size]
        return moving_inputs[:, :1]

    def _build_contrast_guidance(self, fixed_slice, moving_inputs, encode_only=False):
        """Returns the Contrast Guidance Map (CGM), [B,1,h,w].

        Multi/triple-output generation always uses the legacy z_agg-aggregated moving
        stack as guidance. Single-output 2D generation uses the conv-ablation guidance
        (guidance_mode='conv_ablation') or Slice-Window Attention (guidance_mode='swa',
        the reported default).
        """
        cgm_h, cgm_w = self._get_cgm_size(fixed_slice)

        if not self._use_ref_conditioner:
            # legacy multi/triple-output: z_agg(2.5D) / nearest-downsampled(2D) as guidance
            if self.use_25d_style and self.ref_stack_size >= 1:
                base_ref = self._aggregate_moving_stack_legacy(moving_inputs)        # [B,1,H,W] via z_agg
                return F.interpolate(base_ref, size=(cgm_h, cgm_w), mode='nearest')
            moving_map = self._select_moving_stack(moving_inputs)                    # [B,1,H,W]
            return F.interpolate(moving_map, size=(cgm_h, cgm_w), mode='nearest')

        if self.guidance_mode == 'conv_ablation':
            moving_stack = self._select_moving_stack(moving_inputs)                  # [B,K,H,W]
            guidance_map, stats = self.ref_conditioner_conv(
                fixed_slice=fixed_slice, moving_stack=moving_stack, cgm_size=(cgm_h, cgm_w)
            )
            if not encode_only:
                self._last_ref_condition_stats = stats
            return guidance_map

        if self.use_25d_style and self.ref_stack_size >= 1:
            moving_stack = self._select_moving_stack(moving_inputs)                  # [B,K,H,W]
            cgm, stats = self.ref_conditioner_25d(
                fixed_slice=fixed_slice, moving_stack=moving_stack, cgm_size=(cgm_h, cgm_w)
            )
            if not encode_only:
                self._last_ref_condition_stats = stats
                self._last_attn_entropy_for_loss = self.ref_conditioner_25d.last_attn_entropy
            return cgm

        moving_map = self._select_moving_stack(moving_inputs)                        # [B,1,H,W]
        cgm, stats = self.ref_conditioner_2d(
            fixed_slice=fixed_slice, moving_ref=moving_map, cgm_size=(cgm_h, cgm_w)
        )
        if not encode_only:
            self._last_ref_condition_stats = stats
        return cgm

    def forward(self, merged_input, layers=[], encode_only=False):

        fixed_slice = merged_input[:, :1, ...]
        moving_inputs = merged_input[:, 1:, ...]

        if self.is_3d:
            # 3D path: legacy compatibility, unrelated to the reported 2D SWA model
            ref = self._aggregate_moving_stack_legacy(moving_inputs)
            ref = ref.permute(0, 1, 4, 2, 3)
            cgm = F.interpolate(ref, scale_factor=1/16, mode='trilinear', align_corners=False)
            cgm = cgm.permute(0, 1, 3, 4, 2)
        else:
            cgm = self._build_contrast_guidance(
                fixed_slice=fixed_slice, moving_inputs=moving_inputs, encode_only=encode_only
            )

        # SWA: construct a coarse CGM from fixed anatomy and moving contrast.
        # The same CGM spatially modulates every backbone block.
        if self.use_pure_transformer_backbone:
            (stem_feat, enc1_feat, enc2_feat, bottleneck_feat,
             dec1_feat, dec2_feat, refined_feat) = self.pure_backbone(fixed_slice, cgm)

            if encode_only:
                taps = [stem_feat, enc1_feat, enc2_feat, bottleneck_feat,
                        dec1_feat, dec2_feat, refined_feat]
                # per-tap 1x1 projection to a common width: PatchSampleF shares one
                # Linear across layers, so all taps must arrive with equal channels
                return [self.nce_proj[i](taps[i]) for i in layers]
            return torch.tanh(self.conv_final(refined_feat))

        stem_feat = self.conv0(fixed_slice, cgm)

        if self.use_local_attention_backbone:
            # Encoder: MIGConv changes the resolution, the Transformer stage
            # refines the features at that resolution.
            enc1_feat = self.conv11(stem_feat, cgm)                  # 128 -> 64
            enc1_feat = self.enc1_stage(enc1_feat, cgm)

            enc2_feat = self.conv21(enc1_feat, cgm)                  # 64 -> 32
            enc2_feat = self.enc2_stage(enc2_feat, cgm)

            # Bottleneck
            bottleneck_feat = self.bottleneck_stage(enc2_feat, cgm)  # 32

            # Decoder with element-wise skip addition
            dec1_feat = self.conv41(bottleneck_feat + enc2_feat, cgm)  # 32 -> 64
            dec1_feat = self.dec1_stage(dec1_feat, cgm)

            dec2_feat = self.conv51(dec1_feat + enc1_feat, cgm)        # 64 -> 128
            dec2_feat = self.dec2_stage(dec2_feat, cgm)

            # Final refinement
            refined_feat = self.refine_stage(dec2_feat + stem_feat, cgm)
        else:
            # Encoder
            enc1_feat = self.conv11(stem_feat, cgm)
            enc1_feat = self.conv12(enc1_feat, cgm)

            enc2_feat = self.conv21(enc1_feat, cgm)
            enc2_feat = self.conv22(enc2_feat, cgm)

            # Bottleneck
            bottleneck_feat = self.conv31(enc2_feat, cgm)
            bottleneck_feat = self.conv32(bottleneck_feat, cgm)

            # Decoder with element-wise skip addition
            dec1_feat = self.conv41(bottleneck_feat + enc2_feat, cgm)
            dec1_feat = self.conv42(dec1_feat, cgm)

            dec2_feat = self.conv51(dec1_feat + enc1_feat, cgm)
            dec2_feat = self.conv52(dec2_feat, cgm)

            # Final refinement
            refined_feat = self.conv6(dec2_feat + stem_feat, cgm)

        # -------------------------------------------------------------------------
        # Legacy multi-output / triple-output compatibility paths
        # Not used by the reported single-output 2.5D MIGS model.
        # -------------------------------------------------------------------------
        if self.use_separate_style_layers and self.use_triple_outputs:
            feat6_1, feat6_2, feat6_3 = torch.chunk(refined_feat, chunks=3, dim=1)
            style_1 = cgm[:, :1, ...]
            style_2 = cgm[:, 1:2, ...]
            style_3 = cgm[:, 2:3, ...]

            feat7_1 = self.conv7_1(feat6_1, style_1)
            feat7_2 = self.conv7_2(feat6_2, style_2)
            feat7_3 = self.conv7_3(feat6_3, style_3)

            feat8_1 = self.conv8_1(feat7_1, style_1)
            feat8_2 = self.conv8_2(feat7_2, style_2)
            feat8_3 = self.conv8_3(feat7_3, style_3)

            out_1 = torch.tanh(self.conv_final_1(feat8_1))
            out_2 = torch.tanh(self.conv_final_2(feat8_2))
            out_3 = torch.tanh(self.conv_final_3(feat8_3))

            synthesized_slice = torch.cat((out_1, out_2, out_3), dim=1)
        elif self.use_separate_style_layers and self.use_multiple_outputs:
            feat6_1, feat6_2 = torch.chunk(refined_feat, chunks=2, dim=1)
            style_1 = cgm[:, :1, ...]
            style_2 = cgm[:, 1:, ...]

            feat7_1 = self.conv7_1(feat6_1, style_1)
            feat7_2 = self.conv7_2(feat6_2, style_2)

            feat8_1 = self.conv8_1(feat7_1, style_1)
            feat8_2 = self.conv8_2(feat7_2, style_2)

            out_1 = torch.tanh(self.conv_final_1(feat8_1))
            out_2 = torch.tanh(self.conv_final_2(feat8_2))

            synthesized_slice = torch.cat((out_1, out_2), dim=1)
        else:
            # Moving-contrast output aligned with the fixed anatomy
            synthesized_slice = torch.tanh(self.conv_final(refined_feat))

        if encode_only:
            layers_dict = {0: stem_feat, 1: enc1_feat, 2: enc2_feat, 3: bottleneck_feat,
                            4: dec1_feat, 5: dec2_feat, 6: refined_feat}
            if self.use_separate_style_layers and self.use_triple_outputs:
                layers_dict[7] = torch.cat((feat7_1, feat7_2, feat7_3), dim=1)
                layers_dict[8] = torch.cat((feat8_1, feat8_2, feat8_3), dim=1)
            elif self.use_separate_style_layers and self.use_multiple_outputs:
                layers_dict[7] = torch.cat((feat7_1, feat7_2), dim=1)
                layers_dict[8] = torch.cat((feat8_1, feat8_2), dim=1)
            return [layers_dict[i] for i in layers]

        return synthesized_slice


class ProposedSynthesisModule(MIGSGenerator):
    """Backward-compatible name used by the existing model factory
    (networks_define.define_G) and by type-name checks in BaseModule_AtoB /
    BaseModule_AtoB_BtoA (`type(self.netG_A).__name__ == "ProposedSynthesisModule"`).
    The actual implementation lives in MIGSGenerator above.
    """
    pass


class MIGConv(nn.Module):
    """Moving-Image-Guided Convolution (MIG Conv), corresponding to Eq. (4).

    Spatial operation -> affine-free InstanceNorm -> learnable Gaussian noise
    -> CGM-derived gamma/beta modulation -> LeakyReLU.
    """
    def __init__(self,
                 input_nc,
                 feat_ch,
                 kernel_size,
                 upsample=False,
                 downsample=False,
                 activate=False,
                 blur_kernel=[1, 1.5, 1.5, 1],
                 style_denorm=True,
                 eps=1e-8,
                 ch=1,
                 is_3d=False,
                 noise_independent=False):

        super(MIGConv, self).__init__()
        self.eps = eps
        self.input_nc = input_nc
        self.feat_ch = feat_ch
        self.upsample = upsample
        self.downsample = downsample
        self.activate = activate
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.style_denorm = style_denorm
        self.is_3d = is_3d
        self.noise_independent = noise_independent

        Conv, Norm, dim = get_layer_by_dim(is_3d)

        mode = 'trilinear' if is_3d else 'nearest'

        if self.upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                Conv(input_nc, feat_ch, kernel_size=3, padding=1)
            )

        elif self.downsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, (pad0, pad1))
            self.down = nn.Sequential(
                Conv(input_nc, feat_ch, stride=2, kernel_size=1, padding=0)
                )
        else:
            self.conv = Conv(input_nc, feat_ch, kernel_size=3, padding=1)

        self.normalize = Norm(feat_ch, affine=False)

        nhidden = 512

        self.ch = ch
        # CGM encoder shared by the gamma and beta heads (and per-stream heads
        # for legacy multi/triple-output). Attribute names kept for checkpoint
        # compatibility.
        if ch == 1:
            self.mlp_shared = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = Conv(nhidden, feat_ch, kernel_size=3, padding=1)
            self.mlp_beta = Conv(nhidden, feat_ch, kernel_size=3, padding=1)
        elif ch == 2:
            self.mlp_shared = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)
            self.mlp_beta = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)

            self.mlp_shared_2 = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma_2 = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)
            self.mlp_beta_2 = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)
        elif ch == 3:
            self.mlp_shared = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)
            self.mlp_beta = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)

            self.mlp_shared_2 = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma_2 = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)
            self.mlp_beta_2 = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)

            self.mlp_shared_3 = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma_3 = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)
            self.mlp_beta_3 = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.randomize_noise = True
        # noise_independent=True: separate noise_strength per chunk (independent)
        # noise_independent=False: single shared noise_strength as before (noise injected after concatenation)
        if not self.noise_independent:
            self.noise_strength = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            if ch == 1:
                self.noise_strength = nn.Parameter(torch.zeros(1), requires_grad=True)
            elif ch == 2:
                self.noise_strength_1 = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.noise_strength_2 = nn.Parameter(torch.zeros(1), requires_grad=True)
            elif ch == 3:
                self.noise_strength_1 = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.noise_strength_2 = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.noise_strength_3 = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feature, cgm):
        x = feature
        style = cgm

        if self.downsample:
            original_size = x.size()
            if x.size()[2:] != original_size[2:]:  # If the size has changed
                x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=True)
            x = self.down(x)
        elif self.upsample:
            x = self.up(x)
            original_size = x.size()
            if x.size()[2:] != original_size[2:]:  # If the size has changed
                x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=True)
        else:
            x = self.conv(x)

        if not self.noise_independent:
            # normalize -> cat -> noise all at once
            if style.shape[1] == 1:
                x = self.normalize(x)
            elif style.shape[1] == 2:
                x1, x2 = torch.chunk(x, chunks=2, dim=1)
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x = torch.cat((x1, x2), dim=1)
            elif style.shape[1] == 3:
                x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x3 = self.normalize(x3)
                x = torch.cat((x1, x2, x3), dim=1)

            if self.randomize_noise:
                noise = torch.randn_like(x) * self.noise_strength
            else:
                noise = torch.zeros_like(x) * self.noise_strength

            x = x + noise
        else:
            # normalize -> noise injected per chunk -> cat
            if style.shape[1] == 1:
                x = self.normalize(x)
                if self.randomize_noise:
                    noise = torch.randn_like(x) * self.noise_strength
                else:
                    noise = torch.zeros_like(x) * self.noise_strength
                x = x + noise
            elif style.shape[1] == 2:
                x1, x2 = torch.chunk(x, chunks=2, dim=1)
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                if self.randomize_noise:
                    noise1 = torch.randn_like(x1) * self.noise_strength_1
                    noise2 = torch.randn_like(x2) * self.noise_strength_2
                else:
                    noise1 = torch.zeros_like(x1) * self.noise_strength_1
                    noise2 = torch.zeros_like(x2) * self.noise_strength_2
                x1 = x1 + noise1
                x2 = x2 + noise2
                x = torch.cat((x1, x2), dim=1)
            elif style.shape[1] == 3:
                x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x3 = self.normalize(x3)
                if self.randomize_noise:
                    noise1 = torch.randn_like(x1) * self.noise_strength_1
                    noise2 = torch.randn_like(x2) * self.noise_strength_2
                    noise3 = torch.randn_like(x3) * self.noise_strength_3
                else:
                    noise1 = torch.zeros_like(x1) * self.noise_strength_1
                    noise2 = torch.zeros_like(x2) * self.noise_strength_2
                    noise3 = torch.zeros_like(x3) * self.noise_strength_3
                x1 = x1 + noise1
                x2 = x2 + noise2
                x3 = x3 + noise3
                x = torch.cat((x1, x2, x3), dim=1)

        use_cgm_modulation = self.style_denorm
        if use_cgm_modulation:
            # 1. Interpolate the CGM to match the spatial size of x
            mode = 'trilinear' if self.is_3d else 'nearest'
            style = F.interpolate(style, size=x.size()[2:], mode=mode)
            # 2. Project the CGM to x's channel count via the shared MLP
            if style.shape[1] == 1:
                cgm_features = self.mlp_shared(style)
                gamma = self.mlp_gamma(cgm_features)
                beta = self.mlp_beta(cgm_features)
            elif style.shape[1] == 2:
                actv1 = self.mlp_shared(style[:, :1, :, :])
                gamma = self.mlp_gamma(actv1)
                beta = self.mlp_beta(actv1)
                actv_2 = self.mlp_shared_2(style[:, 1:, :, :])
                gamma_2 = self.mlp_gamma_2(actv_2)
                beta_2 = self.mlp_beta_2(actv_2)
                gamma = torch.cat((gamma, gamma_2), dim=1)
                beta = torch.cat((beta, beta_2), dim=1)
            elif style.shape[1] == 3:
                actv1 = self.mlp_shared(style[:, :1, ...])
                gamma = self.mlp_gamma(actv1)
                beta = self.mlp_beta(actv1)
                actv_2 = self.mlp_shared_2(style[:, 1:2, ...])
                gamma_2 = self.mlp_gamma_2(actv_2)
                beta_2 = self.mlp_beta_2(actv_2)
                actv_3 = self.mlp_shared_3(style[:, 2:3, ...])
                gamma_3 = self.mlp_gamma_3(actv_3)
                beta_3 = self.mlp_beta_3(actv_3)
                gamma = torch.cat((gamma, gamma_2, gamma_3), dim=1)
                beta = torch.cat((beta, beta_2, beta_3), dim=1)

            # 3. CGM-derived affine modulation
            normalized_feature = x
            x = gamma * normalized_feature + beta

            if getattr(self, '_log_style_modulation', False):
                with torch.no_grad():
                    eps = 1e-8
                    dbg = {
                        'gamma_abs': gamma.abs().mean().detach(),
                        'beta_abs': beta.abs().mean().detach(),
                        'gamma_spatial_std': gamma.std(dim=(-2, -1)).mean().detach(),
                        'beta_spatial_std': beta.std(dim=(-2, -1)).mean().detach(),
                        'delta_ratio': ((x - normalized_feature).abs().mean() / (normalized_feature.abs().mean() + eps)).detach(),
                        'x_out_over_x_pre': (x.abs().mean() / (normalized_feature.abs().mean() + eps)).detach(),
                        'style_in_std': style.std().detach(),
                        'style_in_spatial_std': style.std(dim=(-2, -1)).mean().detach(),
                    }
                    self.last_style_debug = dbg

        # activation (LeakyReLU)
        if self.activate:
            x = self.activation(x)
        return x



class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super(Blur, self).__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        out = upfirdn2d(x, self.kernel, padding=self.pad)
        return out.to(orig_dtype)

def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

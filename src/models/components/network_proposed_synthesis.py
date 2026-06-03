import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _safe_entropy(prob, dim, norm_base):
    ent = -(prob * prob.clamp_min(1e-8).log()).sum(dim=dim)
    return ent.mean() / max(math.log(norm_base), 1e-8)


class LocalWindowAttentionConditioner2D(nn.Module):
    """B/C. Local QKV attention conditioner for 2D reference conditioning.

    coarse=False (B): source Q attends to same-resolution ref K,V.
    coarse=True  (C): source Q attends to 4x4 pooled (then nearest-up) ref K,V.
    """

    def __init__(self, dim=16, window=3, coarse=False, residual_scale=0.1):
        super().__init__()
        assert window % 2 == 1
        self.dim = dim
        self.window = window
        self.coarse = coarse
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

    def forward(self, source, ref, out_size):
        src_low = F.interpolate(source, size=out_size, mode='bilinear', align_corners=False)
        ref_low = F.interpolate(ref, size=out_size, mode='bilinear', align_corners=False)

        h, w = out_size
        if self.coarse:
            region_size = (max(1, h // 2), max(1, w // 2))
            ref_base = F.interpolate(
                F.adaptive_avg_pool2d(ref_low, region_size), size=out_size, mode='nearest'
            )
        else:
            ref_base = ref_low

        q = self.q_proj(src_low)
        k = self.k_proj(ref_base)
        v = self.v_proj(ref_base)

        ctx, attn = self._local_attention(q, k, v)
        delta = self.out_proj(ctx)
        style = ref_base + self.residual_scale * delta

        center_idx = (self.window * self.window) // 2
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=1).mean()
        entropy_norm = entropy / math.log(self.window * self.window)

        stats = {
            "base_std": ref_low.std().detach(),
            "ref_base_std": ref_base.std().detach(),
            "style_std": style.std().detach(),
            "style_base_delta": ((style - ref_low).abs().mean() / (ref_low.abs().mean() + 1e-8)).detach(),
            "attn_entropy": entropy_norm.detach(),
            "attn_max": attn.max(dim=1).values.mean().detach(),
            "attn_center_weight": attn[:, center_idx:center_idx + 1].mean().detach(),
        }
        return style, stats


class LocalWindowAttentionConditioner25D(nn.Module):
    """B/C. source(2D) query vs ref_stack(K-slice) local window attention.

    coarse=False (B): same-grid local attention across K slices.
    coarse=True  (C): coarse-region local attention across K slices.
    """

    def __init__(self, dim=16, window=3, coarse=False, residual_scale=0.1, center_slice_bias=0.2,
                 blend_mode='none', use_rel_bias=False, use_multihead=False, use_direct_attn=False,
                 use_qk_norm=False, init_temperature=10.0, direct_alpha=0.5):
        super().__init__()
        assert window % 2 == 1, f"window must be odd, got {window}"
        self.dim = dim
        self.window = window
        self.coarse = coarse
        self.residual_scale = residual_scale
        self.center_slice_bias = center_slice_bias
        _valid_blend = ['none', 'center_base', 'center_ref_low']
        if blend_mode not in _valid_blend:
            raise ValueError(f"Unknown blend_mode={blend_mode!r}. Choose from {_valid_blend}")
        self.blend_mode = blend_mode
        self.use_rel_bias = bool(use_rel_bias)
        self.spatial_rel_bias_strength = 0.05
        self.slice_rel_bias_strength = 0.05
        self.use_multihead = bool(use_multihead)
        self.num_heads = 4 if self.use_multihead else 1
        if dim % self.num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={self.num_heads}")
        self.head_dim = dim // self.num_heads
        # direct: attention weights directly mix raw ref values; no V_proj/out_proj shortcut
        self.use_direct_attn = bool(use_direct_attn)
        self.direct_alpha = float(direct_alpha)  # style = center_ref + alpha*(weighted_ref - center_ref)
        # qk_norm: cosine similarity attention + learnable temperature
        # prevents Q,K → 0 collapse; small init temperature amplifies tiny cosine differences
        self.use_qk_norm = bool(use_qk_norm)
        if self.use_qk_norm:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(float(init_temperature))))

        # Manhattan distance for each position in the local window
        r = window // 2
        dist = [abs(dy) + abs(dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        self.register_buffer(
            "spatial_rel_dist",
            torch.tensor(dist, dtype=torch.float32).view(1, 1, window * window, 1, 1),
        )

        self.q_proj = nn.Conv2d(1, dim, 1)
        self.k_proj = nn.Conv2d(1, dim, 1)
        if not self.use_direct_attn:
            self.v_proj = nn.Conv2d(1, dim, 1)
            self.out_proj = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim, 1, 3, padding=1),
            )
            _zero_init_last_conv(self.out_proj)

    def _unfold_same(self, x, win):
        """Unfold preserving spatial size (odd window, symmetric padding)."""
        return F.unfold(x, kernel_size=win, padding=win // 2)

    def forward(self, source, ref_stack, out_size):
        B, K, _, _ = ref_stack.shape
        h, w = out_size
        center_idx = K // 2
        win = self.window
        win2 = win * win
        pad = win // 2

        src_low = F.interpolate(source, size=(h, w), mode='bilinear', align_corners=False)
        ref_low = F.interpolate(ref_stack, size=(h, w), mode='bilinear', align_corners=False)  # [B,K,h,w]

        if self.coarse:
            region_size = (max(1, h // 2), max(1, w // 2))
            ref_base = F.adaptive_avg_pool2d(ref_low.view(B * K, 1, h, w), region_size)
            ref_base = F.interpolate(ref_base, size=(h, w), mode='nearest').view(B, K, h, w)
        else:
            ref_base = ref_low

        q = self.q_proj(src_low)  # [B,dim,h,w]
        k = self.k_proj(ref_base.reshape(B * K, 1, h, w)).view(B, K, self.dim, h, w)

        # QK normalization: cosine similarity, prevents Q,K → 0 collapse
        if self.use_qk_norm:
            q = F.normalize(q, dim=1, eps=1e-8)  # unit norm per spatial position
            k = F.normalize(k, dim=2, eps=1e-8)

        # k unfold for score computation (always needed)
        k_unfold_flat = self._unfold_same(k.reshape(B * K, self.dim, h, w), win)
        k_unfold = k_unfold_flat.view(B, K, self.dim, win2, h, w)

        # ── score computation (shared for both direct / non-direct) ──────────────────
        if not self.use_multihead:
            if self.use_qk_norm:
                temperature = self.log_temperature.exp()
                score_nobias = (q.unsqueeze(1).unsqueeze(3) * k_unfold).sum(dim=2) * temperature
            else:
                score_nobias = (q.unsqueeze(1).unsqueeze(3) * k_unfold).sum(dim=2) / math.sqrt(self.dim)
            score = score_nobias.clone()
            score[:, center_idx:center_idx + 1] += self.center_slice_bias
            if self.use_rel_bias:
                score = score - self.spatial_rel_bias_strength * self.spatial_rel_dist.to(score.device)
                slice_dist = (torch.arange(K, device=score.device, dtype=score.dtype) - center_idx).abs()
                score = score - self.slice_rel_bias_strength * slice_dist.view(1, K, 1, 1, 1)
            attn = torch.softmax(score.view(B, K * win2, h, w), dim=1).view(B, K, win2, h, w)
            attn_for_log = attn.unsqueeze(1)  # [B,1,K,win2,h,w]
        else:
            Hh, Dh = self.num_heads, self.head_dim
            qh = q.view(B, Hh, Dh, h, w)
            kh = k.view(B, K, Hh, Dh, h, w)
            kh_unfold = self._unfold_same(kh.reshape(B * K * Hh, Dh, h, w), win)
            kh_unfold = kh_unfold.view(B, K, Hh, Dh, win2, h, w)
            if self.use_qk_norm:
                temperature = self.log_temperature.exp()
                score_nobias = (qh.unsqueeze(1).unsqueeze(4) * kh_unfold).sum(dim=3) * temperature
            else:
                score_nobias = (qh.unsqueeze(1).unsqueeze(4) * kh_unfold).sum(dim=3) / math.sqrt(Dh)
            score = score_nobias.clone()
            score[:, center_idx:center_idx + 1] += self.center_slice_bias
            if self.use_rel_bias:
                sp_dist = self.spatial_rel_dist.to(score.device)
                score = score - self.spatial_rel_bias_strength * sp_dist.unsqueeze(2)
                slice_dist = (torch.arange(K, device=score.device, dtype=score.dtype) - center_idx).abs()
                score = score - self.slice_rel_bias_strength * slice_dist.view(1, K, 1, 1, 1, 1)
            score_h = score.permute(0, 2, 1, 3, 4, 5).contiguous()
            attn_h = torch.softmax(score_h.view(B, Hh, K * win2, h, w), dim=2).view(B, Hh, K, win2, h, w)
            attn_for_log = attn_h  # [B,Hh,K,win2,h,w]

        center_ref_low = ref_low[:, center_idx:center_idx + 1]
        center_base = ref_base[:, center_idx:center_idx + 1]

        # ── style generation ─────────────────────────────────────────────
        # attn_mean: [B,K,win2,h,w], averaged over heads
        attn_mean = attn_for_log.mean(dim=1)

        if self.use_direct_attn:
            # Direct: attention weights → raw ref_base values directly
            # No V_proj / out_proj. Attention must learn to select slices/windows.
            ref_base_unfold = self._unfold_same(ref_base.reshape(B * K, 1, h, w), win).view(B, K, win2, h, w)
            weighted_ref = (attn_mean * ref_base_unfold).sum(dim=(1, 2)).unsqueeze(1)  # [B,1,h,w]
            # Fixed anchor: center_ref_low + alpha=0.5
            attn_style = weighted_ref
            blend_anchor = center_ref_low
            blend_alpha = self.direct_alpha
            style = blend_anchor + blend_alpha * (attn_style - blend_anchor)
        else:
            # Original: V_proj → out_proj → delta
            v = self.v_proj(ref_base.reshape(B * K, 1, h, w)).view(B, K, self.dim, h, w)
            if not self.use_multihead:
                v_unfold = self._unfold_same(v.reshape(B * K, self.dim, h, w), win)
                v_unfold = v_unfold.view(B, K, self.dim, win2, h, w)
                ctx = (attn.unsqueeze(2) * v_unfold).sum(dim=1).sum(dim=2)
            else:
                vh = v.view(B, K, Hh, Dh, h, w)
                vh_unfold = self._unfold_same(vh.reshape(B * K * Hh, Dh, h, w), win)
                vh_unfold = vh_unfold.view(B, K, Hh, Dh, win2, h, w)
                vh_unfold_h = vh_unfold.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
                ctx_h = (attn_h.unsqueeze(3) * vh_unfold_h).sum(dim=2).sum(dim=3)
                ctx = ctx_h.reshape(B, self.dim, h, w)
            delta = self.out_proj(ctx)
            attn_style = center_base + self.residual_scale * delta
            if self.blend_mode == 'none':
                blend_anchor, blend_alpha, style = attn_style, 1.0, attn_style
            elif self.blend_mode == 'center_base':
                blend_anchor, blend_alpha = center_base, 0.7
                style = blend_anchor + blend_alpha * (attn_style - blend_anchor)
            elif self.blend_mode == 'center_ref_low':
                blend_anchor, blend_alpha = center_ref_low, 0.5
                style = blend_anchor + blend_alpha * (attn_style - blend_anchor)
            else:
                raise RuntimeError(f"Invalid blend_mode={self.blend_mode}")

        # ── unified logging ───────────────────────────────────────────────
        # (attn_mean already computed above)
        slice_prob = attn_mean.sum(dim=2)                         # [B,K,h,w]
        spatial_center_prob = attn_mean[:, :, win2 // 2].sum(dim=1)  # [B,h,w]

        attn_flat = attn_for_log.view(B, self.num_heads, K * win2, h, w)
        attn_entropy = _safe_entropy(attn_flat, dim=2, norm_base=K * win2)
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

            # head diversity (multi-head only, zeros for single-head)
            head_slice_prob = attn_for_log.sum(dim=3)              # [B,Hh,K,h,w]
            head_center_w = head_slice_prob[:, :, center_idx].mean(dim=(0, 2, 3))  # [Hh]
            head_spatial_center_w = attn_for_log[:, :, :, win2 // 2].sum(dim=2).mean(dim=(0, 2, 3))
            head_center_slice_std = head_center_w.std(correction=0)
            head_spatial_center_std = head_spatial_center_w.std(correction=0)

        # QK diagnostics: if qk_score_std_nobias ≈ 0 → Q/K not contributing
        with torch.no_grad():
            qk_score_std_nobias = score_nobias.std().detach()
            qk_score_std = score.std().detach()
            q_abs = q.abs().mean().detach()
            k_abs = k.abs().mean().detach()
            temperature_val = (self.log_temperature.exp().detach()
                               if self.use_qk_norm else torch.zeros(1, device=style.device))

        stats = {
            "base_std": ref_low.std().detach(),
            "ref_base_std": ref_base.std().detach(),
            "style_std": style.std().detach(),
            "style_center_delta": ((style - center_ref_low).abs().mean() / (center_ref_low.abs().mean() + 1e-8)).detach(),
            "style_base_delta": ((style - center_base).abs().mean() / (center_base.abs().mean() + 1e-8)).detach(),
            "blend_to_attn_delta": ((style - attn_style).abs().mean() / (attn_style.abs().mean() + 1e-8)).detach(),
            "blend_anchor_delta": ((style - blend_anchor).abs().mean() / (blend_anchor.abs().mean() + 1e-8)).detach(),
            "blend_alpha": torch.as_tensor(blend_alpha, device=style.device).detach(),
            "direct_attn_enabled": torch.as_tensor(float(self.use_direct_attn), device=style.device).detach(),
            "qk_norm_enabled": torch.as_tensor(float(self.use_qk_norm), device=style.device).detach(),
            "qk_temperature": temperature_val.squeeze().detach(),
            "qk_score_std_nobias": qk_score_std_nobias,
            "qk_score_std": qk_score_std,
            "q_abs": q_abs,
            "k_abs": k_abs,
            "multihead_enabled": torch.as_tensor(float(self.use_multihead), device=style.device).detach(),
            "num_heads": torch.as_tensor(float(self.num_heads), device=style.device).detach(),
            "attn_entropy": attn_entropy.mean().detach(),
            "attn_max": attn_max.detach(),
            "slice_entropy": _safe_entropy(slice_prob, dim=1, norm_base=K).detach(),
            "center_slice_weight": slice_prob[:, center_idx].mean().detach(),
            "spatial_center_weight": spatial_center_prob.mean().detach(),
            "rel_bias_enabled": torch.as_tensor(float(self.use_rel_bias), device=style.device).detach(),
            "expected_abs_slice_offset": expected_abs_slice_offset.detach(),
            "expected_spatial_l1": expected_spatial_l1.detach(),
            "near_slice_weight": near_slice_weight.detach(),
            "near_spatial_weight": near_spatial_weight.detach(),
            "head_center_slice_std": head_center_slice_std.detach(),
            "head_spatial_center_std": head_spatial_center_std.detach(),
        }
        return style, stats


class SliceConvFusionConditioner25D(nn.Module):
    """Conv counterpart of the attention conditioner (matched baseline).

    Same inputs (source + K ref slices), same role (source-conditioned K-slice
    fusion), same anchor blend (center_ref + 0.5*(fused - center_ref)).
    Only difference vs attention: per-slice weights are predicted by a small
    CONV stack over concat([source, ref_slices]) instead of QK dot-product.
    Isolates "dynamic metric matching (attention)" vs "learned conv predictor".
    """

    def __init__(self, dim=16, coarse=True, k=5, blend_alpha=0.5):
        super().__init__()
        self.coarse = coarse
        self.k = k
        self.blend_alpha = blend_alpha
        # conv weight-predictor: concat(src, K ref) -> per-slice logits (K)
        self.weight_net = nn.Sequential(
            nn.Conv2d(1 + k, dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, k, 3, padding=1),
        )
        _zero_init_last_conv(self.weight_net)  # start uniform (softmax(0)) → mean fusion

    def forward(self, source, ref_stack, out_size):
        B, K, H, W = ref_stack.shape
        h, w = out_size
        center_idx = K // 2

        src_low = F.interpolate(source, size=(h, w), mode='bilinear', align_corners=False)
        ref_low = F.interpolate(ref_stack, size=(h, w), mode='bilinear', align_corners=False)
        if self.coarse:
            region_size = (max(1, h // 2), max(1, w // 2))
            ref_base = F.adaptive_avg_pool2d(ref_low.view(B * K, 1, h, w), region_size)
            ref_base = F.interpolate(ref_base, size=(h, w), mode='nearest').view(B, K, h, w)
        else:
            ref_base = ref_low

        # source-conditioned per-slice weights via conv (vs attention's QK)
        logit = self.weight_net(torch.cat([src_low, ref_base], dim=1))  # [B,K,h,w]
        weight = torch.softmax(logit, dim=1)                            # [B,K,h,w]
        fused = (weight * ref_base).sum(dim=1, keepdim=True)            # [B,1,h,w]

        center_ref = ref_low[:, center_idx:center_idx + 1]
        style = center_ref + self.blend_alpha * (fused - center_ref)

        with torch.no_grad():
            stats = {
                "base_std": ref_low.std().detach(),
                "ref_base_std": ref_base.std().detach(),
                "style_std": style.std().detach(),
                "style_center_delta": ((style - center_ref).abs().mean()
                                       / (center_ref.abs().mean() + 1e-8)).detach(),
                "slice_entropy": _safe_entropy(weight, dim=1, norm_base=K).detach(),
                "center_slice_weight": weight[:, center_idx].mean().detach(),
                "max_slice_weight": weight.max(dim=1).values.mean().detach(),
                "conv_fusion_enabled": torch.as_tensor(1.0, device=style.device),
            }
        return style, stats


class ProposedSynthesisModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.input_nc = kwargs['input_nc']
            self.feat_ch = kwargs['feat_ch']
            self.output_nc = kwargs['output_nc']
            self.demodulate = kwargs['demodulate']
            self.use_multiple_outputs = kwargs.get('use_multiple_outputs', None)
            self.use_triple_outputs = kwargs.get('use_triple_outputs', False)
            self.is_3d = kwargs.get('is_3d', False)
            self.use_separate_style_layers = kwargs.get('use_separate_style_layers', False)
            self.noise_independent = kwargs.get('noise_independent', False)
            self.use_25d_style = kwargs.get('use_25d_style', False)
            self.ref_stack_size = kwargs.get('ref_stack_size', 3)
            self.z_agg_deep = kwargs.get('z_agg_deep', False)
            self.ref_condition_mode = kwargs.get('ref_condition_mode', 'original')
            self.ref_condition_coarse = kwargs.get('ref_condition_coarse', True)
            self.ref_condition_blend = kwargs.get('ref_condition_blend', 'none')
            self.ref_condition_rel_bias = kwargs.get('ref_condition_rel_bias', False)
            self.ref_condition_multihead = kwargs.get('ref_condition_multihead', False)
            self.ref_condition_window = kwargs.get('ref_condition_window', 3)
            self.ref_condition_dim = kwargs.get('ref_condition_dim', 16)
            self.ref_condition_center_bias = kwargs.get('ref_condition_center_bias', 0.2)
            self.ref_condition_direct = kwargs.get('ref_condition_direct', False)
            self.ref_condition_qk_norm = kwargs.get('ref_condition_qk_norm', False)
            self.ref_condition_init_temperature = kwargs.get('ref_condition_init_temperature', 10.0)
            self.ref_condition_direct_alpha = kwargs.get('ref_condition_direct_alpha', 0.5)
            self.ref_condition_downsample = kwargs.get('ref_condition_downsample', 16)

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        _valid_ref_condition_modes = ['original', 'C_coarse_attn', 'conv_fusion']
        if self.ref_condition_mode not in _valid_ref_condition_modes:
            raise ValueError(
                f"Unknown ref_condition_mode={self.ref_condition_mode!r}. "
                f"Choose from {_valid_ref_condition_modes}"
            )
        if self.ref_condition_window not in [1, 3, 5]:
            raise ValueError(f"ref_condition_window must be 1, 3 or 5, got {self.ref_condition_window}")

        if self.is_3d and self.ref_condition_mode != 'original':
            raise NotImplementedError("ref_condition_mode C_coarse_attn은 2D 전용입니다.")
        if (self.use_multiple_outputs or self.use_triple_outputs) and self.ref_condition_mode != 'original':
            raise NotImplementedError("ref_condition_mode C_coarse_attn은 단일 출력 모드에서만 지원됩니다.")

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

        # 2.5D: compress ref stack [B, K, H, W] -> [B, 1, H, W] per style stream
        if self.use_25d_style:
            hidden = max(8, self.feat_ch // 8)
            if self.z_agg_deep:
                def _make_z_agg():
                    return nn.Sequential(
                        nn.Conv2d(self.ref_stack_size, hidden, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
                    )
            else:
                def _make_z_agg():
                    return nn.Sequential(
                        nn.Conv2d(self.ref_stack_size, hidden, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
                    )
            self.z_aggs = nn.ModuleList([_make_z_agg() for _ in range(self.num_style_streams)])

        # 2D conditioners (always built when mode != original)
        # 2.5D conditioners (built when use_25d_style=True and ref_stack_size > 1)
        self.ref_conditioner_2d = None
        self.ref_conditioner_25d = None

        _cd = self.ref_condition_dim  # shorthand
        _cb = self.ref_condition_center_bias  # shorthand
        _w2d = self.ref_condition_window  # window is always odd (1/3/5)
        _coarse = self.ref_condition_coarse  # True: region-pooled K,V (coarse) | False: full-res
        if self.ref_condition_mode == 'C_coarse_attn':
            self.ref_conditioner_2d = LocalWindowAttentionConditioner2D(
                dim=_cd, window=_w2d, coarse=_coarse, residual_scale=0.1,
            )
            if self.use_25d_style and self.ref_stack_size >= 1:
                self.ref_conditioner_25d = LocalWindowAttentionConditioner25D(
                    dim=_cd, window=self.ref_condition_window, coarse=_coarse, residual_scale=0.1,
                    center_slice_bias=_cb, blend_mode=self.ref_condition_blend,
                    use_rel_bias=self.ref_condition_rel_bias,
                    use_multihead=self.ref_condition_multihead,
                    use_direct_attn=self.ref_condition_direct,
                    use_qk_norm=self.ref_condition_qk_norm,
                    init_temperature=self.ref_condition_init_temperature,
                    direct_alpha=self.ref_condition_direct_alpha,
                )
        elif self.ref_condition_mode == 'conv_fusion':
            # matched conv baseline (only 2.5D; uses K-slice). coarse/dim shared with C.
            if self.use_25d_style and self.ref_stack_size >= 1:
                self.ref_conditioner_25d = SliceConvFusionConditioner25D(
                    dim=_cd, coarse=_coarse, k=self.ref_stack_size,
                )

        self._last_ref_condition_stats: dict = {}

        self.guide_net = nn.Sequential(
            nn.Conv2d(self.input_nc, int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 일부분에만 style_denorm -> 21, 22, 31, 32에만 적용 #TODO: feat_ch에 ref ch만큼 배수로
        self.conv0 = StyleConv(self.input_nc, self.feat_ch * ch, kernel_size=3,
                                                 activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv11 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv12 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv21 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv22 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv31 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv32 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv41 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, #feat_ch *4는 변치않게
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv42 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv51 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv52 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv6 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, ) # 이걸 빠트렸었어.
                                # activate=False, demodulate=self.demodulate)

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 style 적용
        if self.use_separate_style_layers and self.use_triple_outputs:
            # 각 branch는 feat_ch 채널 (전체의 1/3)
            self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv7_3 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_3 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            # 각각 독립적인 conv_final
            self.conv_final_1 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_3 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
        elif self.use_separate_style_layers and self.use_multiple_outputs:
            # 각 branch는 feat_ch 채널 (전체의 절반)
            self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            # 각각 독립적인 conv_final
            self.conv_final_1 = Conv(self.feat_ch, self.output_nc // 2, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, self.output_nc // 2, kernel_size=3, padding=1)
        
        ## Added for checkerboard artifact
        # if self.use_multiple_outputs:
        #     self.conv7_1 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
        #     self.conv7_2 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
        # else:
        #     self.conv7 = nn.Conv2d(self.feat_ch * ch, self.feat_ch * ch // 2, kernel_size=3, padding=1)

        # if self.use_multiple_outputs:
        #     self.conv8_1 = nn.Conv2d(self.feat_ch * ch // 4, self.feat_ch * ch // 8, kernel_size=3, padding=1)
        #     self.conv8_2 = nn.Conv2d(self.feat_ch * ch // 4, self.feat_ch * ch // 8, kernel_size=3, padding=1)
        # else:
        #     self.conv8 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
                                       
        # if self.use_multiple_outputs:
        #     self.conv_final_1 = nn.Conv2d(self.feat_ch * ch // 8, self.output_nc // 2, kernel_size=3, padding=1)
        #     self.conv_final_2 = nn.Conv2d(self.feat_ch * ch // 8, self.output_nc // 2, kernel_size=3, padding=1)
        # else:
        #     self.conv_final = nn.Conv2d(self.feat_ch * ch // 4, self.output_nc, kernel_size=3, padding=1)
        self.conv_final = Conv(self.feat_ch * ch, self.output_nc, kernel_size=3, padding=1)

    def _aggregate_ref_stack(self, ref_all):
        """2.5D: [B, num_streams*K, H, W] -> [B, num_streams, H, W]"""
        if not self.use_25d_style:
            return ref_all
        chunks = torch.split(ref_all, self.ref_stack_size, dim=1)
        style_maps = [z_agg(chunk) for chunk, z_agg in zip(chunks, self.z_aggs)]
        return torch.cat(style_maps, dim=1)

    def _style_size(self, source):
        H, W = source.shape[-2:]
        ds = self.ref_condition_downsample
        return max(1, H // ds), max(1, W // ds)

    def _get_ref_stack(self, ref_all):
        """Return ref tensor for conditioners.
        2D  : [B,1,H,W]
        2.5D: [B,K,H,W]
        """
        if self.use_25d_style and self.ref_stack_size >= 1:
            return ref_all[:, :self.ref_stack_size]
        return ref_all[:, :1]

    def _make_ref_condition(self, source, ref_all, encode_only=False):
        """Returns (base_style, aux_style), both [B,1,h,w].

        original        : z_agg(2.5D) / nearest-downsampled(2D) ref as style
        C_coarse_attn   : local-window attention conditioner replaces style
                          (coarse=True: region-pooled K,V | coarse=False: full-res)
        aux_style is always None (kept for forward signature compatibility).
        """
        h, w = self._style_size(source)

        # ── 2.5D path ──────────────────────────────────────────────────────
        if self.use_25d_style and self.ref_stack_size >= 1:
            if self.ref_condition_mode == 'original':
                base_ref = self._aggregate_ref_stack(ref_all)        # [B,1,H,W] via z_agg
                return F.interpolate(base_ref, size=(h, w), mode='nearest'), None

            ref_stack = self._get_ref_stack(ref_all)                 # [B,K,H,W]
            cond_style, stats = self.ref_conditioner_25d(
                source=source, ref_stack=ref_stack, out_size=(h, w)
            )
            if not encode_only:
                self._last_ref_condition_stats = stats
            return cond_style, None

        # ── 2D path ────────────────────────────────────────────────────────
        ref_map = self._get_ref_stack(ref_all)                       # [B,1,H,W]
        if self.ref_condition_mode == 'original':
            return F.interpolate(ref_map, size=(h, w), mode='nearest'), None

        cond_style, stats = self.ref_conditioner_2d(
            source=source, ref=ref_map, out_size=(h, w)
        )
        if not encode_only:
            self._last_ref_condition_stats = stats
        return cond_style, None

    def forward(self, merged_input, layers=[], encode_only=False):

        x = merged_input[:, :1, ...]
        ref_all = merged_input[:, 1:, ...]

        if self.is_3d:
            # 3D path: original logic unchanged
            ref = self._aggregate_ref_stack(ref_all)
            ref = ref.permute(0, 1, 4, 2, 3)
            style_guidance_1 = F.interpolate(ref, scale_factor=1/16, mode='trilinear', align_corners=False)
            style_guidance_1 = style_guidance_1.permute(0, 1, 3, 4, 2)
            aux_style = None
        else:
            style_guidance_1, aux_style = self._make_ref_condition(
                source=x, ref_all=ref_all, encode_only=encode_only
            )

        feats = []
        feat0 = self.conv0(x, style_guidance_1, aux_style=aux_style)
        feat1 = self.conv11(feat0, style_guidance_1, aux_style=aux_style)
        feat1 = self.conv12(feat1, style_guidance_1, aux_style=aux_style)
        feat2 = self.conv21(feat1, style_guidance_1, aux_style=aux_style)
        feat2 = self.conv22(feat2, style_guidance_1, aux_style=aux_style)
        feat3 = self.conv31(feat2, style_guidance_1, aux_style=aux_style)
        feat3 = self.conv32(feat3, style_guidance_1, aux_style=aux_style)
        feat4 = self.conv41(feat3 + feat2, style_guidance_1, aux_style=aux_style)
        feat4 = self.conv42(feat4, style_guidance_1, aux_style=aux_style)
        feat5 = self.conv51(feat4 + feat1, style_guidance_1, aux_style=aux_style)
        feat5 = self.conv52(feat5, style_guidance_1, aux_style=aux_style)
        feat6 = self.conv6(feat5 + feat0, style_guidance_1, aux_style=aux_style)

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 처리
        if self.use_separate_style_layers and self.use_triple_outputs:
            # feat6를 3등분으로 분리
            feat6_1, feat6_2, feat6_3 = torch.chunk(feat6, chunks=3, dim=1)  # 각 [B, feat_ch, H, W]
            # style도 분리
            style_1 = style_guidance_1[:, :1, ...]  # [B, 1, ...]
            style_2 = style_guidance_1[:, 1:2, ...]  # [B, 1, ...]
            style_3 = style_guidance_1[:, 2:3, ...]  # [B, 1, ...]

            # conv7: 각각 독립적으로 처리
            feat7_1 = self.conv7_1(feat6_1, style_1)  # [B, feat_ch, H, W]
            feat7_2 = self.conv7_2(feat6_2, style_2)  # [B, feat_ch, H, W]
            feat7_3 = self.conv7_3(feat6_3, style_3)  # [B, feat_ch, H, W]

            # conv8: 각각 독립적으로 처리
            feat8_1 = self.conv8_1(feat7_1, style_1)  # [B, feat_ch, H, W]
            feat8_2 = self.conv8_2(feat7_2, style_2)  # [B, feat_ch, H, W]
            feat8_3 = self.conv8_3(feat7_3, style_3)  # [B, feat_ch, H, W]

            # 각각 conv_final + tanh
            out_1 = torch.tanh(self.conv_final_1(feat8_1))  # [B, output_nc//3, H, W]
            out_2 = torch.tanh(self.conv_final_2(feat8_2))  # [B, output_nc//3, H, W]
            out_3 = torch.tanh(self.conv_final_3(feat8_3))  # [B, output_nc//3, H, W]

            # 마지막에 합쳐서 반환
            out = torch.cat((out_1, out_2, out_3), dim=1)  # [B, output_nc, H, W]
        elif self.use_separate_style_layers and self.use_multiple_outputs:
            # feat6를 반으로 분리
            feat6_1, feat6_2 = torch.chunk(feat6, chunks=2, dim=1)  # 각 [B, feat_ch, H, W]
            # style도 분리
            style_1 = style_guidance_1[:, :1, ...]  # [B, 1, ...]
            style_2 = style_guidance_1[:, 1:, ...]  # [B, 1, ...]

            # conv7: 각각 독립적으로 처리
            feat7_1 = self.conv7_1(feat6_1, style_1)  # [B, feat_ch, H, W]
            feat7_2 = self.conv7_2(feat6_2, style_2)  # [B, feat_ch, H, W]

            # conv8: 각각 독립적으로 처리
            feat8_1 = self.conv8_1(feat7_1, style_1)  # [B, feat_ch, H, W]
            feat8_2 = self.conv8_2(feat7_2, style_2)  # [B, feat_ch, H, W]

            # 각각 conv_final + tanh
            out_1 = torch.tanh(self.conv_final_1(feat8_1))  # [B, output_nc//2, H, W]
            out_2 = torch.tanh(self.conv_final_2(feat8_2))  # [B, output_nc//2, H, W]

            # 마지막에 합쳐서 반환
            out = torch.cat((out_1, out_2), dim=1)  # [B, output_nc, H, W]
        else:
            out = self.conv_final(feat6)
            out = torch.tanh(out)

        if encode_only:
            layers_dict = {0: feat0, 1: feat1, 2: feat2, 3: feat3, 4: feat4, 5: feat5, 6: feat6}
            if self.use_separate_style_layers and self.use_triple_outputs:
                layers_dict[7] = torch.cat((feat7_1, feat7_2, feat7_3), dim=1)
                layers_dict[8] = torch.cat((feat8_1, feat8_2, feat8_3), dim=1)
            elif self.use_separate_style_layers and self.use_multiple_outputs:
                layers_dict[7] = torch.cat((feat7_1, feat7_2), dim=1)
                layers_dict[8] = torch.cat((feat8_1, feat8_2), dim=1)
            return [layers_dict[i] for i in layers]

        return out


class StyleConv(nn.Module):
    def __init__(self,
                 input_nc,
                 feat_ch,
                 kernel_size,
                 upsample=False,
                 downsample=False,
                 activate=False,
                 blur_kernel=[1, 1.5, 1.5, 1],
                 demodulate=True,
                 style_denorm=True,
                 eps=1e-8,
                 ch=1,
                 is_3d=False,
                 noise_independent=False):

        super(StyleConv, self).__init__()
        self.eps = eps
        self.input_nc = input_nc
        self.feat_ch = feat_ch
        self.demodulate = demodulate
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
            
        # self.batch_norm = nn.BatchNorm2d(feat_ch, affine=False)
        self.normalize = Norm(feat_ch, affine=False)
        # self.batch_norm = nn.SyncBatchNorm(feat_ch, affine=False)

        nhidden = 512

        self.ch = ch
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
        # noise_independent=True: chunk별로 개별 noise_strength (독립적)
        # noise_independent=False: 기존처럼 하나의 noise_strength (cat된 상태에서 noise 주입)
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

    def forward(self, x, style, aux_style=None):

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
            # 기존 방식: normalize -> cat -> noise 한꺼번에
            if style.shape[1] == 1:
                x = self.normalize(x)
            elif style.shape[1] == 2:
                x1, x2 = torch.chunk(x, chunks=2, dim=1)  # 각 부분 [B, C/2, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x = torch.cat((x1, x2), dim=1)
            elif style.shape[1] == 3:
                x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)  # 각 부분 [B, C/3, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x3 = self.normalize(x3)
                x = torch.cat((x1, x2, x3), dim=1)

            # Add noise
            if self.randomize_noise:
                noise = torch.randn_like(x) * self.noise_strength
            else:
                noise = torch.zeros_like(x) * self.noise_strength

            x = x + noise
        else:
            # 새로운 방식: normalize -> noise 개별 주입 -> cat
            if style.shape[1] == 1:
                x = self.normalize(x)
                if self.randomize_noise:
                    noise = torch.randn_like(x) * self.noise_strength
                else:
                    noise = torch.zeros_like(x) * self.noise_strength
                x = x + noise
            elif style.shape[1] == 2:
                x1, x2 = torch.chunk(x, chunks=2, dim=1)  # 각 부분 [B, C/2, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                # noise 개별 주입
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
                x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)  # 각 부분 [B, C/3, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x3 = self.normalize(x3)
                # noise 개별 주입
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
        if self.style_denorm:
            # 1. style을 x의 크기와 맞게 interpolation 
            mode = 'trilinear' if self.is_3d else 'nearest'
            style = F.interpolate(style, size=x.size()[2:], mode=mode) # TODO: bilinear 도 시도
            # 2. style을 x의 C과 맞게 mlp
            if style.shape[1] == 1:
                actv = self.mlp_shared(style)
                gamma = self.mlp_gamma(actv)
                beta = self.mlp_beta(actv)
            elif style.shape[1] == 2:
                actv1 = self.mlp_shared(style[:, :1, :, :])
                gamma = self.mlp_gamma(actv1) # B, 256, f_H, f_W
                beta = self.mlp_beta(actv1) # B, 256, f_H, f_W
                actv_2 = self.mlp_shared_2(style[:, 1:, :, :])
                gamma_2 = self.mlp_gamma_2(actv_2)
                beta_2 = self.mlp_beta_2(actv_2)
                gamma = torch.cat((gamma, gamma_2), dim=1) # B, 512, f_H, f_W
                beta = torch.cat((beta, beta_2), dim=1)
            elif style.shape[1] == 3:
                actv1 = self.mlp_shared(style[:, :1, ...])
                gamma = self.mlp_gamma(actv1)  # B, feat_ch//3, f_H, f_W
                beta = self.mlp_beta(actv1)
                actv_2 = self.mlp_shared_2(style[:, 1:2, ...])
                gamma_2 = self.mlp_gamma_2(actv_2)
                beta_2 = self.mlp_beta_2(actv_2)
                actv_3 = self.mlp_shared_3(style[:, 2:3, ...])
                gamma_3 = self.mlp_gamma_3(actv_3)
                beta_3 = self.mlp_beta_3(actv_3)
                gamma = torch.cat((gamma, gamma_2, gamma_3), dim=1)  # B, feat_ch, f_H, f_W
                beta = torch.cat((beta, beta_2, beta_3), dim=1)

            # 3. affine modulation
            x_pre = x
            x = x * gamma + beta

            # D. coarse attention residual branch
            if aux_style is not None:
                aux_style_interp = F.interpolate(aux_style, size=x.size()[2:], mode=mode)
                aux_actv = self.mlp_shared(aux_style_interp)
                aux_beta = self.mlp_beta(aux_actv)
                aux_alpha = 0.2
                x = x + aux_alpha * aux_beta

            if getattr(self, '_log_style_modulation', False):
                with torch.no_grad():
                    eps = 1e-8
                    dbg = {
                        'gamma_abs': gamma.abs().mean().detach(),
                        'beta_abs': beta.abs().mean().detach(),
                        'gamma_spatial_std': gamma.std(dim=(-2, -1)).mean().detach(),
                        'beta_spatial_std': beta.std(dim=(-2, -1)).mean().detach(),
                        'delta_ratio': ((x - x_pre).abs().mean() / (x_pre.abs().mean() + eps)).detach(),
                        'x_out_over_x_pre': (x.abs().mean() / (x_pre.abs().mean() + eps)).detach(),
                        'style_in_std': style.std().detach(),
                        'style_in_spatial_std': style.std(dim=(-2, -1)).mean().detach(),
                    }
                    if aux_style is not None:
                        dbg.update({
                            'aux_style_std': aux_style.std().detach(),
                            'aux_beta_abs': aux_beta.abs().mean().detach(),
                            'aux_beta_spatial_std': aux_beta.std(dim=(-2, -1)).mean().detach(),
                            'aux_applied_over_x_pre': (
                                (aux_alpha * aux_beta).abs().mean() / (x_pre.abs().mean() + eps)
                            ).detach(),
                        })
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
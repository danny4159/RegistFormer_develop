"""Tokenized Slice-Window guidance encoder for the pure-Transformer MIGS generator.

This is the Transformer-native counterpart of `SliceWindowAttention` (SWA). The
*principle* is unchanged -- a fixed-image query picks, among a local window of
candidates across the K moving slices, which moving content corresponds to it --
but everything happens on patch tokens instead of downsampled pixels, and the
Q/K projections are per-patch linear maps instead of convolutions.

Design point that matters (raw-appearance preservation)
-------------------------------------------------------
The original SWA applies the attention weights directly to *raw* moving
intensities, which is what makes the CGM interpretable ("the moving contrast
that corresponds to this fixed location") and what stops the guidance from
degenerating into a re-encoding of the fixed anatomy. That property is kept:

    Q, K  <- learned patch tokens        (correspondence *selection*)
    V     <- raw moving patch intensity  (what is actually transported)

A learned V branch is only *added* on top when ``guidance_dim > 1``, and it is
concatenated with the raw-appearance channel rather than replacing it.

Outputs a guidance pyramid so each generator stage gets guidance at its own
resolution, instead of one map resized everywhere.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer2D(nn.Module):
    """Non-overlapping patch -> token projection (kernel == stride).

    Mathematically a per-patch shared Linear, i.e. the ViT/Swin patch embedding;
    it never mixes information across patch boundaries.
    """

    def __init__(self, in_ch, dim, patch_size):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)     # [B,h,w,dim]
        return self.norm(x)


class TokenizedSliceWindowAttention(nn.Module):
    """Slice-neighborhood cross-attention on patch tokens -> guidance tokens.

    Each fixed token attends over ``K * window**2`` candidates (K moving slices x
    a local window of moving tokens) and aggregates the raw moving appearance of
    the selected candidates.

    Returns ``guidance`` as ``[B, guidance_dim, h, w]`` (BCHW so it drops straight
    into the existing modulation heads) plus the SWA-style diagnostic stats dict.
    """

    def __init__(self, ref_stack_size=3, patch_size=4, token_dim=96, qk_dim=32,
                 window_size=3, guidance_dim=1, center_slice_bias=0.0,
                 temperature_init=10.0, learnable_temperature=True,
                 use_cosine_similarity=True):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be odd, got {window_size}")
        self.K = int(ref_stack_size)
        self.patch_size = int(patch_size)
        self.window = int(window_size)
        self.guidance_dim = int(guidance_dim)
        self.qk_dim = int(qk_dim)
        self.center_slice_bias = float(center_slice_bias)
        self.use_qk_norm = bool(use_cosine_similarity)

        # shared tokenizer for fixed and moving slices (same appearance space)
        self.tokenizer = PatchTokenizer2D(1, token_dim, self.patch_size)
        self.q_proj = nn.Linear(token_dim, self.qk_dim)
        self.k_proj = nn.Linear(token_dim, self.qk_dim)

        if self.use_qk_norm:
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(float(temperature_init))),
                requires_grad=bool(learnable_temperature),
            )

        # raw-appearance value: mean intensity of each moving patch (NOT learned),
        # so the transported quantity stays the actual moving contrast
        self.raw_value_dim = 1
        if self.guidance_dim > 1:
            # learned value branch, concatenated with (never replacing) the raw one
            self.v_proj = nn.Linear(token_dim, self.guidance_dim - self.raw_value_dim)
            self.out_proj = nn.Linear(self.guidance_dim, self.guidance_dim)

        self.last_attn_entropy = None
        r = self.window // 2
        dist = [abs(dy) + abs(dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        self.register_buffer(
            "spatial_rel_dist",
            torch.tensor(dist, dtype=torch.float32).view(1, 1, self.window ** 2, 1, 1),
            persistent=False,
        )

    def reset_special_parameters(self):
        # tokenizer reads 1 channel; init_net's normal(0, 0.02) is far too small there
        nn.init.kaiming_normal_(self.tokenizer.proj.weight, mode='fan_in', nonlinearity='linear')
        if self.tokenizer.proj.bias is not None:
            nn.init.zeros_(self.tokenizer.proj.bias)

    def _neighborhoods(self, x, win):
        """[N,C,h,w] -> [N,C,win*win,h,w] via symmetric-padded unfold."""
        N, C, h, w = x.shape
        return F.unfold(x, kernel_size=win, padding=win // 2).view(N, C, win * win, h, w)

    def forward(self, fixed_slice, moving_stack):
        B, K, H, W = moving_stack.shape
        if K != self.K:
            raise ValueError(f"expected ref_stack_size={self.K} moving slices, got {K}")
        win, win2 = self.window, self.window ** 2
        center_idx = K // 2

        # pad so the patch grid is exact
        p = self.patch_size
        pad_b, pad_r = (p - H % p) % p, (p - W % p) % p
        if pad_b or pad_r:
            fixed_slice = F.pad(fixed_slice, (0, pad_r, 0, pad_b), mode='replicate')
            moving_stack = F.pad(moving_stack, (0, pad_r, 0, pad_b), mode='replicate')
        Hp, Wp = fixed_slice.shape[-2:]
        h, w = Hp // p, Wp // p

        # ---- tokens -------------------------------------------------------
        fixed_tok = self.tokenizer(fixed_slice)                              # [B,h,w,D]
        moving_tok = self.tokenizer(
            moving_stack.reshape(B * K, 1, Hp, Wp)
        ).view(B, K, h, w, -1)                                               # [B,K,h,w,D]

        query = self.q_proj(fixed_tok)                                       # [B,h,w,d]
        keys = self.k_proj(moving_tok)                                       # [B,K,h,w,d]

        if self.use_qk_norm:
            query = F.normalize(query, dim=-1, eps=1e-8)
            keys = F.normalize(keys, dim=-1, eps=1e-8)

        # ---- raw moving appearance per patch (mean pooling, not learned) ---
        raw_value = F.avg_pool2d(moving_stack, kernel_size=p, stride=p)      # [B,K,h,w]
        raw_value = raw_value.unsqueeze(2)                                   # [B,K,1,h,w]

        # ---- slice-neighborhood candidates --------------------------------
        d = self.qk_dim
        keys_bchw = keys.permute(0, 1, 4, 2, 3).reshape(B * K, d, h, w)
        local_keys = self._neighborhoods(keys_bchw, win).view(B, K, d, win2, h, w)

        q = query.permute(0, 3, 1, 2).unsqueeze(1).unsqueeze(3)              # [B,1,d,1,h,w]
        if self.use_qk_norm:
            logits = (q * local_keys).sum(dim=2) * self.log_temperature.exp()
        else:
            logits = (q * local_keys).sum(dim=2) / math.sqrt(d)              # [B,K,win2,h,w]

        logits = logits.clone()
        logits[:, center_idx:center_idx + 1] += self.center_slice_bias
        attn = torch.softmax(logits.reshape(B, K * win2, h, w), dim=1).view(B, K, win2, h, w)

        # ---- aggregate: raw appearance first -------------------------------
        raw_cand = self._neighborhoods(
            raw_value.reshape(B * K, 1, h, w), win
        ).view(B, K, 1, win2, h, w)
        guidance_raw = (attn.unsqueeze(2) * raw_cand).sum(dim=(1, 3))        # [B,1,h,w]

        if self.guidance_dim > 1:
            v = self.v_proj(moving_tok)                                      # [B,K,h,w,Cv]
            cv = v.shape[-1]
            v_bchw = v.permute(0, 1, 4, 2, 3).reshape(B * K, cv, h, w)
            v_cand = self._neighborhoods(v_bchw, win).view(B, K, cv, win2, h, w)
            guidance_learned = (attn.unsqueeze(2) * v_cand).sum(dim=(1, 3))  # [B,Cv,h,w]
            guidance = torch.cat([guidance_raw, guidance_learned], dim=1)
            guidance = self.out_proj(
                guidance.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2).contiguous()
        else:
            guidance = guidance_raw

        # ---- diagnostics (mirrors the pixel-SWA stats keys) ----------------
        eps = 1e-8
        slice_prob = attn.sum(dim=2)                                          # [B,K,h,w]
        flat = attn.reshape(B, K * win2, h, w)
        entropy = -(flat * flat.clamp_min(1e-8).log()).sum(dim=1).mean() / math.log(K * win2)
        self.last_attn_entropy = entropy

        center_moving = raw_value[:, center_idx]                              # [B,1,h,w]
        with torch.no_grad():
            offsets = torch.arange(K, device=attn.device, dtype=attn.dtype) - float(center_idx)
            exp_slice = (slice_prob * offsets.abs().view(1, K, 1, 1)).sum(dim=1).mean()
            sp = self.spatial_rel_dist.to(attn.device)
            exp_spatial = (attn * sp).sum(dim=(1, 2)).mean()

        stats = {
            "token_swa_enabled": torch.as_tensor(1.0, device=guidance.device),
            "guidance_dim": torch.as_tensor(float(self.guidance_dim), device=guidance.device),
            "base_std": raw_value.std().detach(),
            "style_std": guidance.std().detach(),
            "style_center_delta": (
                (guidance_raw - center_moving).abs().mean() / (center_moving.abs().mean() + eps)
            ).detach(),
            "attn_entropy": entropy.detach(),
            "attn_max": flat.max(dim=1).values.mean().detach(),
            "center_slice_weight": slice_prob[:, center_idx].mean().detach(),
            "expected_abs_slice_offset": exp_slice.detach(),
            "expected_spatial_l1": exp_spatial.detach(),
            "qk_temperature": (
                self.log_temperature.exp().detach() if self.use_qk_norm
                else torch.zeros((), device=guidance.device)
            ),
        }
        return guidance, stats


class GuidancePyramid(nn.Module):
    """Patch-merging hierarchy over guidance tokens: G0 -> G1 -> G2 ...

    Mirrors the generator's own Patch Merging so each stage can be conditioned by
    guidance at its own resolution, rather than resizing one map everywhere.
    Level ``l`` has ``dim * 2**l`` channels, matching the doc's G0/G1/G2 widths.
    """

    def __init__(self, dim, num_levels=3):
        super().__init__()
        self.num_levels = int(num_levels)
        self.dims = [dim * (2 ** i) for i in range(self.num_levels)]
        self.merges = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.merges.append(nn.Sequential(
                nn.LayerNorm(4 * self.dims[i]),
                nn.Linear(4 * self.dims[i], self.dims[i + 1], bias=False),
            ))

    def forward(self, g0):
        """g0: [B,dim,h,w] -> list of [B,dim_l,h/2^l,w/2^l]."""
        levels = [g0]
        cur = g0
        for merge in self.merges:
            x = cur.permute(0, 2, 3, 1)                       # BHWC
            B, H, W, C = x.shape
            if H % 2 or W % 2:
                x = F.pad(x.permute(0, 3, 1, 2), (0, W % 2, 0, H % 2)).permute(0, 2, 3, 1)
            x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                           x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], dim=-1)
            cur = merge(x).permute(0, 3, 1, 2).contiguous()   # BCHW
            levels.append(cur)
        return levels

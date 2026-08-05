"""CGM-conditioned local-attention blocks for the MIGS generator.

These blocks are the Transformer half of the CNN+Transformer hybrid backbone
(`backbone_type: local_attention` in the generator config). They keep exactly the
same contract as MIGConv -- ``forward(feature_BCHW, cgm) -> feature_BCHW`` at an
unchanged resolution and channel count -- so the surrounding U-Net skips,
`encode_only` feature taps and PatchNCE stay untouched.

Two local-attention flavours are selectable per stage:
  * ``swin``   -- non-overlapping W-MSA windows, alternating with shifted SW-MSA.
  * ``natten`` -- sliding neighbourhood attention (needs the `natten` package).

Every block is CGM-conditioned: the Contrast Guidance Map produced by
Slice-Window Attention is resampled to the block resolution and turned into a
spatial (gamma, beta) pair that modulates the LayerNorm output, mirroring the
CGM modulation inside MIGConv (Eq. 4).

Tensor-layout convention inside this file: attention operates in ``[B, H, W, C]``
(channels-last), which is what both the Swin reference implementation and the
NATTEN kernels expect. Conversion to/from ``[B, C, H, W]`` happens only at the
block boundary.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth on the residual branch (per-sample)."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:g}"


def _trunc_normal_(tensor, std=0.02):
    with torch.no_grad():
        tensor.normal_(0.0, std).clamp_(-2 * std, 2 * std)
    return tensor


# ---------------------------------------------------------------------------
# CGM-conditioned normalization
# ---------------------------------------------------------------------------
class CGMAdaptiveLayerNorm2D(nn.Module):
    """LayerNorm over channels, followed by CGM-derived spatial affine modulation.

    ``x`` is channels-last ``[B, H, W, C]``; ``cgm`` is ``[B, 1, h, w]`` (the
    coarse Contrast Guidance Map). The CGM is resampled to ``(H, W)`` with the
    same 'nearest' mode MIGConv uses, then projected to per-pixel gamma/beta::

        out = LayerNorm(x) * (1 + gamma) + beta

    The last projection is zero-initialized (see :meth:`reset_special_parameters`)
    so the block starts as a plain pre-LN Transformer and learns the CGM
    conditioning from there.
    """

    def __init__(self, dim, cgm_channels=1, cgm_hidden=64, eps=1e-6, kernel_size=3,
                 zero_init=True):
        """kernel_size=3 mixes CGM neighbours (hybrid backbone, matching MIGConv);
        kernel_size=1 is a per-pixel linear map, used by the convolution-free
        backbone so no spatial kernel enters the network.

        zero_init=True gives adaLN-zero (DiT style): the block starts as a plain
        pre-LN Transformer and the CGM has no effect until it is learned. That is
        fine for the hybrid backbone, where MIGConv already injects the CGM with
        randomly initialized gamma/beta.

        zero_init=False must be used when AdaLN is the ONLY CGM path (the
        convolution-free backbone). With zero_init there, the generator would
        ignore the moving image entirely at initialization -- gradients still
        flow, but the model starts blind to the guidance it is built around.
        """
        super().__init__()
        self.dim = dim
        self.zero_init = bool(zero_init)
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.cgm_encoder = nn.Sequential(
            nn.Conv2d(cgm_channels, cgm_hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        )
        self.cgm_affine = nn.Conv2d(cgm_hidden, 2 * dim, kernel_size=kernel_size, padding=pad)
        self.reset_special_parameters()

    def reset_special_parameters(self):
        """(Re-)init the modulation head.

        Called again after `networks_define.init_net`, which would otherwise
        overwrite this with normal(0, 0.02) like every other conv.
        """
        # The CGM encoder reads a SINGLE channel, so init_net's blanket
        # normal(0, 0.02) is far too small for it (fan_in is 1 or 9, not hundreds)
        # and would squash the guidance signal by ~50x before it reaches gamma.
        # Re-init it at the proper fan-in scale.
        enc_conv = self.cgm_encoder[0]
        nn.init.kaiming_normal_(enc_conv.weight, mode='fan_in', nonlinearity='relu')
        if enc_conv.bias is not None:
            nn.init.zeros_(enc_conv.bias)

        if self.zero_init:
            nn.init.zeros_(self.cgm_affine.weight)      # identity: out = LayerNorm(x)
        else:
            # non-zero so the CGM modulates from step 0
            _trunc_normal_(self.cgm_affine.weight, std=0.02)
        if self.cgm_affine.bias is not None:
            nn.init.zeros_(self.cgm_affine.bias)        # beta starts at 0 either way

    def forward(self, x, cgm):
        B, H, W, C = x.shape
        normalized = self.norm(x)

        cgm_resized = F.interpolate(cgm, size=(H, W), mode='nearest')
        affine = self.cgm_affine(self.cgm_encoder(cgm_resized))     # [B, 2C, H, W]
        gamma, beta = torch.chunk(affine, 2, dim=1)
        gamma = gamma.permute(0, 2, 3, 1)                            # [B, H, W, C]
        beta = beta.permute(0, 2, 3, 1)
        return normalized * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# feed-forward
# ---------------------------------------------------------------------------
class ConvFFN2D(nn.Module):
    """Channels-last FFN with a depthwise 3x3 in the middle (local inductive bias)."""

    def __init__(self, dim, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden = max(1, int(round(dim * mlp_ratio)))
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        # [B,H,W,C] -> [B,C,H,W] -> ... -> [B,H,W,C].
        # .contiguous() keeps the convs in plain NCHW: without it the permuted
        # view is channels-last, and the resulting channels-last weight grads
        # trip DDP's "grad strides do not match bucket view strides" warning.
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.dwconv(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.permute(0, 2, 3, 1)


class MlpFFN2D(nn.Module):
    """Standard Transformer FFN: two per-token linear layers, no spatial kernel.

    Used by the convolution-free backbone, where all spatial mixing must come
    from attention alone.
    """

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = max(1, int(round(dim * mlp_ratio)))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        # channels-last throughout: [B,H,W,C] -> [B,H,W,C]
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


# ---------------------------------------------------------------------------
# Swin window attention
# ---------------------------------------------------------------------------
def window_partition(x, window_size):
    """[B,H,W,C] -> [B*num_windows, window_size**2, C]."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, C)


def window_reverse(windows, window_size, H, W):
    """[B*num_windows, window_size**2, C] -> [B,H,W,C]."""
    C = windows.shape[-1]
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, C)


class WindowSelfAttention2D(nn.Module):
    """Multi-head self-attention inside a window, with relative position bias."""

    def __init__(self, dim, num_heads, window_size, attn_drop=0.0, proj_drop=0.0, qkv_bias=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        )).flatten(1)                                            # [2, ws*ws]
        rel = coords[:, :, None] - coords[:, None, :]            # [2, N, N]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)
        self.reset_special_parameters()

    def reset_special_parameters(self):
        _trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """x: [B_, N, C] where N == window_size**2. mask: [num_windows, N, N] or None."""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # [B_, heads, N, head_dim]

        attn = (q * self.scale) @ k.transpose(-2, -1)             # [B_, heads, N, N]

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


class SwinLocalAttention2D(nn.Module):
    """(Shifted) window attention on a ``[B,H,W,C]`` feature map.

    Handles cyclic shift, the shift attention mask and right/bottom padding when
    the resolution is not a multiple of ``window_size``.
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                f"shift_size ({self.shift_size}) must be in [0, window_size={self.window_size})"
            )
        self.attn = WindowSelfAttention2D(
            dim=dim, num_heads=num_heads, window_size=self.window_size,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self._mask_cache = {}

    def _build_attn_mask(self, H, W, device, dtype):
        """Standard Swin shift mask; cached per (H, W, device)."""
        key = (H, W, str(device), str(dtype))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        ws, ss = self.window_size, self.shift_size
        img_mask = torch.zeros((1, H, W, 1), device=device, dtype=dtype)
        slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for hs in slices:
            for wr in slices:
                img_mask[:, hs, wr, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        self._mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x):
        B, H, W, C = x.shape
        ws = self.window_size

        # pad to a multiple of the window size (right / bottom, as in Swin)
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        if pad_r or pad_b:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1).contiguous()
        Hp, Wp = x.shape[1], x.shape[2]

        # a single window covers the whole map -> shifting is a no-op
        shift = self.shift_size if (Hp > ws and Wp > ws) else 0

        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
            attn_mask = self._build_attn_mask(Hp, Wp, x.device, x.dtype)
        else:
            attn_mask = None

        windows = window_partition(x, ws)
        windows = self.attn(windows, mask=attn_mask)
        x = window_reverse(windows, ws, Hp, Wp)

        if shift > 0:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))

        if pad_r or pad_b:
            x = x[:, :H, :W, :].contiguous()
        return x


# ---------------------------------------------------------------------------
# NATTEN neighborhood attention
# ---------------------------------------------------------------------------
class NattenLocalAttention2D(nn.Module):
    """Thin wrapper around ``natten.NeighborhoodAttention2D`` (channels-last in/out).

    NATTEN is imported lazily so environments without the package can still use
    ``local_attention_type: swin``.
    """

    def __init__(self, dim, num_heads, kernel_size=7, dilation=1,
                 attn_drop=0.0, proj_drop=0.0, rel_pos_bias=True):
        super().__init__()
        try:
            from natten import NeighborhoodAttention2D
        except ImportError as e:                                   # pragma: no cover
            raise ImportError(
                "local_attention_type='natten' requires the `natten` package. "
                "Install a wheel matching this environment's torch/CUDA build "
                "(see https://shi-labs.com/natten), or use "
                "local_attention_type='swin'."
            ) from e

        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)

        kwargs = dict(
            num_heads=num_heads,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rel_pos_bias=rel_pos_bias,
        )
        # NATTEN renamed the first positional arg (dim -> embed_dim) around 0.20.
        try:
            self.attn = NeighborhoodAttention2D(dim=dim, **kwargs)
        except TypeError:
            self.attn = NeighborhoodAttention2D(embed_dim=dim, **kwargs)

    def forward(self, x):
        B, H, W, C = x.shape
        span = self.kernel_size * self.dilation
        if H < span or W < span:
            raise ValueError(
                f"NATTEN needs H,W >= kernel_size*dilation ({span}), got ({H},{W}). "
                f"Reduce natten_kernel_size or natten_stage_dilations for this stage."
            )
        return self.attn(x)


# ---------------------------------------------------------------------------
# block / stage
# ---------------------------------------------------------------------------
class MIGLocalTransformerBlock2D(nn.Module):
    """Pre-LN Transformer block with CGM-conditioned norms and local attention.

        x = x + LayerScale * DropPath(LocalAttention(CGMNorm1(x, cgm)))
        x = x + LayerScale * DropPath(ConvFFN(CGMNorm2(x, cgm)))
    """

    def __init__(self, dim, num_heads, attention_type='swin',
                 window_size=8, shift_size=0,
                 natten_kernel_size=7, natten_dilation=1,
                 mlp_ratio=2.0, cgm_channels=1, cgm_hidden=64,
                 drop=0.0, attn_drop=0.0, drop_path_prob=0.0,
                 layer_scale_init=1e-4, ffn_type='conv', cgm_kernel_size=3,
                 cgm_zero_init=True):
        super().__init__()
        self.attention_type = attention_type

        self.norm1 = CGMAdaptiveLayerNorm2D(dim, cgm_channels=cgm_channels,
                                            cgm_hidden=cgm_hidden, kernel_size=cgm_kernel_size,
                                            zero_init=cgm_zero_init)
        self.norm2 = CGMAdaptiveLayerNorm2D(dim, cgm_channels=cgm_channels,
                                            cgm_hidden=cgm_hidden, kernel_size=cgm_kernel_size,
                                            zero_init=cgm_zero_init)

        if attention_type == 'swin':
            self.attn = SwinLocalAttention2D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=shift_size, attn_drop=attn_drop, proj_drop=drop,
            )
        elif attention_type == 'natten':
            self.attn = NattenLocalAttention2D(
                dim=dim, num_heads=num_heads, kernel_size=natten_kernel_size,
                dilation=natten_dilation, attn_drop=attn_drop, proj_drop=drop,
            )
        else:
            raise ValueError(
                f"Unknown local_attention_type '{attention_type}' (expected 'swin' or 'natten')"
            )

        if ffn_type == 'conv':
            self.ffn = ConvFFN2D(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        elif ffn_type == 'mlp':
            self.ffn = MlpFFN2D(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        else:
            raise ValueError(f"ffn_type must be 'conv' or 'mlp', got '{ffn_type}'")

        self.drop_path1 = DropPath(drop_path_prob)
        self.drop_path2 = DropPath(drop_path_prob)

        self.use_layer_scale = layer_scale_init is not None and layer_scale_init > 0
        if self.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(torch.full((dim,), float(layer_scale_init)))
            self.layer_scale_2 = nn.Parameter(torch.full((dim,), float(layer_scale_init)))

    def forward(self, feature, cgm):
        """feature: [B,C,H,W] -> [B,C,H,W] (same shape); cgm: [B,1,h,w]."""
        x = feature.permute(0, 2, 3, 1).contiguous()              # -> [B,H,W,C]

        residual = self.attn(self.norm1(x, cgm))
        if self.use_layer_scale:
            residual = residual * self.layer_scale_1
        x = x + self.drop_path1(residual)

        residual = self.ffn(self.norm2(x, cgm))
        if self.use_layer_scale:
            residual = residual * self.layer_scale_2
        x = x + self.drop_path2(residual)

        return x.permute(0, 3, 1, 2).contiguous()                 # -> [B,C,H,W]


class MIGLocalTransformerStage2D(nn.Module):
    """A stack of :class:`MIGLocalTransformerBlock2D` at one resolution.

    For ``swin`` the shift alternates 0 / window_size//2 across the stack, so a
    stage needs depth >= 2 to contain a full W-MSA + SW-MSA pair.
    """

    def __init__(self, dim, depth, num_heads, attention_type='swin',
                 window_size=8, natten_kernel_size=7, natten_dilation=1,
                 mlp_ratio=2.0, cgm_channels=1, cgm_hidden=64,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, layer_scale_init=1e-4,
                 ffn_type='conv', cgm_kernel_size=3, cgm_zero_init=True):
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError(f"local stage depth must be >= 1, got {depth}")

        def _is_seq(v):
            # omegaconf ListConfig is iterable but not a list/tuple instance
            return hasattr(v, '__iter__') and not isinstance(v, (str, bytes))

        # natten_dilation may be a scalar (all blocks) or a per-block sequence
        if _is_seq(natten_dilation):
            given = [int(d) for d in natten_dilation]
            if not given:
                raise ValueError("natten_dilation sequence must not be empty")
            dilations = [given[i % len(given)] for i in range(depth)]
        else:
            dilations = [int(natten_dilation)] * depth

        if _is_seq(drop_path):
            given = [float(p) for p in drop_path]
            if not given:
                raise ValueError("drop_path sequence must not be empty")
            drop_paths = (given + [given[-1]] * depth)[:depth]
        else:
            drop_paths = [float(drop_path)] * depth

        self.blocks = nn.ModuleList([
            MIGLocalTransformerBlock2D(
                dim=dim,
                num_heads=num_heads,
                attention_type=attention_type,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                natten_kernel_size=natten_kernel_size,
                natten_dilation=dilations[i],
                mlp_ratio=mlp_ratio,
                cgm_channels=cgm_channels,
                cgm_hidden=cgm_hidden,
                drop=drop,
                attn_drop=attn_drop,
                drop_path_prob=drop_paths[i],
                layer_scale_init=layer_scale_init,
                ffn_type=ffn_type,
                cgm_kernel_size=cgm_kernel_size,
                cgm_zero_init=cgm_zero_init,
            )
            for i in range(depth)
        ])

    def forward(self, feature, cgm):
        for block in self.blocks:
            feature = block(feature, cgm)
        return feature

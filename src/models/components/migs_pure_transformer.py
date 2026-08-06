"""Convolution-free (pure Transformer) backbone for the MIGS generator.

Selected with ``backbone_type: pure_transformer``. Unlike the CNN and hybrid
backbones there is **no MIGConv anywhere**: resolution changes are done with
Patch Embedding / Patch Merging / Patch Expanding, and every same-resolution
block is a CGM-conditioned local-attention Transformer block (Swin or NATTEN),
reusing :class:`MIGLocalTransformerStage2D`.

Guidance is unchanged: Slice-Window Attention still turns
(fixed_slice, moving_stack) into the Contrast Guidance Map, and the CGM
conditions every Transformer block through the adaptive LayerNorm. Only
``fixed_slice`` enters the backbone itself.

U-shape (patch_size p, embed dim C; shown for a 128x128 input with p=2)::

    fixed_slice [B,1,128,128]
      | PatchEmbed(p)                      -> stem       [B, C,64,64]
      | enc1 stage                         -> enc1       [B, C,64,64]
      | PatchMerging                        \
      | enc2 stage                         -> enc2       [B,2C,32,32]
      | PatchMerging                        \
      | bottleneck stage                   -> bottleneck [B,4C,16,16]
      | PatchExpanding + (skip enc2)        /
      | dec1 stage                         -> dec1       [B,2C,32,32]
      | PatchExpanding + (skip enc1)        /
      | dec2 stage                         -> dec2       [B, C,64,64]
      | (skip stem) + FinalPatchExpanding(p)
      | refine stage                       -> refined    [B, C,128,128]

Skip connections stay element-wise additive, matching the CNN backbone
(``conv41(bottleneck + enc2)``, ``conv51(dec1 + enc1)``, ``conv6(dec2 + stem)``);
the addition simply happens after the matching Patch Expanding so the two
operands agree in resolution and channel count.

PatchNCE: the taps now have different channel counts per stage (C / 2C / 4C),
while this repo's ``PatchSampleF`` shares ONE ``nn.Linear(input_nc, nc)`` across
all layers. Per-tap 1x1 projection heads therefore map every tap to
``nce_proj_dim`` (264 by default) so ``netF_A.input_nc`` stays unchanged. The
heads run only on the ``encode_only`` path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.migs_local_attention import MIGLocalTransformerStage2D


class PatchEmbed2D(nn.Module):
    """Non-overlapping patch embedding: [B,in_nc,H,W] -> [B,embed_dim,H/p,W/p]."""

    def __init__(self, in_nc, embed_dim, patch_size):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_nc, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                                   # [B,C,H/p,W/p]
        x = x.permute(0, 2, 3, 1)                          # -> BHWC for the channel norm
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()          # -> BCHW


class PatchMerging2D(nn.Module):
    """Swin patch merging: concat each 2x2 neighbourhood, then project.

    [B,C,H,W] -> [B,out_dim,H/2,W/2] (odd sizes are bottom/right padded).
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)                          # -> BHWC
        B, H, W, C = x.shape
        if H % 2 or W % 2:
            x = F.pad(x.permute(0, 3, 1, 2), (0, W % 2, 0, H % 2))
            x = x.permute(0, 2, 3, 1)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)            # [B,H/2,W/2,4C]
        x = self.reduction(self.norm(x))
        return x.permute(0, 3, 1, 2).contiguous()          # -> BCHW


class PatchExpanding2D(nn.Module):
    """Swin-Unet style patch expanding via pixel shuffle.

    [B,in_dim,H,W] -> [B,out_dim,H*scale,W*scale]. A linear layer produces
    ``scale**2 * out_dim`` channels which pixel shuffle rearranges into space,
    so upsampling carries no convolution.
    """

    def __init__(self, in_dim, out_dim, scale=2):
        super().__init__()
        self.scale = int(scale)
        self.out_dim = int(out_dim)
        self.expand = nn.Linear(in_dim, (self.scale ** 2) * self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)                          # -> BHWC
        x = self.expand(x)                                 # [B,H,W,s^2*out]
        x = x.permute(0, 3, 1, 2)                          # -> [B,s^2*out,H,W]
        x = F.pixel_shuffle(x, self.scale)                 # -> [B,out,H*s,W*s]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()          # -> BCHW


class MIGSPureTransformerBackbone(nn.Module):
    """Convolution-free U-shaped local-attention backbone, CGM-conditioned.

    ``forward`` returns the seven feature taps in the same order and with the
    same semantics as the CNN backbone:
    ``[stem, enc1, enc2, bottleneck, dec1, dec2, refined]``.
    """

    # channel multiplier per stage: [enc1, enc2, bottleneck, dec1, dec2, refine]
    STAGE_MULTIPLIERS = (1, 2, 4, 2, 1, 1)

    # which guidance pyramid level conditions each stage
    # [enc1, enc2, bottleneck, dec1, dec2, refine]
    STAGE_GUIDANCE_LEVEL = (0, 1, 2, 1, 0, 0)

    def __init__(self, in_nc, embed_dim, patch_size, stage_depths, stage_num_heads,
                 attention_type='swin', window_size=8, natten_kernel_size=7,
                 natten_stage_dilations=(1, 1, 1, 1, 1, 1), mlp_ratio=4.0,
                 cgm_channels=1, cgm_hidden=64, drop=0.0, attn_drop=0.0,
                 drop_path=0.1, layer_scale_init=1e-4,
                 modulation='cgm_norm', gate_zero_init=True,
                 guidance_channels_per_stage=None):
        super().__init__()
        if len(stage_depths) != 6:
            raise ValueError(f"stage_depths needs 6 entries, got {len(stage_depths)}")
        if len(stage_num_heads) != 6:
            raise ValueError(f"stage_num_heads needs 6 entries, got {len(stage_num_heads)}")

        C = int(embed_dim)
        self.embed_dim = C
        self.patch_size = int(patch_size)
        # total stride from input to the bottleneck: patch embed + 2 patch merges
        self.size_divisor = self.patch_size * 4

        dims = [C * m for m in self.STAGE_MULTIPLIERS]      # [C, 2C, 4C, 2C, C, C]
        self.stage_dims = dims
        for i, (d, h) in enumerate(zip(dims, stage_num_heads)):
            if d % int(h) != 0:
                raise ValueError(
                    f"stage {i}: dim {d} is not divisible by num_heads {h}. "
                    f"Adjust pure_embed_dim or pure_num_heads."
                )

        self.patch_embed = PatchEmbed2D(in_nc, C, self.patch_size)

        # guidance channels may differ per stage when a guidance pyramid is used
        if guidance_channels_per_stage is None:
            guidance_channels_per_stage = [cgm_channels] * 6
        elif len(guidance_channels_per_stage) != 6:
            raise ValueError(
                f"guidance_channels_per_stage needs 6 entries, got {guidance_channels_per_stage}"
            )
        self.guidance_channels_per_stage = list(guidance_channels_per_stage)

        def _stage(index, dim):
            return MIGLocalTransformerStage2D(
                dim=dim,
                depth=stage_depths[index],
                num_heads=int(stage_num_heads[index]),
                attention_type=attention_type,
                window_size=window_size,
                natten_kernel_size=natten_kernel_size,
                natten_dilation=natten_stage_dilations[index],
                mlp_ratio=mlp_ratio,
                cgm_channels=self.guidance_channels_per_stage[index],
                cgm_hidden=cgm_hidden,
                modulation=modulation,
                gate_zero_init=gate_zero_init,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                layer_scale_init=layer_scale_init,
                # convolution-free: plain Linear FFN and a per-pixel (1x1) CGM
                # head, so attention is the ONLY source of spatial mixing
                ffn_type='mlp',
                cgm_kernel_size=1,
                # AdaLN is the ONLY CGM path here (no MIGConv), so it must not be
                # zero-initialized -- otherwise the generator starts completely
                # blind to the moving image.
                cgm_zero_init=False,
            )

        # encoder
        self.enc1_stage = _stage(0, dims[0])               # C  @ H/p
        self.down1 = PatchMerging2D(dims[0], dims[1])      # C  -> 2C, /2
        self.enc2_stage = _stage(1, dims[1])               # 2C @ H/2p
        self.down2 = PatchMerging2D(dims[1], dims[2])      # 2C -> 4C, /2
        # bottleneck
        self.bottleneck_stage = _stage(2, dims[2])         # 4C @ H/4p
        # decoder
        self.up1 = PatchExpanding2D(dims[2], dims[3], scale=2)   # 4C -> 2C, x2
        self.dec1_stage = _stage(3, dims[3])               # 2C @ H/2p
        self.up2 = PatchExpanding2D(dims[3], dims[4], scale=2)   # 2C -> C,  x2
        self.dec2_stage = _stage(4, dims[4])               # C  @ H/p
        # back to full resolution
        self.final_expand = PatchExpanding2D(dims[4], dims[5], scale=self.patch_size)
        self.refine_stage = _stage(5, dims[5])             # C  @ H

        # tap dims in tap order: [stem, enc1, enc2, bottleneck, dec1, dec2, refined]
        self.tap_dims = [C, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]]

    def _stage_guidance(self, guidance):
        """Accept a single map or a guidance pyramid; return one entry per stage."""
        if torch.is_tensor(guidance):
            return [guidance] * 6
        levels = list(guidance)
        return [levels[min(l, len(levels) - 1)] for l in self.STAGE_GUIDANCE_LEVEL]

    def forward(self, fixed_slice, cgm):
        g = self._stage_guidance(cgm)
        B, _, H, W = fixed_slice.shape

        # pad so every merge/expand is exact; the output is cropped back at the end
        div = self.size_divisor
        pad_b = (div - H % div) % div
        pad_r = (div - W % div) % div
        if pad_b or pad_r:
            fixed_slice = F.pad(fixed_slice, (0, pad_r, 0, pad_b), mode='replicate')

        stem_feat = self.patch_embed(fixed_slice)                         # [B,C,H/p,W/p]

        enc1_feat = self.enc1_stage(stem_feat, g[0])                      # [B,C,H/p,W/p]
        enc2_feat = self.enc2_stage(self.down1(enc1_feat), g[1])          # [B,2C,H/2p,W/2p]
        bottleneck_feat = self.bottleneck_stage(self.down2(enc2_feat), g[2])  # [B,4C,H/4p,W/4p]

        # additive skips, mirroring the CNN backbone
        dec1_feat = self.dec1_stage(self.up1(bottleneck_feat) + enc2_feat, g[3])  # [B,2C,H/2p,..]
        dec2_feat = self.dec2_stage(self.up2(dec1_feat) + enc1_feat, g[4])        # [B,C,H/p,..]
        refined_feat = self.refine_stage(self.final_expand(dec2_feat + stem_feat), g[5])  # [B,C,H,W]

        if pad_b or pad_r:
            refined_feat = refined_feat[:, :, :H, :W].contiguous()

        return [stem_feat, enc1_feat, enc2_feat, bottleneck_feat,
                dec1_feat, dec2_feat, refined_feat]

import torch.nn as nn
from torch.nn import init
import numpy as np
import torch


class PatchSampleF(nn.Module):
    def __init__(self, **kwargs):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        try:
            self.use_mlp = kwargs['use_mlp']
            self.init_type = kwargs['init_type']
            self.init_gain = kwargs['init_gain']
            self.nc = kwargs['nc']
            self.input_nc = kwargs['input_nc']
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        self.l2norm = Normalize(2)
        self.mlp_init = False

        if self.use_mlp:
            # Keep compatibility with configs that still pass input_nc, but build
            # per-layer MLPs lazily from the actual feature channel count.
            self.mlps = nn.ModuleDict()

    def create_mlp(self, feats):
        for feat_id, feat in enumerate(feats):
            key = str(feat_id)
            input_nc = feat.shape[1]
            if key in self.mlps:
                continue

            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc),
                nn.ReLU(),
                nn.Linear(self.nc, self.nc)
            )
            mlp.to(feat.device)
            self.mlps[key] = mlp
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):  # B, C, H, W
        return_ids = []
        return_feats = []
        if self.use_mlp:
            self.create_mlp(feats)

        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # B, H*W, C
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                patch_id = torch.as_tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []

            if self.use_mlp:
                mlp = self.mlps[str(feat_id)]
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

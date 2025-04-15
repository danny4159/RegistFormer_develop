import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_gaussian_filter_3d(sigma, sz):
    coords = torch.arange(sz)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    mid = sz // 2
    d = (x - mid) ** 2 + (y - mid) ** 2 + (z - mid) ** 2
    gauss = torch.exp(-d / (2 * sigma**2)) / ((2 * np.pi * sigma**2) ** 1.5)
    return gauss


def gaussian_filter_3d(img, n, sigma):
    """
    img: image tensor of size (1, 1, D, H, W)
    n: size of the Gaussian filter (n x n x n)
    """
    gauss_kernel = get_gaussian_filter_3d(sigma, n).to(img.device)
    gauss_kernel = gauss_kernel.view(1, 1, n, n, n)
    filtered_img = F.conv3d(img, gauss_kernel, padding=n // 2)
    return filtered_img


def Dp_3d(image, sigma, patch_size, xshift, yshift, zshift):
    shifted = torch.roll(image, shifts=(zshift, yshift, xshift), dims=(-3, -2, -1))
    diff = image - shifted
    diff_sq = diff ** 2
    return gaussian_filter_3d(diff_sq, patch_size, sigma)


def mind_3d(image, sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7):
    D, H, W = image.shape[-3:]
    # reduce_size = (patch_size + neigh_size - 2) // 2
    reduce_size = min((patch_size + neigh_size - 2) // 2, D//2 - 1, H//2 - 1, W//2 - 1) # Solve Nan error in small size of depth


    # Local variance
    Vimg = (
        Dp_3d(image, sigma, patch_size, -1, 0, 0) +
        Dp_3d(image, sigma, patch_size, 1, 0, 0) +
        Dp_3d(image, sigma, patch_size, 0, -1, 0) +
        Dp_3d(image, sigma, patch_size, 0, 1, 0) +
        Dp_3d(image, sigma, patch_size, 0, 0, -1) +
        Dp_3d(image, sigma, patch_size, 0, 0, 1)
    ) / 6 + eps

    Vimg = Vimg + eps + 1e-6

    if torch.isnan(Vimg).any():
        print("🚨 Vimg contains NaN")

    shift_range = np.arange(-neigh_size // 2, neigh_size - neigh_size // 2)
    iter_pos = 0

    for x in shift_range:
        for y in shift_range:
            for z in shift_range:
                if (x, y, z) == (0, 0, 0):
                    continue
                # MIND_tmp = torch.exp(-Dp_3d(image, sigma, patch_size, x, y, z) / Vimg)
                MIND_tmp = torch.exp(torch.clamp(-Dp_3d(image, sigma, patch_size, x, y, z) / Vimg, min=-50))
                if torch.isnan(MIND_tmp).any():
                    print(f"🚨 exp result NaN at shift ({x},{y},{z})")
                tmp = MIND_tmp[..., reduce_size:-reduce_size, reduce_size:-reduce_size, reduce_size:-reduce_size, None]
                if torch.isnan(tmp).any():
                    print(f"🚨 tmp NaN after slicing at shift ({x},{y},{z})")
                output = tmp if iter_pos == 0 else torch.cat((output, tmp), dim=-1)
                iter_pos += 1

    # output = output / torch.max(output, dim=-1, keepdim=True)[0]
    output = output / (torch.max(output, dim=-1, keepdim=True)[0] + 1e-8)
    if torch.isnan(output).any():
        print("🚨 final output contains NaN")
    return output


class MINDLoss3D(nn.Module):
    def __init__(self, sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7):
        super(MINDLoss3D, self).__init__()
        self.sigma = sigma
        self.eps = eps
        self.neigh_size = neigh_size
        self.patch_size = patch_size

    def forward(self, pred, gt):
        pred_mind = mind_3d(pred, self.sigma, self.eps, self.neigh_size, self.patch_size)
        gt_mind = mind_3d(gt, self.sigma, self.eps, self.neigh_size, self.patch_size)
        loss = F.l1_loss(pred_mind, gt_mind)
        if torch.isnan(loss).any():
            print("🚨 NaN 발생! loss:", loss.item())
        return loss

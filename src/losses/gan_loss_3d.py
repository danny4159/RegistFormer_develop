import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GANLoss3D(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0, reduction='mean'):
        super(GANLoss3D, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss(reduction=reduction)
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        elif self.gan_type == 'swd':
            self.loss = self._slicedWassersteinDistance_loss
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_gaussian_kernel_3d(self, device="cpu"):
        kernel = torch.tensor([[[1, 4, 6, 4, 1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4, 6, 4, 1]]], dtype=torch.float32)
        kernel = kernel.repeat(5, 1, 1) / 256.0
        kernel = kernel.reshape(1, 1, 5, 5, 5).to(device)
        return kernel

    def pyramid_down(self, volume, device="cpu"):
        gaussian_k = self.get_gaussian_kernel_3d(device=device)
        multiband = [F.conv3d(volume[:, i:i + 1], gaussian_k, padding=2, stride=2) for i in range(volume.size(1))]
        return torch.cat(multiband, dim=1)

    def pyramid_up(self, volume, device="cpu"):
        gaussian_k = self.get_gaussian_kernel_3d(device=device)
        upsample = F.interpolate(volume, scale_factor=2, mode='trilinear', align_corners=False)
        multiband = [F.conv3d(upsample[:, i:i + 1], gaussian_k, padding=2) for i in range(volume.size(1))]
        return torch.cat(multiband, dim=1)

    def gaussian_pyramid(self, original, n_pyramids, device="cpu"):
        x = original
        pyramids = [original]
        for _ in range(n_pyramids):
            x = self.pyramid_down(x, device=device)
            pyramids.append(x)
        return pyramids

    def laplacian_pyramid(self, original, n_pyramids, device="cpu"):
        pyramids = self.gaussian_pyramid(original, n_pyramids, device=device)
        laplacian = []
        for i in range(len(pyramids) - 1):
            diff = pyramids[i] - self.pyramid_up(pyramids[i + 1], device=device)
            laplacian.append(diff)
        laplacian.append(pyramids[-1])
        return laplacian

    def minibatch_laplacian_pyramid(self, volume, n_pyramids, batch_size, device="cpu"):
        n = volume.size(0) // batch_size + int(volume.size(0) % batch_size != 0)
        pyramids = []
        for i in range(n):
            x = volume[i * batch_size:(i + 1) * batch_size]
            p = self.laplacian_pyramid(x.to(device), n_pyramids, device=device)
            p = [x.cpu() for x in p]
            pyramids.append(p)

        result = []
        for i in range(n_pyramids + 1):
            x = [pyramids[j][i] for j in range(n)]
            result.append(torch.cat(x, dim=0))
        return result

    def extract_patches_3d(self, volume_layer, slice_indices, slice_size=5):
        B, C, D, H, W = volume_layer.shape
        patches = []

        for b in range(B):
            for idx in slice_indices:
                z = idx // ((H - slice_size + 1) * (W - slice_size + 1))
                rem = idx % ((H - slice_size + 1) * (W - slice_size + 1))
                y = rem // (W - slice_size + 1)
                x = rem % (W - slice_size + 1)

                patch = volume_layer[b:b+1, :, z:z+slice_size, y:y+slice_size, x:x+slice_size]
                patches.append(patch)

        x = torch.cat(patches, dim=0)
        x = x.reshape(x.size(0), -1)
        return x

    def _slicedWassersteinDistance_loss(self, input, target, n_pyramids=None, slice_size=5, n_descriptors=64,
                                        n_repeat_projection=64, proj_per_repeat=4, return_by_resolution=False,
                                        pyramid_batchsize=64):

        device = input.device

        assert input.size() == target.size()
        assert input.ndim == 5 and target.ndim == 5

        if n_pyramids is None:
            denom = max(input.size(2) // 16, 1)
            n_pyramids = int(np.rint(np.log2(denom)))

        pyramid1 = self.minibatch_laplacian_pyramid(input, n_pyramids, pyramid_batchsize, device=device)
        pyramid2 = self.minibatch_laplacian_pyramid(target, n_pyramids, pyramid_batchsize, device=device)

        result = []

        for i_pyramid in range(n_pyramids + 1):
            n = (pyramid1[i_pyramid].size(2) - slice_size + 1) * \
                (pyramid1[i_pyramid].size(3) - slice_size + 1) * \
                (pyramid1[i_pyramid].size(4) - slice_size + 1)
            indices = torch.randperm(n)[:n_descriptors]

            p1 = self.extract_patches_3d(pyramid1[i_pyramid], indices, slice_size=slice_size).to(device)
            p2 = self.extract_patches_3d(pyramid2[i_pyramid], indices, slice_size=slice_size).to(device)

            distances = []
            for _ in range(n_repeat_projection):
                rand = torch.randn(p1.size(1), proj_per_repeat).to(device)
                rand = rand / torch.std(rand, dim=0, keepdim=True)

                proj1 = torch.matmul(p1, rand)
                proj2 = torch.matmul(p2, rand)
                proj1, _ = torch.sort(proj1, dim=0)
                proj2, _ = torch.sort(proj2, dim=0)
                distances.append(torch.mean(torch.abs(proj1 - proj2)))

            result.append(torch.mean(torch.stack(distances)))

        result = torch.stack(result) * 1e3
        return result if return_by_resolution else torch.mean(result)

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        target_label = self.get_target_label(input, target_is_real)

        if self.gan_type == 'swd' and not is_disc:
            swd_loss = self.loss(input, target_label)
            return swd_loss * self.loss_weight

        if self.gan_type == 'hinge':
            if is_disc:
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:
                loss = -input.mean()
        else:
            loss = self.loss(input, target_label)

        return loss if is_disc else loss * self.loss_weight

import torch
import torch.nn as nn


class MutualInformation(nn.Module):
    def __init__(self, num_bins=21, sigma_ratio=0.5, eps=1.19209e-07):
        super(MutualInformation, self).__init__()
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.eps = eps

        # bin_centers는 forward에서 t1의 device에 맞게 생성되도록 처리
        self.register_buffer("bin_centers", torch.linspace(0.0, 1.0, num_bins))

    def forward(self, t1, t2):
        bin_centers = self.bin_centers.to(t1.device)
        sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * self.sigma_ratio
        preterm = 1 / (2 * sigma**2)

        # 입력 크기에 따른 shape 변환
        if len(t1.shape) == 3:
            w, h, z = t1.shape
            c, batch = 1, 1
        elif len(t1.shape) == 4:
            c, w, h, z = t1.shape
            batch = 1
        elif len(t1.shape) == 5:
            batch, c, w, h, z = t1.shape
        else:
            raise NotImplementedError(f"Unsupported input shape: {t1.shape}")

        t1 = t1.reshape(batch, c * w * h * z, 1)
        t2 = t2.reshape(batch, c * w * h * z, 1)
        nb_voxels = t1.shape[1] * 1.0

        vbc = bin_centers.view(1, 1, -1)

        I_a = torch.exp(-preterm * (t1 - vbc) ** 2)
        I_a = I_a / (torch.sum(I_a, dim=-1, keepdim=True) + self.eps)
        I_a_T = I_a.transpose(1, 2)

        I_b = torch.exp(-preterm * (t2 - vbc) ** 2)
        I_b = I_b / (torch.sum(I_b, dim=-1, keepdim=True) + self.eps)
        I_b_T = I_b.transpose(1, 2)

        pa = torch.mean(I_a, dim=1, keepdim=True).transpose(1, 2)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.matmul(pa, pb) + self.eps
        pab = torch.matmul(I_a_T, I_b) / nb_voxels

        mi = torch.sum(pab * torch.log(pab / papb + self.eps), dim=[1, 2])
        return torch.mean(mi)

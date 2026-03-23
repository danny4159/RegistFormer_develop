import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleDecompositionLoss(nn.Module):
    """
    Loss functions for Common-Private Style Decomposition

    Components:
    1. Common Agreement Loss: common_b ≈ common_c
    2. Private Decorrelation Loss: private_b ⊥ private_c
    3. Common-Private Separation Loss: common ⊥ private within each branch
    4. Shared Smoothness Loss: TV regularization on shared style
    5. Shared Consistency Loss: common_shared stays close to common_avg
    """
    def __init__(
        self,
        lambda_common=1.0,
        lambda_private=0.2,
        lambda_sep=0.2,
        lambda_smooth=0.05,
        lambda_shared_consistency=0.1,
        use_cosine=True
    ):
        super().__init__()
        self.lambda_common = lambda_common
        self.lambda_private = lambda_private
        self.lambda_sep = lambda_sep
        self.lambda_smooth = lambda_smooth
        self.lambda_shared_consistency = lambda_shared_consistency
        self.use_cosine = use_cosine

    def compute_cosine_similarity(self, x, y):
        """Compute cosine similarity between two tensors (spatial average)"""
        x_flat = x.view(x.size(0), x.size(1), -1)
        y_flat = y.view(y.size(0), y.size(1), -1)

        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)

        sim = (x_norm * y_norm).sum(dim=1).mean()
        return sim

    def compute_covariance_penalty(self, x, y):
        """
        VICReg-style covariance penalty: minimize off-diagonal covariance
        Encourages decorrelation between x and y features
        """
        B, C = x.size(0), x.size(1)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C) if x.dim() == 4 else x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        y_flat = y.permute(0, 2, 3, 1).reshape(-1, C) if y.dim() == 4 else y.permute(0, 2, 3, 4, 1).reshape(-1, C)

        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        y_centered = y_flat - y_flat.mean(dim=0, keepdim=True)

        N = x_centered.size(0)
        cov_xy = (x_centered.T @ y_centered) / (N - 1 + 1e-8)

        cov_loss = (cov_xy ** 2).mean()
        return cov_loss

    def compute_tv_loss(self, x):
        """Total Variation loss for smoothness"""
        if x.dim() == 5:  # 3D
            tv_h = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
            tv_w = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
            tv_d = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
            return tv_h + tv_w + tv_d
        else:  # 2D
            tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
            tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
            return tv_h + tv_w

    def forward(self, decomp_dict):
        """
        Compute all decomposition losses

        Args:
            decomp_dict: dict containing:
                - common_b, common_c: [B, C, H, W]
                - private_b, private_c: [B, C, H, W]
                - common_shared or c_shared: [B, C, H, W]
                - g_common: gate tensor used by the router (optional, not used here)

        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses for logging
        """
        common_b = decomp_dict['common_b']
        common_c = decomp_dict['common_c']
        private_b = decomp_dict['private_b']
        private_c = decomp_dict['private_c']
        c_shared = decomp_dict.get('common_shared', decomp_dict['c_shared'])

        loss_dict = {}

        # 1. Common Agreement Loss: common_b ≈ common_c
        if self.use_cosine:
            common_sim = self.compute_cosine_similarity(common_b, common_c)
            loss_common = (1.0 - common_sim) * self.lambda_common
        else:
            loss_common = F.l1_loss(common_b, common_c) * self.lambda_common
        loss_dict['loss_common'] = loss_common

        # 2. Private Decorrelation Loss: private_b ⊥ private_c
        loss_private = self.compute_covariance_penalty(private_b, private_c) * self.lambda_private
        loss_dict['loss_private'] = loss_private

        # 3. Common-Private Separation Loss: common ⊥ private within each branch
        loss_sep_b = self.compute_covariance_penalty(common_b, private_b)
        loss_sep_c = self.compute_covariance_penalty(common_c, private_c)
        loss_sep = (loss_sep_b + loss_sep_c) * self.lambda_sep
        loss_dict['loss_sep'] = loss_sep

        # 4. Shared Smoothness Loss: TV regularization
        loss_smooth = self.compute_tv_loss(c_shared) * self.lambda_smooth
        loss_dict['loss_smooth'] = loss_smooth

        # 5. Conservative shared fusion: keep common_shared close to common_avg
        common_avg = 0.5 * (common_b + common_c)
        loss_shared_consistency = F.l1_loss(c_shared, common_avg) * self.lambda_shared_consistency
        loss_dict['loss_shared_consistency'] = loss_shared_consistency

        total_loss = loss_common + loss_private + loss_sep + loss_smooth + loss_shared_consistency
        loss_dict['loss_decomp_total'] = total_loss

        return total_loss, loss_dict

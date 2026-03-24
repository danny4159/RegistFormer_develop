import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleDecompositionLoss(nn.Module):
    """Loss functions for Common-Private Style Decomposition.

    This loss ensures:
    1. Common styles are similar across references (common agreement)
    2. Private styles are decorrelated (private decorrelation - VICReg style)
    3. Common and private components are separated (orthogonality)
    4. Shared features are spatially smooth (TV regularization)
    """

    def __init__(
        self,
        lambda_common=1.0,
        lambda_private=0.2,
        lambda_sep=0.2,
        lambda_smooth=0.05,
    ):
        """Initialize StyleDecompositionLoss.

        Args:
            lambda_common: Weight for common agreement loss (common_b ≈ common_c)
            lambda_private: Weight for private decorrelation loss (private_b ⊥ private_c)
            lambda_sep: Weight for common-private separation loss
            lambda_smooth: Weight for spatial smoothness (TV) loss on common_shared
        """
        super().__init__()
        self.lambda_common = lambda_common
        self.lambda_private = lambda_private
        self.lambda_sep = lambda_sep
        self.lambda_smooth = lambda_smooth

    def common_agreement_loss(self, common_b, common_c):
        """Encourage common styles to be similar across references.

        Uses cosine similarity to allow magnitude differences while
        encouraging directional agreement.

        Args:
            common_b: Common style from reference B [B, C, H, W]
            common_c: Common style from reference C [B, C, H, W]

        Returns:
            Scalar loss value
        """
        # Flatten spatial dimensions
        B, C = common_b.shape[:2]
        common_b_flat = common_b.view(B, C, -1)  # [B, C, HW]
        common_c_flat = common_c.view(B, C, -1)  # [B, C, HW]

        # Compute cosine similarity along channel dimension
        common_b_norm = F.normalize(common_b_flat, dim=1, eps=1e-8)
        common_c_norm = F.normalize(common_c_flat, dim=1, eps=1e-8)

        # Mean cosine similarity across spatial locations
        cos_sim = (common_b_norm * common_c_norm).sum(dim=1).mean()

        # Loss: 1 - cos_sim (want to maximize similarity)
        return 1.0 - cos_sim

    def private_decorrelation_loss(self, private_b, private_c):
        """Encourage private styles to be decorrelated (VICReg-style).

        Uses covariance-based decorrelation to ensure private components
        capture different information.

        Args:
            private_b: Private style from reference B [B, C, H, W]
            private_c: Private style from reference C [B, C, H, W]

        Returns:
            Scalar loss value
        """
        B, C = private_b.shape[:2]

        # Flatten to [B, C, HW]
        private_b_flat = private_b.view(B, C, -1)
        private_c_flat = private_c.view(B, C, -1)

        # Compute cross-covariance matrix
        # First, center the features
        private_b_centered = private_b_flat - private_b_flat.mean(dim=2, keepdim=True)
        private_c_centered = private_c_flat - private_c_flat.mean(dim=2, keepdim=True)

        # Cross-covariance: [B, C, C]
        HW = private_b_flat.shape[2]
        cov = torch.bmm(private_b_centered, private_c_centered.transpose(1, 2)) / (HW - 1)

        # We want the covariance to be close to zero (decorrelated)
        # Loss: squared Frobenius norm of covariance
        loss = (cov ** 2).sum(dim=[1, 2]).mean()

        return loss

    def common_private_separation_loss(self, common, private):
        """Encourage orthogonality between common and private components.

        Args:
            common: Common style features [B, C, H, W]
            private: Private style features [B, C, H, W]

        Returns:
            Scalar loss value
        """
        B, C = common.shape[:2]

        # Flatten to [B, C, HW]
        common_flat = common.view(B, C, -1)
        private_flat = private.view(B, C, -1)

        # Center the features
        common_centered = common_flat - common_flat.mean(dim=2, keepdim=True)
        private_centered = private_flat - private_flat.mean(dim=2, keepdim=True)

        # Compute correlation: [B, C, C]
        HW = common_flat.shape[2]
        corr = torch.bmm(common_centered, private_centered.transpose(1, 2)) / (HW - 1)

        # Normalize to get correlation coefficient
        common_std = common_centered.std(dim=2, unbiased=False).clamp(min=1e-8)  # [B, C]
        private_std = private_centered.std(dim=2, unbiased=False).clamp(min=1e-8)  # [B, C]
        std_outer = torch.bmm(common_std.unsqueeze(2), private_std.unsqueeze(1))  # [B, C, C]
        corr_normalized = corr / std_outer.clamp(min=1e-8)

        # Loss: squared Frobenius norm of correlation
        loss = (corr_normalized ** 2).sum(dim=[1, 2]).mean()

        return loss

    def spatial_smoothness_loss(self, features):
        """Total Variation (TV) regularization for spatial smoothness.

        Args:
            features: Feature map [B, C, H, W] or [B, C, D, H, W]

        Returns:
            Scalar loss value
        """
        if features.dim() == 5:
            # 3D case
            diff_d = (features[:, :, 1:, :, :] - features[:, :, :-1, :, :]).abs().mean()
            diff_h = (features[:, :, :, 1:, :] - features[:, :, :, :-1, :]).abs().mean()
            diff_w = (features[:, :, :, :, 1:] - features[:, :, :, :, :-1]).abs().mean()
            return diff_d + diff_h + diff_w
        else:
            # 2D case
            diff_h = (features[:, :, 1:, :] - features[:, :, :-1, :]).abs().mean()
            diff_w = (features[:, :, :, 1:] - features[:, :, :, :-1]).abs().mean()
            return diff_h + diff_w

    def forward(self, decomp_dict):
        """Compute total style decomposition loss.

        Args:
            decomp_dict: Dictionary containing:
                - common_b: Common style from ref B [B, C, H, W]
                - common_c: Common style from ref C [B, C, H, W]
                - private_b: Private style from ref B [B, C, H, W]
                - private_c: Private style from ref C [B, C, H, W]
                - common_shared: Fused common style [B, C, H, W]
                - gate: Gating values [B, C, H, W]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        common_b = decomp_dict['common_b']
        common_c = decomp_dict['common_c']
        private_b = decomp_dict['private_b']
        private_c = decomp_dict['private_c']
        common_shared = decomp_dict['common_shared']

        # Compute individual losses
        loss_common = self.common_agreement_loss(common_b, common_c)
        loss_private = self.private_decorrelation_loss(private_b, private_c)

        # Separation loss for both references
        loss_sep_b = self.common_private_separation_loss(common_b, private_b)
        loss_sep_c = self.common_private_separation_loss(common_c, private_c)
        loss_sep = (loss_sep_b + loss_sep_c) / 2.0

        # Smoothness loss on shared features
        loss_smooth = self.spatial_smoothness_loss(common_shared)

        # Total weighted loss
        total_loss = (
            self.lambda_common * loss_common +
            self.lambda_private * loss_private +
            self.lambda_sep * loss_sep +
            self.lambda_smooth * loss_smooth
        )

        loss_dict = {
            'loss_common': loss_common.item(),
            'loss_private': loss_private.item(),
            'loss_sep': loss_sep.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_style_decomp': total_loss.item(),
        }

        return total_loss, loss_dict


class GateRegularizationLoss(nn.Module):
    """Regularization loss to prevent gates from becoming too large.

    Encourages sparsity in gate activations, which prevents over-aggressive
    common style sharing that can lead to sharp but structurally misaligned outputs.
    """

    def __init__(self, lambda_gate=0.005):
        """Initialize GateRegularizationLoss.

        Args:
            lambda_gate: Weight for gate regularization (default: 0.005)
        """
        super().__init__()
        self.lambda_gate = lambda_gate

    def forward(self, decomp_dict):
        """Compute gate regularization loss.

        Args:
            decomp_dict: Dictionary containing:
                - g_common: Common fusion gate [B, C, H, W]
                - g_b: Branch B receiving gate [B, C, H, W]
                - g_c: Branch C receiving gate [B, C, H, W]

        Returns:
            Scalar loss value (already weighted by lambda_gate)
        """
        g_common = decomp_dict['g_common']
        g_b = decomp_dict['g_b']
        g_c = decomp_dict['g_c']

        # Mean absolute activation (encourage sparsity)
        loss_gate = (
            g_common.abs().mean() +
            g_b.abs().mean() +
            g_c.abs().mean()
        ) / 3.0

        return self.lambda_gate * loss_gate

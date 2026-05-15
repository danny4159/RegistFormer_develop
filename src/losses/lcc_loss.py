import torch
import torch.nn as nn
import torch.nn.functional as F


class LCCLoss(nn.Module):
    """
    Local Cross-Correlation loss for multi-modal image registration.
    Computes normalized cross-correlation within local windows.
    Loss = -mean(LCC), so minimizing drives alignment.
    """

    def __init__(self, win=9, eps=1e-3):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, fixed, warped):
        # fixed, warped: [B, C, H, W, D] (3D) or [B, C, H, W] (2D)
        ndim = fixed.dim() - 2
        assert ndim in (2, 3), f"Expected 2D or 3D input, got {ndim}D"

        win = [self.win] * ndim
        # sum filter via avg_pool (keeps spatial dims with padding)
        padding = [w // 2 for w in win]

        if ndim == 3:
            sum_filt = lambda x: F.avg_pool3d(x, kernel_size=win, stride=1, padding=padding) * (self.win ** 3)
        else:
            sum_filt = lambda x: F.avg_pool2d(x, kernel_size=win, stride=1, padding=padding) * (self.win ** 2)

        I = fixed
        J = warped

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum  = sum_filt(I)
        J_sum  = sum_filt(J)
        I2_sum = sum_filt(I2)
        J2_sum = sum_filt(J2)
        IJ_sum = sum_filt(IJ)

        win_size = self.win ** ndim
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross  = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var  = (I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size).clamp(min=0)
        J_var  = (J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size).clamp(min=0)

        # clamp denominator away from zero to avoid exploding gradients
        denom = (I_var * J_var).clamp(min=self.eps)
        lcc = (cross * cross) / denom

        return -lcc.mean()

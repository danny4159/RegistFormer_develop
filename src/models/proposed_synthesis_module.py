import numpy as np

from typing import Any
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss, VGG_Model
from src.losses.patch_nce_loss import PatchNCELoss
from src.losses.mind_loss import MINDLoss


class PerceptualVGGLoss(nn.Module):
    """VGG feature L2 perceptual loss (Johnson et al.).
    Uses the same VGG layers as contextual loss but computes MSE on features instead of contextual similarity.
    """
    def __init__(self, feat_layers: dict):
        super().__init__()
        self.vgg = VGG_Model(listen_list=list(feat_layers.keys()))
        self.weights = feat_layers

    def forward(self, fake: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if fake.shape[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)
        if ref.shape[1] == 1:
            ref = ref.repeat(1, 3, 1, 1)
        fake_feats = self.vgg(fake)
        with torch.no_grad():
            ref_feats = self.vgg(ref)
        loss = torch.tensor(0.0, device=fake.device)
        for layer, w in self.weights.items():
            loss = loss + w * F.mse_loss(fake_feats[layer], ref_feats[layer])
        return loss

from src import utils
from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src.models.base_module_AtoB import BaseModule_AtoB
# from src.models.base_module_AtoB_multi import BaseModule_AtoB


log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)


# -------------------------------------------------------------------------
# Legacy batch-variable mapping inherited from BaseModule_AtoB
#
# real_a:
#   fixed_slice / fixed image x^k
#   Anatomical source input and PatchNCE structural reference.
#
# real_b:
#   reference_aligned_target
#   Available in controlled settings such as IXI for evaluation only.
#   It is not used as a supervised pixel-aligned training target.
#
# fake_b:
#   synthesized_slice / MIGS output y_hat^k
#   Moving-contrast image synthesized in the fixed-image geometry.
#
# real_b_ref:
#   moving_stack y_adj^k
#   K moving-domain slices after initial rigid pre-alignment;
#   residual through-plane and in-plane mismatch may remain.
#
# The variable names themselves (real_a/real_b/fake_b/real_b_ref) are kept
# as-is because they are the tuple-unpacking interface returned by
# BaseModule_AtoB.model_step(), which is shared with other models and not
# modified here.
# -------------------------------------------------------------------------


class MIGSModule(BaseModule_AtoB):
    """Lightning training module for MIGS (Moving-Image-Guided Synthesis)."""

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        netD_B: torch.nn.Module,
        netF_A: torch.nn.Module,
        netD_C: torch.nn.Module = None,
        optimizer=None,
        params=None,
        scheduler=None,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)

        # assign generator
        self.netG_A = netG_A
        self.netD_A = netD_A
        self.netD_B = netD_B
        self.netD_C = netD_C  # For triple outputs
        self.netF_A = netF_A

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

        if self.params.nce_on_vgg: # vgg for patchNCE
            # choose layers what you want # "conv_1_2", "conv_2_2", "conv_3_4", "conv_4_4", "conv_5_4"
            listen_list = ["conv_4_2", "conv_5_4"] # PatchNCE reflects MR structure; use high-level feature layers. Low-level layers would leave too much raw MR appearance in the features.
            self.vgg = VGG_Model(listen_list=listen_list)

        # assign contextual loss (cx = "contextual", the paper's appearance-matching term)
        cx_feature_layers = {
            "conv_1_2": 1.0,
            "conv_2_1": 1.0,
            "conv_2_2": 1.0,
            "conv_3_2": 1.0,
            "conv_4_2": 1.0,
            "conv_4_4": 1.0
        }
        lambda_cx = self._get_lambda_cx()

        # loss function
        self.criterionContextual = Contextual_Loss(cx_feature_layers) if lambda_cx != 0 else None
        lambda_perceptual = getattr(params, 'lambda_perceptual', 0)
        self.criterionPerceptual = PerceptualVGGLoss(cx_feature_layers) if lambda_perceptual != 0 else None
        self.criterionGAN = GANLoss(gan_type='lsgan')
        self.criterionNCE = PatchNCELoss(False, nce_T=0.07, batch_size=params.batch_size) if params.lambda_nce != 0 else None

        # Loss for ablation
        self.criterionMIND = MINDLoss() if params.lambda_mind != 0 else None
        self.criterionL1 = torch.nn.L1Loss() if params.lambda_l1 != 0 else None


        # PatchNCE specific initializations
        # self.nce_layers = [0,2,4,6] # range: 0~6
        # self.flip_equivariance = params.flip_equivariance

    def _get_lambda_cx(self):
        """lambda_cx (new name) with fallback to the legacy lambda_style key,
        so old override scripts/configs keep working."""
        return float(getattr(self.params, 'lambda_cx', getattr(self.params, 'lambda_style', 0.0)))

    @staticmethod
    def _softmin_contextual(cx_list, shift_penalties=None, tau=0.3):
        """Soft-min over a list of contextual losses (center-biased)."""
        losses = torch.stack(cx_list)
        if shift_penalties is not None:
            penalties = torch.tensor(shift_penalties, device=losses.device, dtype=losses.dtype)
            losses = losses + penalties
        return -tau * torch.logsumexp(-losses / tau, dim=0)

    def _compute_cx_loss_over_moving_stack(self, synthesized_slice, moving_stack, lambda_cx):
        """Compute contextual (cx) loss against the K-slice moving stack.

        Reported MIGS: mean aggregation over all K slices (cx_stack_aggregation='mean').
        Optional ablations: center-only (cx_center_only=True) or center-biased soft-min.
        """
        K = moving_stack.shape[1]
        center_idx = K // 2
        if getattr(self.params, 'cx_center_only', getattr(self.params, 'ctx_center_only', False)):
            return self.criterionContextual(moving_stack[:, center_idx:center_idx+1], synthesized_slice) * lambda_cx
        agg_mode = getattr(self.params, 'cx_stack_aggregation', getattr(self.params, 'ctx_agg_mode', 'softmin'))
        cx_losses = [self.criterionContextual(moving_stack[:, i:i+1], synthesized_slice).squeeze() for i in range(K)]
        if agg_mode == 'mean':
            return torch.stack(cx_losses).mean() * lambda_cx
        # softmin (default)
        tau = getattr(self.params, 'cx_softmin_temperature', getattr(self.params, 'ctx_softmin_tau', 0.3))
        shift_penalty_base = getattr(self.params, 'cx_slice_offset_penalty', getattr(self.params, 'ctx_shift_penalty', 0.05))
        shift_penalties = [abs(i - center_idx) * shift_penalty_base for i in range(K)]
        return self._softmin_contextual(cx_losses, shift_penalties, tau) * lambda_cx

    def _compute_perceptual_loss_over_moving_stack(self, synthesized_slice, moving_stack, lambda_perceptual):
        """VGG perceptual loss for 2.5D moving stack — same softmin aggregation as contextual."""
        K = moving_stack.shape[1]
        center_idx = K // 2
        p_losses = [self.criterionPerceptual(synthesized_slice, moving_stack[:, i:i+1]).squeeze() for i in range(K)]
        tau = getattr(self.params, 'cx_softmin_temperature', getattr(self.params, 'ctx_softmin_tau', 0.3))
        shift_penalty_base = getattr(self.params, 'cx_slice_offset_penalty', getattr(self.params, 'ctx_shift_penalty', 0.05))
        shift_penalties = [abs(i - center_idx) * shift_penalty_base for i in range(K)]
        return self._softmin_contextual(p_losses, shift_penalties, tau) * lambda_perceptual

    def _compute_l1_loss_over_moving_stack(self, synthesized_slice, moving_stack, lambda_l1):
        """L1 loss for 2.5D moving stack — same softmin aggregation as contextual."""
        K = moving_stack.shape[1]
        center_idx = K // 2
        l1_losses = [self.criterionL1(synthesized_slice, moving_stack[:, i:i+1]).squeeze() for i in range(K)]
        tau = getattr(self.params, 'cx_softmin_temperature', getattr(self.params, 'ctx_softmin_tau', 0.3))
        shift_penalty_base = getattr(self.params, 'cx_slice_offset_penalty', getattr(self.params, 'ctx_shift_penalty', 0.05))
        shift_penalties = [abs(i - center_idx) * shift_penalty_base for i in range(K)]
        return self._softmin_contextual(l1_losses, shift_penalties, tau) * lambda_l1

    def _setup_mig_conv_debug_flags(self):
        if not hasattr(self, 'netG_A'):
            return
        log_mig_conv = getattr(self.params, 'log_mig_conv_stats', getattr(self.params, 'log_style_modulation', False))
        for mod in self.netG_A.modules():
            if type(mod).__name__ == 'MIGConv':
                mod._log_style_modulation = log_mig_conv

    def _log_swa_stats(self):
        """Log Slice-Window Attention diagnostics (attention entropy, QK stats, etc.)."""
        stats = getattr(self.netG_A, '_last_ref_condition_stats', None)
        if not stats:
            return
        for k, v in stats.items():
            self.log(f"swa/{k}", v, prog_bar=False)

    def _log_mig_conv_stats(self):
        """Log MIGConv (CGM gamma/beta modulation) diagnostics."""
        if not getattr(self.params, 'log_mig_conv_stats', getattr(self.params, 'log_style_modulation', False)):
            return
        for name, mod in self.netG_A.named_modules():
            if type(mod).__name__ == 'MIGConv' and hasattr(mod, 'last_style_debug'):
                for k, v in mod.last_style_debug.items():
                    self.log(f"mig_conv/{name}/{k}", v, prog_bar=False)

    def compute_generator_loss(self, real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref):
        """Compute the generator's total loss (does not perform backward itself).

        Paper-code mapping for the reported single-output model:
          real_a     -> fixed_slice x^k
          real_b     -> reference_aligned_target (evaluation only)
          fake_b     -> synthesized_slice y_hat^k
          real_b_ref -> moving_stack y_adj^k

        Reported self-supervised objective:
          1. PatchNCE: fixed structure vs. synthesized output
          2. Contextual (cx): moving-stack appearance vs. synthesized output
          3. LSGAN: moving-domain fidelity
        """
        loss_G = torch.tensor(0.0, device=real_a.device)
        use_25d = getattr(self.params, 'use_25d_style', False)

        # Moving-domain references used by the contextual loss and as
        # conditioning inputs during PatchNCE feature extraction.
        # 2.5D: real_b_ref is the K-channel moving stack
        # 2D misalign: real_b_ref is a single misaligned moving reference
        # 2D no misalign: real_b (GT) is used directly
        if use_25d or self.params.use_misalign_simul:
            moving_ref_b, moving_ref_c, moving_ref_d = real_b_ref, real_c_ref, real_d_ref
        else:
            moving_ref_b, moving_ref_c, moving_ref_d = real_b, real_c, real_d

        ##################################################################################################################
        ## 1. GAN Loss
        lambda_gan = float(getattr(self.params, 'lambda_gan', 1))
        if self.criterionGAN and lambda_gan > 0:
            pred_fake = self.netD_A(fake_b)
            loss_gan_b = self.criterionGAN(pred_fake, True) * lambda_gan
            self.log("loss/gan_b", loss_gan_b.detach(), prog_bar=True)
            loss_G += loss_gan_b
            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                pred_fake = self.netD_B(fake_c)
                loss_gan_c = self.criterionGAN(pred_fake, True) * lambda_gan
                self.log("loss/gan_c", loss_gan_c.detach(), prog_bar=True)
                loss_G += loss_gan_c
                if self.params.use_triple_outputs and fake_d is not None:
                    pred_fake = self.netD_C(fake_d)
                    loss_gan_d = self.criterionGAN(pred_fake, True) * lambda_gan
                    self.log("loss/gan_d", loss_gan_d.detach(), prog_bar=True)
                    loss_G += loss_gan_d

        ##################################################################################################################
        ## 2. Contextual (cx) loss
        if self.criterionContextual:
            lambda_cx = self._get_lambda_cx()
            if use_25d and moving_ref_b is not None:
                loss_cx_b = self._compute_cx_loss_over_moving_stack(fake_b, moving_ref_b, lambda_cx)
            else:
                loss_cx_b = self.criterionContextual(moving_ref_b, fake_b) * lambda_cx
            self.log("loss/cx_b", loss_cx_b.detach(), prog_bar=True)
            loss_G += loss_cx_b.squeeze()

            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                if use_25d and moving_ref_c is not None:
                    loss_cx_c = self._compute_cx_loss_over_moving_stack(fake_c, moving_ref_c, lambda_cx)
                else:
                    loss_cx_c = self.criterionContextual(moving_ref_c, fake_c) * lambda_cx
                self.log("loss/cx_c", loss_cx_c.detach(), prog_bar=True)
                loss_G += loss_cx_c.squeeze()
                if self.params.use_triple_outputs and fake_d is not None:
                    if use_25d and moving_ref_d is not None:
                        loss_cx_d = self._compute_cx_loss_over_moving_stack(fake_d, moving_ref_d, lambda_cx)
                    else:
                        loss_cx_d = self.criterionContextual(moving_ref_d, fake_d) * lambda_cx
                    self.log("loss/cx_d", loss_cx_d.detach(), prog_bar=True)
                    loss_G += loss_cx_d.squeeze()

        ##################################################################################################################
        ## 2b. Perceptual loss (VGG feature L2, same 2.5D softmin aggregation as contextual)
        if self.criterionPerceptual:
            lambda_perceptual = getattr(self.params, 'lambda_perceptual', 0)
            if use_25d and moving_ref_b is not None:
                loss_perceptual_b = self._compute_perceptual_loss_over_moving_stack(fake_b, moving_ref_b, lambda_perceptual)
            else:
                loss_perceptual_b = self.criterionPerceptual(fake_b, moving_ref_b) * lambda_perceptual
            self.log("loss/perceptual_b", loss_perceptual_b.detach(), prog_bar=True)
            loss_G += loss_perceptual_b

        ##################################################################################################################
        ## 2c. L1 stack loss (2.5D softmin, replaces old center-slice-only L1)
        if self.criterionL1 and use_25d and moving_ref_b is not None:
            loss_l1_b = self._compute_l1_loss_over_moving_stack(fake_b, moving_ref_b, self.params.lambda_l1)
            self.log("loss/l1_b", loss_l1_b.detach(), prog_bar=True)
            loss_G += loss_l1_b

        ##################################################################################################################
        ## 3. PatchNCE loss: computed once, jointly, for fake_b and fake_c. fake_d support still needs to be implemented.
        if self.criterionNCE:
            if self.params.nce_on_vgg: # not used
                real_rgb = real_a.repeat(1, 3, 1, 1)
                if self.params.nce_independent:
                    fake_rgb_b = fake_b.repeat(1, 3, 1, 1)
                    fake_rgb_c = fake_c.repeat(1, 3, 1, 1)
                elif fake_d is not None:
                    fake_rgb = torch.cat((fake_b, fake_c, fake_d), dim=1)
                elif fake_c is not None:
                    fake_rgb = torch.cat((fake_b, fake_c, torch.zeros_like(fake_b)), dim=1) # Last channel is average -> zero (For checkerboard artifact but not sure)
                else:
                    fake_rgb = torch.cat((fake_b, fake_b, fake_b), dim=1) # Last channel is average -> zero (For checkerboard artifact but not sure)
                self.vgg.to(real_a.device)

                if self.params.nce_independent:
                    fixed_features = self.vgg(real_rgb)
                    fixed_features = list(fixed_features.values())

                    synthesized_features_b = self.vgg(fake_rgb_b)
                    synthesized_features_b = list(synthesized_features_b.values())

                    synthesized_features_c = self.vgg(fake_rgb_c)
                    synthesized_features_c = list(synthesized_features_c.values())

                    fixed_patch_features, shared_patch_ids = self.netF_A(fixed_features, 256, None)
                    synthesized_patch_features_b, _ = self.netF_A(synthesized_features_b, 256, shared_patch_ids)
                    synthesized_patch_features_c, _ = self.netF_A(synthesized_features_c, 256, shared_patch_ids)

                    total_nce_loss = 0.0

                    for f_a, f_b, f_c in zip(fixed_patch_features, synthesized_patch_features_b, synthesized_patch_features_c):
                        loss = (self.criterionNCE(f_a, f_b) + self.criterionNCE(f_a, f_c)) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / (len(synthesized_features_b) + len(synthesized_features_c))
                    self.log("loss/nce_b", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b

                else:
                    synthesized_features = self.vgg(fake_rgb)
                    synthesized_features = list(synthesized_features.values()) # [0]:8,512,16,16 [1]:8,512,8,8

                    fixed_features = self.vgg(real_rgb)
                    fixed_features = list(fixed_features.values())

                    fixed_patch_features, shared_patch_ids = self.netF_A(fixed_features, 256, None)
                    synthesized_patch_features, _ = self.netF_A(synthesized_features, 256, shared_patch_ids)

                    total_nce_loss = 0.0

                    for f_a, f_b in zip(fixed_patch_features, synthesized_patch_features):
                        loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / len(synthesized_features)
                    self.log("loss/nce_b", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b

            else:
                # A fixed-filled dummy moving stack is used only to satisfy the generator
                # input format while extracting features from the synthesized slice.
                if use_25d:
                    K = self.params.ref_stack_size
                    dummy_moving_stack = real_a.repeat(1, K, 1, 1)
                else:
                    dummy_moving_stack = real_a  # 2D: use real_a as neutral guidance

                if self.params.use_triple_outputs:
                    n_layers = len(self.params.nce_layers)
                    synthesized_feature_input_b = torch.cat((fake_b, dummy_moving_stack, dummy_moving_stack, dummy_moving_stack), dim=1)
                    synthesized_features_b = self.netG_A(synthesized_feature_input_b, self.params.nce_layers, encode_only=True)

                    synthesized_feature_input_c = torch.cat((fake_c, dummy_moving_stack, dummy_moving_stack, dummy_moving_stack), dim=1)
                    synthesized_features_c = self.netG_A(synthesized_feature_input_c, self.params.nce_layers, encode_only=True)

                    synthesized_feature_input_d = torch.cat((fake_d, dummy_moving_stack, dummy_moving_stack, dummy_moving_stack), dim=1)
                    synthesized_features_d = self.netG_A(synthesized_feature_input_d, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        synthesized_features_b = [torch.flip(fb, [3]) for fb in synthesized_features_b]
                        synthesized_features_c = [torch.flip(fc, [3]) for fc in synthesized_features_c]
                        synthesized_features_d = [torch.flip(fd, [3]) for fd in synthesized_features_d]

                    fixed_feature_input = torch.cat((real_a, moving_ref_b, moving_ref_c, moving_ref_d), dim=1)
                    fixed_features = self.netG_A(fixed_feature_input, self.params.nce_layers, encode_only=True)
                    fixed_patch_features, shared_patch_ids = self.netF_A(fixed_features, 256, None)
                    synthesized_patch_features_b, _ = self.netF_A(synthesized_features_b, 256, shared_patch_ids)
                    synthesized_patch_features_c, _ = self.netF_A(synthesized_features_c, 256, shared_patch_ids)
                    synthesized_patch_features_d, _ = self.netF_A(synthesized_features_d, 256, shared_patch_ids)

                    total_nce_loss = 0.0
                    for f_a, f_b, f_c, f_d in zip(fixed_patch_features, synthesized_patch_features_b, synthesized_patch_features_c, synthesized_patch_features_d):
                        loss = (self.criterionNCE(f_a, f_b) + self.criterionNCE(f_a, f_c) + self.criterionNCE(f_a, f_d)) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / n_layers
                    self.log("loss/nce_b", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b
                    assert not torch.isnan(loss_nce_b).any(), "NCE Loss is NaN"
                elif self.params.use_multiple_outputs:
                    n_layers = len(self.params.nce_layers)
                    synthesized_feature_input_b = torch.cat((fake_b, dummy_moving_stack, dummy_moving_stack), dim=1)
                    synthesized_features_b = self.netG_A(synthesized_feature_input_b, self.params.nce_layers, encode_only=True)

                    synthesized_feature_input_c = torch.cat((fake_c, dummy_moving_stack, dummy_moving_stack), dim=1)
                    synthesized_features_c = self.netG_A(synthesized_feature_input_c, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        synthesized_features_b = [torch.flip(fb, [3]) for fb in synthesized_features_b]
                        synthesized_features_c = [torch.flip(fc, [3]) for fc in synthesized_features_c]

                    fixed_feature_input = torch.cat((real_a, moving_ref_b, moving_ref_c), dim=1)
                    fixed_features = self.netG_A(fixed_feature_input, self.params.nce_layers, encode_only=True)
                    fixed_patch_features, shared_patch_ids = self.netF_A(fixed_features, 256, None)
                    synthesized_patch_features_b, _ = self.netF_A(synthesized_features_b, 256, shared_patch_ids)
                    synthesized_patch_features_c, _ = self.netF_A(synthesized_features_c, 256, shared_patch_ids)

                    total_nce_loss = 0.0
                    for f_a, f_b, f_c in zip(fixed_patch_features, synthesized_patch_features_b, synthesized_patch_features_c):
                        loss = (self.criterionNCE(f_a, f_b) + self.criterionNCE(f_a, f_c)) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / n_layers
                    self.log("loss/nce_b", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b
                    assert not torch.isnan(loss_nce_b).any(), "NCE Loss is NaN"
                else:
                    n_layers = len(self.params.nce_layers)
                    synthesized_feature_input = torch.cat((fake_b, dummy_moving_stack), dim=1)
                    synthesized_features = self.netG_A(synthesized_feature_input, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        synthesized_features = [torch.flip(fb, [3]) for fb in synthesized_features]

                    fixed_feature_input = torch.cat((real_a, moving_ref_b), dim=1)
                    fixed_features = self.netG_A(fixed_feature_input, self.params.nce_layers, encode_only=True)
                    fixed_patch_features, shared_patch_ids = self.netF_A(fixed_features, 256, None)
                    synthesized_patch_features, _ = self.netF_A(synthesized_features, 256, shared_patch_ids)

                    total_nce_loss = 0.0
                    for f_a, f_b in zip(fixed_patch_features, synthesized_patch_features):
                        loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / n_layers
                    self.log("loss/nce_b", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b
                    assert not torch.isnan(loss_nce_b).any(), "NCE Loss is NaN"

        if self.criterionMIND:
            loss_mind_b = self.criterionMIND(real_a, fake_b) * self.params.lambda_mind
            self.log("loss/mind_b", loss_mind_b.detach(), prog_bar=True)
            loss_G += loss_mind_b

            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                loss_mind_c = self.criterionMIND(real_a, fake_c) * self.params.lambda_mind
                self.log("loss/mind_c", loss_mind_c.detach(), prog_bar=True)
                loss_G += loss_mind_c

            if self.params.use_triple_outputs and fake_d is not None:
                loss_mind_d = self.criterionMIND(real_a, fake_d) * self.params.lambda_mind
                self.log("loss/mind_d", loss_mind_d.detach(), prog_bar=True)
                loss_G += loss_mind_d

        if self.criterionL1 and not use_25d:
            # 2D case: L1 uses center slice (2.5D case handled above in section 2c)
            def _center_slice(t):
                if t is not None and use_25d and t.shape[1] > 1:
                    return t[:, t.shape[1] // 2: t.shape[1] // 2 + 1]
                return t
            loss_l1_b = self.criterionL1(_center_slice(real_b_ref), fake_b) * self.params.lambda_l1
            self.log("loss/l1_b", loss_l1_b.detach(), prog_bar=True)
            loss_G += loss_l1_b

            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                loss_l1_c = self.criterionL1(_center_slice(real_c_ref), fake_c) * self.params.lambda_l1
                self.log("loss/l1_c", loss_l1_c.detach(), prog_bar=True)
                loss_G += loss_l1_c

            if self.params.use_triple_outputs and fake_d is not None and real_d_ref is not None:
                loss_l1_d = self.criterionL1(_center_slice(real_d_ref), fake_d) * self.params.lambda_l1
                self.log("loss/l1_d", loss_l1_d.detach(), prog_bar=True)
                loss_G += loss_l1_d

        ##################################################################################################################
        ## Attention entropy regularization (encourage selective / peaked SWA attention).
        ## Minimizing normalized entropy pushes attention away from uniform toward query-similar candidates.
        lambda_swa_entropy = float(getattr(self.params, 'lambda_swa_entropy', getattr(self.params, 'lambda_attn_entropy', 0)))
        if lambda_swa_entropy != 0:
            ent = getattr(self.netG_A, '_last_attn_entropy_for_loss', None)
            if ent is not None:
                loss_swa_entropy = lambda_swa_entropy * ent
                self.log("loss/swa_entropy", loss_swa_entropy.detach(), prog_bar=True)
                self.log("diag/swa_entropy", ent.detach(), prog_bar=False)
                loss_G += loss_swa_entropy

        self.log("loss/g_total", loss_G.detach(), prog_bar=True)
        return loss_G
        # assert not torch.isnan(loss_G).any(), "Total Loss is NaN"

    def compute_generator_loss_3d_legacy(self, real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref):
        loss_G = torch.tensor(0.0, device=real_a.device)
        D = real_a.shape[-1]  # number of slices

        loss_logs = {}

        for d in range(D):
            ra = real_a[..., d]
            rb = real_b[..., d]
            rb_ref = real_b_ref[..., d] if real_b_ref is not None else None
            fb = fake_b[..., d]

            ## GAN
            lambda_gan = float(getattr(self.params, 'lambda_gan', 1))
            if self.criterionGAN and lambda_gan > 0:
                pred_fake = self.netD_A(fb)
                loss_gan_b = self.criterionGAN(pred_fake, True) * lambda_gan / D
                loss_G += loss_gan_b
                loss_logs.setdefault("loss/gan_b", 0.0)
                loss_logs["loss/gan_b"] += loss_gan_b.detach()

            # Contextual (cx)
            if self.criterionContextual:
                lambda_cx = self._get_lambda_cx()
                loss_cx_b = self.criterionContextual(rb, fb) * lambda_cx / D
                loss_G += loss_cx_b.squeeze()
                loss_logs.setdefault("loss/cx_b", 0.0)
                loss_logs["loss/cx_b"] += loss_cx_b.detach()

            ## PatchNCE (slice-wise)
            if self.criterionNCE:
                if self.params.nce_on_vgg:
                    real_rgb = ra.repeat(1, 3, 1, 1)
                    fake_rgb = fb.repeat(1, 3, 1, 1)
                    self.vgg.to(ra.device)

                    synthesized_features = self.vgg(fake_rgb)
                    synthesized_features = list(synthesized_features.values()) # [0]:8,512,16,16 [1]:8,512,8,8

                    fixed_features = self.vgg(real_rgb)
                    fixed_features = list(fixed_features.values())

                    fixed_patch_features, shared_patch_ids = self.netF_A(fixed_features, 256, None)
                    synthesized_patch_features, _ = self.netF_A(synthesized_features, 256, shared_patch_ids)

                    total_nce_loss = 0.0

                    for f_a, f_b in zip(fixed_patch_features, synthesized_patch_features):
                        loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce / D
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / len(synthesized_features)
                    loss_logs.setdefault("loss/nce_b", 0.0)
                    loss_logs["loss/nce_b"] += loss_nce_b.detach()
                    loss_G += loss_nce_b

                else:
                    n_layers = len(self.params.nce_layers)
                    synthesized_feature_input = torch.cat((fb, ra), dim=1)
                    synthesized_features = self.netG_A(synthesized_feature_input, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        synthesized_features = [torch.flip(fb_, [3]) for fb_ in synthesized_features]

                    fixed_feature_input = torch.cat((ra, rb), dim=1)
                    fixed_features = self.netG_A(fixed_feature_input, self.params.nce_layers, encode_only=True)
                    fixed_patch_features, shared_patch_ids = self.netF_A(fixed_features, 256, None)
                    synthesized_patch_features, _ = self.netF_A(synthesized_features, 256, shared_patch_ids)

                    total_nce_loss = 0.0
                    for f_a, f_b in zip(fixed_patch_features, synthesized_patch_features):
                        loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce / D
                        total_nce_loss += loss.mean()

                    loss_nce_b = total_nce_loss / n_layers
                    loss_logs.setdefault("loss/nce_b", 0.0)
                    loss_logs["loss/nce_b"] += loss_nce_b.detach()
                    loss_G += loss_nce_b
        # log the averaged losses
        for key, val in loss_logs.items():
            self.log(key, val, prog_bar=True)
        self.log("loss/g_total", loss_G.detach(), prog_bar=True)

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):

        self._setup_mig_conv_debug_flags()

        real_c = real_d = fake_c = fake_d = None
        real_b_ref = real_c_ref = real_d_ref = None
        use_25d = getattr(self.params, 'use_25d_style', False)
        need_ref = self.params.use_misalign_simul or use_25d

        lambda_gan = float(getattr(self.params, 'lambda_gan', 1))
        use_gan = lambda_gan > 0

        if self.params.use_triple_outputs:
            if use_gan:
                optimizer_G_A, optimizer_D_A, optimizer_D_B, optimizer_D_C, optimizer_F_A = self.optimizers()
            else:
                optimizer_G_A, optimizer_F_A = self.optimizers()
                optimizer_D_A = optimizer_D_B = optimizer_D_C = None
            if need_ref:
                real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref = self.model_step(batch)
            else:
                real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d = self.model_step(batch)
        elif self.params.use_multiple_outputs:
            if use_gan:
                optimizer_G_A, optimizer_D_A, optimizer_D_B, optimizer_F_A = self.optimizers()
            else:
                optimizer_G_A, optimizer_F_A = self.optimizers()
                optimizer_D_A = optimizer_D_B = None
            if need_ref:
                real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref = self.model_step(batch)
            else:
                real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d = self.model_step(batch)
        else:
            if use_gan:
                optimizer_G_A, optimizer_D_A, optimizer_F_A = self.optimizers()
            else:
                optimizer_G_A, optimizer_F_A = self.optimizers()
                optimizer_D_A = None
            if need_ref:
                # Single-output 2.5D MIGS:
                # real_a     = fixed slice
                # real_b     = reference-aligned target for evaluation
                # fake_b     = synthesized aligned slice
                # real_b_ref = K-slice moving stack used by SWA and contextual loss
                real_a, real_b, fake_b, real_b_ref = self.model_step(batch)
            else:
                real_a, real_b, fake_b = self.model_step(batch)

        with optimizer_G_A.toggle_model():
            if self.params.use_triple_outputs:
                loss_G = self.compute_generator_loss(real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref)
            elif self.params.use_multiple_outputs:
                loss_G = self.compute_generator_loss(real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref)
            else:
                if self.params.is_3d and not use_25d:
                    loss_G = self.compute_generator_loss_3d_legacy(real_a, real_b, None, None, fake_b, None, None, None, None, None)
                else:
                    loss_G = self.compute_generator_loss(real_a, real_b, None, None, fake_b, None, None, real_b_ref, None, None)

            if getattr(self.params, 'log_swa_stats', getattr(self.params, 'log_ref_condition', False)) or \
               getattr(self.params, 'log_mig_conv_stats', getattr(self.params, 'log_style_modulation', False)):
                interval = int(getattr(self.params, 'diagnostic_log_interval', getattr(self.params, 'log_z_select_interval', 200)))
                if self.global_step % interval == 0:
                    self._log_swa_stats()
                    self._log_mig_conv_stats()

            self.manual_backward(loss_G)
            self.clip_gradients(
                optimizer_G_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            self.clip_gradients(
                optimizer_F_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_G_A.step()
            optimizer_F_A.step()
            optimizer_G_A.zero_grad()
            optimizer_F_A.zero_grad()

        # LSGAN real sample: center slice of the moving stack, matching the
        # moving-domain discriminator setup in the paper.
        def _select_discriminator_real_slice(ref, gt):
            if use_25d and ref is not None and ref.shape[1] > 1:
                return ref[:, ref.shape[1] // 2: ref.shape[1] // 2 + 1]
            return ref if self.params.use_misalign_simul else gt

        if use_gan:
            moving_center_for_discriminator = _select_discriminator_real_slice(real_b_ref, real_b)
            moving_center_for_discriminator_c = _select_discriminator_real_slice(real_c_ref, real_c) if (self.params.use_multiple_outputs or self.params.use_triple_outputs) else real_c
            moving_center_for_discriminator_d = _select_discriminator_real_slice(real_d_ref, real_d) if self.params.use_triple_outputs else real_d

            with optimizer_D_A.toggle_model():
                if self.params.is_3d:
                    loss_D_A = self.backward_D_A_3D(moving_center_for_discriminator, fake_b)
                else:
                    loss_D_A = self.backward_D_A(moving_center_for_discriminator, fake_b)
                self.manual_backward(loss_D_A)
                self.clip_gradients(
                    optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_D_A.step()
                optimizer_D_A.zero_grad()
            self.log("loss/d_a", loss_D_A.detach(), prog_bar=True)

            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                with optimizer_D_B.toggle_model():
                    loss_D_B = self.backward_D_B(moving_center_for_discriminator_c, fake_c)
                    self.manual_backward(loss_D_B)
                    self.clip_gradients(
                        optimizer_D_B, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                    )
                    optimizer_D_B.step()
                    optimizer_D_B.zero_grad()
                self.log("loss/d_b", loss_D_B.detach(), prog_bar=True)

            if self.params.use_triple_outputs:
                with optimizer_D_C.toggle_model():
                    loss_D_C = self.backward_D_C(moving_center_for_discriminator_d, fake_d)
                    self.manual_backward(loss_D_C)
                    self.clip_gradients(
                        optimizer_D_C, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                    )
                    optimizer_D_C.step()
                    optimizer_D_C.zero_grad()
                self.log("loss/d_c", loss_D_C.detach(), prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizers = []
        schedulers = []

        use_gan = float(getattr(self.params, 'lambda_gan', 1)) > 0

        optimizer_G_A = self.hparams.optimizer(params=self.netG_A.parameters())
        optimizers.append(optimizer_G_A)
        if use_gan:
            optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())
            optimizers.append(optimizer_D_A)
            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())
                optimizers.append(optimizer_D_B)
            if self.params.use_triple_outputs:
                optimizer_D_C = self.hparams.optimizer(params=self.netD_C.parameters())
                optimizers.append(optimizer_D_C)
        optimizer_F_A = self.hparams.optimizer(params=self.netF_A.parameters())
        optimizers.append(optimizer_F_A)

        if self.hparams.scheduler is not None:
            scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
            schedulers.append(scheduler_G_A)
            if use_gan:
                scheduler_D_A = self.hparams.scheduler(optimizer=optimizer_D_A)
                schedulers.append(scheduler_D_A)
                if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                    scheduler_D_B = self.hparams.scheduler(optimizer=optimizer_D_B)
                    schedulers.append(scheduler_D_B)
                if self.params.use_triple_outputs:
                    scheduler_D_C = self.hparams.scheduler(optimizer=optimizer_D_C)
                    schedulers.append(scheduler_D_C)
            scheduler_F_A = self.hparams.scheduler(optimizer=optimizer_F_A)
            schedulers.append(scheduler_F_A)
            return optimizers, schedulers

        return optimizers


# Backward-compatible alias: nothing currently imports this by name (hydra
# instantiates via the _target_ string in the yaml), but kept just in case.
ProposedSynthesisModule = MIGSModule

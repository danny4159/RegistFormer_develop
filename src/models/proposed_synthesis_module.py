import numpy as np

from typing import Any
import itertools

import torch
import torch.nn.functional as F
from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss, VGG_Model
from src.losses.patch_nce_loss import PatchNCELoss
from src.losses.mind_loss import MINDLoss

from src import utils
from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src.models.base_module_AtoB import BaseModule_AtoB
# from src.models.base_module_AtoB_multi import BaseModule_AtoB


log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)



class ProposedSynthesisModule(BaseModule_AtoB):

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
            listen_list = ["conv_4_2", "conv_5_4"] # PatchNCE는 MR을 반영하는 것. high level feature layer를 선택. low lever로 하면 mr의 feature가 그대로 많이 남을것
            self.vgg = VGG_Model(listen_list=listen_list)
            
        # assign contextual loss
        style_feat_layers = {
            "conv_1_2": 1.0,
            "conv_2_1": 1.0,
            "conv_2_2": 1.0,
            "conv_3_2": 1.0,
            "conv_4_2": 1.0,
            "conv_4_4": 1.0
        }

        # loss function
        self.criterionContextual = Contextual_Loss(style_feat_layers) if params.lambda_style != 0 else None
        self.criterionGAN = GANLoss(gan_type='lsgan')
        self.criterionNCE = PatchNCELoss(False, nce_T=0.07, batch_size=params.batch_size) if params.lambda_nce != 0 else None

        # Loss for ablation
        self.criterionMIND = MINDLoss() if params.lambda_mind != 0 else None
        self.criterionL1 = torch.nn.L1Loss() if params.lambda_l1 != 0 else None


        # PatchNCE specific initializations
        # self.nce_layers = [0,2,4,6] # range: 0~6
        # self.flip_equivariance = params.flip_equivariance

    @staticmethod
    def _softmin_contextual(cx_list, shift_penalties=None, tau=0.3):
        """Soft-min over a list of contextual losses (center-biased)."""
        losses = torch.stack(cx_list)
        if shift_penalties is not None:
            penalties = torch.tensor(shift_penalties, device=losses.device, dtype=losses.dtype)
            losses = losses + penalties
        return -tau * torch.logsumexp(-losses / tau, dim=0)

    def _contextual_stack_loss(self, fake_img, ref_stack, lambda_style):
        """Contextual loss for 2.5D ref stack.
        ctx_center_only=True : center slice only (2D equivalent).
        ctx_agg_mode='softmin': center-biased soft-min over K slices (default).
        ctx_agg_mode='mean'   : simple mean over all K slices.
        """
        K = ref_stack.shape[1]
        center_idx = K // 2
        if getattr(self.params, 'ctx_center_only', False):
            return self.criterionContextual(ref_stack[:, center_idx:center_idx+1], fake_img) * lambda_style
        agg_mode = getattr(self.params, 'ctx_agg_mode', 'softmin')
        cx_losses = [self.criterionContextual(ref_stack[:, i:i+1], fake_img).squeeze() for i in range(K)]
        if agg_mode == 'mean':
            return torch.stack(cx_losses).mean() * lambda_style
        # softmin (default)
        tau = getattr(self.params, 'ctx_softmin_tau', 0.3)
        shift_penalty_base = getattr(self.params, 'ctx_shift_penalty', 0.05)
        shift_penalties = [abs(i - center_idx) * shift_penalty_base for i in range(K)]
        return self._softmin_contextual(cx_losses, shift_penalties, tau) * lambda_style

    def _setup_zselect_debug_flags(self):
        if not hasattr(self, 'netG_A'):
            return
        log_style = getattr(self.params, 'log_style_modulation', False)
        for mod in self.netG_A.modules():
            if type(mod).__name__ == 'StyleConv':
                mod._log_style_modulation = log_style

    def _log_zselect_stats(self):
        stats = getattr(self.netG_A, '_last_zselect_stats', None)
        if not stats:
            return
        for k, v in stats.items():
            self.log(f"zselect/{k}", v, prog_bar=False)

    def _log_style_modulation_stats(self):
        if not getattr(self.params, 'log_style_modulation', False):
            return
        for name, mod in self.netG_A.named_modules():
            if type(mod).__name__ != 'StyleConv':
                continue
            dbg = getattr(mod, 'last_style_debug', None)
            if dbg is None:
                continue
            for k, v in dbg.items():
                self.log(f"style/{name}/{k}", v, prog_bar=False)

    def _set_randomize_noise(self, value):
        old_states = []
        for mod in self.netG_A.modules():
            if hasattr(mod, 'randomize_noise'):
                old_states.append((mod, mod.randomize_noise))
                mod.randomize_noise = value
        return old_states

    def _restore_randomize_noise(self, old_states):
        for mod, old_value in old_states:
            mod.randomize_noise = old_value

    def _forward_no_zdebug(self, real_a, ref_stack):
        old_store = getattr(self.netG_A, 'store_zselect_debug', True)
        self.netG_A.store_zselect_debug = False
        try:
            merged = torch.cat([real_a, ref_stack], dim=1)
            return self.netG_A(merged)
        finally:
            self.netG_A.store_zselect_debug = old_store

    def _log_zselect_sensitivity(self, real_a, real_b_ref):
        if not getattr(self.params, 'log_z_select_sensitivity', False):
            return
        interval = int(getattr(self.params, 'log_z_select_interval', 200))
        if self.global_step % interval != 0:
            return
        use_25d = getattr(self.params, 'use_25d_style', False)
        if not use_25d or real_b_ref is None or real_b_ref.shape[1] < 3:
            return
        K = real_b_ref.shape[1]
        center_idx = K // 2
        center_ref = real_b_ref[:, center_idx:center_idx+1].expand_as(real_b_ref)
        zero_ref   = torch.zeros_like(real_b_ref)
        idx        = torch.arange(K - 1, -1, -1, device=real_b_ref.device)
        perm_ref   = real_b_ref[:, idx]
        old_noise  = self._set_randomize_noise(False)
        try:
            with torch.no_grad():
                fake_default = self._forward_no_zdebug(real_a, real_b_ref)
                fake_center  = self._forward_no_zdebug(real_a, center_ref)
                fake_zero    = self._forward_no_zdebug(real_a, zero_ref)
                fake_perm    = self._forward_no_zdebug(real_a, perm_ref)
                eps   = 1e-8
                denom = fake_default.abs().mean() + eps
                self.log("zselect_sens/out_center_l1", ((fake_default - fake_center).abs().mean() / denom).item(), prog_bar=False)
                self.log("zselect_sens/out_zero_l1",   ((fake_default - fake_zero).abs().mean()   / denom).item(), prog_bar=False)
                self.log("zselect_sens/out_perm_l1",   ((fake_default - fake_perm).abs().mean()   / denom).item(), prog_bar=False)
        finally:
            self._restore_randomize_noise(old_noise)

    def _log_global_local_stats(self):
        if not getattr(self.params, 'log_global_local_style', False):
            return
        stats = getattr(self.netG_A, '_last_global_local_stats', None)
        if not stats:
            return
        for k, v in stats.items():
            self.log(f"global_local/{k}", v, prog_bar=False)

    def _set_global_local_mode(self, mode):
        old_mode = getattr(self.netG_A, 'global_local_mode', 'off')
        self.netG_A.global_local_mode = mode
        return old_mode

    def _restore_global_local_mode(self, old_mode):
        self.netG_A.global_local_mode = old_mode

    def _log_global_local_sensitivity(self, real_a, real_b_ref):
        if not getattr(self.params, 'log_global_local_sensitivity', False):
            return
        interval = int(getattr(self.params, 'log_z_select_interval', 200))
        if self.global_step % interval != 0:
            return
        if real_b_ref is None:
            return

        old_noise = self._set_randomize_noise(False)
        old_store = getattr(self.netG_A, 'store_zselect_debug', True)
        self.netG_A.store_zselect_debug = False
        old_mode = self._set_global_local_mode(getattr(self.netG_A, 'global_local_mode', 'off'))

        try:
            with torch.no_grad():
                merged = torch.cat([real_a, real_b_ref], dim=1)

                self.netG_A.global_local_mode = old_mode
                fake_current = self.netG_A(merged)

                self.netG_A.global_local_mode = 'global_only'
                fake_global = self.netG_A(merged)

                self.netG_A.global_local_mode = 'off'
                fake_local = self.netG_A(merged)

                eps = 1e-8
                denom = fake_current.abs().mean() + eps
                self.log("global_local_sens/current_vs_global_l1",
                         (fake_current - fake_global).abs().mean() / denom, prog_bar=False)
                self.log("global_local_sens/current_vs_local_l1",
                         (fake_current - fake_local).abs().mean() / denom, prog_bar=False)
                self.log("global_local_sens/global_vs_local_l1",
                         (fake_global - fake_local).abs().mean() / denom, prog_bar=False)
        finally:
            self._restore_global_local_mode(old_mode)
            self.netG_A.store_zselect_debug = old_store
            self._restore_randomize_noise(old_noise)

    def _log_layerwise_style_stats(self):
        stats = getattr(self.netG_A, '_last_layerwise_style_stats', None)
        if not stats:
            return
        for k, v in stats.items():
            self.log(f"layerwise_style/{k}", v, prog_bar=False)

    def _log_global_local_alpha_grad(self):
        if not getattr(self.params, 'log_global_local_style', False):
            return
        interval = int(getattr(self.params, 'log_z_select_interval', 200))
        if self.global_step % interval != 0:
            return
        p = getattr(self.netG_A, 'global_local_alpha_logit', None)
        if p is None or p.grad is None:
            return
        self.log("global_local/alpha_grad_abs", p.grad.detach().abs().mean(), prog_bar=False)

    def _log_zselect_alpha_grad(self):
        if not getattr(self.params, 'log_z_select', False):
            return
        interval = int(getattr(self.params, 'log_z_select_interval', 200))
        if self.global_step % interval != 0:
            return
        p = getattr(self.netG_A, 'z_select_alpha_logit', None)
        if p is None or p.grad is None:
            return
        self.log("zselect/alpha_grad_abs", p.grad.detach().abs().mean(), prog_bar=False)

    def backward_G(self, real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref): # real_a, real_b, fake_b
        loss_G = torch.tensor(0.0, device=real_a.device)
        use_25d = getattr(self.params, 'use_25d_style', False)

        # Effective style refs for contextual and NCE real-side
        # 2.5D: real_b_ref is K-channel stack
        # 2D misalign: real_b_ref is single misaligned ref
        # 2D no misalign: real_b is GT
        if use_25d or self.params.use_misalign_simul:
            eff_b, eff_c, eff_d = real_b_ref, real_c_ref, real_d_ref
        else:
            eff_b, eff_c, eff_d = real_b, real_c, real_d

        ##################################################################################################################
        ## 1. GAN Loss
        lambda_gan = float(getattr(self.params, 'lambda_gan', 1))
        if self.criterionGAN and lambda_gan > 0:
            pred_fake = self.netD_A(fake_b)
            loss_gan_b = self.criterionGAN(pred_fake, True) * lambda_gan
            self.log("Gan_b_Loss", loss_gan_b.detach(), prog_bar=True)
            loss_G += loss_gan_b
            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                pred_fake = self.netD_B(fake_c)
                loss_gan_c = self.criterionGAN(pred_fake, True) * lambda_gan
                self.log("Gan_c_Loss", loss_gan_c.detach(), prog_bar=True)
                loss_G += loss_gan_c
                if self.params.use_triple_outputs and fake_d is not None:
                    pred_fake = self.netD_C(fake_d)
                    loss_gan_d = self.criterionGAN(pred_fake, True) * lambda_gan
                    self.log("Gan_d_Loss", loss_gan_d.detach(), prog_bar=True)
                    loss_G += loss_gan_d

        ##################################################################################################################
        ## 2. Contextual loss
        if self.criterionContextual:
            if use_25d and eff_b is not None:
                loss_style_b = self._contextual_stack_loss(fake_b, eff_b, self.params.lambda_style)
            else:
                loss_style_b = self.criterionContextual(eff_b, fake_b) * self.params.lambda_style
            self.log("Context_b_Loss", loss_style_b.detach(), prog_bar=True)
            loss_G += loss_style_b.squeeze()

            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                if use_25d and eff_c is not None:
                    loss_style_c = self._contextual_stack_loss(fake_c, eff_c, self.params.lambda_style)
                else:
                    loss_style_c = self.criterionContextual(eff_c, fake_c) * self.params.lambda_style
                self.log("Context_c_Loss", loss_style_c.detach(), prog_bar=True)
                loss_G += loss_style_c.squeeze()
                if self.params.use_triple_outputs and fake_d is not None:
                    if use_25d and eff_d is not None:
                        loss_style_d = self._contextual_stack_loss(fake_d, eff_d, self.params.lambda_style)
                    else:
                        loss_style_d = self.criterionContextual(eff_d, fake_d) * self.params.lambda_style
                    self.log("Context_d_Loss", loss_style_d.detach(), prog_bar=True)
                    loss_G += loss_style_d.squeeze()

        ##################################################################################################################
        ## 3. PatchNCE loss: 이건 fake_b, fake_c 한꺼번에 해서 한번만 해. 이건 fake_d에 대한 코드 구현 필요.
        if self.criterionNCE:
            if self.params.nce_on_vgg: # 이거 안써
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
                    feat_a = self.vgg(real_rgb)
                    feat_a = list(feat_a.values())

                    feat_b = self.vgg(fake_rgb_b)
                    feat_b = list(feat_b.values())

                    feat_c = self.vgg(fake_rgb_c)
                    feat_c = list(feat_c.values())

                    feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
                    feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)
                    feat_c_pool, _ = self.netF_A(feat_c, 256, sample_ids)

                    total_nce_loss = 0.0

                    for f_a, f_b, f_c in zip(feat_a_pool, feat_b_pool, feat_c_pool):
                        loss = (self.criterionNCE(f_a, f_b) + self.criterionNCE(f_a, f_c)) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / (len(feat_b) + len(feat_c))
                    self.log("NCE_b_Loss", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b

                else:
                    feat_b = self.vgg(fake_rgb)
                    feat_b = list(feat_b.values()) # [0]:8,512,16,16 [1]:8,512,8,8

                    feat_a = self.vgg(real_rgb)
                    feat_a = list(feat_a.values())

                    feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
                    feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)

                    total_nce_loss = 0.0

                    for f_a, f_b in zip(feat_a_pool,feat_b_pool):
                        loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / len(feat_b)
                    self.log("NCE_b_Loss", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b

            else:
                # For 2.5D: fake side uses dummy stacks; real side uses eff_b/c/d (already K-ch stacks)
                if use_25d:
                    K = self.params.ref_stack_size
                    dummy = real_a.repeat(1, K, 1, 1)
                else:
                    dummy = real_a  # 2D: use real_a as neutral style

                if self.params.use_triple_outputs:
                    n_layers = len(self.params.nce_layers)
                    merged_input_1 = torch.cat((fake_b, dummy, dummy, dummy), dim=1)
                    feat_b = self.netG_A(merged_input_1, self.params.nce_layers, encode_only=True)

                    merged_input_2 = torch.cat((fake_c, dummy, dummy, dummy), dim=1)
                    feat_c = self.netG_A(merged_input_2, self.params.nce_layers, encode_only=True)

                    merged_input_3 = torch.cat((fake_d, dummy, dummy, dummy), dim=1)
                    feat_d = self.netG_A(merged_input_3, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        feat_b = [torch.flip(fb, [3]) for fb in feat_b]
                        feat_c = [torch.flip(fc, [3]) for fc in feat_c]
                        feat_d = [torch.flip(fd, [3]) for fd in feat_d]

                    merged_input_real = torch.cat((real_a, eff_b, eff_c, eff_d), dim=1)
                    feat_a = self.netG_A(merged_input_real, self.params.nce_layers, encode_only=True)
                    feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
                    feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)
                    feat_c_pool, _ = self.netF_A(feat_c, 256, sample_ids)
                    feat_d_pool, _ = self.netF_A(feat_d, 256, sample_ids)

                    total_nce_loss = 0.0
                    for f_a, f_b, f_c, f_d in zip(feat_a_pool, feat_b_pool, feat_c_pool, feat_d_pool):
                        loss = (self.criterionNCE(f_a, f_b) + self.criterionNCE(f_a, f_c) + self.criterionNCE(f_a, f_d)) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / n_layers
                    self.log("NCE_b_Loss", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b
                    assert not torch.isnan(loss_nce_b).any(), "NCE Loss is NaN"
                elif self.params.use_multiple_outputs:
                    n_layers = len(self.params.nce_layers)
                    merged_input_1 = torch.cat((fake_b, dummy, dummy), dim=1)
                    feat_b = self.netG_A(merged_input_1, self.params.nce_layers, encode_only=True)

                    merged_input_2 = torch.cat((fake_c, dummy, dummy), dim=1)
                    feat_c = self.netG_A(merged_input_2, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        feat_b = [torch.flip(fb, [3]) for fb in feat_b]
                        feat_c = [torch.flip(fc, [3]) for fc in feat_c]

                    merged_input_real = torch.cat((real_a, eff_b, eff_c), dim=1)
                    feat_a = self.netG_A(merged_input_real, self.params.nce_layers, encode_only=True)
                    feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
                    feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)
                    feat_c_pool, _ = self.netF_A(feat_c, 256, sample_ids)

                    total_nce_loss = 0.0
                    for f_a, f_b, f_c in zip(feat_a_pool, feat_b_pool, feat_c_pool):
                        loss = (self.criterionNCE(f_a, f_b) + self.criterionNCE(f_a, f_c)) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / n_layers
                    self.log("NCE_b_Loss", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b
                    assert not torch.isnan(loss_nce_b).any(), "NCE Loss is NaN"
                else:
                    n_layers = len(self.params.nce_layers)
                    merged_input_1 = torch.cat((fake_b, dummy), dim=1)
                    feat_b = self.netG_A(merged_input_1, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        feat_b = [torch.flip(fb, [3]) for fb in feat_b]

                    merged_input_real = torch.cat((real_a, eff_b), dim=1)
                    feat_a = self.netG_A(merged_input_real, self.params.nce_layers, encode_only=True)
                    feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
                    feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)

                    total_nce_loss = 0.0
                    for f_a, f_b in zip(feat_a_pool, feat_b_pool):
                        loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / n_layers
                    self.log("NCE_b_Loss", loss_nce_b.detach(), prog_bar=True)
                    loss_G += loss_nce_b
                    assert not torch.isnan(loss_nce_b).any(), "NCE Loss is NaN"

        if self.criterionMIND:
            loss_mind_b = self.criterionMIND(real_a, fake_b) * self.params.lambda_mind
            self.log("MIND_b_Loss", loss_mind_b.detach(), prog_bar=True)
            loss_G += loss_mind_b

            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                loss_mind_c = self.criterionMIND(real_a, fake_c) * self.params.lambda_mind
                self.log("MIND_c_Loss", loss_mind_c.detach(), prog_bar=True)
                loss_G += loss_mind_c

            if self.params.use_triple_outputs and fake_d is not None:
                loss_mind_d = self.criterionMIND(real_a, fake_d) * self.params.lambda_mind
                self.log("MIND_d_Loss", loss_mind_d.detach(), prog_bar=True)
                loss_G += loss_mind_d

        if self.criterionL1:
            # For 2.5D: L1 uses center slice of stack
            def _center_slice(t):
                if t is not None and use_25d and t.shape[1] > 1:
                    return t[:, t.shape[1] // 2: t.shape[1] // 2 + 1]
                return t
            loss_l1_b = self.criterionL1(_center_slice(real_b_ref), fake_b) * self.params.lambda_l1
            self.log("L1_b_Loss", loss_l1_b.detach(), prog_bar=True)
            loss_G += loss_l1_b

            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                loss_l1_c = self.criterionL1(_center_slice(real_c_ref), fake_c) * self.params.lambda_l1
                self.log("L1_c_Loss", loss_l1_c.detach(), prog_bar=True)
                loss_G += loss_l1_c

            if self.params.use_triple_outputs and fake_d is not None and real_d_ref is not None:
                loss_l1_d = self.criterionL1(_center_slice(real_d_ref), fake_d) * self.params.lambda_l1
                self.log("L1_d_Loss", loss_l1_d.detach(), prog_bar=True)
                loss_G += loss_l1_d

        self.log("G_loss", loss_G.detach(), prog_bar=True)
        return loss_G
        # assert not torch.isnan(loss_G).any(), "Total Loss is NaN"

    def backward_G_3D(self, real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref):
        loss_G = torch.tensor(0.0, device=real_a.device)
        D = real_a.shape[-1]  # 슬라이스 수

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
                loss_logs.setdefault("Gan_b_Loss", 0.0)
                loss_logs["Gan_b_Loss"] += loss_gan_b.detach()

            # Contextual
            if self.criterionContextual:
                loss_style_b = self.criterionContextual(rb, fb) * self.params.lambda_style / D
                loss_G += loss_style_b.squeeze()
                loss_logs.setdefault("Context_b_Loss", 0.0)
                loss_logs["Context_b_Loss"] += loss_style_b.detach()

            ## PatchNCE (slice-wise)
            if self.criterionNCE:
                if self.params.nce_on_vgg:
                    real_rgb = ra.repeat(1, 3, 1, 1)
                    fake_rgb = fb.repeat(1, 3, 1, 1)
                    self.vgg.to(ra.device)

                    feat_b = self.vgg(fake_rgb)
                    feat_b = list(feat_b.values()) # [0]:8,512,16,16 [1]:8,512,8,8

                    feat_a = self.vgg(real_rgb)
                    feat_a = list(feat_a.values())

                    feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
                    feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)

                    total_nce_loss = 0.0

                    for f_a, f_b in zip(feat_a_pool, feat_b_pool):
                        loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce / D
                        total_nce_loss = total_nce_loss + loss.mean()
                    loss_nce_b = total_nce_loss / len(feat_b)
                    loss_logs.setdefault("NCE_b_Loss", 0.0)
                    loss_logs["NCE_b_Loss"] += loss_nce_b.detach()
                    loss_G += loss_nce_b

                else:
                    n_layers = len(self.params.nce_layers)
                    merged_input_1 = torch.cat((fb, ra), dim=1)
                    feat_b = self.netG_A(merged_input_1, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        feat_b = [torch.flip(fb_, [3]) for fb_ in feat_b]

                    merged_input_2 = torch.cat((ra, rb), dim=1)
                    feat_a = self.netG_A(merged_input_2, self.params.nce_layers, encode_only=True)
                    feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
                    feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)

                    total_nce_loss = 0.0
                    for f_a, f_b in zip(feat_a_pool, feat_b_pool):
                        loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce / D
                        total_nce_loss += loss.mean()

                    loss_nce_b = total_nce_loss / n_layers
                    loss_logs.setdefault("NCE_b_Loss", 0.0)
                    loss_logs["NCE_b_Loss"] += loss_nce_b.detach()
                    loss_G += loss_nce_b
        # 평균 로그 출력
        for key, val in loss_logs.items():
            self.log(key, val, prog_bar=True)
        self.log("G_loss", loss_G.detach(), prog_bar=True)

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):

        self._setup_zselect_debug_flags()

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
                real_a, real_b, fake_b, real_b_ref = self.model_step(batch)
            else:
                real_a, real_b, fake_b = self.model_step(batch)

        # z-selection / style modulation / global-local logging
        do_log_z      = getattr(self.params, 'log_z_select', False)
        do_log_st     = getattr(self.params, 'log_style_modulation', False)
        do_log_gl     = getattr(self.params, 'log_global_local_style', False)
        do_log_glsens = getattr(self.params, 'log_global_local_sensitivity', False)
        if do_log_z or do_log_st or do_log_gl or do_log_glsens:
            interval = int(getattr(self.params, 'log_z_select_interval', 200))
            if self.global_step % interval == 0:
                if do_log_z:
                    self._log_zselect_stats()
                    self._log_zselect_sensitivity(real_a, real_b_ref)
                if do_log_st:
                    self._log_style_modulation_stats()
                if do_log_gl:
                    self._log_global_local_stats()
                    self._log_layerwise_style_stats()
                if do_log_glsens:
                    self._log_global_local_sensitivity(real_a, real_b_ref)

        with optimizer_G_A.toggle_model():
            if self.params.use_triple_outputs:
                loss_G = self.backward_G(real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref)
            elif self.params.use_multiple_outputs:
                loss_G = self.backward_G(real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref)
            else:
                if self.params.is_3d and not use_25d:
                    loss_G = self.backward_G_3D(real_a, real_b, None, None, fake_b, None, None, None, None, None)
                else:
                    loss_G = self.backward_G(real_a, real_b, None, None, fake_b, None, None, real_b_ref, None, None)

            self.manual_backward(loss_G)
            self._log_global_local_alpha_grad()
            self._log_zselect_alpha_grad()
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

        # Discriminator real target: for 2.5D use center slice of stack
        def _disc_real(ref, gt):
            if use_25d and ref is not None and ref.shape[1] > 1:
                return ref[:, ref.shape[1] // 2: ref.shape[1] // 2 + 1]
            return ref if self.params.use_misalign_simul else gt

        if use_gan:
            real_b_disc = _disc_real(real_b_ref, real_b)
            real_c_disc = _disc_real(real_c_ref, real_c) if (self.params.use_multiple_outputs or self.params.use_triple_outputs) else real_c
            real_d_disc = _disc_real(real_d_ref, real_d) if self.params.use_triple_outputs else real_d

            with optimizer_D_A.toggle_model():
                if self.params.is_3d:
                    loss_D_A = self.backward_D_A_3D(real_b_disc, fake_b)
                else:
                    loss_D_A = self.backward_D_A(real_b_disc, fake_b)
                self.manual_backward(loss_D_A)
                self.clip_gradients(
                    optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_D_A.step()
                optimizer_D_A.zero_grad()
            self.log("D_A_Loss", loss_D_A.detach(), prog_bar=True)

            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                with optimizer_D_B.toggle_model():
                    loss_D_B = self.backward_D_B(real_c_disc, fake_c)
                    self.manual_backward(loss_D_B)
                    self.clip_gradients(
                        optimizer_D_B, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                    )
                    optimizer_D_B.step()
                    optimizer_D_B.zero_grad()
                self.log("D_B_Loss", loss_D_B.detach(), prog_bar=True)

            if self.params.use_triple_outputs:
                with optimizer_D_C.toggle_model():
                    loss_D_C = self.backward_D_C(real_d_disc, fake_d)
                    self.manual_backward(loss_D_C)
                    self.clip_gradients(
                        optimizer_D_C, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                    )
                    optimizer_D_C.step()
                    optimizer_D_C.zero_grad()
                self.log("D_C_Loss", loss_D_C.detach(), prog_bar=True)

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

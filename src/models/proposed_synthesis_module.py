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
    def _conf_tv_loss(x):
        """Total-variation smoothness over the confidence map. x: [B,C,H,W]."""
        loss_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        loss_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return loss_h + loss_w

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
        ctx_center_only=True: use center slice only (2D equivalent).
        ctx_center_only=False: center-biased soft-min over K slices.
        """
        K = ref_stack.shape[1]
        center_idx = K // 2
        if getattr(self.params, 'ctx_center_only', False):
            return self.criterionContextual(ref_stack[:, center_idx:center_idx+1], fake_img) * lambda_style
        tau = getattr(self.params, 'ctx_softmin_tau', 0.3)
        shift_penalty_base = getattr(self.params, 'ctx_shift_penalty', 0.05)
        cx_losses, shift_penalties = [], []
        for i in range(K):
            cx_losses.append(self.criterionContextual(ref_stack[:, i:i+1], fake_img).squeeze())
            shift_penalties.append(abs(i - center_idx) * shift_penalty_base)
        return self._softmin_contextual(cx_losses, shift_penalties, tau) * lambda_style

    def backward_G(self, real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref): # real_a, real_b, fake_b
        loss_G = torch.tensor(0.0, device=real_a.device)
        use_25d = getattr(self.params, 'use_25d_style', False)

        # Capture confidence map from the main forward (in model_step) BEFORE
        # NCE encode_only calls below overwrite netG_A.last_conf_map. Holding the
        # tensor reference keeps gradients alive for the regularization terms.
        conf_for_reg = getattr(self.netG_A, 'last_conf_map', None)

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
        if self.criterionGAN:
            pred_fake = self.netD_A(fake_b)
            loss_gan_b = self.criterionGAN(pred_fake, True)
            self.log("Gan_b_Loss", loss_gan_b.detach(), prog_bar=True)
            loss_G += loss_gan_b
            # assert not torch.isnan(loss_gan_b).any(), "GAN Loss is NaN"
            
            if self.params.use_multiple_outputs or self.params.use_triple_outputs:
                pred_fake = self.netD_B(fake_c)
                loss_gan_c = self.criterionGAN(pred_fake, True)
                self.log("Gan_c_Loss", loss_gan_c.detach(), prog_bar=True)
                loss_G += loss_gan_c
                # assert not torch.isnan(loss_gan_c).any(), "GAN Loss is NaN"
                if self.params.use_triple_outputs and fake_d is not None:
                    pred_fake = self.netD_C(fake_d)
                    loss_gan_d = self.criterionGAN(pred_fake, True)
                    self.log("Gan_d_Loss", loss_gan_d.detach(), prog_bar=True)
                    loss_G += loss_gan_d
                    # assert not torch.isnan(loss_gan_d).any(), "GAN Loss is NaN"

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

        # Confidence regularization (active only when lambdas > 0 AND gating ran).
        if conf_for_reg is not None:
            lambda_conf_smooth = float(getattr(self.params, 'lambda_conf_smooth', 0.0) or 0.0)
            lambda_conf_mean = float(getattr(self.params, 'lambda_conf_mean', 0.0) or 0.0)
            if lambda_conf_smooth > 0:
                loss_conf_smooth = self._conf_tv_loss(conf_for_reg) * lambda_conf_smooth
                self.log("ConfSmooth_Loss", loss_conf_smooth.detach(), prog_bar=True)
                loss_G += loss_conf_smooth
            if lambda_conf_mean > 0:
                conf_min_mean = float(getattr(self.params, 'conf_min_mean', 0.5))
                loss_conf_mean = F.relu(conf_min_mean - conf_for_reg.mean()).pow(2) * lambda_conf_mean
                self.log("ConfMean_Loss", loss_conf_mean.detach(), prog_bar=True)
                loss_G += loss_conf_mean

        attn_aux = getattr(self.netG_A, "last_attn_aux", None)
        if attn_aux is not None:
            for scale_name in ("s3", "s4", "s5"):
                aux = attn_aux.get(scale_name, None)
                if aux is None:
                    continue
                if "entropy" in aux:
                    self.log(f"{scale_name}_attn_entropy", aux["entropy"], prog_bar=False)
                if "gate_attn" in aux:
                    self.log(f"{scale_name}_attn_gate", aux["gate_attn"], prog_bar=False)
                if "gate_ffn" in aux:
                    self.log(f"{scale_name}_ffn_gate", aux["gate_ffn"], prog_bar=False)
                if "exp_dx" in aux and "exp_dy" in aux and "exp_dz" in aux:
                    self.log(f"{scale_name}_abs_dx", aux["exp_dx"].abs().mean().detach(), prog_bar=False)
                    self.log(f"{scale_name}_abs_dy", aux["exp_dy"].abs().mean().detach(), prog_bar=False)
                    self.log(f"{scale_name}_abs_dz", aux["exp_dz"].abs().mean().detach(), prog_bar=False)

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
            if self.criterionGAN:
                pred_fake = self.netD_A(fb)
                loss_gan_b = self.criterionGAN(pred_fake, True) / D
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

        real_c = real_d = fake_c = fake_d = None
        real_b_ref = real_c_ref = real_d_ref = None
        use_25d = getattr(self.params, 'use_25d_style', False)
        need_ref = self.params.use_misalign_simul or use_25d

        if self.params.use_triple_outputs:
            optimizer_G_A, optimizer_D_A, optimizer_D_B, optimizer_D_C, optimizer_F_A = self.optimizers()
            if need_ref:
                real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref = self.model_step(batch)
            else:
                real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d = self.model_step(batch)
        elif self.params.use_multiple_outputs:
            optimizer_G_A, optimizer_D_A, optimizer_D_B, optimizer_F_A = self.optimizers()
            if need_ref:
                real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, real_b_ref, real_c_ref, real_d_ref = self.model_step(batch)
            else:
                real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d = self.model_step(batch)
        else:
            optimizer_G_A, optimizer_D_A, optimizer_F_A = self.optimizers()
            if need_ref:
                real_a, real_b, fake_b, real_b_ref = self.model_step(batch)
            else:
                real_a, real_b, fake_b = self.model_step(batch)

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

        optimizer_G_A = self.hparams.optimizer(params=self.netG_A.parameters())
        optimizers.append(optimizer_G_A)
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

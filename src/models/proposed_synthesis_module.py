import numpy as np

from typing import Any

import torch
import torch.nn.functional as F
from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss, VGG_Model
from src.losses.patch_nce_loss import PatchNCELoss
from src.losses.mind_loss import MINDLoss

from src import utils
from src.models.base_module_AtoB import BaseModule_AtoB
# from src.models.base_module_AtoB_multi import BaseModule_AtoB


log = utils.get_pylogger(__name__)

class ProposedSynthesisModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        netD_B: torch.nn.Module,
        netF_A: torch.nn.Module,
        optimizer,
        params,
        scheduler,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        
        # assign generator
        self.netG_A = netG_A
        self.netD_A = netD_A
        self.netD_B = netD_B
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
            # "conv_1_2": 1.0,
            # "conv_2_1": 1.0,
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

        # Style decomposition loss weights (for Shared-Private Style Decomposition)
        self.lambda_shared = getattr(params, 'lambda_shared', 1.0)
        self.lambda_ortho = getattr(params, 'lambda_ortho', 0.1)
        self.lambda_leak = getattr(params, 'lambda_leak', 0.05)
        self.use_style_decomposition = getattr(params, 'use_style_decomposition', False)

    def backward_G(self, real_a, real_b, real_c, fake_b, fake_c, real_b_ref, real_c_ref):
        loss_G = torch.tensor(0.0, device=real_a.device)
        if self.params.use_misalign_simul:
            real_b, real_c = real_b_ref, real_c_ref
        ##################################################################################################################
        ## 1. GAN Loss
        if self.criterionGAN:
            pred_fake = self.netD_A(fake_b)
            loss_gan_b = self.criterionGAN(pred_fake, True)
            self.log("Gan_b_Loss", loss_gan_b.detach(), prog_bar=True)
            loss_G += loss_gan_b
            # assert not torch.isnan(loss_gan_b).any(), "GAN Loss is NaN"
            
            if self.params.use_multiple_outputs:
                pred_fake = self.netD_B(fake_c)
                loss_gan_c = self.criterionGAN(pred_fake, True)
                self.log("Gan_c_Loss", loss_gan_c.detach(), prog_bar=True)
                loss_G += loss_gan_c
                # assert not torch.isnan(loss_gan_c).any(), "GAN Loss is NaN"

        ##################################################################################################################
        ## 2. Contextual loss: fake_b, fake_c 각각 따로
        if self.criterionContextual:
            loss_style_b = self.criterionContextual(real_b, fake_b)
            loss_style_b =  loss_style_b * self.params.lambda_style
            self.log("Context_b_Loss", loss_style_b.detach(), prog_bar=True)
            loss_G += loss_style_b.squeeze()
            # assert not torch.isnan(loss_style_b).any(), "Contextual Loss is NaN"

            if self.params.use_multiple_outputs:
                loss_style_c = self.criterionContextual(real_c, fake_c)
                loss_style_c =  loss_style_c * self.params.lambda_style
                self.log("Context_c_Loss", loss_style_c.detach(), prog_bar=True)
                loss_G += loss_style_c.squeeze()
                # assert not torch.isnan(loss_style_c).any(), "Contextual Loss is NaN"

        ##################################################################################################################
        ## 3. PatchNCE loss
        if self.criterionNCE:
            if self.params.nce_on_vgg: # 이거 안써
                real_rgb = real_a.repeat(1, 3, 1, 1)
                if self.params.nce_independent:
                    fake_rgb_b = fake_b.repeat(1, 3, 1, 1)
                    fake_rgb_c = fake_c.repeat(1, 3, 1, 1)
                elif fake_c is not None:
                    fake_rgb = torch.cat((fake_b, fake_c, torch.zeros_like(fake_b)), dim=1)
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
                if self.params.use_multiple_outputs:
                    n_layers = len(self.params.nce_layers)
                    if self.use_style_decomposition and hasattr(self.netG_A, 'encode_content_only'):
                        allowed_layers = {0, 1, 2}
                        if not set(self.params.nce_layers).issubset(allowed_layers):
                            raise ValueError("When use_style_decomposition=True, nce_layers must be a subset of [0, 1, 2].")

                        feat_a_all = self.netG_A.encode_content_only(real_a)
                        feat_b_all = self.netG_A.encode_content_only(fake_b)
                        feat_c_all = self.netG_A.encode_content_only(fake_c)
                        feat_a = [feat_a_all[i] for i in self.params.nce_layers]
                        feat_b = [feat_b_all[i] for i in self.params.nce_layers]
                        feat_c = [feat_c_all[i] for i in self.params.nce_layers]
                    else:
                        merged_input_1 = torch.cat((fake_b, real_a, real_a), dim=1)
                        feat_b = self.netG_A(merged_input_1, self.params.nce_layers, encode_only=True)

                        merged_input_2 = torch.cat((fake_c, real_a, real_a), dim=1)
                        feat_c = self.netG_A(merged_input_2, self.params.nce_layers, encode_only=True)

                        merged_input_2 = torch.cat((real_a, real_b, real_c), dim=1)
                        feat_a = self.netG_A(merged_input_2, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        feat_b = [torch.flip(fb, [3]) for fb in feat_b]
                        feat_c = [torch.flip(fc, [3]) for fc in feat_c]

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
                    merged_input_1 = torch.cat((fake_b, real_a), dim=1)
                    feat_b = self.netG_A(merged_input_1, self.params.nce_layers, encode_only=True)

                    flipped_for_equivariance = np.random.random() < 0.5
                    if self.params.flip_equivariance and flipped_for_equivariance:
                        feat_b = [torch.flip(fb, [3]) for fb in feat_b]

                    merged_input_2 = torch.cat((real_a, real_b), dim=1)
                    feat_a = self.netG_A(merged_input_2, self.params.nce_layers, encode_only=True)
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

            if self.params.use_multiple_outputs:
                loss_mind_c = self.criterionMIND(real_a, fake_c) * self.params.lambda_mind
                self.log("MIND_c_Loss", loss_mind_c.detach(), prog_bar=True)
                loss_G += loss_mind_c

        if self.criterionL1:
            loss_l1_b = self.criterionL1(real_b_ref, fake_b) * self.params.lambda_l1
            self.log("L1_b_Loss", loss_l1_b.detach(), prog_bar=True)
            loss_G += loss_l1_b

            if self.params.use_multiple_outputs:
                loss_l1_c = self.criterionL1(real_c_ref, fake_c) * self.params.lambda_l1
                self.log("L1_c_Loss", loss_l1_c.detach(), prog_bar=True)
                loss_G += loss_l1_c

        ##################################################################################################################
        ## Style Decomposition Losses (Shared-Private Style Decomposition + Conservative Routing)
        if self.use_style_decomposition and hasattr(self.netG_A, 'aux') and self.netG_A.aux:
            aux = self.netG_A.aux

            # 1. Shared Consistency Loss: compare the recovered shared components directly
            # This ensures the "shared" component captures truly common information
            if 'common_b' in aux and 'common_c' in aux:
                loss_shared = F.l1_loss(aux['common_b'], aux['common_c']) * self.lambda_shared
                self.log("Shared_Loss", loss_shared.detach(), prog_bar=True)
                loss_G += loss_shared

            # 2. Orthogonality Loss: private and shared components should be decorrelated
            # Prevents private from containing shared info and vice versa
            if 's_priv_b' in aux and 's_priv_c' in aux and 's_common' in aux:
                # Normalize features for cosine similarity computation
                def cosine_sq(a, b):
                    a_flat = a.view(a.size(0), -1)
                    b_flat = b.view(b.size(0), -1)
                    a_norm = F.normalize(a_flat, dim=1)
                    b_norm = F.normalize(b_flat, dim=1)
                    cos_sim = (a_norm * b_norm).sum(dim=1)
                    return (cos_sim ** 2).mean()

                s_common_flat = aux['s_common']
                ortho_b = cosine_sq(aux['s_priv_b'], s_common_flat)
                ortho_c = cosine_sq(aux['s_priv_c'], s_common_flat)
                loss_ortho = (ortho_b + ortho_c) * self.lambda_ortho
                self.log("Ortho_Loss", loss_ortho.detach(), prog_bar=True)
                loss_G += loss_ortho

            # 3. Structure Leakage Suppression Loss
            # Penalize correlation between source structure and the shared contribution
            # that is actually injected into each synthesis branch.
            if self.lambda_leak > 0 and ('shared_contrib_b' in aux or 's_common' in aux):
                if hasattr(self.netG_A, 'encode_content_only'):
                    struct_feat, _, _ = self.netG_A.encode_content_only(real_a)
                else:
                    merged_struct = torch.cat((real_a, real_a, real_a), dim=1)
                    struct_feats = self.netG_A(merged_struct, layers=[0], encode_only=True)
                    struct_feat = struct_feats[0] if struct_feats else None

                if struct_feat is not None:
                    if 'shared_contrib_b' in aux and 'shared_contrib_c' in aux:
                        shared_target = 0.5 * (aux['shared_contrib_b'] + aux['shared_contrib_c'])
                    else:
                        shared_target = aux['s_common']

                    if self.params.is_3d:
                        if struct_feat.shape[2:] != shared_target.shape[2:]:
                            struct_feat = F.adaptive_avg_pool3d(struct_feat, shared_target.shape[2:])
                        s_gap = F.adaptive_avg_pool3d(shared_target, 1).flatten(1)
                        f_gap = F.adaptive_avg_pool3d(struct_feat, 1).flatten(1)
                    else:
                        if struct_feat.shape[2:] != shared_target.shape[2:]:
                            struct_feat = F.adaptive_avg_pool2d(struct_feat, shared_target.shape[2:])
                        s_gap = F.adaptive_avg_pool2d(shared_target, 1).flatten(1)
                        f_gap = F.adaptive_avg_pool2d(struct_feat, 1).flatten(1)

                    # Compute correlation using normalized features
                    # We want to penalize if style and structure features have similar patterns
                    s_norm = F.normalize(s_gap, dim=1)
                    f_norm = F.normalize(f_gap, dim=1)

                    # Compute cross-correlation matrix between style and structure channels
                    # If they're uncorrelated, this should be close to zero
                    corr_matrix = torch.matmul(s_norm.unsqueeze(2), f_norm.unsqueeze(1))  # (B, C_s, C_f)

                    # Penalize high correlation
                    loss_leak = (corr_matrix ** 2).mean() * self.lambda_leak
                    self.log("Leak_Loss", loss_leak.detach(), prog_bar=True)
                    loss_G += loss_leak

            # Log gate activations for monitoring
            if 'alpha_b' in aux and 'alpha_c' in aux:
                alpha_b_mean = aux['alpha_b'].mean().detach()
                alpha_c_mean = aux['alpha_c'].mean().detach()
                self.log("Gate_b", alpha_b_mean, prog_bar=False)
                self.log("Gate_c", alpha_c_mean, prog_bar=False)

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

            ## GAN - Remove detach() so generator receives gradient
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

    def backward_D_B_3D(self, real_c, fake_c):
        """
        3D volume discriminator loss for domain C using the 2D discriminator slice-wise.
        """
        b, c, h, w, d = real_c.shape
        loss_D_total = 0.0

        if self.criterionGAN3D:
            pred_real = self.netD_B(real_c)
            loss_real = self.criterionGAN(pred_real, True)

            pred_fake = self.netD_B(fake_c.detach())
            loss_fake = self.criterionGAN(pred_fake, False)

            loss_D_total = (loss_real + loss_fake) * 0.5
        else:
            for i in range(d):
                real_slice = real_c[..., i]
                fake_slice = fake_c[..., i]

                pred_real = self.netD_B(real_slice)
                loss_real = self.criterionGAN(pred_real, True)

                pred_fake = self.netD_B(fake_slice.detach())
                loss_fake = self.criterionGAN(pred_fake, False)

                loss_D_total += (loss_real + loss_fake) * 0.5

            loss_D_total = loss_D_total / d
        return loss_D_total

    def training_step(self, batch: Any, batch_idx: int):
        
        if self.params.use_multiple_outputs:
            optimizer_G_A, optimizer_D_A, optimizer_D_B, optimizer_F_A = self.optimizers()

            if self.params.use_misalign_simul:
                real_a, real_b, real_c, _, fake_b, fake_c, _, real_b_ref, real_c_ref, _ = self.model_step(batch)
            else:
                real_a, real_b, real_c, _, fake_b, fake_c, _ = self.model_step(batch)
        else:
            optimizer_G_A, optimizer_D_A, optimizer_F_A = self.optimizers()
            if self.params.use_misalign_simul:
                real_a, real_b, fake_b, real_b_ref = self.model_step(batch)
            else:
                real_a, real_b, fake_b = self.model_step(batch)

        
        with optimizer_G_A.toggle_model():
            if self.params.use_multiple_outputs:
                if self.params.use_misalign_simul:
                    loss_G = self.backward_G(real_a, real_b, real_c, fake_b, fake_c, real_b_ref, real_c_ref)
                else:
                    loss_G = self.backward_G(real_a, real_b, real_c, fake_b, fake_c, None, None)
            else:
                if self.params.use_misalign_simul:
                    loss_G = self.backward_G(real_a, real_b, None, fake_b, None, real_b_ref, None)
                else:
                    if self.params.is_3d:
                        loss_G = self.backward_G_3D(real_a, real_b, None, None, fake_b, None, None, None, None, None)
                    else:
                        loss_G = self.backward_G(real_a, real_b, None, fake_b, None, None, None)
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

        with optimizer_D_A.toggle_model():
            if self.params.is_3d:
                loss_D_A = self.backward_D_A_3D(real_b, fake_b)
            else:
                loss_D_A = self.backward_D_A(real_b, fake_b)
            self.manual_backward(loss_D_A)
            self.clip_gradients(
                optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()
        self.log("D_A_Loss", loss_D_A.detach(), prog_bar=True)
        
        if self.params.use_multiple_outputs:
            with optimizer_D_B.toggle_model(): 
                if self.params.is_3d:
                    loss_D_B = self.backward_D_B_3D(real_c, fake_c)
                else:
                    loss_D_B = self.backward_D_B(real_c, fake_c)
                self.manual_backward(loss_D_B)
                self.clip_gradients(
                    optimizer_D_B, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_D_B.step()
                optimizer_D_B.zero_grad()
            self.log("D_B_Loss", loss_D_B.detach(), prog_bar=True)

        # with optimizer_D_D.toggle_model(): 
        #     loss_D_D = self.backward_D_D(real_d, fake_d)
        #     self.manual_backward(loss_D_D)
        #     self.clip_gradients(
        #         optimizer_D_D, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        #     )
        #     optimizer_D_D.step()
        #     optimizer_D_D.zero_grad()
        # self.log("D_D_Loss", loss_D_D.detach(), prog_bar=True)

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
        if self.params.use_multiple_outputs:
            optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())
            optimizers.append(optimizer_D_B)
        # optimizer_D_D = self.hparams.optimizer(params=self.netD_D.parameters())
        # optimizers.append(optimizer_D_D)
        optimizer_F_A = self.hparams.optimizer(params=self.netF_A.parameters())
        optimizers.append(optimizer_F_A)

        if self.hparams.scheduler is not None:
            scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
            schedulers.append(scheduler_G_A)
            scheduler_D_A = self.hparams.scheduler(optimizer=optimizer_D_A)
            schedulers.append(scheduler_D_A)
            if self.params.use_multiple_outputs:
                scheduler_D_B = self.hparams.scheduler(optimizer=optimizer_D_B)
                schedulers.append(scheduler_D_B)
            # # scheduler_D_D = self.hparams.scheduler(optimizer=optimizer_D_D)
            # schedulers.append(scheduler_D_D)
            scheduler_F_A = self.hparams.scheduler(optimizer=optimizer_F_A)
            schedulers.append(scheduler_F_A)
            return optimizers, schedulers
        
        return optimizers
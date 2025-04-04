from typing import Any
import numpy as np

import torch
from src.losses.gan_loss import GANLoss
from src.losses.mind_loss import MINDLoss
from src.losses.contextual_loss import Contextual_Loss, VGG_Model
from src.losses.occlusion_contextual_loss import OcclusionContextualLoss
from src.losses.patch_nce_loss import PatchNCELoss

from src.models.base_module_AtoB import BaseModule_AtoB
from src import utils

# from torch_ema import ExponentialMovingAverage

log = utils.get_pylogger(__name__)

class RegistFormerModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        # netD_B: torch.nn.Module,
        netF_A: torch.nn.Module,
        optimizer,
        params,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        self.netG_A = netG_A
        self.netD_A = netD_A
        # self.netD_B = netD_B
        self.netF_A = netF_A
        self.save_hyperparameters(logger=False, ignore=["netG_A", "netD_A"])
        self.automatic_optimization = False  # perform manual
        self.optimizer = optimizer
        self.params = params

        if self.params.nce_on_vgg: # vgg for patchNCE
            # choose layers what you want # "conv_1_2", "conv_2_2", "conv_3_4", "conv_4_4", "conv_5_4"
            listen_list = ["conv_4_2", "conv_5_4"] # PatchNCE는 MR을 반영하는 것. high level feature layer를 선택. low lever로 하면 mr의 feature가 그대로 많이 남을것
            self.vgg = VGG_Model(listen_list=listen_list)

        # loss function
        # style_feat_layers = {"conv_4_4": 1.0} 
        # style_feat_layers = {"conv_2_2": 1.0, "conv_3_2": 1.0, "conv_4_2": 1.0}
        style_feat_layers = {"conv_1_2": 1.0, "conv_2_2": 1.0, "conv_3_2": 1.0} # JBHI version # CTXLoss는 feature에서 matching point를 찾는거니 low lever로 해서 CT의 texture가 그대로 남아있어야 잘 반영될 것
        # style_feat_layers = {"conv_2_2": 1.0, "conv_3_2": 1.0, "conv_4_2": 1.0, "conv_4_4": 1.0} # Miccai version # Memory saving
        
        if params.flag_occlusionCTX:
            self.criterionCTX = OcclusionContextualLoss(flow_model_path=self.params.flow_model_path) if params.lambda_ctx != 0 else None
        else:
            self.criterionCTX = Contextual_Loss(style_feat_layers) if params.lambda_ctx != 0 else None

        self.criterionGAN = GANLoss(gan_type="lsgan") if params.lambda_gan != 0 else None  # gan_type = wgan, lsgan, wgangp ..

        self.criterionMIND = MINDLoss() if params.lambda_mind != 0 else None

        self.criterionNCE = PatchNCELoss(False, nce_T=0.07, batch_size=params.batch_size) if params.lambda_nce != 0 else None 
        
        self.criterionL1 = torch.nn.L1Loss() if params.lambda_l1 != 0 else None

        # PatchNCE specific initializations
        self.flip_equivariance = params.flip_equivariance

    def backward_G(self, real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d):
        loss_G = 0.0

        if self.criterionCTX:
            # Random translation to Ground truth
            if self.params.ran_shift > 0:
                shift_range_x, shift_range_y = 0, 0
                while shift_range_x == 0 and shift_range_y == 0:
                    shift_range_x = torch.randint(-self.params.ran_shift, self.params.ran_shift, (1,)).item()
                    shift_range_y = torch.randint(-self.params.ran_shift, self.params.ran_shift, (1,)).item()
                
                real_b = torch.roll(real_b, shifts=(shift_range_y,shift_range_x), dims=(2,3))
                if self.params.use_multiple_outputs:
                    real_c = torch.roll(real_c, shifts=(shift_range_y,shift_range_x), dims=(2,3))

            loss_CTX_b = self.criterionCTX(fake_b, real_b) * self.params.lambda_ctx
            loss_G += loss_CTX_b
            self.log("CTX_b_Loss", loss_CTX_b.detach(), prog_bar=True)

            if self.params.use_multiple_outputs:
                loss_CTX_c = self.criterionCTX(fake_c, real_c) * self.params.lambda_ctx
                loss_G += loss_CTX_c
                self.log("CTX_c_Loss", loss_CTX_c.detach(), prog_bar=True)

        if self.criterionGAN:
            pred_fake = self.netD_A(fake_b)
            loss_GAN_b = self.criterionGAN(pred_fake, True) * self.params.lambda_gan
            self.log("GAN_b_Loss", loss_GAN_b.detach(), prog_bar=True)
            loss_G += loss_GAN_b

            if self.params.use_multiple_outputs:
                pred_fake = self.netD_B(fake_c)
                loss_GAN_c = self.criterionGAN(pred_fake, True) * self.params.lambda_gan
                self.log("GAN_c_Loss", loss_GAN_c.detach(), prog_bar=True)
                loss_G += loss_GAN_c

        if self.criterionMIND:
            loss_MIND = self.criterionMIND(real_a, fake_b) * self.params.lambda_mind
            self.log("MIND_Loss", loss_MIND.detach(), prog_bar=True)
            loss_G += loss_MIND

        if self.criterionNCE:
            if self.params.nce_on_vgg:
                if real_a.shape[1] == 1 and fake_b.shape[1] == 1:
                    real_a_rgb = real_a.repeat(1, 3, 1, 1)
                    fake_b_rgb = fake_b.repeat(1, 3, 1, 1)
                    if self.params.use_multiple_outputs:
                        fake_c_rgb = fake_c.repeat(1, 3, 1, 1)
                self.vgg.to(real_a.device)
                feat_b = self.vgg(fake_b_rgb)
                feat_b = list(feat_b.values())
                if self.params.use_multiple_outputs:
                    feat_c = self.vgg(fake_c_rgb)
                    feat_c = list(feat_c.values())
            else:
                feat_b = self.netG_A(real_a, fake_b, for_nce=True, for_src=False)

            flipped_for_equivariance = np.random.random() < 0.5
            if self.flip_equivariance and flipped_for_equivariance:
                feat_b = [torch.flip(fb, [3]) for fb in feat_b]

            if self.params.nce_on_vgg:
                feat_a = self.vgg(real_a_rgb)
                feat_a = list(feat_a.values())
            else:
                feat_a = self.netG_A(real_a, real_b, for_nce=True, for_src=True)

            feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
            feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)
            if self.params.use_multiple_outputs:
                feat_c_pool, _ = self.netF_A(feat_c, 256, sample_ids)

            total_nce_loss = 0.0
            if self.params.use_multiple_outputs:
                for f_a, f_b, f_c in zip(feat_a_pool, feat_b_pool, feat_c_pool):
                    loss_NCE_ab = self.criterionNCE(f_a, f_b)
                    loss_NCE_ac = self.criterionNCE(f_a, f_c)
                    loss = (loss_NCE_ab + loss_NCE_ac) * self.params.lambda_nce
                    total_nce_loss += loss.mean()
                loss_NCE = total_nce_loss / (len(feat_b) + len(feat_c))
                self.log("NCE_ab_Loss", loss_NCE_ab.mean().detach(), prog_bar=True)
                self.log("NCE_ac_Loss", loss_NCE_ac.mean().detach(), prog_bar=True)
            else:
                for f_a, f_b in zip(feat_a_pool, feat_b_pool):
                    loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce
                    total_nce_loss += loss.mean()
                loss_NCE = total_nce_loss / len(feat_b)
                self.log("NCE_Loss", loss_NCE.detach(), prog_bar=True)
            loss_G += loss_NCE

        if self.criterionL1:
            loss_L1 = self.criterionL1(real_b, fake_b) * self.params.lambda_l1
            self.log("L1_Loss", loss_L1.detach(), prog_bar=True)
            loss_G += loss_L1

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):
        if self.params.lambda_gan != 0:
            if self.params.lambda_nce == 0:
                optimizer_G_A, optimizer_D_A = self.optimizers()
            else:
                if self.params.use_multiple_outputs:
                    optimizer_G_A, optimizer_D_A, optimizer_D_B, optimizer_F_A = self.optimizers()
                else:
                    optimizer_G_A, optimizer_D_A, optimizer_F_A = self.optimizers()
        else: #TODO: 이캐스는 multiple에 대한 코딩 덜됨
            if self.params.lambda_nce == 0:
                optimizer_G_A = self.optimizers()
            else:
                optimizer_G_A, optimizer_F_A = self.optimizers()

        if self.params.use_multiple_outputs:
            real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d = self.model_step(batch)
        else:
            real_a, real_b, fake_b = self.model_step(batch)

        # Renew
        with optimizer_G_A.toggle_model():
            optimizer_G_A.zero_grad() # 이것만 위로 올림
            if self.params.lambda_nce != 0:
                optimizer_F_A.zero_grad()
            if self.params.use_multiple_outputs:
                loss_G = self.backward_G(real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d)
            else:
                loss_G = self.backward_G(real_a, real_b, None, None, fake_b, None, None)
            self.manual_backward(loss_G)
            self.clip_gradients(
                optimizer_G_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            if self.params.lambda_nce != 0:
                self.clip_gradients(
                        optimizer_F_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
            optimizer_G_A.step()
            if self.params.lambda_nce != 0:
                optimizer_F_A.step()

        self.log("G_loss", loss_G.detach(), prog_bar=True)
        
        if self.params.lambda_gan != 0:
            with optimizer_D_A.toggle_model():
                optimizer_D_A.zero_grad()
                loss_D_A = self.backward_D_A(real_b, fake_b)
                self.manual_backward(loss_D_A)
                self.clip_gradients(
                    optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_D_A.step()

            self.log("Disc_Loss", loss_D_A.detach(), prog_bar=True)
        
            if self.params.use_multiple_outputs:
                with optimizer_D_B.toggle_model():
                    optimizer_D_B.zero_grad()
                    loss_D_B = self.backward_D_B(real_c, fake_c)
                    self.manual_backward(loss_D_B)
                    self.clip_gradients(
                        optimizer_D_B, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                    )
                    optimizer_D_B.step()
                
                self.log("Disc_Loss", loss_D_B.detach(), prog_bar=True)

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        
        optimizer_G_A = self.hparams.optimizer(params=self.netG_A.parameters())
        optimizers.append(optimizer_G_A)

        if self.params.lambda_gan != 0:
            optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())
            optimizers.append(optimizer_D_A)
            if self.params.use_multiple_outputs:
                optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())
                optimizers.append(optimizer_D_B)
        
        if self.params.lambda_nce != 0:
            optimizer_F_A = self.hparams.optimizer(params=self.netF_A.parameters())
            optimizers.append(optimizer_F_A)

        if self.hparams.scheduler is not None:
                scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
                schedulers.append(scheduler_G_A)
                if self.params.lambda_gan != 0:
                    scheduler_D_B = self.hparams.scheduler(optimizer=optimizer_D_A)
                    schedulers.append(scheduler_D_B)
                    if self.params.use_multiple_outputs:
                        scheduler_D_B = self.hparams.scheduler(optimizer=optimizer_D_B)
                        schedulers.append(scheduler_D_B)
                if self.params.lambda_nce != 0:
                    scheduler_F_A = self.hparams.scheduler(optimizer=optimizer_F_A)
                    schedulers.append(scheduler_F_A)
                return optimizers, schedulers
        
        return optimizers 
    
    # def load_state_dict(self, state_dict, strict: bool = True):
    #     # 1. DAM 파트는 strict=False로 따로 로딩
    #     if "DAM" in self.model.__dict__:
    #         dam_keys = [k for k in state_dict.keys() if "DAM" in k]
    #         dam_state = {k.replace("model.", ""): state_dict[k] for k in dam_keys}
    #         _ = self.model.DAM.load_state_dict(dam_state, strict=False)

    #     # 2. 나머지는 원래대로
    #     new_state = {k: v for k, v in state_dict.items() if "DAM" not in k}
    #     return super().load_state_dict(new_state, strict)
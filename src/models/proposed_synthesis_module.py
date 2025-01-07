import numpy as np

from typing import Any
import itertools

import torch
from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss, VGG_Model
from src.losses.patch_nce_loss import PatchNCELoss

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
        self.criterionContextual = Contextual_Loss(style_feat_layers)
        self.criterionGAN = GANLoss(gan_type='lsgan')
        self.criterionNCE = PatchNCELoss(False, nce_T=0.07, batch_size=params.batch_size)

        # PatchNCE specific initializations
        # self.nce_layers = [0,2,4,6] # range: 0~6
        # self.flip_equivariance = params.flip_equivariance

    def backward_G(self, real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, lambda_style, lambda_nce): # real_a, real_b, fake_b
        ##################################################################################################################
        ## 1. GAN Loss
        pred_fake = self.netD_A(fake_b.detach())
        loss_gan_b = self.criterionGAN(pred_fake, True)
        # assert not torch.isnan(loss_gan_b).any(), "GAN Loss is NaN"
        
        if self.params.use_multiple_outputs:
            pred_fake = self.netD_B(fake_c.detach())
            loss_gan_c = self.criterionGAN(pred_fake, True)
            # assert not torch.isnan(loss_gan_c).any(), "GAN Loss is NaN"
            if fake_d is not None:
                pred_fake = self.netD_D(fake_d.detach())
                loss_gan_d = self.criterionGAN(pred_fake, True)
                # assert not torch.isnan(loss_gan_d).any(), "GAN Loss is NaN"

        ##################################################################################################################
        ## 2. Contextual loss: fake_b, fake_c 각각 따로
        loss_style_b = self.criterionContextual(real_b, fake_b)
        loss_style_b =  loss_style_b * lambda_style
        # assert not torch.isnan(loss_style_b).any(), "Contextual Loss is NaN"

        if self.params.use_multiple_outputs:
            loss_style_c = self.criterionContextual(real_c, fake_c)
            loss_style_c =  loss_style_c * lambda_style
            # assert not torch.isnan(loss_style_c).any(), "Contextual Loss is NaN"
            if fake_d is not None:
                loss_style_d = self.criterionContextual(real_d, fake_d)
                loss_style_d =  loss_style_d * lambda_style
            # assert not torch.isnan(loss_style_d).any(), "Contextual Loss is NaN"

        ##################################################################################################################
        ## 3. PatchNCE loss: 이건 fake_b, fake_c 한꺼번에 해서 한번만 해. 이건 fake_d에 대한 코드 구현 필요.
        if self.params.nce_on_vgg:
            real_rgb = real_a.repeat(1, 3, 1, 1)
            if fake_d is not None:
                fake_rgb = torch.cat((fake_b, fake_c, fake_d), dim=1)
            else:
                fake_rgb = torch.cat((fake_b, fake_c, torch.zeros_like(fake_b)), dim=1) # Last channel is average -> zero (For checkerboard artifact but not sure)
            self.vgg.to(real_a.device)

            feat_b = self.vgg(fake_rgb)
            feat_b = list(feat_b.values()) # [0]:8,512,16,16 [1]:8,512,8,8

            feat_a = self.vgg(real_rgb)
            feat_a = list(feat_a.values())

            feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
            feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)

            total_nce_loss = 0.0

            for f_a, f_b in zip(feat_b_pool, feat_a_pool):
                loss = self.criterionNCE(f_a, f_b) * self.params.lambda_nce
                total_nce_loss = total_nce_loss + loss.mean()
            loss_nce_b = total_nce_loss / len(feat_b)

        else: # 이건 구현 덜됨. 나중에 필요할때 구현.
            n_layers = len(self.params.nce_layers)
            merged_input_1 = torch.cat((fake_b, real_a, real_a), dim=1)
            feat_b = self.netG_A(merged_input_1, self.params.nce_layers, encode_only=True)

            flipped_for_equivariance = np.random.random() < 0.5
            if self.params.flip_equivariance and flipped_for_equivariance:
                feat_b = [torch.flip(fb, [3]) for fb in feat_b]

            merged_input_2 = torch.cat((real_a, real_b, real_c), dim=1)
            feat_a = self.netG_A(merged_input_2, self.params.nce_layers, encode_only=True)
            feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
            feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)

            total_nce_loss = 0.0
            for f_a, f_b in zip(feat_b_pool, feat_a_pool):
                loss = self.criterionNCE(f_a, f_b) * lambda_nce
                total_nce_loss = total_nce_loss + loss.mean()
            loss_nce_b = total_nce_loss / n_layers
            assert not torch.isnan(loss_nce_b).any(), "NCE Loss is NaN"

        if self.params.use_multiple_outputs:
            loss_G = loss_gan_b + loss_gan_c + loss_style_b + loss_style_c + loss_nce_b # fake_d에 대한 loss는 추가안함. 필요할때.
            return loss_G, loss_gan_b, loss_gan_c, loss_style_b, loss_style_c, loss_nce_b
        else:
            loss_G = loss_gan_b + loss_style_b + loss_nce_b
            return loss_G, loss_gan_b, loss_style_b, loss_nce_b
        # assert not torch.isnan(loss_G).any(), "Total Loss is NaN"

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G_A, optimizer_D_A, optimizer_D_B, optimizer_F_A = self.optimizers()
        
        if self.params.use_multiple_outputs:
            real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d = self.model_step(batch)
        else:
            real_a, real_b, fake_b = self.model_step(batch)

        
        with optimizer_G_A.toggle_model():
            if self.params.use_multiple_outputs:
                loss_G, loss_gan_b, loss_gan_c, loss_style_b, loss_style_c, loss_nce_b = self.backward_G(real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d, self.params.lambda_style, self.params.lambda_nce)
            else:
                loss_G, loss_gan_b, loss_style_b, loss_nce_b = self.backward_G(real_a, real_b, fake_b, self.params.lambda_style, self.params.lambda_nce)
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

        # self.loss_G = loss_G.detach() * 0.1 + self.loss_G * 0.9
        self.log("G_loss", loss_G.detach(), prog_bar=True)
        self.log("loss_gan_b", loss_gan_b.detach(), prog_bar=True)
        if self.params.use_multiple_outputs:
            self.log("loss_gan_c", loss_gan_c.detach(), prog_bar=True)
        # self.log("loss_gan", loss_gan_d.detach(), prog_bar=True)
        self.log("loss_style_b", loss_style_b.detach(), prog_bar=True)
        if self.params.use_multiple_outputs:
            self.log("loss_style_c", loss_style_c.detach(), prog_bar=True)
        # self.log("loss_style", loss_style_d.detach(), prog_bar=True)
        self.log("loss_nce", loss_nce_b.detach(), prog_bar=True)
        # self.log("loss_nce", loss_nce_c.detach(), prog_bar=True)
        # self.log("loss_nce", loss_nce_d.detach(), prog_bar=True)

        with optimizer_D_A.toggle_model(): 
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
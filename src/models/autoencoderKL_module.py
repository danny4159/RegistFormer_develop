import numpy as np

from typing import Any
import itertools

import torch
from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss, VGG_Model
from src.losses.patch_nce_loss import PatchNCELoss
from src.losses.mind_loss import MINDLoss

from src import utils
from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src.models.base_module_AtoB import BaseModule_AtoB
# from src.models.base_module_AtoB_multi import BaseModule_AtoB

from monai.inferers import LatentDiffusionInferer
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from monai.networks.schedulers import DDPMScheduler


log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class AutoencoderKLModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        # netD_B: torch.nn.Module,
        # netF_A: torch.nn.Module,
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
        # self.netD_B = netD_B
        # self.netF_A = netF_A

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

        # loss function
        # self.criterionContextual = Contextual_Loss(style_feat_layers) if params.lambda_style != 0 else None
        # self.criterionNCE = PatchNCELoss(False, nce_T=0.07, batch_size=params.batch_size) if params.lambda_nce != 0 else None
        self.criterionGAN = PatchAdversarialLoss(criterion="least_squares") 
        self.criterionPeceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2) if params.lambda_percept !=0 else None
        self.criterionL1 = torch.nn.L1Loss() if params.lambda_recon != 0 else None
        # self.criterionMIND = MINDLoss() if params.lambda_mind != 0 else None
        
    def kl_loss(self, z_mu, z_sigma):
        klloss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(klloss) / klloss.shape[0]


    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G_A, optimizer_D_A = self.optimizers()
        optimizer_G_A.zero_grad(set_to_none=True)
        real_a, real_b, fake_a, z_mu, z_sigma = self.model_step(batch)

        loss_recon = self.criterionL1(fake_a.float().detach(), real_a.float().detach()) * self.params.lambda_recon
        loss_kl = self.kl_loss(z_mu, z_sigma) * self.params.lambda_kl
        loss_percept = self.criterionPeceptual(fake_a.float().detach(), real_a.float().detach()) * self.params.lambda_percept
        loss_g = loss_recon + loss_kl + loss_percept

        # Epoch 5까지는 GAN에서 Generator만 학습
        if self.current_epoch > 5: # autoencoder_warm_up_n_epochs
            logit_fake = self.netD_A(fake_a.float())[-1]
            generator_loss = self.criterionGAN(logit_fake, target_is_real=True, for_discriminator=False) * self.params.lambda_adv
            loss_g += generator_loss

        loss_g.backward()
        torch.cuda.synchronize()
        optimizer_G_A.step()

        self.log("loss_recon", loss_recon.detach(), prog_bar=True)
        self.log("loss_percept", loss_percept.detach(), prog_bar=True)
        self.log("loss_kl", loss_kl.detach(), prog_bar=True)
        self.log("loss_g", loss_g.detach(), prog_bar=True)

        del real_a, real_b, fake_a, z_mu, z_sigma, loss_g, loss_recon, loss_kl, loss_percept
        torch.cuda.empty_cache()


        if self.current_epoch > 5: # autoencoder_warm_up_n_epochs
            optimizer_D_A.zero_grad(set_to_none=True)
            logits_fake = self.netD_A(fake_a.detach())[-1]
            loss_d_fake = self.criterionGAN(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.netD_A(real_b.detach())[-1]
            loss_d_real = self.criterionGAN(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d = self.params.lambda_adv * loss_d
            loss_d.backward()
            optimizer_D_A.step()

            self.log("loss_d_fake", loss_d_fake.detach(), prog_bar=True)
            self.log("loss_d_real", loss_d_real.detach(), prog_bar=True)
            self.log("loss_d", loss_d.detach(), prog_bar=True)
        
            del loss_d, loss_d_fake, loss_d_real
            torch.cuda.empty_cache()
        
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

        if self.hparams.scheduler is not None:
            scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
            schedulers.append(scheduler_G_A)
            scheduler_D_A = self.hparams.scheduler(optimizer=optimizer_D_A)
            schedulers.append(scheduler_D_A)
            return optimizers, schedulers
        
        return optimizers
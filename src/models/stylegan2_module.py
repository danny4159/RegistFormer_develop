from typing import Any
import itertools

import torch
from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss
from src.losses.mind_loss import MINDLoss

from src import utils
from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src.models.base_module_AtoB import BaseModule_AtoB


log = utils.get_pylogger(__name__)

# Code: DAM Based
# Changed to GAN Based model
class StyleGAN2Module(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        optimizer,
        params,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        
        # assign generator
        self.netG_A = netG_A
        self.netD_A = netD_A

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params

        # assign contextual loss
        style_feat_layers = {
            "conv_2_2": 1.0,
            "conv_3_2": 1.0,
            "conv_4_2": 1.0,
            "conv_4_4": 1.0
        }

        # loss function
        self.style_loss = Contextual_Loss(style_feat_layers)
        self.criterionGAN = GANLoss(gan_type='lsgan')

    def backward_G(self, real_a, real_b, fake_b, lambda_style):
        
        #GAN Loss
        pred_fake = self.netD_A(fake_b.detach())
        loss_gan = self.criterionGAN(pred_fake, True)
        assert not torch.isnan(loss_gan).any(), "GAN Loss is NaN"

        ## Contextual loss
        loss_style_B = self.style_loss(real_b, fake_b)
        loss_style = (loss_style_B) * lambda_style
        assert not torch.isnan(loss_style).any(), "Contextual Loss is NaN"

        loss_G = loss_gan + loss_style

        return loss_G, loss_gan, loss_style

    def training_step(self, batch: Any, batch_idx: int):

        optimizer_G_A, optimizer_D_A = self.optimizers()
        real_a, real_b, fake_b = self.model_step(batch)

        with optimizer_G_A.toggle_model():
            loss_G, loss_gan, loss_style = self.backward_G(real_a, real_b, fake_b, self.params.lambda_style)
            self.manual_backward(loss_G)
            self.clip_gradients(
                    optimizer_G_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
            optimizer_G_A.step()
            optimizer_G_A.zero_grad()

        self.log("G_loss", loss_G.detach(), prog_bar=True)
        self.log("loss_gan", loss_gan.detach(), prog_bar=True)
        self.log("loss_style", loss_style.detach(), prog_bar=True)
            
        # self.loss_G = loss_G.detach() * 0.1 + self.loss_G * 0.9

        with optimizer_D_A.toggle_model(): 
            loss_D_A = self.backward_D_A(real_b, fake_b)
            self.manual_backward(loss_D_A)
            self.clip_gradients(
                optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()
        self.log("D_Loss", loss_D_A.detach(), prog_bar=True)


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
            scheduler_D_B = self.hparams.scheduler(optimizer=optimizer_D_A)
            schedulers.append(scheduler_D_B)
            return optimizers, schedulers
        
        return optimizers

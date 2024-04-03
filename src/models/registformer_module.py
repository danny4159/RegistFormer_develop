from typing import Any

import torch
from src.losses.gan_loss import GANLoss
from src.losses.scgan_loss import MINDLoss
from src.models.base_module_A_to_B import BaseModule_A_to_B
from src import utils

from src.losses.contextual_loss import (
    Contextual_Loss,
)  # this is the CX loss

log = utils.get_pylogger(__name__)

class RegistFormerModule(BaseModule_A_to_B):
    # 1. Regist: for real_B or fake_A
    # 2. meta-learning: batch or spatial
    # 3. Feature-descriptor: VGG or ResNet50(RadNet) or self-supervised

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        optimizer,
        params,
        **kwargs: Any
    ):
        super().__init__()
        self.netG_A = netG_A
        self.netD_A = netD_A
        self.save_hyperparameters(logger=False, ignore=["netG_A", "netD_A"])
        self.automatic_optimization = False  # perform manual
        self.params = params
        self.optimizer = optimizer

        # loss function
        style_feat_layers = {"conv_2_2": 1.0, "conv_3_2": 1.0, "conv_4_2": 1.0}
        self.criterionCTX = Contextual_Loss(style_feat_layers)

        self.criterionGAN = GANLoss(gan_mode="lsgan", reduce=False)

        self.criterionMIND = MINDLoss()

    def backward_G(self, real_a, real_b, fake_b):

        loss_CTX = self.criterionCTX(fake_b, real_b) * self.params.lambda_ctx

        pred_fake = self.netD_A(fake_b)
        loss_GAN = self.criterionGAN(pred_fake, True).mean() * self.params.lambda_gan

        loss_MIND = self.criterionMIND(real_a, fake_b) * self.params.lambda_mind

        loss_G = loss_CTX + loss_GAN + loss_MIND

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G, optimizer_D_A = self.optimizers()
        real_a, real_b, fake_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, fake_b)
            self.manual_backward(loss_G)
            self.clip_gradients(
                optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():
            loss_D_A = self.backward_D_A(real_b, fake_b)
            self.manual_backward(loss_D_A)
            self.clip_gradients(
                optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        self.log("G_loss", loss_G.detach(), prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer_G = self.hparams.optimizer(params=self.netG_A.parameters())

        optimizer_D = self.hparams.optimizer(params=self.netD_A.parameters())

        return optimizer_G, optimizer_D

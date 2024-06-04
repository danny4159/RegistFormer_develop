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

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
# class ProposedSynthesisModule(BaseModule_AtoB_BtoA):
class ProposedSynthesisModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        # netG_B: torch.nn.Module,
        netD_A: torch.nn.Module,
        optimizer,
        params,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        
        # assign generator
        self.netG_A = netG_A
        # self.netG_B = netG_B
        self.netD_A = netD_A

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params

        # assign contextual loss
        style_feat_layers = {
            # "conv_2_2": 1.0,
            # "conv_3_2": 1.0,
            # "conv_4_2": 1.0,
            "conv_4_4": 1.0
        }

        # loss function
        self.style_loss = Contextual_Loss(style_feat_layers)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMindFeature = MINDLoss()       
        self.criterionGAN = GANLoss(gan_type='lsgan')


    # def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_style, lambda_cycle_a, lambda_cycle_b, lambda_sc):
    def backward_G(self, real_a, real_b, fake_b, lambda_style, lambda_cycle_a, lambda_cycle_b, lambda_sc):
        ## GAN Loss 내가 추가했음
        pred_fake = self.netD_A(fake_b.detach())
        loss_gan = self.criterionGAN(pred_fake, True)

        ## Cycle loss
        # # MR > CT > MR
        # # rec_a = self.netG_A(fake_b, fake_a)
        # rec_a = self.netG_A(fake_b, real_a)
        # loss_cycle_A = self.criterionL1(real_a, rec_a) * lambda_cycle_a
        # # CT > MR > CT
        # # rec_b = self.netG_B(fake_a, fake_b) 
        # rec_b = self.netG_B(fake_a, real_b) 
        # loss_cycle_B = self.criterionL1(real_b, rec_b) * lambda_cycle_b
        
        ## Contextual loss
        # loss_style_A = self.style_loss(real_a, fake_a)
        loss_style_B = self.style_loss(real_b, fake_b)
        # loss_style = (loss_style_A + loss_style_B) * lambda_style
        loss_style =  loss_style_B * lambda_style

        ## Cycle Contextual loss (오히려 성능저하)
        # loss_cycle_style_A = self.style_loss(rec_b, real_b)
        # loss_cycle_style_B = self.style_loss(rec_a, real_a)
        # loss_cycle_style = (loss_cycle_style_A + loss_cycle_style_B) * lambda_cycle_style

        ## MIND feature loss
        loss_sc_A = self.criterionMindFeature(real_a, fake_b)
        # loss_sc_B = self.criterionMindFeature(real_b, fake_a)
        # loss_sc = (loss_sc_A + loss_sc_B) * lambda_sc
        loss_sc = loss_sc_A * lambda_sc

        ## L1 loss
        loss_l1 = self.criterionL1(real_b, fake_b)

        # loss_G = loss_style + loss_cycle_A + loss_cycle_B + loss_sc # + loss_cycle_style
        # loss_G = loss_style # + loss_sc # + loss_cycle_style
        # loss_G = loss_gan + loss_style + loss_sc
        loss_G = loss_gan + loss_style + loss_sc + loss_l1

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):

        optimizer_G = self.optimizers()
        optimizer_G, optimizer_D_A = self.optimizers()
        # real_a, real_b, fake_a, fake_b = self.model_step(batch)
        real_a, real_b, fake_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            # loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, self.params.lambda_style, self.params.lambda_cycle_a, self.params.lambda_cycle_b, self.params.lambda_sc)
            loss_G = self.backward_G(real_a, real_b, fake_b, self.params.lambda_style, self.params.lambda_cycle_a, self.params.lambda_cycle_b, self.params.lambda_sc)
            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()
            
        # self.loss_G = loss_G.detach() * 0.1 + self.loss_G * 0.9
        self.log("G_loss", loss_G.detach(), prog_bar=True)

        with optimizer_D_A.toggle_model(): 
            loss_D_A = self.backward_D_A(real_b, fake_b)
            self.manual_backward(loss_D_A)
            # self.clip_gradients(
            #     optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            # )
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()
        self.log("Disc_A_Loss", loss_D_A.detach(), prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        # optimizer_G = self.hparams.optimizer(params=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
        optimizer_G = self.hparams.optimizer(params=self.netG_A.parameters())
        optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())

        return optimizer_G, optimizer_D_A

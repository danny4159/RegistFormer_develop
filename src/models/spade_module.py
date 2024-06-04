from typing import Any

import torch
from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss
from src.losses.kld_loss import KLDLoss
from src.losses.mind_loss import MINDLoss

from src.models.base_module_AtoB import BaseModule_AtoB
from src import utils

log = utils.get_pylogger(__name__)

class SPADEModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        netE_A: torch.nn.Module,
        optimizer,
        params,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        self.netG_A = netG_A
        self.netD_A = netD_A
        self.netE_A = netE_A
        self.save_hyperparameters(logger=False, ignore=["netG_A", "netD_A", "netE_A"])
        self.automatic_optimization = False  # perform manual
        self.params = params
        self.optimizer = optimizer

        # loss function
        style_feat_layers = {"conv_4_4": 1.0} 
        
        self.criterionCTX = Contextual_Loss(style_feat_layers) if params.lambda_ctx != 0 else None
        self.criterionGAN = GANLoss(gan_type="lsgan") if params.lambda_gan != 0 else None  # gan_type = wgan, lsgan, wgangp ..
        self.criterionKLD = KLDLoss() if self.netG_A.use_vae else None
        self.criterionL1 = torch.nn.L1Loss() if params.lambda_l1 != 0 else None
        self.criterionMIND = MINDLoss() if params.lambda_mind != 0 else None

    def backward_G(self, real_a, real_b, fake_b, mu, logvar):
      
        loss_G = 0.0

        if self.criterionCTX:
            # Random translation to Ground truth
            if self.params.ran_shift is not None:
                shift_range_x = torch.randint(-self.params.ran_shift, self.params.ran_shift, (1,)).item()
                shift_range_y = torch.randint(-self.params.ran_shift, self.params.ran_shift, (1,)).item()
                real_b = torch.roll(real_b, shifts=(shift_range_y,shift_range_x), dims=(2,3))

                while shift_range_x == 0 and shift_range_y == 0:
                    shift_range_x = torch.randint(-self.params.ran_shift, self.params.ran_shift, (1,)).item()
                    shift_range_y = torch.randint(-self.params.ran_shift, self.params.ran_shift, (1,)).item()
        
            loss_CTX = self.criterionCTX(fake_b, real_b) * self.params.lambda_ctx
            self.log("CTX_Loss", loss_CTX.item(), prog_bar=True)
            loss_G += loss_CTX

        if self.criterionGAN:
            pred_fake = self.netD_A(fake_b)
            loss_GAN = self.criterionGAN(pred_fake, True) * self.params.lambda_gan
            self.log("GAN_Loss", loss_GAN.item(), prog_bar=True)
            loss_G += loss_GAN

        if self.criterionKLD:
            loss_KLD = self.criterionKLD(mu, logvar) * self.params.lambda_kld
            self.log("KLD_Loss", loss_KLD.item(), prog_bar=True)
            loss_G += loss_KLD

        if self.criterionL1:
            loss_L1 = self.criterionL1(fake_b, real_b) * self.params.lambda_l1
            self.log("L1_Loss", loss_L1.item(), prog_bar=True)
            loss_G += loss_L1
        
        if self.criterionMIND:
            loss_MIND = self.criterionMIND(real_a, fake_b) * self.params.lambda_mind
            self.log("MIND_Loss", loss_MIND.item(), prog_bar=True)
            loss_G += loss_MIND

        return loss_G
    
    def model_step(self, batch: Any):
        real_a, real_b = batch

        mu, logvar = self.netE_A(real_b)

        z = self.reparameterize(mu, logvar)
        
        fake_b = self.netG_A(real_a, z) #TODO: netG_A가 real_b를 잘 처리하도록. 원래는 seg map

        return real_a, real_b, fake_b, mu, logvar
    
    # def model_step(self, batch: Any):
    #     real_a, real_b = batch

    #     mu, logvar = self.netE_A(real_a)

    #     z = self.reparameterize(mu, logvar)
        
    #     fake_b = self.netG_A(real_b, z) #TODO: netG_A가 real_b를 잘 처리하도록. 원래는 seg map

    #     return real_a, real_b, fake_b, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu
    
    def training_step(self, batch: Any, batch_idx: int):
        if self.params.lambda_gan != 0:
            optimizer_G, optimizer_D_A  = self.optimizers()
        else:
            optimizer_G = self.optimizers()
        real_a, real_b, fake_b, mu, logvar = self.model_step(batch)
    
        # Renew
        with optimizer_G.toggle_model():
            optimizer_G.zero_grad() 
            loss_G = self.backward_G(real_a, real_b, fake_b, mu, logvar)
            self.manual_backward(loss_G)
            self.clip_gradients(
                optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_G.step()
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

    def configure_optimizers(self):
        G_params = list(self.netG_A.parameters())
        if self.netG_A.use_vae:
            G_params += list(self.netE_A.parameters())
        optimizer_G = self.hparams.optimizer(params=G_params)
        optimizers = [optimizer_G]
        # schedulers = []

        if self.params.lambda_gan != 0:
            optimizer_D = self.hparams.optimizer(params=self.netD_A.parameters())
            optimizers.append(optimizer_D)

        # if self.hparams.scheduler is not None:
        #     scheduler_G = self.hparams.scheduler(optimizer=optimizer_G)
        #     schedulers.append(scheduler_G)

        #     if self.params.lambda_gan != 0:
        #         scheduler_D = self.hparams.scheduler(optimizer=optimizer_D)
        #         schedulers.append(scheduler_D)
            
            # return optimizers, schedulers

        return optimizers
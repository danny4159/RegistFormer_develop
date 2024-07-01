import torch
from typing import Any
import itertools
from torch.autograd import Variable
import random

from src.losses.gan_loss import GANLoss
from src.losses.perceptual_loss import Perceptual_Loss
from src.losses.contextual_loss import Contextual_Loss

from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src import utils
# from torchsummary import summary


log = utils.get_pylogger(__name__)

class UnitModule(BaseModule_AtoB_BtoA):
    def __init__(
        self,
        netG_A: torch.nn.Module,
        netG_B: torch.nn.Module,
        netD_A: torch.nn.Module,
        netD_B: torch.nn.Module,
        optimizer,
        params,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)

        # assign generator
        self.netG_A = netG_A
        self.netG_B = netG_B
        # assign discriminator
        self.netD_A = netD_A
        self.netD_B = netD_B

        self.save_hyperparameters(logger=False, ignore=["netG_A", "netD_A","netG_B", "netD_B"])
        self.automatic_optimization = False  # perform manual
        self.optimizer = optimizer
        self.params = params

        # fix the noise used in sampling
        valid_size = 1 # TODO: hyperparameters['valid_size']
        self.style_dim = 8 # TODO:  hyperparameters['gen']['style_dim']
        self.s_a = torch.randn(valid_size, self.style_dim)
        self.s_b = torch.randn(valid_size, self.style_dim)
        
        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(gan_type='lsgan')
        self.criterionRecon = torch.nn.L1Loss()
        self.criterionPerceptual = Perceptual_Loss(spatial_dims=2, network_type="radimagenet_resnet50")
        style_feat_layers = {
            "conv_2_2": 1.0,
            "conv_3_2": 1.0,
            "conv_4_2": 1.0
        }
        # self.contextual_loss = Contextual_Loss(style_feat_layers)

        
        # Image Pool
        self.fake_A_pool = ImagePool(params.pool_size)
        self.fake_B_pool = ImagePool(params.pool_size)
    
    def __compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
        
    def backward_G(self, real_a, real_b, hidden_a, hidden_b, recon_a, recon_b, recon_b_a, recon_a_b, hidden_b_a, hidden_a_b, recon_a_b_a, recon_b_a_b):
        loss_G = 0.0

        # loss GAN
        pred_fake = self.netD_A(recon_b_a)
        loss_G_adv_a = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B(recon_a_b)
        loss_G_adv_b = self.criterionGAN(pred_fake, True)
        loss_GAN = (loss_G_adv_a + loss_G_adv_b)
        self.log("GAN_Loss", loss_GAN.detach(), prog_bar=True)
        loss_G += loss_GAN

        # loss content (recon)
        loss_G_recon_a = self.criterionRecon(recon_a, real_a)
        loss_G_recon_b = self.criterionRecon(recon_b, real_b)
        loss_content = (loss_G_recon_a + loss_G_recon_b) * self.params.lambda_content
        self.log("Content_Loss", loss_content.detach(), prog_bar=True)
        loss_G += loss_content

        # loss kl
        loss_G_recon_kl_a = self.__compute_kl(hidden_a)
        loss_G_recon_kl_b = self.__compute_kl(hidden_b)
        loss_kl = (loss_G_recon_kl_a + loss_G_recon_kl_b) * self.params.lambda_kl
        self.log("KL_Loss", loss_kl.detach(), prog_bar=True)
        loss_G += loss_kl
        # loss cycle
        loss_G_cyc_recon_a = self.criterionRecon(recon_a_b_a, real_a)
        loss_G_cyc_recon_b = self.criterionRecon(recon_b_a_b, real_b)
        loss_cycle = (loss_G_cyc_recon_a + loss_G_cyc_recon_b) * self.params.lambda_cycle
        self.log("Cycle_Loss", loss_cycle.detach(), prog_bar=True)
        loss_G += loss_cycle

        # loss kl cross
        loss_G_recon_kl_ab = self.__compute_kl(hidden_a_b)
        loss_G_recon_kl_ba = self.__compute_kl(hidden_b_a)
        loss_kl_cross = (loss_G_recon_kl_ab + loss_G_recon_kl_ba) * self.params.lambda_kl_cross
        self.log("KL_cross_Loss", loss_kl_cross.detach(), prog_bar=True)
        loss_G += loss_kl_cross

        # perceptual loss (extra)
        if self.params.lambda_perceptual != 0:
            loss_G_vgg_a = self.criterionPerceptual(recon_b_a, real_b)
            loss_G_vgg_b = self.criterionPerceptual(recon_a_b, real_a)
            loss_perceptual = (loss_G_vgg_a + loss_G_vgg_b) * self.params.lambda_perceptual
            self.log("Percept_Loss", loss_perceptual.detach(), prog_bar=True)
            loss_G += loss_perceptual

        # contextual loss (extra)
        if self.params.lambda_contextual != 0:
            loss_contextual_a = torch.mean(self.contextual_loss(real_a, recon_b_a))
            loss_contextual_b = torch.mean(self.contextual_loss(real_b, recon_a_b))
            loss_contextual = (loss_contextual_a + loss_contextual_b) * self.params.lambda_contextual
            self.log("Context_Loss", loss_contextual.detach(), prog_bar=True)
            loss_G += loss_contextual

        return loss_G

    def model_step_unit(self, batch: Any):
        real_a, real_b = batch

        hidden_a, noise_a = self.netG_A.encode(real_a)
        hidden_b, noise_b = self.netG_B.encode(real_b)
        # decode (within domain)
        recon_a = self.netG_A.decode(hidden_a + noise_a)
        recon_b = self.netG_B.decode(hidden_b + noise_b)
        # decode (cross domain)
        recon_b_a = self.netG_A.decode(hidden_b + noise_b)
        recon_a_b = self.netG_B.decode(hidden_a + noise_a)
        # encode again
        hidden_b_a, noise_b_a = self.netG_A.encode(recon_b_a)
        hidden_a_b, noise_a_b = self.netG_B.encode(recon_a_b)
        # decode again (if needed)
        recon_a_b_a = self.netG_A.decode(hidden_a_b + noise_a_b)
        recon_b_a_b = self.netG_B.decode(hidden_b_a + noise_b_a)

        return real_a, real_b, hidden_a, hidden_b, recon_a, recon_b, recon_b_a, recon_a_b, hidden_b_a, hidden_a_b, recon_a_b_a, recon_b_a_b

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, hidden_a, hidden_b, recon_a, recon_b, recon_b_a, recon_a_b, hidden_b_a, hidden_a_b, recon_a_b_a, recon_b_a_b = self.model_step_unit(batch)
        
        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, hidden_a, hidden_b, recon_a, recon_b, recon_b_a, recon_a_b, hidden_b_a, hidden_a_b, recon_a_b_a, recon_b_a_b)
            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()
        self.log("G_loss", loss_G.detach(), prog_bar=True)

        with optimizer_D_A.toggle_model():        
            loss_D_A = self.backward_D_A(real_a, recon_b_a)
            self.manual_backward(loss_D_A)
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()
        self.log("Disc_A_Loss", loss_D_A.detach(), prog_bar=True)

        with optimizer_D_B.toggle_model():
            loss_D_B = self.backward_D_B(real_b, recon_a_b)
            self.manual_backward(loss_D_B)
            optimizer_D_B.step()
            optimizer_D_B.zero_grad()
        self.log("Disc_B_Loss", loss_D_B.detach(), prog_bar=True)
        

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer_G = self.hparams.optimizer(params=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
        optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())
        optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())

        return optimizer_G, optimizer_D_A, optimizer_D_B


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


if __name__ == "__main__":
    _ = UnitModule(None, None, None)
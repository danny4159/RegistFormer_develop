import torch
from typing import Any
import itertools
from torch.autograd import Variable
import random

from src.losses.gan_loss import GANLoss
from src.losses.mind_loss import MINDLoss

from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src import utils
# from torchsummary import summary


log = utils.get_pylogger(__name__)

class SCCycleGANModule(BaseModule_AtoB_BtoA):
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
        
        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(gan_type='lsgan')
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdentity = torch.nn.L1Loss()
        # Added in SC-GAN
        self.criterionMindFeature = MINDLoss()
        
        # Image Pool
        self.fake_A_pool = ImagePool(params.pool_size)
        self.fake_B_pool = ImagePool(params.pool_size)
    
    def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_cycle=100, lambda_identity=1, lambda_sc=1):
        # GAN loss D_A(G_A(A))
        pred_fake = self.netD_A(fake_b)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        pred_fake = self.netD_B(fake_a)
        loss_G_B = self.criterionGAN(pred_fake, True)

        loss_G = (loss_G_A + loss_G_B)
        
        # Forward cycle loss
        rec_A = self.netG_B(fake_b)
        loss_cycle_A = self.criterionCycle(rec_A, real_a) * lambda_cycle
        
        # Backward cycle loss
        rec_B = self.netG_A(fake_a)
        loss_cycle_B = self.criterionCycle(rec_B, real_b) * lambda_cycle
        loss_cycle = (loss_cycle_A + loss_cycle_B)
        
        # Identity loss
        rec_A = self.netG_B(fake_b)
        loss_identity_A = self.criterionCycle(rec_A, real_a) * lambda_identity

        # Identity loss
        rec_B = self.netG_A(fake_a)
        loss_identity_B = self.criterionCycle(rec_B, real_b) * lambda_identity
        
        loss_identity = (loss_identity_A + loss_identity_B)
        
        # SC loss (L1 distance of Mind feature between image 1 and 2)\
        loss_sc_A = self.criterionMindFeature(real_a, fake_b) * lambda_sc
        loss_sc_B = self.criterionMindFeature(real_b, fake_a) * lambda_sc
        loss_sc = (loss_sc_A + loss_sc_B)
        
        # combined loss
        loss_G = loss_G + loss_cycle + loss_identity + loss_sc
        
        return loss_G
    

    def training_step(self, batch: Any, batch_idx: int):
        
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, fake_a, fake_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, self.params.lambda_cycle, self.params.lambda_identity, self.params.lambda_sc)
            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():        
            loss_D_A = self.backward_D_A(real_b, fake_b)
            self.manual_backward(loss_D_A)
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()
        
        with optimizer_D_B.toggle_model():                
            loss_D_B = self.backward_D_B(real_a, fake_a)
            self.manual_backward(loss_D_B)
            optimizer_D_B.step()
            optimizer_D_B.zero_grad()
        self.loss_G = loss_G.detach() * 0.1 + self.loss_G * 0.9
        self.log("G_loss", self.loss_G, prog_bar=True)


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
    _ = SCCycleGANModule(None, None, None)
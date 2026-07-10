from typing import Any
import itertools
import torch.nn as nn
import torch.nn.functional as F

import torch
from src.losses.gan_loss import GANLoss
from torch.autograd import Variable

from src import utils

import random
from torchvision import models
from src.models.base_module_AtoB_BtoA import (
    BaseModule_AtoB_BtoA,
    clip_to_valid_range,
    norm_to_uint8,
)
from src.models.components.component_regpgan import *

from reprlib import recursive_repr

log = utils.get_pylogger(__name__)

gray2rgb = lambda x: torch.cat((x, x, x), dim=1)


class ResViTModule(BaseModule_AtoB_BtoA):
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

        self.automatic_optimization = False  # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.params = params
        self.optimizer = optimizer

        # Image Pool
        self.fake_AB_pool = ImagePool(params.pool_size)
        self.fake_BA_pool = ImagePool(params.pool_size)

        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(gan_type='lsgan')
        self.criterionL1 = torch.nn.L1Loss()


    def _unpack_batch(self, batch):
        """Normal batch: (real_a, real_b). When params.use_eval_ref is set
        (data.data_group_3 carries the true, registered T2, used only for
        metric computation, never for the training loss), batch instead is
        (real_a, real_b, eval_ref)."""
        if getattr(self.params, "use_eval_ref", False) and len(batch) == 3:
            real_a, real_b, eval_ref = batch
            return real_a, real_b, eval_ref
        real_a, real_b = batch
        return real_a, real_b, None

    def model_step(self, batch: Any):
        """Override: strip eval_ref (if present) before delegating to the
        normal ResViT forward pass, so the training loss always uses real_b
        (the actual training target, e.g. T2_proposed/T2_moved), never eval_ref."""
        real_a, real_b, _eval_ref = self._unpack_batch(batch)
        fake_b, fake_a = self.forward(real_a, real_b)
        return real_a, real_b, fake_a, fake_b

    def validation_step(self, batch: Any, batch_idx: int):
        real_a, real_b, eval_ref = self._unpack_batch(batch)
        fake_b, fake_a = self.forward(real_a, real_b)
        metric_ref_b = eval_ref if eval_ref is not None else real_b

        self.val_ssim_A.update(real_a, fake_a)
        self.val_psnr_A.update(real_a, fake_a)
        self.psnr_values_A.append(self.val_psnr_A.compute().item())
        self.val_psnr_A.reset()
        self.val_lpips_A.update(gray2rgb(clip_to_valid_range(real_a)), gray2rgb(clip_to_valid_range(fake_a)))
        self.lpips_values_A.append(self.val_lpips_A.compute().item())
        self.val_lpips_A.reset()
        self.val_sharpness_A.update(norm_to_uint8(fake_a).float())

        self.val_ssim_B.update(metric_ref_b, fake_b)
        self.val_psnr_B.update(metric_ref_b, fake_b)
        self.psnr_values_B.append(self.val_psnr_B.compute().item())
        self.val_psnr_B.reset()
        self.val_lpips_B.update(gray2rgb(clip_to_valid_range(metric_ref_b)), gray2rgb(clip_to_valid_range(fake_b)))
        self.lpips_values_B.append(self.val_lpips_B.compute().item())
        self.val_lpips_B.reset()
        self.val_sharpness_B.update(norm_to_uint8(fake_b).float())

    def test_step(self, batch: Any, batch_idx: int):
        real_a, real_b, eval_ref = self._unpack_batch(batch)
        fake_b, fake_a = self.forward(real_a, real_b)
        metric_ref_b = eval_ref if eval_ref is not None else real_b

        self.test_ssim_A.update(real_a, fake_a)
        self.test_psnr_A.update(real_a, fake_a)
        self.psnr_values_A.append(self.test_psnr_A.compute().item())
        self.test_psnr_A.reset()
        self.test_lpips_A.update(gray2rgb(clip_to_valid_range(real_a)), gray2rgb(clip_to_valid_range(fake_a)))
        self.lpips_values_A.append(self.test_lpips_A.compute().item())
        self.test_lpips_A.reset()
        self.test_sharpness_A.update(norm_to_uint8(fake_a).float())

        self.test_ssim_B.update(metric_ref_b, fake_b)
        self.test_psnr_B.update(metric_ref_b, fake_b)
        self.psnr_values_B.append(self.test_psnr_B.compute().item())
        self.test_psnr_B.reset()
        self.test_lpips_B.update(gray2rgb(clip_to_valid_range(metric_ref_b)), gray2rgb(clip_to_valid_range(fake_b)))
        self.lpips_values_B.append(self.test_lpips_B.compute().item())
        self.test_lpips_B.reset()
        self.test_sharpness_B.update(norm_to_uint8(fake_b).float())

    def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_l1):
        fake_ab = torch.cat((real_a, fake_b), 1) # [12, 1, 256, 256] + [12, 1, 256, 256]
        pred_fake_ab = self.netD_B(fake_ab)
        loss_GAN_AB = self.criterionGAN(pred_fake_ab, True)

        fake_ba = torch.cat((real_b, fake_a), 1)
        pred_fake_ba = self.netD_A(fake_ba)
        loss_GAN_BA = self.criterionGAN(pred_fake_ba, True)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) * 0.5

        loss_L1 = (self.criterionL1(fake_a, real_a) + self.criterionL1(fake_b, real_b) * lambda_l1) * 0.5
       
        loss_G = loss_GAN + loss_L1

        return loss_G

    def backward_D_A(self, real_a, real_b, fake_a, fake_b):
        fake_ba = self.fake_BA_pool.query(torch.cat((real_b, fake_a), 1).data)
        pred_fake_ba = self.netD_A(fake_ba.detach())
        loss_D_A_fake = self.criterionGAN(pred_fake_ba, False)

        real_ba = torch.cat((real_b, real_a), 1)
        pred_real_ba = self.netD_A(real_ba)
        loss_D_A_real = self.criterionGAN(pred_real_ba, True)

        loss_D_A = (loss_D_A_fake + loss_D_A_real) * 0.5
        
        return loss_D_A
    
    def backward_D_B(self, real_a, real_b, fake_a, fake_b):
        fake_ab = self.fake_AB_pool.query(torch.cat((real_a,fake_b), 1).data)
        pred_fake_ab = self.netD_B(fake_ab.detach())
        loss_D_B_fake = self.criterionGAN(pred_fake_ab, False)

        real_ab = torch.cat((real_a, real_b), 1)
        pred_real_ab = self.netD_B(real_ab)
        loss_D_B_real = self.criterionGAN(pred_real_ab, True)

        loss_D_B = (loss_D_B_fake + loss_D_B_real) * 0.5
    
        return loss_D_B


    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, fake_a, fake_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, self.params.lambda_l1)
            self.manual_backward(loss_G)
            self.clip_gradients(optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():
            loss_D_A = self.backward_D_A(real_a, real_b, fake_a, fake_b)
            self.manual_backward(loss_D_A)
            self.clip_gradients(optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        with optimizer_D_B.toggle_model():
            loss_D_B = self.backward_D_B(real_a, real_b, fake_a, fake_b)
            self.manual_backward(loss_D_B)
            self.clip_gradients(optimizer_D_B, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_D_B.step()
            optimizer_D_B.zero_grad()
        self.log("G_loss", loss_G.detach(), prog_bar=True)

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
    _ = ResViTModule(None, None, None)

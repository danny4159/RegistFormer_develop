import numpy as np

from typing import Any
import itertools
from torch.autograd import Variable
import random

import torch
from src.losses.gan_loss import GANLoss
from src.losses.perceptual_loss import Perceptual_Loss

from src import utils
from src.models.base_module_AtoB import BaseModule_AtoB


 
log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class PixelGANModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        netD_B: torch.nn.Module,
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

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

        # loss function
        self.criterionGAN = GANLoss(gan_type='lsgan')
        self.criterionL1 = torch.nn.L1Loss(reduction="none")
        self.criterionPerceptual = Perceptual_Loss(spatial_dims=2, network_type="radimagenet_resnet50")
        style_feat_layers = {
            "conv_2_2": 1.0,
            "conv_3_2": 1.0,
            "conv_4_2": 1.0
        }

    def backward_G(self, real_a, real_b, real_c, fake_b, fake_c, lambda_l1, lambda_percept):
        
        if self.params.use_multiple_outputs:
            ## GAN loss
            pred_fake_b = self.netD_A(fake_b.detach())
            loss_gan_b = self.criterionGAN(pred_fake_b, True)
            loss_gan_b = torch.mean(loss_gan_b)
            assert not torch.isnan(loss_gan_b).any(), "GAN Loss is NaN"
            
            pred_fake_c = self.netD_B(fake_c.detach())
            loss_gan_c = self.criterionGAN(pred_fake_c, True)
            loss_gan_c = torch.mean(loss_gan_c)
            assert not torch.isnan(loss_gan_c).any(), "GAN Loss is NaN"

            loss_gan = (loss_gan_b + loss_gan_c) / 2 

            ## l1 loss
            loss_l1_b = self.criterionL1(fake_b, real_b) * lambda_l1
            loss_l1_b = torch.mean(loss_l1_b)
            assert not torch.isnan(loss_l1_b).any(), "L1 Loss is NaN"
            
            loss_l1_c = self.criterionL1(fake_c, real_c) * lambda_l1
            loss_l1_c = torch.mean(loss_l1_c)
            assert not torch.isnan(loss_l1_c).any(), "L1 Loss is NaN"

            loss_l1 = (loss_l1_b + loss_l1_c) / 2 

            ## Percept loss
            loss_percept_b = self.criterionPerceptual(real_b, fake_b)
            loss_percept_b =  loss_percept_b * lambda_percept
            loss_percept_b = torch.mean(loss_percept_b)
            assert not torch.isnan(loss_percept_b).any(), "Percept Loss is NaN"
            
            loss_percept_c = self.criterionPerceptual(real_c, fake_c)
            loss_percept_c =  loss_percept_c * lambda_percept
            loss_percept_c = torch.mean(loss_percept_c)
            assert not torch.isnan(loss_percept_c).any(), "Percept Loss is NaN"

            loss_percept = (loss_percept_b + loss_percept_c) / 2

            ## Total loss
            loss_G = loss_gan + loss_l1 + loss_percept
            assert not torch.isnan(loss_G).any(), "Total Loss is NaN"

        else:

            ## GAN loss
            pred_fake = self.netD_A(fake_b.detach())
            loss_gan = self.criterionGAN(pred_fake, True)
            loss_gan = torch.mean(loss_gan)
            assert not torch.isnan(loss_gan).any(), "GAN Loss is NaN"

            ## l1 loss
            loss_l1 = self.criterionL1(fake_b, real_b) * lambda_l1
            loss_l1 = torch.mean(loss_l1)
            assert not torch.isnan(loss_l1).any(), "L1 Loss is NaN"

            ## Percept loss
            loss_percept = self.criterionPerceptual(real_b, fake_b)
            loss_percept =  loss_percept * lambda_percept
            loss_percept = torch.mean(loss_percept)
            assert not torch.isnan(loss_percept).any(), "Percept Loss is NaN"

            ## Total loss
            loss_G = loss_gan + loss_l1 + loss_percept
            assert not torch.isnan(loss_G).any(), "Total Loss is NaN"

        return loss_G, loss_gan, loss_l1, loss_percept

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G_A, optimizer_D_A, optimizer_D_B = self.optimizers()

        if self.params.use_multiple_outputs:
            real_a, real_b, real_c, fake_b, fake_c = self.model_step(batch)
            
            with optimizer_G_A.toggle_model():
                loss_G, loss_gan, loss_l1, loss_percept = self.backward_G(real_a, real_b, real_c, fake_b, fake_c, self.params.lambda_l1, self.params.lambda_percept)
                self.manual_backward(loss_G)
                self.clip_gradients(
                    optimizer_G_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_G_A.step()
                optimizer_G_A.zero_grad()

            self.log("G_loss", loss_G.detach(), prog_bar=True)
            self.log("loss_gan", loss_gan.detach(), prog_bar=True)
            self.log("loss_l1", loss_l1.detach(), prog_bar=True)
            self.log("loss_percept", loss_percept.detach(), prog_bar=True)

            with optimizer_D_A.toggle_model(): 
                loss_D_A = self.backward_D_A(real_b, fake_b)
                self.manual_backward(loss_D_A)
                self.clip_gradients(
                    optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_D_A.step()
                optimizer_D_A.zero_grad()
            self.log("D_Loss", loss_D_A.detach(), prog_bar=True)
            
            with optimizer_D_B.toggle_model(): 
                loss_D_B = self.backward_D_B(real_c, fake_c)
                self.manual_backward(loss_D_B)
                self.clip_gradients(
                    optimizer_D_B, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_D_B.step()
                optimizer_D_B.zero_grad()
            self.log("D_Loss", loss_D_B.detach(), prog_bar=True)


        else:
            real_a, real_b, fake_b = self.model_step(batch)
        
            with optimizer_G_A.toggle_model():
                loss_G, loss_gan, loss_l1, loss_percept = self.backward_G(real_a, real_b, fake_b, self.params.lambda_l1, self.params.lambda_percept)
                self.manual_backward(loss_G)
                self.clip_gradients(
                    optimizer_G_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
                )
                optimizer_G_A.step()
                optimizer_G_A.zero_grad()

            self.log("G_loss", loss_G.detach(), prog_bar=True)
            self.log("loss_gan", loss_gan.detach(), prog_bar=True)
            self.log("loss_l1", loss_l1.detach(), prog_bar=True)
            self.log("loss_percept", loss_percept.detach(), prog_bar=True)

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
        if self.params.use_multiple_outputs:
            optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())
            optimizers.append(optimizer_D_B)

        if self.hparams.scheduler is not None:
            scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
            schedulers.append(scheduler_G_A)
            scheduler_D_A = self.hparams.scheduler(optimizer=optimizer_D_A)
            schedulers.append(scheduler_D_A)
            if self.params.use_multiple_outputs:
                scheduler_D_B = self.hparams.scheduler(optimizer=optimizer_D_B)
                schedulers.append(scheduler_D_B)
            return optimizers, schedulers
        
        return optimizers
            


import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
import itertools
import warnings

class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s

def fmt(t: object, print_stats=False) -> str:
    if isinstance(t, torch.Tensor):
        s = f"torch.tensor(..., size={tuple(t.shape)}, dtype={t.dtype}, device='{t.device}')"
        if print_stats and t.is_floating_point():
            s += f" [with stats min={t.min()}, max={t.max()}, mean={t.mean()}]"
        return Lit(s)
    else:
        return t

class NaNErrorMode(TorchDispatchMode):
    def __init__(
        self, enabled=True, raise_error=False, print_stats=True, print_nan_index=False
    ):
        self.enabled = enabled
        self.raise_error = raise_error
        self.print_stats = print_stats
        self.print_nan_index = print_nan_index

    def __torch_dispatch__(self, func, types, args, kwargs):
        out = func(*args, **kwargs)
        if self.enabled:
            if isinstance(out, torch.Tensor):
                if not torch.isfinite(out).all():
                    fmt_lambda = lambda t: fmt(t, self.print_stats)
                    fmt_args = ", ".join(
                        itertools.chain(
                            (repr(tree_map(fmt_lambda, a)) for a in args),
                            (
                                f"{k}={tree_map(fmt_lambda, v)}"
                                for k, v in kwargs.items()
                            ),
                        )
                    )
                    msg = f"NaN outputs in out = {func}({fmt_args})"
                    if self.print_nan_index:
                        msg += f"\nInvalid values detected at:\n{(~out.isfinite()).nonzero()}"
                    if self.raise_error:
                        raise RuntimeError(msg)
                    else:
                        warnings.warn(msg)
        return out
    
def check_for_nan(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
        return True
    return False
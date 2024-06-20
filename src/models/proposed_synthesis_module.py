import numpy as np

from typing import Any
import itertools

import torch
from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss
from src.losses.patch_nce_loss import PatchNCELoss

from src import utils
from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src.models.base_module_AtoB import BaseModule_AtoB



log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class ProposedSynthesisModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
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
        self.netF_A = netF_A

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

        # assign contextual loss
        style_feat_layers = {
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

    def backward_G(self, real_a, real_b, fake_b, lambda_style, lambda_nce):

        ## GAN Loss 내가 추가했음
        pred_fake = self.netD_A(fake_b.detach())
        loss_gan = self.criterionGAN(pred_fake, True)
        assert not torch.isnan(loss_gan).any(), "GAN Loss is NaN"

        ## Contextual loss
        loss_style_B = self.criterionContextual(real_b, fake_b)
        loss_style =  loss_style_B * lambda_style
        assert not torch.isnan(loss_style).any(), "Contextual Loss is NaN"

        # PatchNCE loss (real_a, fake_b)
        n_layers = len(self.params.nce_layers)
        feat_b = self.netG_A(fake_b, real_a, self.params.nce_layers, encode_only=True)

        flipped_for_equivariance = np.random.random() < 0.5
        if self.params.flip_equivariance and flipped_for_equivariance:
            feat_b = [torch.flip(fb, [3]) for fb in feat_b]

        feat_a = self.netG_A(real_a, real_b, self.params.nce_layers, encode_only=True)
        feat_a_pool, sample_ids = self.netF_A(feat_a, 256, None)
        feat_b_pool, _ = self.netF_A(feat_b, 256, sample_ids)

        total_nce_loss = 0.0
        for f_a, f_b in zip(feat_b_pool, feat_a_pool):
            loss = self.criterionNCE(f_a, f_b) * lambda_nce
            total_nce_loss += loss.mean()
        loss_nce = total_nce_loss / n_layers
        assert not torch.isnan(loss_nce).any(), "NCE Loss is NaN"

        loss_G = loss_gan + loss_style + loss_nce
        assert not torch.isnan(loss_G).any(), "Total Loss is NaN"

        return loss_G, loss_gan, loss_style, loss_nce

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G_A, optimizer_D_A, optimizer_F_A = self.optimizers()
        real_a, real_b, fake_b = self.model_step(batch)
        
        # with NaNErrorMode(enabled=True, raise_error=False, print_stats=True, print_nan_index=True):
        with optimizer_G_A.toggle_model():
            # with optimizer_F_A.toggle_model():
            loss_G, loss_gan, loss_style, loss_nce = self.backward_G(real_a, real_b, fake_b, self.params.lambda_style, self.params.lambda_nce)
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
        self.log("loss_gan", loss_gan.detach(), prog_bar=True)
        self.log("loss_style", loss_style.detach(), prog_bar=True)
        self.log("loss_nce", loss_nce.detach(), prog_bar=True)

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
        optimizer_F_A = self.hparams.optimizer(params=self.netF_A.parameters())
        optimizers.append(optimizer_F_A)

        if self.hparams.scheduler is not None:
            scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
            schedulers.append(scheduler_G_A)
            scheduler_D_B = self.hparams.scheduler(optimizer=optimizer_D_A)
            schedulers.append(scheduler_D_B)
            scheduler_F_A = self.hparams.scheduler(optimizer=optimizer_F_A)
            schedulers.append(scheduler_F_A)
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
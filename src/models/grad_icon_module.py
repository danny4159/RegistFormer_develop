import numpy as np

from typing import Any
import itertools

import torch
import torch.nn.functional as F

from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss
from src.losses.patch_nce_loss import PatchNCELoss

from src import utils
from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src.models.base_module_AtoB import BaseModule_AtoB
from src.models.base_module_registration import BaseModule_Registration


log = utils.get_pylogger(__name__)

class GradICONModule(BaseModule_Registration):

    def __init__(
        self,
        netR_A: torch.nn.Module,

        optimizer,
        params,
        scheduler,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        
        # assign generator
        self.netR_A = netR_A

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

    def pad_to_128(self, tensor, padding_value=-1):
        # tensor shape: [batch, channel, height, width, slice]
        slices = tensor.shape[-1]
        if slices < 128:
            padding = (0, 128 - slices)  # padding only on one side
            tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        return tensor
    
    def pad_to_128_and_384(self, tensor, padding_value=-1):
        # tensor shape: [batch, channel, height, width, slice]
        slices = tensor.shape[-1]
        height = tensor.shape[-2]
        if slices < 128:
            slice_padding = (0, 128 - slices)  # padding only on one side
        else:
            slice_padding = (0, 0)
        
        if height < 384:
            height_padding = (0, 384 - height)  # padding only on one side
        else:
            height_padding = (0, 0)
        
        padding = slice_padding + height_padding
        tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        return tensor

    # def crop_to_original(self, tensor, original_slices):
    #     # tensor shape: [batch, channel, height, width, slice]
    #     return tensor[..., :original_slices]

    def crop_to_original(self, tensor, original_slices, original_height):
        # tensor shape: [batch, channel, height, width, slice]
        return tensor[..., :original_height, :original_slices]
    
    def model_step(self, batch: Any, return_loss=False):
        evaluation_img, moving_img, fixed_img = batch # MR, CT, syn_CT

        original_slices = evaluation_img.shape[-1]
        original_height = evaluation_img.shape[-2]
        # original_slices = evaluation_img.shape[-1]

        evaluation_img = self.pad_to_128_and_384(evaluation_img)
        moving_img = self.pad_to_128_and_384(moving_img)
        fixed_img = self.pad_to_128_and_384(fixed_img)
        
        # evaluation_img = self.pad_to_128(evaluation_img)
        # moving_img = self.pad_to_128(moving_img)
        # fixed_img = self.pad_to_128(fixed_img)
        
        loss, a, b, c, flips, transform_vector, warped_img = self.netR_A(moving_img, fixed_img)

        evaluation_img = self.crop_to_original(evaluation_img, original_slices, original_height)
        moving_img = self.crop_to_original(moving_img, original_slices, original_height)
        fixed_img = self.crop_to_original(fixed_img, original_slices, original_height)
        warped_img = self.crop_to_original(warped_img, original_slices, original_height)
        
        # evaluation_img = self.crop_to_original(evaluation_img, original_slices)
        # moving_img = self.crop_to_original(moving_img, original_slices)
        # fixed_img = self.crop_to_original(fixed_img, original_slices)
        # warped_img = self.crop_to_original(warped_img, original_slices)
        

        if return_loss:
            return loss
        else:
            return evaluation_img, moving_img, fixed_img, warped_img

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_R_A, optimizer_D_A, optimizer_F_A = self.optimizers()
        # loss, a, b, c, flips, transform_vector, warped_image, moving_img, fixed_img, evaluation_img = self.model_step(batch)
        
        with optimizer_R_A.toggle_model():
            loss = self.model_step(batch, return_loss=True)
            self.manual_backward(loss)
            self.clip_gradients(
                optimizer_R_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_R_A.step()
            optimizer_R_A.zero_grad()

        self.log("loss", loss.detach(), prog_bar=True)


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizers = []
        schedulers = []
        
        optimizer_R_A = self.hparams.optimizer(params=self.netR_A.parameters())
        optimizers.append(optimizer_R_A)

        if self.hparams.scheduler is not None:
            scheduler_R_A = self.hparams.scheduler(optimizer=optimizer_R_A)
            schedulers.append(scheduler_R_A)
            return optimizers, schedulers
        
        return optimizers
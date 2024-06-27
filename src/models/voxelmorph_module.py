import torch.nn as nn
from typing import Any

import torch
import torch.nn.functional as F

from src import utils

from src.models.base_module_registration import BaseModule_Registration


log = utils.get_pylogger(__name__)

class VoxelmorphModule(BaseModule_Registration):

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

        self.criterionL2 = nn.MSELoss()

    def pad_slice_to_128(self, tensor, padding_value=-1):
        # tensor shape: [batch, channel, height, width, slice]
        slices = tensor.shape[-1]
        if slices < 128:
            padding = (0, 128 - slices)  # padding only on one side
            tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        return tensor
    
    def crop_slice_to_original(self, tensor, original_slices):
        # tensor shape: [batch, channel, height, width, slice]
        return tensor[..., :original_slices]
    
    def backward_G(self, fixed_img, warped_img):
        loss_G = 0

        loss_l2 = self.criterionL2(warped_img, fixed_img)
        loss_G += loss_l2
        
        return loss_G

    def model_step(self, batch: Any, is_3d=False):
        evaluation_img, moving_img, fixed_img = batch
        original_slices = evaluation_img.shape[-1]
        moving_img_pad = self.pad_slice_to_128(moving_img)
        fixed_img_pad = self.pad_slice_to_128(fixed_img)
        warped_img_pad = self.netR_A(moving_img_pad, fixed_img_pad)
        warped_img = self.crop_slice_to_original(warped_img_pad, original_slices)
        return evaluation_img, moving_img, fixed_img, warped_img


    def training_step(self, batch: Any, batch_idx: int):
        optimizer_R_A = self.optimizers()
        evaluation_img, moving_img, fixed_img, warped_img = self.model_step(batch)
        
        with optimizer_R_A.toggle_model():
            loss_G = self.backward_G(fixed_img, warped_img)
            self.manual_backward(loss_G)
            self.clip_gradients(
                optimizer_R_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_R_A.step()
            optimizer_R_A.zero_grad()

        self.log("loss_G", loss_G.detach(), prog_bar=True)


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
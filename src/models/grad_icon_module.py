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
from src.models.components.component_grad_icon import register_pair

import itk

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
    
    def model_step_for_train(self, batch: Any):
        evaluation_img, moving_img, fixed_img = batch # MR, CT, syn_CT
        original_slices = evaluation_img.shape[-1]
        moving_img = self.pad_slice_to_128(moving_img)
        fixed_img = self.pad_slice_to_128(fixed_img)
        loss, transform_vector, warped_img = self.netR_A(moving_img, fixed_img)
        return loss

    def model_step(self, batch: Any, is_3d=False):
        evaluation_img, moving_img, fixed_img = batch
        if is_3d:
            moving_img_np = moving_img.cpu().detach().squeeze().numpy()
            fixed_img_np = fixed_img.cpu().detach().squeeze().numpy()
            moving_img_np = moving_img_np.transpose(2, 1, 0) # itk: D, W, H
            fixed_img_np = fixed_img_np.transpose(2, 1, 0) 
            
            moving_img_itk = itk.image_from_array(moving_img_np)
            fixed_img_itk = itk.image_from_array(fixed_img_np)
            phi_AB, phi_BA = register_pair(self.netR_A, moving_img_itk, fixed_img_itk)
            interpolator = itk.LinearInterpolateImageFunction.New(moving_img_itk)
            warped_img = itk.resample_image_filter(moving_img_itk, 
                                                    transform=phi_AB, 
                                                    interpolator=interpolator,
                                                    size=itk.size(fixed_img_itk),
                                                    output_spacing=itk.spacing(fixed_img_itk),
                                                    output_direction=fixed_img_itk.GetDirection(),
                                                    output_origin=fixed_img_itk.GetOrigin()
                                                    )
            warped_img_np = itk.array_from_image(warped_img) # D, W, H -> H, W, D
            warped_img_tensor = torch.from_numpy(warped_img_np).unsqueeze(0).unsqueeze(0)
            warped_img_tensor = warped_img_tensor.to(evaluation_img.device)
            return evaluation_img, moving_img, fixed_img, warped_img_tensor
    
        else:
            original_slices = evaluation_img.shape[-1]
            moving_img_pad = self.pad_slice_to_128(moving_img)
            fixed_img_pad = self.pad_slice_to_128(fixed_img)
            loss, transform_vector, warped_img_pad = self.netR_A(moving_img_pad, fixed_img_pad)
            warped_img = self.crop_slice_to_original(warped_img_pad, original_slices)
            return evaluation_img, moving_img, fixed_img, warped_img


        
    def training_step(self, batch: Any, batch_idx: int):
        optimizer_R_A = self.optimizers()
        
        with optimizer_R_A.toggle_model():
            loss = self.model_step_for_train(batch)
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
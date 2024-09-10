from typing import Any

import torch
import torch.nn.functional as F

from src.losses.grad_loss import GradLoss, SmoothLoss
from src.losses.mask_mse_loss import MaskMSELoss

from src import utils
from src.models.base_module_registration import BaseModule_Registration
from src.models.components.network_transMorph import *

log = utils.get_pylogger(__name__)

class TransMorph_Module(BaseModule_Registration):
    def __init__(
        self,
        netR_A: torch.nn.Module,
        optimizer,
        params,
        # scheduler,
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
        # self.scheduler = scheduler

        self.criterionMSE = nn.MSELoss()
        self.criterionGrad3d = Grad3d(penalty='l2')

    def pad_slice_to_96(self, tensor, padding_value=-1):
        # tensor shape: [batch, channel, height, width, slice]
        slices = tensor.shape[-1]
        if slices < 96:
            padding = (0, 96 - slices)  # padding only on one side
            tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        return tensor
    
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
    
    def backward_R(self, fixed_img, warped_img, deform_field): # grid_4 
        loss_MSE = self.criterionMSE(fixed_img, warped_img)

        loss_Grad3d = self.criterionGrad3d(deform_field, fixed_img) * self.params.lambda_grad

        loss_R = loss_MSE + loss_Grad3d
        return loss_R

    def resize_tensor(self, tensor, size):
        return F.interpolate(tensor, size=size, mode='trilinear', align_corners=False)

    # 메모리 이슈만 없으면 이거 그냥 돌리면 되는데 메모리 이슈로 아래 코드로 갈아타자.
    # def model_step(self, batch: Any, is_3d=False, is_train=False):
    #     evaluation_img, moving_img, fixed_img = batch
    #     if is_3d:

    #         original_slices = evaluation_img.shape[-1]
    #         moving_img = self.pad_slice_to_128(moving_img)
    #         fixed_img = self.pad_slice_to_128(fixed_img)
    #         moving_fixed_cat = torch.cat((moving_img, fixed_img), dim=1)
    #         warped_img, deform_field = self.netR_A(moving_fixed_cat)
            
    #         moving_img = self.crop_slice_to_original(moving_img, original_slices)
    #         fixed_img = self.crop_slice_to_original(fixed_img, original_slices)
    #         warped_img = self.crop_slice_to_original(warped_img, original_slices)

    #     if is_train:
    #         return evaluation_img, moving_img, fixed_img, warped_img, deform_field
    #     else: 
    #         return evaluation_img, moving_img, fixed_img, warped_img
        
    #
    def model_step(self, batch: Any, is_3d=False, is_train=False):
        evaluation_img, moving_img, fixed_img = batch
        if is_3d:

            _, _, H, W, D = evaluation_img.shape
            # scale_factor = 0.7
            # new_size = [int(H * scale_factor), int(W * scale_factor), int(D)]
            new_size = [int(320), int(256), int(D)]

            evaluation_img = F.interpolate(evaluation_img, size=new_size, mode='trilinear', align_corners=False)
            moving_img = F.interpolate(moving_img, size=new_size, mode='trilinear', align_corners=False)
            fixed_img = F.interpolate(fixed_img, size=new_size, mode='trilinear', align_corners=False)

            original_slices = evaluation_img.shape[-1]
            
            evaluation_img = self.pad_slice_to_128(evaluation_img)
            moving_img = self.pad_slice_to_128(moving_img)
            fixed_img = self.pad_slice_to_128(fixed_img)

            # interpolation
            _, _, H, W, D = evaluation_img.shape
            scale_factor = 0.8
            new_size = [int(H * scale_factor), int(W * scale_factor), int(D * scale_factor)]

            moving_fixed_cat = torch.cat((moving_img, fixed_img), dim=1)
            warped_img, deform_field = self.netR_A(moving_fixed_cat)
            
            evaluation_img = self.crop_slice_to_original(evaluation_img, original_slices)
            moving_img = self.crop_slice_to_original(moving_img, original_slices)
            fixed_img = self.crop_slice_to_original(fixed_img, original_slices)
            warped_img = self.crop_slice_to_original(warped_img, original_slices)

        if is_train:
            return evaluation_img, moving_img, fixed_img, warped_img, deform_field
        else: 
            return evaluation_img, moving_img, fixed_img, warped_img
                
    def training_step(self, batch: Any, batch_idx: int):
        optimizer_R_A = self.optimizers()
        if self.params.flag_train_fixed_moving:
            evaluation_img, moving_img, fixed_img, warped_img = self.model_step_for_swap_moving_fixed(batch, is_3d=self.params.is_3d)
        else:
            evaluation_img, moving_img, fixed_img, warped_img, deform_field = self.model_step(batch, is_3d=self.params.is_3d, is_train=True)
    
        with optimizer_R_A.toggle_model():
            loss = self.backward_R(fixed_img, warped_img, deform_field)
            self.manual_backward(loss)
            # self.clip_gradients(
            #     optimizer_R_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            # )
            optimizer_R_A.step()
            optimizer_R_A.zero_grad()

        # self.log("loss", loss.detach(), prog_bar=True)

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

        # if self.hparams.scheduler is not None:
        #     scheduler_R_A = self.hparams.scheduler(optimizer=optimizer_R_A)
        #     schedulers.append(scheduler_R_A)
        #     return optimizers, schedulers
        
        return optimizers
from typing import Any

import torch
import torch.nn.functional as F

from src.losses.grad_loss import GradLoss, SmoothLoss
from src.losses.mask_mse_loss import MaskMSELoss

from src import utils
from src.models.base_module_registration import BaseModule_Registration
from src.models.components.network_lapIRN import *

log = utils.get_pylogger(__name__)

class LapIRN_Lv1_Module(BaseModule_Registration):
    def __init__(
        self,
        netR_1: torch.nn.Module, # TODO: Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel=start_channel, is_train=True, imgshape=imgshape_4,range_flow=range_flow)
        optimizer,
        params,
        # scheduler,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        
        # assign generator
        self.netR_1 = netR_1

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        # self.scheduler = scheduler

        self.loss_similarity = NCC(win=3)
        self.loss_similarity_multi_res = multi_resolution_NCC(win=5, scale=2)
        self.loss_similarity_multi_res_lv3 = multi_resolution_NCC(win=7, scale=3)
        self.loss_Jdet = neg_Jdet_loss
        self.loss_smooth = smoothloss

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
    
    def backward_R_lv1(self, displacement_field, warped_img, down_fixed_img): # grid_4 
        loss_NCC = self.loss_similarity(warped_img, down_fixed_img)
        # F_X_Y_norm = transform_unit_flow_to_flow_cuda(displacement_field.permute(0,2,3,4,1).clone())
        # loss_Jacobian = self.loss_Jdet(F_X_Y_norm, grid_4)

        # reg2 - use velocity
        _, _, x, y, z = displacement_field.shape
        displacement_field[:, 0, :, :, :] = displacement_field[:, 0, :, :, :] * (z-1)
        displacement_field[:, 1, :, :, :] = displacement_field[:, 1, :, :, :] * (y-1)
        displacement_field[:, 2, :, :, :] = displacement_field[:, 2, :, :, :] * (x-1)
        loss_regulation = self.loss_smooth(displacement_field)

        loss_lv1 = loss_NCC + self.params.lambda_smooth * loss_regulation # antifold*loss_Jacobian #TODO: antifold가 0이라서 loss_Jacobian 구현은 미룸.
        return loss_lv1

    def resize_tensor(self, tensor, size):
        return F.interpolate(tensor, size=size, mode='trilinear', align_corners=False)

    def model_step(self, batch: Any, is_3d=False, is_train=False):
        evaluation_img, moving_img, fixed_img = batch
        if is_3d:

            # grid_4 = generate_grid(self.imgshape_4)
            # grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

            original_slices = evaluation_img.shape[-1]
            moving_img = self.pad_slice_to_128(moving_img)
            fixed_img = self.pad_slice_to_128(fixed_img)

            deform_field, warped_img, down_fixed_img, _, _ = self.netR_1(moving_img, fixed_img) #  F_X_Y, X_Y, Y_4x, F_xy, _ = deform_field, warped_img, down_fixed_img, displacement field, feature map
            
            #TODO: 이건 lv1,lv2에만 적용
            evaluation_img = self.resize_tensor(evaluation_img, warped_img.shape[2:])
            moving_img = self.resize_tensor(moving_img, warped_img.shape[2:])
            fixed_img = self.resize_tensor(fixed_img, warped_img.shape[2:])

            # lv3에는 위에거 지우고 이것만 실행.
            # deform_field = self.crop_slice_to_original(deform_field, original_slices)
            # down_fixed_img = self.crop_slice_to_original(down_fixed_img, original_slices)
            # moving_img = self.crop_slice_to_original(moving_img, original_slices)
            # fixed_img = self.crop_slice_to_original(fixed_img, original_slices)
            # warped_img = self.crop_slice_to_original(warped_img, original_slices)

            #TODO: 그래서 이 loss를 backward
            # return deform_field, warped_img, down_fixed_img
        if is_train:
            return evaluation_img, moving_img, fixed_img, warped_img, deform_field, down_fixed_img
        else: 
            return evaluation_img, moving_img, fixed_img, warped_img
                
    def training_step(self, batch: Any, batch_idx: int): #TODO: epoch 10까진 lv1, 20까진 lv2, 40까진 lv3
        optimizer_R_A = self.optimizers()
        if self.params.flag_train_fixed_moving:
            evaluation_img, moving_img, fixed_img, warped_img = self.model_step_for_swap_moving_fixed(batch, is_3d=self.params.is_3d)
        else:
            evaluation_img, moving_img, fixed_img, warped_img, deform_field, down_fixed_img = self.model_step(batch, is_3d=self.params.is_3d, is_train=True)
    
        with optimizer_R_A.toggle_model():
            loss = self.backward_R_lv1(deform_field, warped_img, down_fixed_img)
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
        
        optimizer_R_A = self.hparams.optimizer(params=self.netR_1.parameters())
        optimizers.append(optimizer_R_A)

        # if self.hparams.scheduler is not None:
        #     scheduler_R_A = self.hparams.scheduler(optimizer=optimizer_R_A)
        #     schedulers.append(scheduler_R_A)
        #     return optimizers, schedulers
        
        return optimizers
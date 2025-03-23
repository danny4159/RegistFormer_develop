import numpy as np

from typing import Any
import itertools

import torch.nn.functional as F
import torch
from torch.amp import GradScaler, autocast


from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss, VGG_Model
from src.losses.patch_nce_loss import PatchNCELoss
from src.losses.mind_loss import MINDLoss

from src import utils
from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src.models.base_module_AtoB import BaseModule_AtoB
# from src.models.base_module_AtoB_multi import BaseModule_AtoB

from monai.inferers import LatentDiffusionInferer
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from monai.networks.schedulers import DDPMScheduler


log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class LatentDiffusionModule(BaseModule_AtoB):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        # netD_A: torch.nn.Module,
        # netD_B: torch.nn.Module,
        # netF_A: torch.nn.Module,
        optimizer,
        params,
        scheduler,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)

        self.initialized = False # training_step 에서 
        self.scaler = GradScaler()        
        # assign generator
        self.netG_A = netG_A
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

        # loss function
        # self.criterionContextual = Contextual_Loss(style_feat_layers) if params.lambda_style != 0 else None
        # self.criterionNCE = PatchNCELoss(False, nce_T=0.07, batch_size=params.batch_size) if params.lambda_nce != 0 else None
        self.criterionGAN = PatchAdversarialLoss(criterion="least_squares") 
        self.criterionPeceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2) if params.lambda_percept !=0 else None
        self.criterionL1 = torch.nn.L1Loss() if params.lambda_recon != 0 else None

    def training_step(self, batch: Any, batch_idx: int):
        
        real_a, real_b = batch

        if not self.initialized and self.current_epoch == 0 and batch_idx == 0:
            self.autoencoder = AutoencoderKL(
                                spatial_dims=3,
                                in_channels=1,
                                out_channels=1,
                                channels=(32, 64, 64),
                                latent_channels=3,
                                num_res_blocks=1,
                                norm_num_groups=16,
                                attention_levels=(False, False, True),
                            )
            #TODO: 학습된 weight 불러오기
            self.autoencoder.to(real_a.device)

            with torch.no_grad():
                with autocast("cuda", enabled=True):
                    z = self.autoencoder.encode_stage_2_inputs(real_a) # check_data = first(train_loader)

            print(f"Scaling factor set to {1/torch.std(z)}")
            scale_factor = 1 / torch.std(z)

            self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=scale_factor)

        optimizer_G_A = self.optimizers()

        z = self.autoencoder.encode_stage_2_inputs(real_a)

        optimizer_G_A.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=True):
            # Generate random noise
            noise = torch.randn_like(z).to(real_a.device)

            # Create timesteps
            timesteps = torch.randint(
                0, self.inferer.scheduler.num_train_timesteps, (real_a.shape[0],), device=real_a.device
            ).long()
        
            # Get model prediction
            noise_pred = self.inferer(
                inputs=real_a, autoencoder_model=self.autoencoder, diffusion_model=self.netG_A, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log("loss_mse", loss.detach(), prog_bar=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer_G_A)
        self.scaler.update()

        
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

        if self.hparams.scheduler is not None:
            scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
            schedulers.append(scheduler_G_A)
            return optimizers, schedulers
        
        return optimizers
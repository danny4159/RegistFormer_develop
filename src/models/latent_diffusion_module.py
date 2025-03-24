import numpy as np

from typing import Any
import itertools
import random

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
        # scheduler,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)

        self.initialized = False # training_step 에서 
        self.scaler = GradScaler()        
        # assign generator
        self.netG_A = netG_A
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)


        ################################################3
        # ✅ real_a 역할의 첫 HDF5 데이터셋을 로딩하여 inferer 초기화
        import h5py
        h5_path = "/home/sumin/registformer/RegistFormer_develop/data/SynthRAD_MR_CT_Pelvis/train/Ver3_AllPatientSameSize_final_2.h5"
        with h5py.File(h5_path, "r") as f:
            mr_data = f["MR"]
            first_key = list(mr_data.keys())[0]  # 첫 번째 데이터셋 이름
            raw = mr_data[first_key][()]  # H, W, D (numpy array)

        # 🟡 정규화: [-1,1] → [0,1]
        raw = (raw + 1.0) / 2.0  # float32로 자동 변환

        # 🟡 Tensor 변환 및 shape 맞추기: (1, 1, H, W, D)
        tensor = torch.tensor(raw, dtype=torch.float32)  # shape: HWD
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # BCHWD

        # 🟡 D=128로 padding
        pad_d = 128 - tensor.shape[-1]
        if pad_d > 0:
            tensor = F.pad(tensor, (0, pad_d), mode="constant", value=0)

        _, _, H, W, D = tensor.shape
        crop_h, crop_w = 96, 96
        # 🟡 시작점 랜덤 설정
        start_h = random.randint(0, max(0, H - crop_h))
        start_w = random.randint(0, max(0, W - crop_w))
        # 🟡 crop 수행
        tensor = tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w, :]

        tensor = torch.randn(1,1,96,96,64).to(tensor.device)
        # print("tensor shape@@@@@@@@@@@@@@", tensor.shape)
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
        self.autoencoder.to(self.device)

        # 2. 체크포인트 로딩
        ckpt_path = "/home/sumin/registformer/RegistFormer_develop/logs/Model_AutoencoderKL_Data_SynthRAD_MR_CT_Pelvis/AutoencoderKL_MrCtPelvis_TrainVer5/runs/2025-03-22_12-43-33/checkpoints/epoch099-fid339.81-kid0.28-gc0.3716-nmi0.3371-sharp382.46.ckpt"
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # 3. 저장된 netG_A weight만 추출
        ckpt_state_dict = ckpt["state_dict"]
        all_keys = list(ckpt_state_dict.keys())

        # 🔍 key에서 netG_A.가 붙어있으면 그걸 기준으로 추출
        if any(k.startswith("netG_A.") for k in all_keys):
            netG_A_state_dict = {k.replace("netG_A.", ""): v for k, v in ckpt_state_dict.items() if k.startswith("netG_A.")}
        else:
            # 만약 그냥 autoencoder만 따로 저장된 케이스라면 통째로 사용
            print("⚠️ 'netG_A.' prefix 없음. 전체 state_dict를 사용합니다.")
            netG_A_state_dict = ckpt_state_dict  # 그대로 사용

        # 4. 현재 모델의 state_dict와 비교
        model_state_dict = self.autoencoder.state_dict()

        print("🔍 [현재 모델 key 수] :", len(model_state_dict))
        print("🔍 [체크포인트 key 수] :", len(netG_A_state_dict))

        # 5. 매칭 key 개수 출력
        matched_keys = [k for k in model_state_dict.keys() if k in netG_A_state_dict]
        print(f"✅ 매칭되는 key 수: {len(matched_keys)} / {len(model_state_dict)}")

        # 6. weight 로드 (strict=False)
        load_result = self.autoencoder.load_state_dict(netG_A_state_dict, strict=False)

        # 7. 로드 결과 출력
        print("💡 불러오지 못한 key 목록:")
        print("  - missing_keys:", load_result.missing_keys)
        print("  - unexpected_keys:", load_result.unexpected_keys)

        with torch.no_grad():
            with autocast("cuda", enabled=True):
                z = self.autoencoder.encode_stage_2_inputs(tensor) # check_data = first(train_loader)

        print(f"Scaling factor set to {1/torch.std(z)}")
        scale_factor = 1 / torch.std(z)

        self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=scale_factor)


        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        # self.scheduler = scheduler



        # loss function
        # self.criterionContextual = Contextual_Loss(style_feat_layers) if params.lambda_style != 0 else None
        # self.criterionNCE = PatchNCELoss(False, nce_T=0.07, batch_size=params.batch_size) if params.lambda_nce != 0 else None
        self.criterionGAN = PatchAdversarialLoss(criterion="least_squares") 
        self.criterionPeceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2) if params.lambda_percept !=0 else None
        self.criterionL1 = torch.nn.L1Loss() if params.lambda_recon != 0 else None

    def training_step(self, batch: Any, batch_idx: int):
        
        real_a, real_b = batch

        # if not self.initialized and self.current_epoch == 0 and batch_idx == 0:
        #     self.autoencoder = AutoencoderKL(
        #                         spatial_dims=3,
        #                         in_channels=1,
        #                         out_channels=1,
        #                         channels=(32, 64, 64),
        #                         latent_channels=3,
        #                         num_res_blocks=1,
        #                         norm_num_groups=16,
        #                         attention_levels=(False, False, True),
        #                     )
        #     #TODO: 학습된 weight 불러오기
        #     self.autoencoder.to(real_a.device)

        #     with torch.no_grad():
        #         with autocast("cuda", enabled=True):
        #             z = self.autoencoder.encode_stage_2_inputs(real_a) # check_data = first(train_loader)

        #     print(f"Scaling factor set to {1/torch.std(z)}")
        #     scale_factor = 1 / torch.std(z)

        #     self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=scale_factor)

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
        # schedulers = []
        
        optimizer_G_A = self.hparams.optimizer(params=self.netG_A.parameters())
        optimizers.append(optimizer_G_A)

        # if self.hparams.scheduler is not None:
        #     scheduler_G_A = self.hparams.scheduler(optimizer=optimizer_G_A)
        #     schedulers.append(scheduler_G_A)
        #     return optimizers, schedulers
        
        return optimizers
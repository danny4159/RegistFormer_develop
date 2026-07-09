from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from src.models.base_module_AtoB import BaseModule_AtoB, gray2rgb, norm_to_uint8


class SwinUNETRSynthesisModule(BaseModule_AtoB):
    def __init__(
        self,
        netG_A: torch.nn.Module,
        optimizer,
        params,
        scheduler=None,
        *args,
        **kwargs: Any,
    ):
        super().__init__(params, *args, **kwargs)

        self.netG_A = netG_A
        self.save_hyperparameters(logger=False, ignore=["netG_A"])
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

    @staticmethod
    def _to_monai_layout(tensor: torch.Tensor) -> torch.Tensor:
        # Current codebase uses [B, C, H, W, D], MONAI SwinUNETR expects [B, C, D, H, W].
        return tensor.permute(0, 1, 4, 2, 3).contiguous()

    @staticmethod
    def _from_monai_layout(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 1, 3, 4, 2).contiguous()

    @staticmethod
    def _center_crop_like(tensor: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        _, _, h, w, d = tensor.shape
        target_h, target_w, target_d = target_shape
        start_h = max((h - target_h) // 2, 0)
        start_w = max((w - target_w) // 2, 0)
        start_d = max((d - target_d) // 2, 0)
        return tensor[
            :,
            :,
            start_h:start_h + target_h,
            start_w:start_w + target_w,
            start_d:start_d + target_d,
        ]

    def _restore_original_shape(
        self,
        cbct_volume: torch.Tensor,
        real_ct: torch.Tensor,
        fake_ct: torch.Tensor,
        original_shape: Optional[torch.Tensor],
    ):
        if original_shape is None:
            return cbct_volume, real_ct, fake_ct

        if original_shape.ndim == 2:
            target_shape = tuple(int(v.item()) for v in original_shape[0])
        else:
            target_shape = tuple(int(v.item()) for v in original_shape)

        cbct_volume = self._center_crop_like(cbct_volume, target_shape)
        real_ct = self._center_crop_like(real_ct, target_shape)
        fake_ct = self._center_crop_like(fake_ct, target_shape)
        return cbct_volume, real_ct, fake_ct

    def _sample_training_patch(self, cbct: torch.Tensor, ct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_h, patch_w, patch_d = tuple(self.params.patch_size)
        _, _, h, w, d = cbct.shape

        if patch_h > h or patch_w > w or patch_d > d:
            raise ValueError(
                f"Patch size {self.params.patch_size} is larger than input volume {(h, w, d)}."
            )

        start_h = torch.randint(0, h - patch_h + 1, size=(1,), device=cbct.device).item()
        start_w = torch.randint(0, w - patch_w + 1, size=(1,), device=cbct.device).item()
        start_d = torch.randint(0, d - patch_d + 1, size=(1,), device=cbct.device).item()

        slices = (
            slice(None),
            slice(None),
            slice(start_h, start_h + patch_h),
            slice(start_w, start_w + patch_w),
            slice(start_d, start_d + patch_d),
        )
        return cbct[slices], ct[slices]

    def forward(self, cbct_volume: torch.Tensor) -> torch.Tensor:
        pred = self.netG_A(self._to_monai_layout(cbct_volume))
        return self._from_monai_layout(pred)

    def predict_full_volume(self, cbct_volume: torch.Tensor) -> torch.Tensor:
        roi_size = tuple(int(v) for v in self.params.patch_size_monai)
        pred = sliding_window_inference(
            inputs=self._to_monai_layout(cbct_volume),
            roi_size=roi_size,
            sw_batch_size=int(self.params.sw_batch_size),
            predictor=self.netG_A,
            overlap=float(self.params.infer_overlap),
            mode="gaussian",
        )
        pred = self._from_monai_layout(pred)
        return pred

    def clamp_for_eval(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.params.norm_ZeroToOne:
            return torch.clamp(tensor, 0.0, 1.0)
        return torch.clamp(tensor, -1.0, 1.0)

    def build_body_mask(self, cbct: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        threshold = self.params.mask_threshold
        mask = (ct > threshold) | (cbct > threshold)
        return mask.float()

    def masked_l1_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(pred - target) * mask
        return loss.sum() / (mask.sum() + 1e-6)

    def model_step(self, batch: Any, is_3d=False):
        if len(batch) == 3:
            cbct_volume, real_ct, original_shape = batch
        else:
            cbct_volume, real_ct = batch
            original_shape = None
        fake_ct = self.predict_full_volume(cbct_volume) if (is_3d or cbct_volume.ndim == 5) else self.forward(cbct_volume)
        cbct_volume, real_ct, fake_ct = self._restore_original_shape(
            cbct_volume, real_ct, fake_ct, original_shape
        )
        if not self.training:
            fake_ct = self.clamp_for_eval(fake_ct)
        return cbct_volume, real_ct, fake_ct

    def shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        if len(batch) == 3:
            cbct_volume, real_ct, original_shape = batch
        else:
            cbct_volume, real_ct = batch
            original_shape = None

        if stage == "train":
            cbct_input, ct_target = self._sample_training_patch(cbct_volume, real_ct)
            fake_ct = self.forward(cbct_input)
            mask = self.build_body_mask(cbct_input, ct_target)
            loss = self.masked_l1_loss(fake_ct, ct_target, mask)
        else:
            fake_ct = self.predict_full_volume(cbct_volume)
            cbct_volume, real_ct, fake_ct = self._restore_original_shape(
                cbct_volume, real_ct, fake_ct, original_shape
            )
            fake_ct = self.clamp_for_eval(fake_ct)
            mask = self.build_body_mask(cbct_volume, real_ct)
            loss = self.masked_l1_loss(fake_ct, real_ct, mask)

        self.log(f"{stage}/loss", loss.detach(), prog_bar=True, sync_dist=(stage != "train"))
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        _ = self.shared_step(batch, "val")

        real_a, real_b, fake_b = self.model_step(batch, is_3d=True)

        if self.params.eval_on_align:
            depth = real_a.size(4)
            for i in range(depth):
                real_slice = real_b[:, :, :, :, i]
                fake_slice = fake_b[:, :, :, :, i]
                self.val_ssim_B.update(real_slice, fake_slice)
                self.val_psnr_B.update(real_slice, fake_slice)
                self.psnr_values_B.append(self.val_psnr_B.compute().item())
                self.val_psnr_B.reset()
                self.val_lpips_B.update(gray2rgb(real_slice), gray2rgb(fake_slice))
                self.lpips_values_B.append(self.val_lpips_B.compute().item())
                self.val_lpips_B.reset()
                self.val_sharpness_B.update(norm_to_uint8(fake_slice).float())
            return

        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int):
        _ = self.shared_step(batch, "test")

        real_a, real_b, fake_b = self.model_step(batch, is_3d=True)

        if self.params.eval_on_align:
            depth = real_a.size(4)
            for i in range(depth):
                real_slice = real_b[:, :, :, :, i]
                fake_slice = fake_b[:, :, :, :, i]
                self.test_ssim_B.update(real_slice, fake_slice)
                self.test_psnr_B.update(real_slice, fake_slice)
                self.psnr_values_B.append(self.test_psnr_B.compute().item())
                self.test_psnr_B.reset()
                self.test_lpips_B.update(gray2rgb(real_slice), gray2rgb(fake_slice))
                self.lpips_values_B.append(self.test_lpips_B.compute().item())
                self.test_lpips_B.reset()
                self.test_sharpness_B.update(norm_to_uint8(fake_slice).float())
            return

        return super().test_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.netG_A.parameters())
        if self.hparams.scheduler is None:
            return optimizer

        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

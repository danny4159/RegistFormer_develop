from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from src.losses.gan_loss import GANLoss
from src.models.base_module_AtoB import BaseModule_AtoB, gray2rgb, norm_to_uint8
from src.models.unetplusplus_synthesis_module import VGG19PerceptualLoss


class SwinUNETRSynthesisModule(BaseModule_AtoB):
    def __init__(
        self,
        netG_A: torch.nn.Module,
        optimizer,
        params,
        scheduler=None,
        netD_A: Optional[torch.nn.Module] = None,
        *args,
        **kwargs: Any,
    ):
        super().__init__(params, *args, **kwargs)

        self.netG_A = netG_A
        self.netD_A = netD_A
        self.save_hyperparameters(logger=False, ignore=["netG_A", "netD_A"])
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler

        self.use_adversarial_loss = bool(getattr(params, "use_adversarial_loss", False)) and netD_A is not None
        self.use_perceptual_loss = bool(getattr(params, "use_perceptual_loss", False))
        self.automatic_optimization = not self.use_adversarial_loss

        self.criterionGAN = GANLoss(gan_type=getattr(params, "gan_type", "lsgan")) if self.use_adversarial_loss else None
        if self.use_perceptual_loss:
            intensity_range = "0_1" if getattr(params, "norm_ZeroToOne", False) else "-1_1"
            self.criterionPerceptual = VGG19PerceptualLoss(
                layer_ids=tuple(getattr(params, "perceptual_layers", (4, 9, 18))),
                normalized_range=intensity_range,
            )
        else:
            self.criterionPerceptual = None

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
        eval_ref: Optional[torch.Tensor] = None,
    ):
        if original_shape is None:
            return cbct_volume, real_ct, fake_ct, eval_ref

        if original_shape.ndim == 2:
            target_shape = tuple(int(v.item()) for v in original_shape[0])
        else:
            target_shape = tuple(int(v.item()) for v in original_shape)

        cbct_volume = self._center_crop_like(cbct_volume, target_shape)
        real_ct = self._center_crop_like(real_ct, target_shape)
        fake_ct = self._center_crop_like(fake_ct, target_shape)
        if eval_ref is not None:
            eval_ref = self._center_crop_like(eval_ref, target_shape)
        return cbct_volume, real_ct, fake_ct, eval_ref

    def _unpack_batch(self, batch: Any):
        """Standard batch: (cbct, real_ct[, original_shape]).
        When params.use_eval_ref is set (data.data_group_3 carries the true,
        registered T2 used only for metric computation, never for the training
        loss), batch instead is (cbct, real_ct, eval_ref[, original_shape])."""
        if getattr(self.params, "use_eval_ref", False):
            if len(batch) == 4:
                cbct_volume, real_ct, eval_ref, original_shape = batch
            else:
                cbct_volume, real_ct, eval_ref = batch
                original_shape = None
        else:
            eval_ref = None
            if len(batch) == 3:
                cbct_volume, real_ct, original_shape = batch
            else:
                cbct_volume, real_ct = batch
                original_shape = None
        return cbct_volume, real_ct, eval_ref, original_shape

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

    def pixel_loss(self, fake_ct: torch.Tensor, ct_target: torch.Tensor, cbct_input: torch.Tensor) -> torch.Tensor:
        if getattr(self.params, "use_masked_loss", True):
            mask = self.build_body_mask(cbct_input, ct_target)
            return self.masked_l1_loss(fake_ct, ct_target, mask)
        return F.l1_loss(fake_ct, ct_target)

    def perceptual_loss(self, fake_ct: torch.Tensor, ct_target: torch.Tensor) -> torch.Tensor:
        # VGG19PerceptualLoss expects 2D images; treat each depth slice as one image.
        b, c, h, w, d = fake_ct.shape
        fake_2d = fake_ct.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
        real_2d = ct_target.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
        return self.criterionPerceptual(fake_2d, real_2d)

    def model_step(self, batch: Any, is_3d=False):
        """Returns (cbct, metric_reference, fake_ct). metric_reference is the true
        registered T2 (params.use_eval_ref) when available, otherwise it falls back
        to the training target (real_ct) as before."""
        cbct_volume, real_ct, eval_ref, original_shape = self._unpack_batch(batch)
        fake_ct = self.predict_full_volume(cbct_volume) if (is_3d or cbct_volume.ndim == 5) else self.forward(cbct_volume)
        cbct_volume, real_ct, fake_ct, eval_ref = self._restore_original_shape(
            cbct_volume, real_ct, fake_ct, original_shape, eval_ref
        )
        if not self.training:
            fake_ct = self.clamp_for_eval(fake_ct)
        metric_ref = eval_ref if eval_ref is not None else real_ct
        return cbct_volume, metric_ref, fake_ct

    def shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        """Used for val/test (no discriminator update), and for train when
        adversarial loss is disabled (plain automatic-optimization path).
        Loss is always computed against the training target (real_ct), never
        against eval_ref (the true T2 is for metric monitoring only)."""
        cbct_volume, real_ct, eval_ref, original_shape = self._unpack_batch(batch)

        if stage == "train":
            cbct_input, ct_target = self._sample_training_patch(cbct_volume, real_ct)
            fake_ct = self.forward(cbct_input)
        else:
            fake_ct = self.predict_full_volume(cbct_volume)
            cbct_volume, real_ct, fake_ct, eval_ref = self._restore_original_shape(
                cbct_volume, real_ct, fake_ct, original_shape, eval_ref
            )
            fake_ct = self.clamp_for_eval(fake_ct)
            cbct_input, ct_target = cbct_volume, real_ct

        loss_pix = self.pixel_loss(fake_ct, ct_target, cbct_input)
        loss = self.params.lambda_mae * loss_pix
        self.log(f"{stage}/loss_pix", loss_pix.detach(), prog_bar=(stage == "train"), sync_dist=(stage != "train"))

        if self.criterionPerceptual is not None:
            loss_perc = self.perceptual_loss(fake_ct, ct_target)
            loss = loss + self.params.lambda_perceptual * loss_perc
            self.log(f"{stage}/loss_perc", loss_perc.detach(), sync_dist=(stage != "train"))

        self.log(f"{stage}/loss", loss.detach(), prog_bar=True, sync_dist=(stage != "train"))
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        if not self.use_adversarial_loss:
            return self.shared_step(batch, "train")

        cbct_volume, real_ct, _eval_ref, _original_shape = self._unpack_batch(batch)

        cbct_input, ct_target = self._sample_training_patch(cbct_volume, real_ct)
        fake_ct = self.forward(cbct_input)

        optimizer_G, optimizer_D = self.optimizers()

        with optimizer_D.toggle_model():
            pred_real = self.netD_A(ct_target)
            loss_D_real = self.criterionGAN(pred_real, True)
            pred_fake = self.netD_A(fake_ct.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            self.manual_backward(loss_D)
            optimizer_D.step()
            optimizer_D.zero_grad()

        with optimizer_G.toggle_model():
            loss_pix = self.pixel_loss(fake_ct, ct_target, cbct_input)
            loss_G = self.params.lambda_mae * loss_pix

            if self.criterionPerceptual is not None:
                loss_perc = self.perceptual_loss(fake_ct, ct_target)
                loss_G = loss_G + self.params.lambda_perceptual * loss_perc

            pred_fake_for_g = self.netD_A(fake_ct)
            loss_adv = self.criterionGAN(pred_fake_for_g, True)
            loss_G = loss_G + self.params.lambda_adversarial * loss_adv

            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()

        self.log("train/loss_pix", loss_pix.detach(), prog_bar=True)
        self.log("train/loss_D", loss_D.detach(), prog_bar=True)
        self.log("train/loss_adv", loss_adv.detach())
        if self.criterionPerceptual is not None:
            self.log("train/loss_perc", loss_perc.detach())
        self.log("train/loss", loss_G.detach(), prog_bar=True)
        return loss_G

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
                self.val_lpips_B.update(
                    gray2rgb(self.clamp_for_eval(real_slice)), gray2rgb(self.clamp_for_eval(fake_slice))
                )
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
                self.test_lpips_B.update(
                    gray2rgb(self.clamp_for_eval(real_slice)), gray2rgb(self.clamp_for_eval(fake_slice))
                )
                self.lpips_values_B.append(self.test_lpips_B.compute().item())
                self.test_lpips_B.reset()
                self.test_sharpness_B.update(norm_to_uint8(fake_slice).float())
            return

        return super().test_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.use_adversarial_loss:
            optimizer_G = self.hparams.optimizer(params=self.netG_A.parameters())
            optimizer_D = self.hparams.optimizer(params=self.netD_A.parameters())
            return optimizer_G, optimizer_D

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

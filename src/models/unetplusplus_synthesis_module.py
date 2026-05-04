from typing import Any

import torch
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19

from src.models.base_module_AtoB import BaseModule_AtoB


class VGG19PerceptualLoss(torch.nn.Module):
    def __init__(self, layer_ids=(4, 9, 18), normalized_range="-1_1"):
        super().__init__()
        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        for param in features.parameters():
            param.requires_grad = False

        self.features = features
        self.layer_ids = set(layer_ids)
        self.normalized_range = normalized_range

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalized_range == "-1_1":
            x = (x + 1.0) / 2.0

        x = torch.clamp(x, 0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.preprocess(pred)
        target = self.preprocess(target)

        loss = pred.new_tensor(0.0)
        for idx, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            if idx in self.layer_ids:
                loss = loss + F.l1_loss(pred, target)

        return loss


class UnetPlusPlusSynthesisModule(BaseModule_AtoB):
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

        if getattr(self.params, "lambda_perc", 0.0) > 0:
            intensity_range = "0_1" if self.params.norm_ZeroToOne else "-1_1"
            self.criterionPerceptual = VGG19PerceptualLoss(
                layer_ids=tuple(self.params.perceptual_layers),
                normalized_range=intensity_range,
            )
        else:
            self.criterionPerceptual = None

    def forward(self, cbct_stack: torch.Tensor) -> torch.Tensor:
        return self.netG_A(cbct_stack)

    def clamp_for_eval(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.params.norm_ZeroToOne:
            return torch.clamp(tensor, 0.0, 1.0)
        return torch.clamp(tensor, -1.0, 1.0)

    def model_step(self, batch: Any, is_3d=False):
        center_cbct, real_ct, cbct_stack = batch
        fake_ct = self.forward(cbct_stack)
        if not self.training:
            fake_ct = self.clamp_for_eval(fake_ct)
        return center_cbct, real_ct, fake_ct

    def build_body_mask(self, center_cbct: torch.Tensor, real_ct: torch.Tensor) -> torch.Tensor:
        threshold = self.params.mask_threshold
        mask = (real_ct > threshold) | (center_cbct > threshold)
        return mask.float()

    def masked_mae_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(pred - target) * mask
        return loss.sum() / (mask.sum() + 1e-6)

    def apply_mask_background(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bg_value = 0.0 if self.params.norm_ZeroToOne else -1.0
        return tensor * mask + (1.0 - mask) * bg_value

    def shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        center_cbct, real_ct, cbct_stack = batch
        fake_ct = self.forward(cbct_stack)
        mask = self.build_body_mask(center_cbct, real_ct)

        loss_mae = self.masked_mae_loss(fake_ct, real_ct, mask)
        loss = self.params.lambda_mae * loss_mae
        self.log(f"{stage}/loss_mae", loss_mae.detach(), prog_bar=(stage == "train"), sync_dist=(stage != "train"))

        if self.criterionPerceptual is not None:
            masked_fake = self.apply_mask_background(fake_ct, mask)
            masked_real = self.apply_mask_background(real_ct, mask)
            loss_perc = self.criterionPerceptual(masked_fake, masked_real)
            loss = loss + self.params.lambda_perc * loss_perc
            self.log(
                f"{stage}/loss_perc",
                loss_perc.detach(),
                prog_bar=(stage == "train"),
                sync_dist=(stage != "train"),
            )

        self.log(f"{stage}/loss", loss.detach(), prog_bar=True, sync_dist=(stage != "train"))
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        _ = self.shared_step(batch, "val")
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int):
        _ = self.shared_step(batch, "test")
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

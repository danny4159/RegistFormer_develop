from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from src.metrics.gradient_correlation import GradientCorrelationMetric
from torchmetrics.clustering import NormalizedMutualInfoScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from src.metrics.sharpness import SharpnessMetric

from torchmetrics.aggregation import CatMetric #TODO: 나중에 필요하면 Catmetric 사용. std 구하기 용도

gray2rgb = lambda x: torch.cat((x, x, x), dim=1)
# norm_0_to_1 = lambda x: (x + 1) / 2
flatten_to_1d = lambda x: x.view(-1)
norm_to_uint8 = lambda x: ((x + 1) / 2 * 255).to(torch.uint8)

class BaseModule_AtoB(LightningModule):  # single direction
    def __init__(self, params, *args: Any, **kwargs: Any):
        super().__init__()
        self.params = params

        self.val_gc_B, self.val_nmi_B, self.val_fid_B, self.val_kid_B, self.val_sharpness_B = self.define_metrics()
        self.test_gc_B, self.test_nmi_B, self.test_fid_B, self.test_kid_B, self.test_sharpness_B = self.define_metrics()

        self.val_metrics = [self.val_gc_B, self.val_nmi_B, self.val_fid_B, self.val_kid_B, self.val_sharpness_B]
        self.test_metrics = [self.test_gc_B, self.test_nmi_B, self.test_fid_B, self.test_kid_B, self.test_sharpness_B]

        self.nmi_scores = []

    @staticmethod
    def define_metrics():
        gc = GradientCorrelationMetric()
        nmi = NormalizedMutualInfoScore()
        fid = FrechetInceptionDistance()
        kid = KernelInceptionDistance(kernel_size=2)
        sharpness = SharpnessMetric()

        return gc, nmi, fid, kid, sharpness

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return self.netG_A(a, b)

    def model_step(self, batch: Any):
        real_a, real_b = batch
        fake_b = self.forward(real_a, real_b)
        return real_a, real_b, fake_b

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for metrics in self.val_metrics:
            metrics.reset()
        for metrics in self.test_metrics:
            metrics.reset()
        return super().on_train_start()

    def backward_G(self, real_a, real_b, fake_b, *args, **kwargs):
        pass

    def backward_D_A(self, real_b, fake_b):
        loss_D_A = self.backward_D_basic(self.netD_A, real_b, fake_b)
        return loss_D_A

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def training_step(self, batch: Any, batch_idx: int):
        return super().training_step(batch, batch_idx)

    def on_train_epoch_start(self) -> None:
        self.loss_G = 0
        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    def validation_step(self, batch: Any, batch_idx: int):
        if self.params.use_split_inference:
            half_size = batch[0].shape[2] // 2
            first_half = [x[:, :, :half_size, :] for x in batch]
            second_half = [x[:, :, half_size:, :] for x in batch]

            res_first_half = self.model_step(first_half)
            res_second_half = self.model_step(second_half)

            real_A = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
            real_B = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
            fake_B = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
        else: 
            real_A, real_B, fake_B = self.model_step(batch)
        
        self.val_gc_B.update(norm_to_uint8(real_A), norm_to_uint8(fake_B))
        nmi_score = self.val_nmi_B(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_B)))
        self.nmi_scores.append(nmi_score)
        self.val_fid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
        self.val_fid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
        self.val_kid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
        self.val_kid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
        self.val_sharpness_B.update(norm_to_uint8(fake_B))

    def on_validation_epoch_end(self):
        gc = self.val_gc_B.compute()
        nmi = torch.mean(torch.stack(self.nmi_scores))
        fid = self.val_fid_B.compute()
        kid_mean, _ = self.val_kid_B.compute()
        sharpness = self.val_sharpness_B.compute()

        self.log("val/gc_B", gc.detach(), sync_dist=True)
        self.log("val/nmi_B", nmi.detach(), sync_dist=True)
        self.log("val/fid_B", fid.detach(), sync_dist=True)
        self.log("val/kid_B", kid_mean.detach(), sync_dist=True)
        self.log("val/sharpness_B", sharpness.detach(), sync_dist=True)

        for metrics in self.val_metrics:
            metrics.reset()
        self.nmi_scores = []

    def test_step(self, batch: Any, batch_idx: int):
        if self.params.use_split_inference:
            half_size = batch[0].shape[2] // 2
            first_half = [x[:, :, :half_size, :] for x in batch]
            second_half = [x[:, :, half_size:, :] for x in batch]

            res_first_half = self.model_step(first_half)
            res_second_half = self.model_step(second_half)

            real_A = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
            real_B = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
            fake_B = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
        else: 
            real_A, real_B, fake_B = self.model_step(batch)

        self.test_gc_B.update(norm_to_uint8(real_A), norm_to_uint8(fake_B))
        nmi_score = self.test_nmi_B(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_B)))
        self.nmi_scores.append(nmi_score)
        self.test_fid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
        self.test_fid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
        self.test_kid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
        self.test_kid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
        self.test_sharpness_B.update(norm_to_uint8(fake_B))



    def on_test_epoch_end(self):
        gc = self.test_gc_B.compute()
        nmi = torch.mean(torch.stack(self.nmi_scores))
        fid = self.test_fid_B.compute()
        kid_mean, kid_std = self.test_kid_B.compute()
        sharpness = self.test_sharpness_B.compute()
        
        self.log("test/gc_B_mean", gc.detach(), sync_dist=True)
        self.log("test/nmi_B_mean", nmi.detach(), sync_dist=True)
        self.log("test/fid_B_mean", fid.detach(), sync_dist=True)
        self.log("test/kid_B_mean", kid_mean.detach(), sync_dist=True)
        self.log("test/sharpness_B_mean", sharpness.detach(), sync_dist=True)

        gc_std = torch.std(self.test_gc_B.correlations)
        nmi_std = torch.std(torch.stack(self.nmi_scores))
        sharpness_std = torch.std(self.test_sharpness_B.scores)

        self.log("test/gc_B_std", gc_std.detach(), sync_dist=True)
        self.log("test/nmi_B_std", nmi_std.detach(), sync_dist=True)
        self.log("test/fid_B_std", torch.tensor(float('nan'), device=self.device).detach(), sync_dist=True)
        self.log("test/kid_B_std", kid_std.detach(), sync_dist=True)
        self.log("test/sharpness_B_std", sharpness_std.detach(), sync_dist=True)

        for metrics in self.test_metrics:
            metrics.reset()
        self.nmi_scores = []

    def configure_optimizers(self):
        pass

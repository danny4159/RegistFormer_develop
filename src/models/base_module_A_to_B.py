from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.clustering import NormalizedMutualInfoScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.aggregation import CatMetric
from src.metrics.sharpness import SharpnessMetric

gray2rgb = lambda x: torch.cat((x, x, x), dim=1)
# norm_0_to_1 = lambda x: (x + 1) / 2
flatten_to_1d = lambda x: x.view(-1)
norm_to_255 = lambda x: ((x + 1) / 2 * 255).to(torch.uint8)

class BaseModule_A_to_B(LightningModule):  # single direction
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        # define metrics (for image B)
        self.val_nmi_B, self.val_fid_B, self.val_sharpness_B = self.define_metrics()
        self.test_nmi_B, self.test_fid_B, self.test_sharpness_B = self.define_metrics()
        # (
        #     self.stats_nmi_B,
        #     self.stats_fid_B,
        #     self.stats_sharpness_B,
        # ) = self.define_cat_metrics()

        # create list (for reset)
        self.val_metrics = [self.val_nmi_B, self.val_fid_B, self.val_sharpness_B]
        self.test_metrics = [self.test_nmi_B, self.test_fid_B, self.test_sharpness_B]

    @staticmethod
    def define_metrics():
        nmi = NormalizedMutualInfoScore()
        fid = FrechetInceptionDistance()
        sharpness = SharpnessMetric()

        # psnr = PeakSignalNoiseRatio()
        # ssim = StructuralSimilarityIndexMeasure()
        # lpips = LearnedPerceptualImagePatchSimilarity()
        # return psnr, ssim, lpips
        return nmi, fid, sharpness

    # @staticmethod
    # def define_cat_metrics():
    #     nmi = CatMetric()
    #     fid = CatMetric()
    #     sharpness = CatMetric()
    #     return nmi, fid, sharpness

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
        loss_D_real = self.criterionGAN(pred_real, True).mean()
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False).mean()
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
        real_A, real_B, fake_B = self.model_step(batch)
        loss = F.l1_loss(real_B, fake_B) #TODO: l1 loss 말고 다른거로 바꾸기

        # Perform metric
        self.val_nmi_B(flatten_to_1d(norm_to_255(real_A)), flatten_to_1d(norm_to_255(fake_B)))
        self.val_fid_B.update(gray2rgb(norm_to_255(real_B)), real=True)
        self.val_fid_B.update(gray2rgb(norm_to_255(fake_B)), real=False)
        # self.val_fid_B.compute()

        # self.val_fid_B(gray2rgb(norm_to_255(real_B)), gray2rgb(norm_to_255(fake_B)))
        self.val_sharpness_B(fake_B)

        self.log("val/loss", loss.detach(), prog_bar=True)

    def on_validation_epoch_end(self):
        nmi = self.val_nmi_B.compute()
        fid = self.val_fid_B.compute()
        sharpness = self.val_sharpness_B.compute()

        self.log("val/nmi_B", nmi.detach())
        self.log("val/fid_B", fid.detach())
        self.log("val/sharpness_B", sharpness.detach())

        for metrics in self.val_metrics:
            metrics.reset()

    def test_step(self, batch: Any, batch_idx: int):
        real_A, real_B, fake_B = self.model_step(batch)

        loss = F.l1_loss(real_B, fake_B)

        _nmi_B = self.test_nmi_B(flatten_to_1d(norm_to_255(real_A)), flatten_to_1d(norm_to_255(fake_B)))
        self.test_fid_B.update(gray2rgb(norm_to_255(real_B)), real=True)
        self.test_fid_B.update(gray2rgb(norm_to_255(fake_B)), real=False)
        # _fid_B = self.test_fid_B.compute()
        
        # _fid_B = self.test_fid_B(gray2rgb(norm_to_255(real_B)), gray2rgb(norm_to_255(fake_B)))
        self.test_sharpness_B(fake_B)
        # _sharpness_B = self.test_sharpness_B(fake_B)

        # self.stats_nmi_B.update(_nmi_B)

        # self.stats_fid_B.update(_fid_B)

        # self.stats_sharpness_B.update(_sharpness_B)

        self.log("test/loss", loss.detach(), prog_bar=True)

    def on_test_epoch_end(self):

        nmi = self.test_nmi_B.compute()
        fid = self.test_fid_B.compute()
        sharpness = self.test_sharpness_B.compute()

        self.log("test/nmi_B_mean", nmi.mean())
        self.log("test/fid_B_mean", fid.mean())
        self.log("test/sharpness_B_mean", sharpness.mean())

        # nmi = self.stats_nmi_B.compute()
        # fid = self.stats_fid_B.compute()
        # sharpness = self.stats_sharpness_B.compute()

        # self.log("test/nmi_B_mean", nmi.mean())
        # self.log("test/fid_B_mean", fid.mean())
        # self.log("test/sharpness_B_mean", sharpness.mean())

        # self.log("test/nmi_B_std", nmi.std())
        # self.log("test/fid_B_std", fid.std())
        # self.log("test/sharpness_B_std", sharpness.std())

        for metrics in self.test_metrics:
            metrics.reset()

    def configure_optimizers(self):
        pass

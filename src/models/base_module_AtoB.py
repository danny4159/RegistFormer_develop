from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from src.metrics.gradient_correlation import GradientCorrelationMetric
from torchmetrics.clustering import NormalizedMutualInfoScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from src.metrics.sharpness import SharpnessMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from monai.inferers import SlidingWindowInferer

gray2rgb = lambda x: torch.cat((x, x, x), dim=1) if x.shape[1] == 1 else x
# norm_0_to_1 = lambda x: (x + 1) / 2
flatten_to_1d = lambda x: x.view(-1)
norm_to_uint8 = lambda x: ((x + 1) / 2 * 255).to(torch.uint8)

class BaseModule_AtoB(LightningModule):  # single direction
    def __init__(self, params, *args: Any, **kwargs: Any):
        super().__init__()
        self.params = params

        if self.params.eval_on_align:
            self.val_ssim_B, self.val_psnr_B, self.val_lpips_B, self.val_sharpness_B = self.define_metrics()
            self.test_ssim_B, self.test_psnr_B, self.test_lpips_B, self.test_sharpness_B = self.define_metrics()
            self.val_metrics = [self.val_ssim_B, self.val_psnr_B, self.val_lpips_B, self.val_sharpness_B]
            self.test_metrics = [self.test_ssim_B, self.test_psnr_B, self.test_lpips_B, self.test_sharpness_B]
            self.psnr_values_B = []
            self.lpips_values_B = []

            if self.params.use_multiple_outputs:
                self.val_ssim_C, self.val_psnr_C, self.val_lpips_C, self.val_sharpness_C = self.define_metrics()
                self.val_ssim_D, self.val_psnr_D, self.val_lpips_D, self.val_sharpness_D = self.define_metrics()
                self.test_ssim_C, self.test_psnr_C, self.test_lpips_C, self.test_sharpness_C = self.define_metrics()
                self.test_ssim_D, self.test_psnr_D, self.test_lpips_D, self.test_sharpness_D = self.define_metrics()
                self.val_metrics.extend([self.val_ssim_C, self.val_psnr_C, self.val_lpips_C, self.val_sharpness_C, 
                                         self.val_ssim_D, self.val_psnr_D, self.val_lpips_D, self.val_sharpness_D])
                self.test_metrics.extend([self.test_ssim_C, self.test_psnr_C, self.test_lpips_C, self.test_sharpness_C,
                                          self.test_ssim_D, self.test_psnr_D, self.test_lpips_D, self.test_sharpness_D])
                self.psnr_values_C, self.psnr_values_D = [], []
                self.lpips_values_C, self.lpips_values_D = [], []

            return

        self.val_gc_B, self.val_nmi_B, self.val_fid_B, self.val_kid_B, self.val_sharpness_B = self.define_metrics()
        self.test_gc_B, self.test_nmi_B, self.test_fid_B, self.test_kid_B, self.test_sharpness_B = self.define_metrics()
        self.val_metrics = [self.val_gc_B, self.val_nmi_B, self.val_fid_B, self.val_kid_B, self.val_sharpness_B]
        self.test_metrics = [self.test_gc_B, self.test_nmi_B, self.test_fid_B, self.test_kid_B, self.test_sharpness_B]
        self.nmi_scores_B = []

        if self.params.use_multiple_outputs:
            self.val_gc_C, self.val_nmi_C, self.val_fid_C, self.val_kid_C, self.val_sharpness_C = self.define_metrics()
            self.val_gc_D, self.val_nmi_D, self.val_fid_D, self.val_kid_D, self.val_sharpness_D = self.define_metrics()
            self.test_gc_C, self.test_nmi_C, self.test_fid_C, self.test_kid_C, self.test_sharpness_C = self.define_metrics()
            self.test_gc_D, self.test_nmi_D, self.test_fid_D, self.test_kid_D, self.test_sharpness_D = self.define_metrics()
            self.val_metrics.extend([self.val_gc_C, self.val_nmi_C, self.val_fid_C, self.val_kid_C, self.val_sharpness_C,
                                     self.val_gc_D, self.val_nmi_D, self.val_fid_D, self.val_kid_D, self.val_sharpness_D])
            self.test_metrics.extend([self.test_gc_C, self.test_nmi_C, self.test_fid_C, self.test_kid_C, self.test_sharpness_C,
                                      self.test_gc_D, self.test_nmi_D, self.test_fid_D, self.test_kid_D, self.test_sharpness_D])
            self.nmi_scores_C, self.nmi_scores_D = [], []

    def define_metrics(self):
        if self.params.eval_on_align:
            # Following Pytorch lightning metric
            # PSNR, LPIPS: 'update', 'compute', and 'append' at each step, calculate 'mean and std' at the end of epoch
            # SSIM: initialize with reduction='none', 'update' at each step, 'compute' at the end of epoch, then calculate 'mean and std'
            ssim = StructuralSimilarityIndexMeasure(reduction="none")
            psnr = PeakSignalNoiseRatio()
            lpips = LearnedPerceptualImagePatchSimilarity()
            sharpness = SharpnessMetric()
            return ssim, psnr, lpips, sharpness
        
        gc = GradientCorrelationMetric()
        nmi = NormalizedMutualInfoScore()
        fid = FrechetInceptionDistance()
        kid = KernelInceptionDistance(subset_size=2)
        sharpness = SharpnessMetric()

        return gc, nmi, fid, kid, sharpness

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        
        # Case1. Refernece guided generation
        if type(self.netG_A).__name__ in ["RegistFormer", "ProposedSynthesisModule"]:
            merged_input = torch.cat((a, b), dim=1)
            
            # Sliding window on inference
            if self.params.use_sliding_inference and not self.training:
                inferer = SlidingWindowInferer(roi_size=(128, 128)) # 96, 96
                pred = inferer(inputs=merged_input, network=self.netG_A) #inputs 손봐야해
                return pred
        
            return self.netG_A(merged_input)
        
        # Case2. Normal generation
        return self.netG_A(a)

    def model_step(self, batch: Any):
        if self.params.use_multiple_outputs:
            if len(batch) == 3: # Ref cont 2
                real_a, real_b, real_c = batch
                real_merged = torch.cat((real_b, real_c), dim=1)
                fake_merged = self.forward(real_a, real_merged)
                fake_b, fake_c = fake_merged[:, :1, :, :], fake_merged[:, 1:, :, :]
                real_d, fake_d = None, None
            elif len(batch) == 4: # Ref cont 3
                real_a, real_b, real_c, real_d = batch
                real_merged = torch.cat((real_b, real_c, real_d), dim=1)
                fake_merged = self.forward(real_a, real_merged)
                fake_b, fake_c, fake_d = fake_merged[:, :1, :, :], fake_merged[:, 1:2, :, :], fake_merged[:, 2:, :, :]

            return real_a, real_b, real_c, real_d, fake_b, fake_c, fake_d

        else: 
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
    
    def backward_D_B(self, real_b, fake_b):
        loss_D_A = self.backward_D_basic(self.netD_B, real_b, fake_b)
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
        if self.params.use_multiple_outputs:
            real_A, real_B, real_C, real_D, fake_B, fake_C, fake_D = self.model_step(batch)
        else:
            real_A, real_B, fake_B = self.model_step(batch)

        if self.params.eval_on_align:
            self.val_ssim_B.update(real_B, fake_B)
            self.val_psnr_B.update(real_B, fake_B)
            self.psnr_values_B.append(self.val_psnr_B.compute().item())
            self.val_psnr_B.reset()
            self.val_lpips_B.update(gray2rgb(real_B), gray2rgb(fake_B))
            self.lpips_values_B.append(self.val_lpips_B.compute().item())
            self.val_lpips_B.reset()
            self.val_sharpness_B.update(norm_to_uint8(fake_B).float())

            if self.params.use_multiple_outputs:
                self.val_ssim_C.update(real_C, fake_C)
                self.val_psnr_C.update(real_C, fake_C)
                self.psnr_values_C.append(self.val_psnr_C.compute().item())
                self.val_psnr_C.reset()
                self.val_lpips_C.update(gray2rgb(real_C), gray2rgb(fake_C))
                self.lpips_values_C.append(self.val_lpips_C.compute().item())
                self.val_lpips_C.reset()
                self.val_sharpness_C.update(norm_to_uint8(fake_C).float())

                if fake_D is not None:
                    self.val_ssim_D.update(real_D, fake_D)
                    self.val_psnr_D.update(real_D, fake_D)
                    self.psnr_values_D.append(self.val_psnr_D.compute().item())
                    self.val_psnr_D.reset()
                    self.val_lpips_D.update(gray2rgb(real_D), gray2rgb(fake_D))
                    self.lpips_values_D.append(self.val_lpips_D.compute().item())
                    self.val_lpips_D.reset()
                    self.val_sharpness_D.update(norm_to_uint8(fake_D).float())


        else:
            self.val_gc_B.update(norm_to_uint8(real_A), norm_to_uint8(fake_B))
            nmi_score = self.val_nmi_B(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_B)))
            self.nmi_scores_B.append(nmi_score)
            self.val_fid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
            self.val_fid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
            self.val_kid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
            self.val_kid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
            self.val_sharpness_B.update(norm_to_uint8(fake_B))

            if self.params.use_multiple_outputs:
                self.val_gc_C.update(norm_to_uint8(real_A), norm_to_uint8(fake_C))
                nmi_score = self.val_nmi_C(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_C)))
                self.nmi_scores_C.append(nmi_score)
                self.val_fid_C.update(gray2rgb(norm_to_uint8(real_C)), real=True)
                self.val_fid_C.update(gray2rgb(norm_to_uint8(fake_C)), real=False)
                self.val_kid_C.update(gray2rgb(norm_to_uint8(real_C)), real=True)
                self.val_kid_C.update(gray2rgb(norm_to_uint8(fake_C)), real=False)
                self.val_sharpness_C.update(norm_to_uint8(fake_C))

                if fake_D is not None:
                    self.val_gc_D.update(norm_to_uint8(real_A), norm_to_uint8(fake_D))
                    nmi_score = self.val_nmi_D(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_D)))
                    self.nmi_scores_D.append(nmi_score)
                    self.val_fid_D.update(gray2rgb(norm_to_uint8(real_D)), real=True)
                    self.val_fid_D.update(gray2rgb(norm_to_uint8(fake_D)), real=False)
                    self.val_kid_D.update(gray2rgb(norm_to_uint8(real_D)), real=True)
                    self.val_kid_D.update(gray2rgb(norm_to_uint8(fake_D)), real=False)
                    self.val_sharpness_D.update(norm_to_uint8(fake_D))

    def on_validation_epoch_end(self):
        if self.params.eval_on_align:
            ssim_B = self.val_ssim_B.compute().mean()
            psnr_B = torch.mean(torch.tensor(self.psnr_values_B, device=self.device))
            lpips_B = torch.mean(torch.tensor(self.lpips_values_B, device=self.device))
            sharpness_B = self.val_sharpness_B.compute()
            self.log("val/ssim_B", ssim_B.detach(), sync_dist=True)
            self.log("val/psnr_B", psnr_B.detach(), sync_dist=True)
            self.log("val/lpips_B", lpips_B.detach(), sync_dist=True)
            self.log("val/sharpness_B", sharpness_B.detach(), sync_dist=True)

            if self.params.use_multiple_outputs:
                ssim_C = self.val_ssim_C.compute().mean()
                psnr_C = torch.mean(torch.tensor(self.psnr_values_C, device=self.device))
                lpips_C = torch.mean(torch.tensor(self.lpips_values_C, device=self.device))
                sharpness_C = self.val_sharpness_C.compute()
                self.log("val/ssim_C", ssim_C.detach(), sync_dist=True)
                self.log("val/psnr_C", psnr_C.detach(), sync_dist=True)
                self.log("val/lpips_C", lpips_C.detach(), sync_dist=True)
                self.log("val/sharpness_C", sharpness_C.detach(), sync_dist=True)
                
                if self.psnr_values_D:
                    ssim_D = self.val_ssim_D.compute().mean()
                    psnr_D = torch.mean(torch.tensor(self.psnr_values_D, device=self.device))
                    lpips_D = torch.mean(torch.tensor(self.lpips_values_D, device=self.device))
                    sharpness_D = self.val_sharpness_D.compute()
                    self.log("val/ssim_D", ssim_D.detach(), sync_dist=True)
                    self.log("val/psnr_D", psnr_D.detach(), sync_dist=True)
                    self.log("val/lpips_D", lpips_D.detach(), sync_dist=True)
                    self.log("val/sharpness_D", sharpness_D.detach(), sync_dist=True)

            for metrics in self.val_metrics:
                metrics.reset()
            self.psnr_values_B = []
            self.lpips_values_B = []

            if self.params.use_multiple_outputs:
                self.psnr_values_C = []
                self.psnr_values_D = []
                self.lpips_values_C = []
                self.lpips_values_D = []

        else:
            gc = self.val_gc_B.compute()
            nmi = torch.mean(torch.stack(self.nmi_scores_B))
            fid = self.val_fid_B.compute()
            kid_mean, _ = self.val_kid_B.compute()
            sharpness = self.val_sharpness_B.compute()
            self.log("val/gc_B", gc.detach(), sync_dist=True)
            self.log("val/nmi_B", nmi.detach(), sync_dist=True)
            self.log("val/fid_B", fid.detach(), sync_dist=True)
            self.log("val/kid_B", kid_mean.detach(), sync_dist=True)
            self.log("val/sharpness_B", sharpness.detach(), sync_dist=True)

            if self.params.use_multiple_outputs:
                gc = self.val_gc_C.compute()
                nmi = torch.mean(torch.stack(self.nmi_scores_C))
                fid = self.val_fid_C.compute()
                kid_mean, _ = self.val_kid_C.compute()
                sharpness = self.val_sharpness_C.compute()
                self.log("val/gc_C", gc.detach(), sync_dist=True)
                self.log("val/nmi_C", nmi.detach(), sync_dist=True)
                self.log("val/fid_C", fid.detach(), sync_dist=True)
                self.log("val/kid_C", kid_mean.detach(), sync_dist=True)
                self.log("val/sharpness_C", sharpness.detach(), sync_dist=True)

                if self.nmi_scores_D:
                    gc = self.val_gc_D.compute()
                    nmi = torch.mean(torch.stack(self.nmi_scores_D))
                    fid = self.val_fid_D.compute()
                    kid_mean, _ = self.val_kid_D.compute()
                    sharpness = self.val_sharpness_D.compute()
                    self.log("val/gc_D", gc.detach(), sync_dist=True)
                    self.log("val/nmi_D", nmi.detach(), sync_dist=True)
                    self.log("val/fid_D", fid.detach(), sync_dist=True)
                    self.log("val/kid_D", kid_mean.detach(), sync_dist=True)
                    self.log("val/sharpness_D", sharpness.detach(), sync_dist=True)

            for metrics in self.val_metrics:
                metrics.reset()
            self.nmi_scores_B = []

            if self.params.use_multiple_outputs:
                self.nmi_scores_C = []
                self.nmi_scores_D = []

    def test_step(self, batch: Any, batch_idx: int):
 
        if self.params.use_multiple_outputs:
            real_A, real_B, real_C, real_D, fake_B, fake_C, fake_D = self.model_step(batch)
        else:
            real_A, real_B, fake_B = self.model_step(batch)

        if self.params.eval_on_align:
            self.test_ssim_B.update(real_B, fake_B)
            self.test_psnr_B.update(real_B, fake_B)
            self.psnr_values_B.append(self.test_psnr_B.compute().item())
            self.test_psnr_B.reset()
            # self.test_lpips_B.update(real_B2, fake_B)
            self.test_lpips_B.update(gray2rgb(real_B), gray2rgb(fake_B))
            self.lpips_values_B.append(self.test_lpips_B.compute().item())
            self.test_lpips_B.reset()
            self.test_sharpness_B.update(norm_to_uint8(fake_B).float())
            
            if self.params.use_multiple_outputs:
                self.test_ssim_C.update(real_C, fake_C)
                self.test_psnr_C.update(real_C, fake_C)
                self.psnr_values_C.append(self.test_psnr_C.compute().item())
                self.test_psnr_C.reset()
                self.test_lpips_C.update(gray2rgb(real_C), gray2rgb(fake_C))
                self.lpips_values_C.append(self.test_lpips_C.compute().item())
                self.test_lpips_C.reset()

                if fake_D is not None:
                    self.test_ssim_D.update(real_D, fake_D)
                    self.test_psnr_D.update(real_D, fake_D)
                    self.psnr_values_D.append(self.test_psnr_D.compute().item())
                    self.test_psnr_D.reset()
                    self.test_lpips_D.update(gray2rgb(real_D), gray2rgb(fake_D))
                    self.lpips_values_D.append(self.test_lpips_D.compute().item())
                    self.test_lpips_D.reset()
                    self.test_sharpness_D.update(norm_to_uint8(fake_D).float())

        else:
            self.test_gc_B.update(norm_to_uint8(real_A), norm_to_uint8(fake_B))
            nmi_score = self.test_nmi_B(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_B)))
            self.nmi_scores_B.append(nmi_score)
            self.test_fid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
            self.test_fid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
            self.test_kid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
            self.test_kid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
            self.test_sharpness_B.update(norm_to_uint8(fake_B))

            if self.params.use_multiple_outputs:
                self.test_gc_C.update(norm_to_uint8(real_A), norm_to_uint8(fake_C))
                nmi_score = self.test_nmi_C(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_C)))
                self.nmi_scores_C.append(nmi_score)
                self.test_fid_C.update(gray2rgb(norm_to_uint8(real_C)), real=True)
                self.test_fid_C.update(gray2rgb(norm_to_uint8(fake_C)), real=False)
                self.test_kid_C.update(gray2rgb(norm_to_uint8(real_C)), real=True)
                self.test_kid_C.update(gray2rgb(norm_to_uint8(fake_C)), real=False)
                self.test_sharpness_C.update(norm_to_uint8(fake_C))

                if fake_D is not None:
                    self.test_gc_D.update(norm_to_uint8(real_A), norm_to_uint8(fake_D))
                    nmi_score = self.test_nmi_D(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_D)))
                    self.nmi_scores_D.append(nmi_score)
                    self.test_fid_D.update(gray2rgb(norm_to_uint8(real_D)), real=True)
                    self.test_fid_D.update(gray2rgb(norm_to_uint8(fake_D)), real=False)
                    self.test_kid_D.update(gray2rgb(norm_to_uint8(real_D)), real=True)
                    self.test_kid_D.update(gray2rgb(norm_to_uint8(fake_D)), real=False)
                    self.test_sharpness_D.update(norm_to_uint8(fake_D))

    def on_test_epoch_end(self):

        if self.params.eval_on_align:
            ssim_B = self.test_ssim_B.compute().mean()
            psnr_B = torch.mean(torch.tensor(self.psnr_values_B, device=self.device))
            lpips_B = torch.mean(torch.tensor(self.lpips_values_B, device=self.device))
            sharpness_B = self.test_sharpness_B.compute()
            self.log("test/ssim_B", ssim_B.detach(), sync_dist=True)
            self.log("test/psnr_B", psnr_B.detach(), sync_dist=True)
            self.log("test/lpips_B", lpips_B.detach(), sync_dist=True)
            self.log("test/sharpness_B", sharpness_B.detach(), sync_dist=True)
            ssim_B_std = torch.std(torch.tensor([metric.item() for metric in self.test_ssim_B.similarity], device=self.device))
            psnr_B_std = torch.std(torch.tensor(self.psnr_values_B, device=self.device))
            lpips_B_std = torch.std(torch.tensor(self.lpips_values_B, device=self.device))
            sharpness_B_std = torch.std(self.test_sharpness_B.scores)
            self.log("test/ssim_B_std", ssim_B_std.detach(), sync_dist=True)
            self.log("test/psnr_B_std", psnr_B_std.detach(), sync_dist=True)
            self.log("test/lpips_B_std", lpips_B_std.detach(), sync_dist=True)
            self.log("test/sharpness_B_std", sharpness_B_std.detach(), sync_dist=True)

            if self.params.use_multiple_outputs:
                ssim_C = self.test_ssim_C.compute().mean()
                psnr_C = torch.mean(torch.tensor(self.psnr_values_C, device=self.device))
                lpips_C = torch.mean(torch.tensor(self.lpips_values_C, device=self.device))
                sharpness_C = self.test_sharpness_C.compute()
                self.log("test/ssim_C", ssim_C.detach(), sync_dist=True)
                self.log("test/psnr_C", psnr_C.detach(), sync_dist=True)
                self.log("test/lpips_C", lpips_C.detach(), sync_dist=True)
                self.log("test/sharpness_C", sharpness_C.detach(), sync_dist=True)
                ssim_C_std = torch.std(torch.tensor([metric.item() for metric in self.test_ssim_C.similarity], device=self.device))
                psnr_C_std = torch.std(torch.tensor(self.psnr_values_C, device=self.device))
                lpips_C_std = torch.std(torch.tensor(self.lpips_values_C, device=self.device))
                sharpness_C_std = torch.std(self.test_sharpness_C.scores)
                self.log("test/ssim_C_std", ssim_C_std.detach(), sync_dist=True)
                self.log("test/psnr_C_std", psnr_C_std.detach(), sync_dist=True)
                self.log("test/lpips_C_std", lpips_C_std.detach(), sync_dist=True)
                self.log("test/sharpness_C_std", sharpness_C_std.detach(), sync_dist=True)

                if self.psnr_values_D:
                    ssim_D = self.test_ssim_D.compute().mean()
                    psnr_D = torch.mean(torch.tensor(self.psnr_values_D, device=self.device))
                    lpips_D = torch.mean(torch.tensor(self.lpips_values_D, device=self.device))
                    sharpness_D = self.test_sharpness_D.compute()
                    self.log("test/ssim_D", ssim_D.detach(), sync_dist=True)
                    self.log("test/psnr_D", psnr_D.detach(), sync_dist=True)
                    self.log("test/lpips_D", lpips_D.detach(), sync_dist=True)
                    self.log("test/sharpness_D", sharpness_D.detach(), sync_dist=True)
                    ssim_D_std = torch.std(torch.tensor([metric.item() for metric in self.test_ssim_D.similarity], device=self.device))
                    psnr_D_std = torch.std(torch.tensor(self.psnr_values_D, device=self.device))
                    lpips_D_std = torch.std(torch.tensor(self.lpips_values_D, device=self.device))
                    sharpness_D_std = torch.std(self.test_sharpness_D.scores)
                    self.log("test/ssim_D_std", ssim_D_std.detach(), sync_dist=True)
                    self.log("test/psnr_D_std", psnr_D_std.detach(), sync_dist=True)
                    self.log("test/lpips_D_std", lpips_D_std.detach(), sync_dist=True)
                    self.log("test/sharpness_D_std", sharpness_D_std.detach(), sync_dist=True)

            for metrics in self.test_metrics:
                metrics.reset()
            self.psnr_values_B = []
            self.lpips_values_B = []

            if self.params.use_multiple_outputs:
                self.psnr_values_C = []
                self.psnr_values_D = []
                self.lpips_values_C = []
                self.lpips_values_D = []
        
        else:
            gc = self.test_gc_B.compute()
            nmi = torch.mean(torch.stack(self.nmi_scores_B))
            fid = self.test_fid_B.compute()
            kid_mean, kid_std = self.test_kid_B.compute()
            sharpness = self.test_sharpness_B.compute()
            self.log("test/gc_B_mean", gc.detach(), sync_dist=True)
            self.log("test/nmi_B_mean", nmi.detach(), sync_dist=True)
            self.log("test/fid_B_mean", fid.detach(), sync_dist=True)
            self.log("test/kid_B_mean", kid_mean.detach(), sync_dist=True)
            self.log("test/sharpness_B_mean", sharpness.detach(), sync_dist=True)
            gc_std = torch.std(self.test_gc_B.correlations)
            nmi_std = torch.std(torch.stack(self.nmi_scores_B))
            sharpness_std = torch.std(self.test_sharpness_B.scores)
            self.log("test/gc_B_std", gc_std.detach(), sync_dist=True)
            self.log("test/nmi_B_std", nmi_std.detach(), sync_dist=True)
            self.log("test/fid_B_std", torch.tensor(float('nan'), device=self.device).detach(), sync_dist=True)
            self.log("test/kid_B_std", kid_std.detach(), sync_dist=True)
            self.log("test/sharpness_B_std", sharpness_std.detach(), sync_dist=True)

            if self.params.use_multiple_outputs:
                gc = self.test_gc_C.compute()
                nmi = torch.mean(torch.stack(self.nmi_scores_C))
                fid = self.test_fid_C.compute()
                kid_mean, kid_std = self.test_kid_C.compute()
                sharpness = self.test_sharpness_C.compute()
                self.log("test/gc_C_mean", gc.detach(), sync_dist=True)
                self.log("test/nmi_C_mean", nmi.detach(), sync_dist=True)
                self.log("test/fid_C_mean", fid.detach(), sync_dist=True)
                self.log("test/kid_C_mean", kid_mean.detach(), sync_dist=True)
                self.log("test/sharpness_C_mean", sharpness.detach(), sync_dist=True)
                gc_std = torch.std(self.test_gc_C.correlations)
                nmi_std = torch.std(torch.stack(self.nmi_scores_C))
                sharpness_std = torch.std(self.test_sharpness_C.scores)
                self.log("test/gc_C_std", gc_std.detach(), sync_dist=True)
                self.log("test/nmi_C_std", nmi_std.detach(), sync_dist=True)
                self.log("test/fid_C_std", torch.tensor(float('nan'), device=self.device).detach(), sync_dist=True)
                self.log("test/kid_C_std", kid_std.detach(), sync_dist=True)
                self.log("test/sharpness_C_std", sharpness_std.detach(), sync_dist=True)
                
                if self.nmi_scores_D:
                    gc = self.test_gc_D.compute()
                    nmi = torch.mean(torch.stack(self.nmi_scores_D))
                    fid = self.test_fid_D.compute()
                    kid_mean, kid_std = self.test_kid_D.compute()
                    sharpness = self.test_sharpness_D.compute()
                    self.log("test/gc_D_mean", gc.detach(), sync_dist=True)
                    self.log("test/nmi_D_mean", nmi.detach(), sync_dist=True)
                    self.log("test/fid_D_mean", fid.detach(), sync_dist=True)
                    self.log("test/kid_D_mean", kid_mean.detach(), sync_dist=True)
                    self.log("test/sharpness_D_mean", sharpness.detach(), sync_dist=True)
                    gc_std = torch.std(self.test_gc_D.correlations)
                    nmi_std = torch.std(torch.stack(self.nmi_scores_D))
                    sharpness_std = torch.std(self.test_sharpness_D.scores)
                    self.log("test/gc_D_std", gc_std.detach(), sync_dist=True)
                    self.log("test/nmi_D_std", nmi_std.detach(), sync_dist=True)
                    self.log("test/fid_D_std", torch.tensor(float('nan'), device=self.device).detach(), sync_dist=True)
                    self.log("test/kid_D_std", kid_std.detach(), sync_dist=True)
                    self.log("test/sharpness_D_std", sharpness_std.detach(), sync_dist=True)

            for metrics in self.test_metrics:
                metrics.reset()
            self.nmi_scores_B = []

            if self.params.use_multiple_outputs:
                self.nmi_scores_C = []
                self.nmi_scores_D = []

    def configure_optimizers(self):
        pass

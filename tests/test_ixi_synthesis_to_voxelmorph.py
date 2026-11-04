import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import nibabel as nib
import numpy as np
import torch

from src.ixi_synthesis_to_voxelmorph import H5_NAME, export_split, select_best_checkpoint
from src.models.voxelmorph_original_module import VoxelmorphOriginalModule


class ExportTests(unittest.TestCase):
    def test_nifti_to_h5_preserves_values_axes_ids_and_source_groups(self):
        class Synthesis(torch.nn.Module):
            def forward(self, source, reference):
                return source * 0.25 + reference * 0.5

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "source.h5"
            source = np.linspace(-0.9, 0.9, 8 * 12 * 3, dtype=np.float32).reshape(8, 12, 3)
            reference = np.flip(source, axis=0).copy()
            target = np.full_like(source, -0.75)
            with h5py.File(original, "w") as file:
                for name, value in [("T1", source), ("T2", target), ("T2_moved_3mm_RigidReg", reference)]:
                    file.create_group(name).create_dataset("IXI-patient-id", data=value)
            result = export_split(Synthesis(), original, root / "out", "train",
                ["T1", "T2", "T2_moved_3mm_RigidReg"], "checkpoint-hash", "cpu", 2)
            expected = source * 0.25 + reference * 0.5
            image = nib.load(result["patients"][0]["nifti"])
            np.testing.assert_array_equal(image.get_fdata(dtype=np.float32), expected)
            with h5py.File(root / "out/train" / H5_NAME, "r") as file:
                np.testing.assert_array_equal(file["T1/IXI-patient-id"][...], source)
                np.testing.assert_array_equal(file["T2/IXI-patient-id"][...], target)
                np.testing.assert_array_equal(file["synth_T2/IXI-patient-id"][...], expected)
                self.assertTrue(file.attrs["complete"])
                self.assertEqual(file.attrs["synthesis_checkpoint_sha256"], "checkpoint-hash")
            self.assertEqual(result["slice_count"], 3)

    def test_best_checkpoint_uses_callback_state_and_requires_finished_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            best = root / "best.ckpt"
            best.touch()
            last = root / "last.ckpt"
            state = {"epoch": 99, "callbacks": {"ModelCheckpoint{'monitor': 'val/ssim_B'}":
                {"best_model_path": str(best), "best_model_score": torch.tensor(0.85)}}}
            torch.save(state, last)
            selected, score, epoch = select_best_checkpoint(last, required_epoch=99)
            self.assertEqual(selected, best)
            self.assertAlmostEqual(score, 0.85)
            self.assertEqual(epoch, 99)
            state["epoch"] = 98
            torch.save(state, last)
            with self.assertRaisesRegex(RuntimeError, "expected >= 99"):
                select_best_checkpoint(last, required_epoch=99)


class RegistrationDirectionTests(unittest.TestCase):
    def test_moving_is_rigid_t2_fixed_is_synthesis_and_t2_is_evaluation_only(self):
        class Registration(torch.nn.Module):
            def forward(self, source, target, registration=False):
                self.inputs = (source, target)
                return source + 0.1, torch.zeros(source.shape[0], 2, *source.shape[2:])

        module = VoxelmorphOriginalModule.__new__(VoxelmorphOriginalModule)
        torch.nn.Module.__init__(module)
        module.params = SimpleNamespace(use_misalign_simul=False)
        module.netR_A = Registration()
        evaluation_t2 = torch.full((2, 1, 32, 32), 0.2)
        moving_rigid_t2 = torch.full_like(evaluation_t2, 0.4)
        fixed_synth_t2 = torch.full_like(evaluation_t2, 0.6)
        result = module.model_step((evaluation_t2, moving_rigid_t2, fixed_synth_t2), is_train=True)
        self.assertIs(module.netR_A.inputs[0], moving_rigid_t2)
        self.assertIs(module.netR_A.inputs[1], fixed_synth_t2)
        self.assertIs(result[0], evaluation_t2)
        self.assertIs(result[2], fixed_synth_t2)
        self.assertEqual(tuple(result[4].shape), (2, 2, 32, 32))

    def test_l1_objective_value_and_gradient_with_other_losses_disabled(self):
        module = VoxelmorphOriginalModule.__new__(VoxelmorphOriginalModule)
        torch.nn.Module.__init__(module)
        module.params = SimpleNamespace(lambda_l1=1, lambda_l2=0, lambda_grad=0,
            lambda_mask_l2=0, lambda_smooth=0, lambda_lcc=0)
        module.criterionL1 = torch.nn.L1Loss()
        module.criterionL2 = torch.nn.MSELoss()
        module.criterionGrad = module.criterionMaskL2 = module.criterionSmooth = None
        module.log = lambda *args, **kwargs: None
        fixed = torch.zeros(1, 1, 1, 2)
        warped = torch.tensor([[[[0., 3.]]]], requires_grad=True)
        loss = module.backward_R(fixed, warped, torch.zeros(1, 2, 1, 2))
        self.assertAlmostEqual(loss.item(), 1.5)
        loss.backward()
        torch.testing.assert_close(warped.grad, torch.tensor([[[[0., 0.5]]]]))

    def test_registration_test_lpips_clamps_interpolation_overshoot(self):
        class Metric:
            def update(self, *values):
                self.values = values

            def compute(self):
                return torch.tensor(1.)

            def reset(self):
                pass

        module = VoxelmorphOriginalModule.__new__(VoxelmorphOriginalModule)
        torch.nn.Module.__init__(module)
        module.params = SimpleNamespace(is_3d=False, eval_on_align=True)
        for name in ['test_ssim_B', 'test_psnr_B', 'test_lpips_B', 'test_sharpness_B']:
            setattr(module, name, Metric())
        module.psnr_values_B, module.lpips_values_B = [], []
        evaluation = torch.full((1, 1, 32, 32), 1.05)
        warped = torch.full_like(evaluation, -1.05)
        module.model_step = lambda *args, **kwargs: (evaluation, warped, evaluation, warped)
        module.test_step(None, 0)
        for value in module.test_lpips_B.values:
            self.assertGreaterEqual(value.min().item(), -1)
            self.assertLessEqual(value.max().item(), 1)


if __name__ == "__main__":
    unittest.main()

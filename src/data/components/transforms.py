from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import h5py
from src import utils
import os
from torch.utils.data import Dataset
from monai.transforms import RandFlipd, RandRotate90d, Compose, RandCropd
from monai.utils.type_conversion import convert_to_tensor
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as nnF

# Misalign
import torchio as tio
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from monai.transforms import Affine
import random
import math
from tqdm import tqdm


def _register_3d_rigid(fixed_np: np.ndarray, moving_np: np.ndarray, z_pad: int = 20) -> np.ndarray:
    """Rigid 3D registration of moving → fixed using SimpleITK.

    z_pad slices are reflect-padded before registration and cropped after,
    preventing diagonal boundary artifacts caused by z-direction translation/rotation.

    Args:
        fixed_np:  (H, W, D) float32 array, [-1, 1] normalized
        moving_np: (H, W, D) float32 array, [-1, 1] normalized
        z_pad:     number of slices to pad on each end in z before registration
    Returns:
        registered moving volume (H, W, D) float32, background filled with -1
    """
    import SimpleITK as sitk

    D_orig = fixed_np.shape[2]

    # replicate-pad in z (edge slice repeated) so registration sees continuous signal at boundaries
    fixed_pad  = np.pad(fixed_np,  ((0, 0), (0, 0), (z_pad, z_pad)), mode='edge')
    moving_pad = np.pad(moving_np, ((0, 0), (0, 0), (z_pad, z_pad)), mode='edge')

    # numpy (H, W, D) → SimpleITK expects (z, y, x) = (D, H, W)
    fixed_sitk  = sitk.GetImageFromArray(fixed_pad.transpose(2, 0, 1).astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving_pad.transpose(2, 0, 1).astype(np.float32))

    initial_tf = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200
    )
    R.SetOptimizerScalesFromIndexShift()
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(initial_tf, inPlace=False)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_tf = R.Execute(fixed_sitk, moving_sitk)

    resampled = sitk.Resample(
        moving_sitk, fixed_sitk, final_tf,
        sitk.sitkLinear, -1.0, moving_sitk.GetPixelID(),
    )

    # SimpleITK (D, H, W) → numpy (H, W, D), then crop back to original z
    result_pad = sitk.GetArrayFromImage(resampled).transpose(1, 2, 0).astype(np.float32)
    return result_pad[:, :, z_pad:z_pad + D_orig]

def _collect_present_tensors(*tensors):
    return tuple(tensor for tensor in tensors if tensor is not None)


def padding_height_width_depth(
    tensorA,
    tensorB,
    tensorC=None,
    tensorD=None,
    tensorE=None,
    tensorF=None,
    tensorG=None,
    target_size=(256, 256),
    pad_value=-1,
):
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    elif isinstance(target_size, (list, tuple)) and len(target_size) not in (2, 3):
        raise ValueError("target_size는 int, (H, W), 또는 (H, W, D) 형식이어야 합니다.")

    tensors = [tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG]
    tensors = [tensor for tensor in tensors if tensor is not None]

    if len(tensorA.shape) == 3:
        _, h, w = tensorA.shape
        expected_ndim = 3
    elif len(tensorA.shape) == 4:
        _, h, w, d = tensorA.shape
        expected_ndim = 4
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    for idx, tensor in enumerate(tensors, start=1):
        assert len(tensor.shape) == expected_ndim, f"Input tensor {idx} must have {expected_ndim} dimensions"

    pad_top = (target_size[0] - h + 1) // 2 if h < target_size[0] else 0
    pad_bottom = (target_size[0] - h) // 2 if h < target_size[0] else 0
    pad_left = (target_size[1] - w + 1) // 2 if w < target_size[1] else 0
    pad_right = (target_size[1] - w) // 2 if w < target_size[1] else 0

    if len(target_size) == 2:
        if pad_top != 0 or pad_left != 0:
            padded = []
            for tensor in tensors:
                padded.append(nnF.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value))
            tensors = padded
    else:
        pad_front = (target_size[2] - d + 1) // 2 if d < target_size[2] else 0
        pad_back = (target_size[2] - d) // 2 if d < target_size[2] else 0
        if pad_top != 0 or pad_left != 0 or pad_front != 0:
            padded = []
            for tensor in tensors:
                padded.append(nnF.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back), value=pad_value))
            tensors = padded

    return tuple(tensors)


def random_crop(
    tensorA,
    tensorB,
    tensorC=None,
    tensorD=None,
    tensorE=None,
    tensorF=None,
    tensorG=None,
    target_size=(128, 128),
):
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    elif isinstance(target_size, (list, tuple)) and len(target_size) not in (2, 3):
        raise ValueError("target_size는 int, (H, W), 또는 (H, W, D) 형식이어야 합니다.")

    tensors = [tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG]
    tensors = [tensor for tensor in tensors if tensor is not None]

    if len(tensorA.shape) == 3:
        _, h, w = tensorA.shape
        expected_ndim = 3
    elif len(tensorA.shape) == 4:
        _, h, w, d = tensorA.shape
        expected_ndim = 4
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    for idx, tensor in enumerate(tensors, start=1):
        assert len(tensor.shape) == expected_ndim, f"Input tensor {idx} must have {expected_ndim} dimensions"

    top = torch.randint(0, h - target_size[0] + 1, size=(1,)).item() if h > target_size[0] else 0
    left = torch.randint(0, w - target_size[1] + 1, size=(1,)).item() if w > target_size[1] else 0

    if expected_ndim == 3:
        cropped = [F.crop(tensor, top, left, target_size[0], target_size[1]) for tensor in tensors]
    else:
        if len(target_size) == 2:
            cropped = [tensor[:, top:top + target_size[0], left:left + target_size[1], :] for tensor in tensors]
        else:
            depth = torch.randint(0, d - target_size[2] + 1, size=(1,)).item() if d > target_size[2] else 0
            cropped = [tensor[:, top:top + target_size[0], left:left + target_size[1], depth:depth + target_size[2]] for tensor in tensors]

    return tuple(cropped)


def even_crop_height_width(
    tensorA,
    tensorB,
    tensorC=None,
    tensorD=None,
    tensorE=None,
    tensorF=None,
    tensorG=None,
    multiple=(16, 16),
):
    tensors = [tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG]
    tensors = [tensor for tensor in tensors if tensor is not None]

    if len(tensorA.shape) == 3:
        _, h, w = tensorA.shape
        expected_ndim = 3
    elif len(tensorA.shape) == 4:
        _, h, w, _ = tensorA.shape
        expected_ndim = 4
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    new_h = (h // multiple[0]) * multiple[0]
    new_w = (w // multiple[1]) * multiple[1]
    top = (h - new_h) // 2
    left = (w - new_w) // 2

    if expected_ndim == 3:
        cropped = [F.crop(tensor, top, left, new_h, new_w) for tensor in tensors]
    else:
        cropped = [tensor[:, top:top + new_h, left:left + new_w, :] for tensor in tensors]

    return tuple(cropped)


log = utils.get_pylogger(__name__)

class dataset_SynthRAD(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_group_1: str,
        data_group_2: str,
        data_group_3: Optional[str] = None,
        data_group_4: Optional[str] = None,
        data_group_5: Optional[str] = None,
        data_group_6: Optional[str] = None,  # For triple outputs (MRA)
        data_group_7: Optional[str] = None,  # For triple outputs (MRA_moved)
        is_3d: bool = False,
        padding_size: Optional[Tuple[int, int]] = None,
        flip_prob: float = 0.0,
        rot_prob: float = 0.0,
        crop_size: Optional[Tuple[int, int]] = None,
        reverse: bool = False,
        norm_ZeroToOne: bool = False,
        use_25d_style: bool = False,
        ref_stack_size: int = 3,
        apply_rigid_registration: bool = False,
        registration_targets: list = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_group_1 = data_group_1
        self.data_group_2 = data_group_2
        self.data_group_3 = data_group_3
        self.data_group_4 = data_group_4
        self.data_group_5 = data_group_5
        self.data_group_6 = data_group_6
        self.data_group_7 = data_group_7
        self.is_3d = is_3d
        self.padding_size = padding_size
        self.crop_size = crop_size
        self.reverse = reverse
        self.norm_ZeroToOne = norm_ZeroToOne
        self.use_25d_style = use_25d_style
        self.ref_stack_size = ref_stack_size
        self.apply_rigid_registration = apply_rigid_registration
        self.registration_targets = registration_targets or []

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"

        self.patient_keys = []
        self.aug_keys = ["A", "B"]
        if self.data_group_3:
            self.aug_keys.append("C")
        if self.data_group_4:
            self.aug_keys.append("D")
        if self.data_group_5:
            self.aug_keys.append("E")
        if self.data_group_6:
            self.aug_keys.append("F")
        if self.data_group_7:
            self.aug_keys.append("G")

        if self.is_3d:
            with h5py.File(self.data_dir, "r") as file:
                self.patient_keys = list(file[self.data_group_1].keys())

            self.aug_func = Compose(
                [
                    RandFlipd(keys=self.aug_keys, prob=flip_prob, spatial_axis=[0, 1]),
                    RandRotate90d(keys=self.aug_keys, prob=rot_prob, spatial_axes=[0, 1]),
                ]
            )
        else:
            with h5py.File(self.data_dir, "r") as file:
                self.patient_keys = list(file[self.data_group_1].keys())
                self.slice_counts = [file[self.data_group_1][key].shape[-1] for key in self.patient_keys]
                self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)

            self.aug_func = Compose(
                [
                    RandFlipd(keys=self.aug_keys, prob=flip_prob, spatial_axis=[0, 1]),
                    RandRotate90d(keys=self.aug_keys, prob=rot_prob, spatial_axes=[0, 1]),
                ]
            )

        # 3D rigid registration cache: registration_targets에 명시된 group → group_1 기준으로 정합
        _group_map = {
            2: ('B', self.data_group_2), 3: ('C', self.data_group_3),
            4: ('D', self.data_group_4), 5: ('E', self.data_group_5),
            6: ('F', self.data_group_6), 7: ('G', self.data_group_7),
        }
        self.reg_cache = {}
        if self.apply_rigid_registration and self.registration_targets:
            # target 번호 중 해당 group이 실제로 정의된 것만 사용
            moving_group_map = {
                cache_key: group
                for t in self.registration_targets
                if t in _group_map
                for cache_key, group in [_group_map[t]]
                if group is not None
            }
            if moving_group_map:
                log.info(f"[RigidReg] Running 3D rigid registration for {len(self.patient_keys)} patients "
                         f"(targets: {self.registration_targets}, active: {list(moving_group_map.values())}) ...")
                with h5py.File(self.data_dir, 'r') as file:
                    for patient_key in tqdm(self.patient_keys, desc="[RigidReg]"):
                        fixed_vol = file[self.data_group_1][patient_key][...]
                        self.reg_cache[patient_key] = {}
                        for cache_key, group in moving_group_map.items():
                            moving_vol = file[group][patient_key][...]
                            self.reg_cache[patient_key][cache_key] = _register_3d_rigid(fixed_vol, moving_vol)
                log.info("[RigidReg] Done.")

    def _load_slice_stack(self, file, group, patient_key, slice_idx):
        """Load K neighboring slices centered at slice_idx; repeat at volume boundaries."""
        total = file[group][patient_key].shape[-1]
        N = self.ref_stack_size // 2
        slices = []
        for k in range(-N, N + 1):
            s = min(max(slice_idx + k, 0), total - 1)
            slices.append(file[group][patient_key][..., s])
        return np.stack(slices, axis=0)  # [K, H, W]

    def _load_slice_stack_from_vol(self, vol, slice_idx):
        """Load K neighboring slices from a cached (H, W, D) volume."""
        total = vol.shape[-1]
        N = self.ref_stack_size // 2
        slices = []
        for k in range(-N, N + 1):
            s = min(max(slice_idx + k, 0), total - 1)
            slices.append(vol[..., s])
        return np.stack(slices, axis=0)  # [K, H, W]

    def __len__(self):
        if self.is_3d:
            return len(self.patient_keys)
        else:
            return self.cumulative_slice_counts[-1]

    def __getitem__(self, idx):
        if self.is_3d:
            patient_key = self.patient_keys[idx]
            with h5py.File(self.data_dir, "r") as file:
                A = file[self.data_group_1][patient_key][...]
                B = file[self.data_group_2][patient_key][...]
                if self.data_group_3:
                    C = self.reg_cache[patient_key]['C'] if self.apply_rigid_registration else file[self.data_group_3][patient_key][...]
                if self.data_group_4:
                    D = file[self.data_group_4][patient_key][...]
                if self.data_group_5:
                    E = self.reg_cache[patient_key]['E'] if self.apply_rigid_registration else file[self.data_group_5][patient_key][...]
                if self.data_group_6:
                    F = file[self.data_group_6][patient_key][...]
                if self.data_group_7:
                    G = self.reg_cache[patient_key]['G'] if self.apply_rigid_registration else file[self.data_group_7][patient_key][...]

        else:
            patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
            slice_idx = idx - self.cumulative_slice_counts[patient_idx]
            patient_key = self.patient_keys[patient_idx]
            with h5py.File(self.data_dir, "r") as file:
                A = file[self.data_group_1][patient_key][..., slice_idx]
                B = file[self.data_group_2][patient_key][..., slice_idx]
                if self.data_group_3:
                    if self.apply_rigid_registration:
                        vol_C = self.reg_cache[patient_key]['C']
                        C = self._load_slice_stack_from_vol(vol_C, slice_idx) if self.use_25d_style else vol_C[..., slice_idx]
                    elif self.use_25d_style:
                        C = self._load_slice_stack(file, self.data_group_3, patient_key, slice_idx)
                    else:
                        C = file[self.data_group_3][patient_key][..., slice_idx]
                if self.data_group_4:
                    # group_4 is GT (even) → always single slice
                    D = file[self.data_group_4][patient_key][..., slice_idx]
                if self.data_group_5:
                    # group_5 is moved ref (odd) → K-stack when use_25d_style
                    if self.apply_rigid_registration:
                        vol_E = self.reg_cache[patient_key]['E']
                        E = self._load_slice_stack_from_vol(vol_E, slice_idx) if self.use_25d_style else vol_E[..., slice_idx]
                    elif self.use_25d_style:
                        E = self._load_slice_stack(file, self.data_group_5, patient_key, slice_idx)
                    else:
                        E = file[self.data_group_5][patient_key][..., slice_idx]
                if self.data_group_6:
                    # group_6 is GT (even) → always single slice
                    F = file[self.data_group_6][patient_key][..., slice_idx]
                if self.data_group_7:
                    # group_7 is moved ref (odd) → K-stack when use_25d_style
                    if self.apply_rigid_registration:
                        vol_G = self.reg_cache[patient_key]['G']
                        G = self._load_slice_stack_from_vol(vol_G, slice_idx) if self.use_25d_style else vol_G[..., slice_idx]
                    elif self.use_25d_style:
                        G = self._load_slice_stack(file, self.data_group_7, patient_key, slice_idx)
                    else:
                        G = file[self.data_group_7][patient_key][..., slice_idx]

        A = torch.from_numpy(A).unsqueeze(0).float()
        B = torch.from_numpy(B).unsqueeze(0).float()
        if self.data_group_3:
            C = torch.from_numpy(C).float() if self.use_25d_style else torch.from_numpy(C).unsqueeze(0).float()
        if self.data_group_4:
            D = torch.from_numpy(D).unsqueeze(0).float()  # GT, always single
        if self.data_group_5:
            E = torch.from_numpy(E).float() if self.use_25d_style else torch.from_numpy(E).unsqueeze(0).float()
        if self.data_group_6:
            F = torch.from_numpy(F).unsqueeze(0).float()  # GT, always single
        if self.data_group_7:
            G = torch.from_numpy(G).float() if self.use_25d_style else torch.from_numpy(G).unsqueeze(0).float()

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}
        if self.data_group_3:
            data_dict["C"] = C
        if self.data_group_4:
            data_dict["D"] = D
        if self.data_group_5:
            data_dict["E"] = E
        if self.data_group_6:
            data_dict["F"] = F
        if self.data_group_7:
            data_dict["G"] = G

        A = data_dict["A"]
        B = data_dict["B"]
        if self.data_group_3:
            C = data_dict["C"]
        if self.data_group_4:
            D = data_dict["D"]
        if self.data_group_5:
            E = data_dict["E"]
        if self.data_group_6:
            F = data_dict["F"]
        if self.data_group_7:
            G = data_dict["G"]

        if self.padding_size:
            if self.data_group_7:
                A, B, C, D, E, F, G = padding_height_width_depth(A, B, C, D, E, F, G, target_size=self.padding_size)
            elif self.data_group_6:
                A, B, C, D, E, F = padding_height_width_depth(A, B, C, D, E, F, target_size=self.padding_size)
            elif self.data_group_5:
                A, B, C, D, E = padding_height_width_depth(A, B, C, D, E, target_size=self.padding_size)
            elif self.data_group_4:
                A, B, C, D = padding_height_width_depth(A, B, C, D, target_size=self.padding_size)
            elif self.data_group_3:
                A, B, C = padding_height_width_depth(A, B, C, target_size=self.padding_size)
            else:
                A, B = padding_height_width_depth(A, B, target_size=self.padding_size)

        data_dict["A"] = A
        data_dict["B"] = B
        if self.data_group_3:
            data_dict["C"] = C
        if self.data_group_4:
            data_dict["D"] = D
        if self.data_group_5:
            data_dict["E"] = E
        if self.data_group_6:
            data_dict["F"] = F
        if self.data_group_7:
            data_dict["G"] = G

        data_dict = self.aug_func(data_dict)
        A = data_dict["A"]
        B = data_dict["B"]
        if self.data_group_3:
            C = data_dict["C"]
        if self.data_group_4:
            D = data_dict["D"]
        if self.data_group_5:
            E = data_dict["E"]
        if self.data_group_6:
            F = data_dict["F"]
        if self.data_group_7:
            G = data_dict["G"]

        if self.crop_size:
            if self.data_group_7:
                A, B, C, D, E, F, G = random_crop(A, B, C, D, E, F, G, target_size=self.crop_size)
            elif self.data_group_6:
                A, B, C, D, E, F = random_crop(A, B, C, D, E, F, target_size=self.crop_size)
            elif self.data_group_5:
                A, B, C, D, E = random_crop(A, B, C, D, E, target_size=self.crop_size)
            elif self.data_group_4:
                A, B, C, D = random_crop(A, B, C, D, target_size=self.crop_size)
            elif self.data_group_3:
                A, B, C = random_crop(A, B, C, target_size=self.crop_size)
            else:
                A, B = random_crop(A, B, target_size=self.crop_size)
        else:
            if self.data_group_7:
                A, B, C, D, E, F, G = even_crop_height_width(A, B, C, D, E, F, G, multiple=(16, 16))
            elif self.data_group_6:
                A, B, C, D, E, F = even_crop_height_width(A, B, C, D, E, F, multiple=(16, 16))
            elif self.data_group_5:
                A, B, C, D, E = even_crop_height_width(A, B, C, D, E, multiple=(16, 16))
            elif self.data_group_4:
                A, B, C, D = even_crop_height_width(A, B, C, D, multiple=(16, 16))
            elif self.data_group_3:
                A, B, C = even_crop_height_width(A, B, C, multiple=(16, 16))
            else:
                A, B = even_crop_height_width(A, B, multiple=(16, 16))

        # Normalize A, B, C, D, E, F, G to range [0,1]
        if self.norm_ZeroToOne:
            A = (A - A.min()) / (A.max() - A.min() + 1e-8)
            B = (B - B.min()) / (B.max() - B.min() + 1e-8)

            if self.data_group_3:
                C = (C - C.min()) / (C.max() - C.min() + 1e-8)
            if self.data_group_4:
                D = (D - D.min()) / (D.max() - D.min() + 1e-8)
            if self.data_group_5:
                E = (E - E.min()) / (E.max() - E.min() + 1e-8)
            if self.data_group_6:
                F = (F - F.min()) / (F.max() - F.min() + 1e-8)
            if self.data_group_7:
                G = (G - G.min()) / (G.max() - G.min() + 1e-8)

        # Return based on which data groups are available
        # For triple outputs without misalign_simul: A, B, D, F (T2, T1, PD, MRA)
        # For triple outputs with misalign_simul: A, B, C, D, E, F, G
        if self.reverse:
            if self.data_group_7:
                return G, F, E, D, C, B, A
            elif self.data_group_6:
                return F, E, D, C, B, A
            elif self.data_group_5:
                return E, D, C, B, A
            elif self.data_group_4:
                return D, C, B, A
            elif self.data_group_3:
                return C, B, A
            else:
                return B, A
        else:
            if self.data_group_7:
                return A, B, C, D, E, F, G
            elif self.data_group_6:
                return A, B, C, D, E, F
            elif self.data_group_5:
                return A, B, C, D, E
            elif self.data_group_4:
                return A, B, C, D
            elif self.data_group_3:
                return A, B, C
            else:
                return A, B

    def get_patient_slice_idx(self, idx):
        if self.is_3d:
            patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
            slice_idx = idx - self.cumulative_slice_counts[patient_idx]
            return patient_idx, slice_idx
        else:
            return idx, None

class dataset_Histological(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_group_1: str,
        data_group_2: str,
        data_group_3: str,
        data_group_4: str,
        padding_size: Optional[Tuple[int, int]] = None,
        flip_prob: float = 0.0,
        rot_prob: float = 0.0,
        crop_size: Optional[Tuple[int, int]] = None,
        reverse: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_group_1 = data_group_1
        self.data_group_2 = data_group_2
        self.data_group_3 = data_group_3
        self.data_group_4 = data_group_4
        self.padding_size = padding_size
        self.crop_size = crop_size
        self.reverse = reverse

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"

        with h5py.File(self.data_dir, "r") as file:
            self.patient_keys = list(file[self.data_group_1].keys())

        self.aug_func = Compose(
            [
                RandFlipd(keys=["A", "B", "C" ,"D"], prob=flip_prob, spatial_axis=[0, 1]),
                RandRotate90d(keys=["A", "B", "C", "D"], prob=rot_prob, spatial_axes=[0, 1]),
            ]
        )

    def __len__(self):
        return len(self.patient_keys)

    def __getitem__(self, idx):
        patient_key = self.patient_keys[idx]
        with h5py.File(self.data_dir, "r") as file:
            A = file[self.data_group_1][patient_key][...]
            B = file[self.data_group_2][patient_key][...]
            C = file[self.data_group_3][patient_key][...]
            D = file[self.data_group_4][patient_key][...]

        # 2D 이미지는 unsqueeze로 채널 추가
        A = torch.from_numpy(A).unsqueeze(0).float()  # (H, W) -> (1, H, W)
        B = torch.from_numpy(B).unsqueeze(0).float()  # (H, W) -> (1, H, W)
        C = torch.from_numpy(C).permute(2, 0, 1).float()  # RGB 이미지 채널을 맨 앞으로 이동 (H, W, 3) -> (3, H, W)
        D = torch.from_numpy(D).permute(2, 0, 1).float()  # RGB 이미지 채널을 맨 앞으로 이동 (H, W, 3) -> (3, H, W)

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B, "C": C, "D": D}

        if self.padding_size:
            A, B, C, D = padding_height_width_depth(A, B, C, D, target_size=self.padding_size)

        data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        B = data_dict["B"]
        C = data_dict["C"]
        D = data_dict["D"]

        if self.crop_size:
            A, B, C, D = random_crop(A, B, C, D, target_size=self.crop_size) 
        else:
            A, B, C, D = even_crop_height_width(A, B, C, D) 

        if self.reverse:
            return C, D, A, B
        else:
            return A, B, C, D
        


def download_process_MR_3T_7T(data_dir: str,
                         write_dir: str,
                         misalign_x: float = 0.0,  # maximum misalignment in x direction (float)
                         misalign_y: float = 0.0,  # maximum misalignment in y direction (float)
                         degree: float = 0.0,      # maximum rotation in z direction (float)
                         motion_prob: float = 0.0, # the probability of occurrence of motion.
                         deform_prob: float = 0.0, # the probability of performing deformation.
                         ret: bool = False
                        ):
    """Downloads, preprocesses and saves the IXI dataset. The images are randomly translated.

    Args:
        data_dir (str): The directory where the input dataset is located.
        write_dir (str): The directory where the processed dataset will be saved.
        misalign_x (float): Maximum allowable misalignment along the x dimension. Defaults to 0.0.
        misalign_y (float): Maximum allowable misalignment along the y dimension. Defaults to 0.0.
        degree (float): Maximum rotation in z direction. Defaults to 0.0.
        motion_prob (float): The probability of occurrence of motion. Defaults to 0.0.
        deform_prob (float): The probability of performing deformation. Defaults to 0.0.
        ret (bool): If True, the function also returns the processed dataset. Defaults to False.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor]]: A pair of tensors representing the processed datasets
        for A and B. This is returned only when 'ret' is set to True.
    """

    with h5py.File(data_dir, "r") as f:
        data_A = np.array(f["data_x"]) # H, W, S
        data_B = np.array(f["data_y"])

    # create torch tensors
    data_A = np.transpose(data_A, (2, 0, 1)) # H, W, Slice -> # Slice, H, W
    # data_A = np.flipud(data_A)
    data_A = np.expand_dims(data_A, axis=0) # 1, Slice, H, W
    print('data_A shape: ', data_A.shape)

    data_B = np.transpose(data_B, (2, 0, 1))
    # data_B = np.flipud(data_B)
    data_B = np.expand_dims(data_B, axis=0)
    print('data_B shape: ', data_B.shape)

    # ensure that data is in range [-1,1]
    data_A[data_A < 0] = 0
    data_B[data_B < 0] = 0

    data_A, data_B = Deformation(data_A.copy(), data_B.copy(), deform_prob)
    data_A, data_B = Motion_artifacts(data_A, data_B, motion_prob)

    data_A = (data_A - 0.5) / 0.5
    data_B = (data_B - 0.5) / 0.5

    log.info(f"Preparing the misalignment from <{data_dir}>")

    data_A, data_B = Rotate_images(data_A, data_B, degree)

    num_slices_per_patient = 200
    num_patients = data_A.shape[1] // num_slices_per_patient

    log.info(f"Saving the prepared dataset to <{write_dir}>")

    with h5py.File(write_dir, "w") as hw:
        data_A_group = hw.create_group("3T")
        data_B_group = hw.create_group("7T")

        for patient_idx in range(num_patients):
            start_idx = patient_idx * num_slices_per_patient
            end_idx = start_idx + num_slices_per_patient

            patient_data_A_list = []
            patient_data_B_list = []

            for sl in range(start_idx, end_idx): # Translation each slice
                A = torch.from_numpy(data_A[:, sl, :, :])
                B = torch.from_numpy(data_B[:, sl, :, :])

                A, B = Translate_images(A, B, misalign_x, misalign_y)

                patient_data_A_list.append(A)
                patient_data_B_list.append(B)

            patient_data_A = torch.stack(patient_data_A_list) # num_slices, 1, H, W
            patient_data_B = torch.stack(patient_data_B_list)

            patient_data_A = patient_data_A.squeeze(1) # num_slices, H, W
            patient_data_B = patient_data_B.squeeze(1)

            patient_data_A = patient_data_A.permute(1, 2, 0) # H, W, num_slices
            patient_data_B = patient_data_B.permute(1, 2, 0)

            data_A_group.create_dataset(str(patient_idx + 1), data=patient_data_A.numpy())
            data_B_group.create_dataset(str(patient_idx + 1), data=patient_data_B.numpy())

    if ret:
        return patient_data_A, patient_data_B
    else:
        return

################################################################################################################################
################################################################################################################################

def Deformation(A:np.ndarray,
                    B:np.ndarray,
                    deform_prob:float):
    """Apply dense random elastic deformation.

    Args:
        A (np.ndarray: An input image ndarray.
        deform_prob (float): The probability of performing deformation.

    Returns:
        Tuple[np.ndarray]: A deformed image.
    """

    elastic_transform = tio.transforms.RandomElasticDeformation( ## Tuning parameters while observing the results
        num_control_points=6,  # Number of control points along each dimension.
        max_displacement=(30,30,0),    # Maximum displacement along each dimension at each control point.
    )

    elastic_A = elastic_transform(A)
    elastic_B = elastic_transform(B)
    num_samples = A.shape[1]
    num_transform_samples = int(num_samples * deform_prob)
    indices_A = random.sample(range(num_samples), num_transform_samples)
    indices_B = random.sample(range(num_samples), num_transform_samples)

    for index in indices_A:
        A[:,index,:,:] = elastic_A[:,index,:,:]

    for index in indices_B:
        B[:,index,:,:] = elastic_B[:,index,:,:]

    return A, B

def Motion_region(raw_img, motion_img, prob):
    raw_k = fftshift(fftn(raw_img))
    motion_k = fftshift(fftn(motion_img))
    diff = math.ceil(2.56*prob)

    raw_k[:,128-diff : 128,:] = motion_k[:,128-diff : 128,:]

    res = abs(ifftn(ifftshift(raw_k)))
    return res

def Motion_artifacts(A:np.ndarray,
                         B:np.ndarray,
                         motion_prob:float):
    """Simulates motion artifacts.

    Args:
        A (np.ndarray): An input image ndarray.
        motion_prob (float): The probability of occurrence of motion.

    Returns:
        Tuple[np.ndarray]: A motion artifacts-injected tensor image.
    """

    # Define the 3D-RandomMotion transform
    random_motion = tio.RandomMotion(
        degrees=(5,5),              # Maximum rotation angle in degrees
        translation=(10,10),         # Maximum translation in mm
        num_transforms=10         # Number of motion transformations to apply
    )

    A = np.transpose(A, [0,2,3,1])
    B = np.transpose(B, [0,2,3,1])
    motion_A = random_motion(A)
    motion_A_prob = Motion_region(A, motion_A, prob=6)
    motion_B = random_motion(B)
    motion_B_prob = Motion_region(B, motion_B, prob=6)

    num_samples = A.shape[3]
    num_transform_samples = int(num_samples * motion_prob)
    indices_A = random.sample(range(num_samples), num_transform_samples)
    indices_B = random.sample(range(num_samples), num_transform_samples)

    for index in indices_A:
        A[:,:,:,index] = motion_A_prob[:,:,:,index]

    for index in indices_B:
        B[:,:,:,index] = motion_B_prob[:,:,:,index]

    A = np.transpose(A, [0,3,1,2])
    B = np.transpose(B, [0,3,1,2])

    return A, B

def Rotate_images(A:np.ndarray,
                      B:np.ndarray,
                      degree:float):
    """Rotates a given image (A) by random degree along z dimensions.

    Args:
        A (np.ndarray): An input image ndarray.
        degree (float): Maximum allowable degree along the z dimension.

    Returns:
        Tuple[np.ndarray]: A rotated tensor image.
    """

    # rotation
    transform = tio.RandomAffine(
        scales=0,
        degrees=(degree,0,0),        # z-axis 3D-Rotation range in degrees.
        translation=(0,0,0),
        default_pad_value='otsu',  # edge control, fill value is the mean of the values at the border that lie under an Otsu threshold.
    )

    rotated_A = transform(A)
    rotated_B = transform(B)
    
    return rotated_A, rotated_B

def Translate_images(A: torch.Tensor,
              B: torch.Tensor,
              misalign_x: float,
              misalign_y: float):
    """Translates two given images (A and B) by random misalignments along x and y dimensions.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        misalign_x (float): Maximum allowable misalignment along the x dimension.
        misalign_y (float): Maximum allowable misalignment along the y dimension.

    Returns:
        Tuple[torch.Tensor]: A pair of tensors representing the translated images.
    """

    # translation (changed : misalign is the smae but magnitude is different)
    _misalign_x = np.random.uniform(-1, 1, size=2)
    _misalign_y = np.random.uniform(-1, 1, size=2)

    misalign_x = misalign_x * _misalign_x
    misalign_y = misalign_y * _misalign_y

    # misalign_x = np.random.uniform(-misalign_x, misalign_x, size=2) # This is the previous version
    # misalign_y = np.random.uniform(-misalign_y, misalign_y, size=2)

    translate_params_A = (misalign_y[0], misalign_x[0])  # (y, x) for image A
    translate_params_B = (misalign_y[1], misalign_x[1])  # (y, x) for image B

    # create affine transform
    affine_A = Affine(translate_params=translate_params_A)
    affine_B = Affine(translate_params=translate_params_B)

    moved_A = affine_A(A)
    moved_B = affine_B(B)

    return moved_A[0], moved_B[0]
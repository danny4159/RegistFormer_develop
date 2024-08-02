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

def padding_height_width(tensorA, tensorB, tensorC=None, tensorD=None, target_size=(256, 256), pad_value=-1):
    # 기존 코드와 동일
    # Determine if the input is 2D or 3D based on the number of dimensions
    if len(tensorA.shape) == 3:
        # 2D case
        _, h, w = tensorA.shape
        assert h >= target_size[0] and w >= target_size[1], "Input tensor size must be larger than min_size"
        assert len(tensorB.shape) == 3, "Input tensor B must have 3 dimensions (C, H, W)"
        if tensorC is not None:
            assert len(tensorC.shape) == 3, "Input tensor C must have 3 dimensions (C, H, W)"
        if tensorD is not None:
            assert len(tensorD.shape) == 3, "Input tensor D must have 3 dimensions (C, H, W)"
    elif len(tensorA.shape) == 4:
        # 3D case
        _, h, w, d = tensorA.shape
        assert h >= target_size[0] and w >= target_size[1], "Input tensor size must be larger than min_size"
        assert len(tensorB.shape) == 4, "Input tensor B must have 4 dimensions (C, H, W, D)"
        if tensorC is not None:
            assert len(tensorC.shape) == 4, "Input tensor C must have 4 dimensions (C, H, W, D)"
        if tensorD is not None:
            assert len(tensorD.shape) == 4, "Input tensor D must have 4 dimensions (C, H, W, D)"
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    # 기존 코드와 동일
    # Calculate padding
    pad_top = (target_size[0] - h + 1) // 2 if h < target_size[0] else 0
    pad_bottom = (target_size[0] - h) // 2 if h < target_size[0] else 0
    pad_left = (target_size[1] - w + 1) // 2 if w < target_size[1] else 0
    pad_right = (target_size[1] - w) // 2 if w < target_size[1] else 0

    # Apply padding
    if pad_top != 0 or pad_left != 0:
        tensorA = nnF.pad(tensorA, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        tensorB = nnF.pad(tensorB, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        if tensorC is not None:
            tensorC = nnF.pad(tensorC, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        if tensorD is not None:
            tensorD = nnF.pad(tensorD, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    if tensorC is not None and tensorD is not None:
        return tensorA, tensorB, tensorC, tensorD
    elif tensorC is not None:
        return tensorA, tensorB, tensorC
    else:
        return tensorA, tensorB

def random_crop_height_width(tensorA, tensorB, tensorC=None, tensorD=None, target_size=(128, 128)):
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # Determine if the input is 2D or 3D based on the number of dimensions
    if len(tensorA.shape) == 3:
        # 2D case
        _, h, w = tensorA.shape
        assert len(tensorB.shape) == 3, "Input tensor B must have 3 dimensions (C, H, W)"
        if tensorC is not None:
            assert len(tensorC.shape) == 3, "Input tensor C must have 3 dimensions (C, H, W)"
        if tensorD is not None:
            assert len(tensorD.shape) == 3, "Input tensor D must have 3 dimensions (C, H, W)"
    elif len(tensorA.shape) == 4:
        # 3D case
        _, h, w, d = tensorA.shape
        assert len(tensorB.shape) == 4, "Input tensor B must have 4 dimensions (C, H, W, D)"
        if tensorC is not None:
            assert len(tensorC.shape) == 4, "Input tensor C must have 4 dimensions (C, H, W, D)"
        if tensorD is not None:
            assert len(tensorD.shape) == 4, "Input tensor D must have 4 dimensions (C, H, W, D)"
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    # Calculate the top left corner of the random crop
    top = (
        torch.randint(0, h - target_size[0] + 1, size=(1,)).item()
        if h > target_size[0]
        else 0
    )
    left = (
        torch.randint(0, w - target_size[1] + 1, size=(1,)).item()
        if w > target_size[1]
        else 0
    )

    if len(tensorA.shape) == 3:
        # Perform the crop for 2D tensors
        tensorA = F.crop(tensorA, top, left, target_size[0], target_size[1])
        tensorB = F.crop(tensorB, top, left, target_size[0], target_size[1])
        if tensorC is not None:
            tensorC = F.crop(tensorC, top, left, target_size[0], target_size[1])
        if tensorD is not None:
            tensorD = F.crop(tensorD, top, left, target_size[0], target_size[1])
    elif len(tensorA.shape) == 4:
        # Perform the crop for 3D tensors
        tensorA = tensorA[:, top:top + target_size[0], left:left + target_size[1], :]
        tensorB = tensorB[:, top:top + target_size[0], left:left + target_size[1], :]
        if tensorC is not None:
            tensorC = tensorC[:, top:top + target_size[0], left:left + target_size[1], :]
        if tensorD is not None:
            tensorD = tensorD[:, top:top + target_size[0], left:left + target_size[1], :]

    if tensorC is not None and tensorD is not None:
        return tensorA, tensorB, tensorC, tensorD
    elif tensorC is not None:
        return tensorA, tensorB, tensorC
    else:
        return tensorA, tensorB
    
def even_crop_height_width(tensorA, tensorB, tensorC=None, tensorD=None, multiple=(16, 16)):
    """
    Crop the image to the target size evenly from all sides.

    Args:
        tensorA (Tensor): Image to be cropped.
        tensorB (Tensor): Second Image to be cropped.
        target_size (tuple): Desired output size (height, width).
        tensorC (Tensor, optional): Third image to be cropped.
        tensorD (Tensor, optional): Fourth image to be cropped.

    Returns:
        Tensor: Cropped images.
    """
    if len(tensorA.shape) == 3:
        # 2D case
        _, h, w = tensorA.shape
    elif len(tensorA.shape) == 4:
        # 3D case
        _, h, w, _ = tensorA.shape
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    new_h = (h // multiple[0]) * multiple[0]
    new_w = (w // multiple[1]) * multiple[1]

    # Calculate cropping dimensions
    top = (h - new_h) // 2
    bottom = h - new_h - top
    left = (w - new_w) // 2
    right = w - new_w - left

    if len(tensorA.shape) == 3:
        # Crop images
        tensorA = F.crop(tensorA, top, left, new_h, new_w)
        tensorB = F.crop(tensorB, top, left, new_h, new_w)
        if tensorC is not None:
            tensorC = F.crop(tensorC, top, left, new_h, new_w)
        if tensorD is not None:
            tensorD = F.crop(tensorD, top, left, new_h, new_w)
    elif len(tensorA.shape) == 4:
        # Crop images
        tensorA = tensorA[:, top:top + new_h, left:left + new_w, :]
        tensorB = tensorB[:, top:top + new_h, left:left + new_w, :]
        if tensorC is not None:
            tensorC = tensorC[:, top:top + new_h, left:left + new_w, :]
        if tensorD is not None:
            tensorD = tensorD[:, top:top + new_h, left:left + new_w, :]

    if tensorC is not None and tensorD is not None:
        return tensorA, tensorB, tensorC, tensorD
    elif tensorC is not None:
        return tensorA, tensorB, tensorC
    else:
        return tensorA, tensorB


log = utils.get_pylogger(__name__)

class dataset_SynthRAD(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_group_1: str,
        data_group_2: str,
        data_group_3: Optional[str] = None,
        is_3d: bool = False,
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
        self.is_3d = is_3d
        self.padding_size = padding_size
        self.crop_size = crop_size
        self.reverse = reverse

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"

        self.patient_keys = []

        if self.is_3d:
            with h5py.File(self.data_dir, "r") as file:
                self.patient_keys = list(file[self.data_group_1].keys())

            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B"] if not self.data_group_3 else ["A", "B", "C"], prob=flip_prob, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B"] if not self.data_group_3 else ["A", "B", "C"], prob=rot_prob, spatial_axes=[0, 1]),
                ]
            )
        else:
            with h5py.File(self.data_dir, "r") as file:
                self.patient_keys = list(file[self.data_group_1].keys())
                self.slice_counts = [file[self.data_group_1][key].shape[-1] for key in self.patient_keys]
                self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)

            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B"] if not self.data_group_3 else ["A", "B", "C"], prob=flip_prob, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B"] if not self.data_group_3 else ["A", "B", "C"], prob=rot_prob, spatial_axes=[0, 1]),
                ]
            )

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
                    C = file[self.data_group_3][patient_key][...]
        else:
            patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
            slice_idx = idx - self.cumulative_slice_counts[patient_idx]
            patient_key = self.patient_keys[patient_idx]
            with h5py.File(self.data_dir, "r") as file:
                A = file[self.data_group_1][patient_key][..., slice_idx]
                B = file[self.data_group_2][patient_key][..., slice_idx]
                if self.data_group_3:
                    C = file[self.data_group_3][patient_key][..., slice_idx]

        A = torch.from_numpy(A).unsqueeze(0).float()
        B = torch.from_numpy(B).unsqueeze(0).float()
        if self.data_group_3:
            C = torch.from_numpy(C).unsqueeze(0).float()

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}
        if self.data_group_3:
            data_dict["C"] = C

        A = data_dict["A"]
        B = data_dict["B"]
        if self.data_group_3:
            C = data_dict["C"]

        if self.padding_size:
            if self.data_group_3:
                A, B, C = padding_height_width(A, B, C, target_size=self.padding_size)
            else:
                A, B = padding_height_width(A, B, target_size=self.padding_size)

        data_dict = self.aug_func(data_dict)

        if self.crop_size:
            if self.data_group_3:
                A, B, C = random_crop_height_width(A, B, C, target_size=self.crop_size) 
            else:
                A, B = random_crop_height_width(A, B, target_size=self.crop_size) 
        else:
            if self.data_group_3:
                A, B, C = even_crop_height_width(A, B, C, multiple=(16, 16)) # 16의 배수로 Crop
            else:
                A, B = even_crop_height_width(A, B, multiple=(16, 16)) # 16의 배수로 Crop
                
        if self.reverse:
            if self.data_group_3:
                return C, B, A
            else:
                return B, A
        else:
            if self.data_group_3:
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
            A, B, C, D = padding_height_width(A, B, C, D, target_size=self.padding_size)

        data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        B = data_dict["B"]
        C = data_dict["C"]
        D = data_dict["D"]

        if self.crop_size:
            A, B, C, D = random_crop_height_width(A, B, C, D, target_size=self.crop_size) 
        else:
            A, B, C, D = even_crop_height_width(A, B, C, D) 

        if self.reverse:
            return C, D, A, B
        else:
            return A, B, C, D
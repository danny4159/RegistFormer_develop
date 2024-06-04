from typing import Any, Dict, Optional
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



def padding_target_size(
    tensorA, tensorB, min_size=(256, 256), pad_value=-1
):  
    """
    Pad and crop the image tensors to the minimum size with padding value.
    If the image size is less than min_size, pad it to min_size.
    If padding needs to be odd, add the extra padding to the top and left.

    Args:
        tensorA (Tensor): Image to be processed.
        tensorB (Tensor): Second Image to be processed.
        min_size (int): Minimum size to pad and crop the image.
        pad_value (float): Value to use for padding.

    Returns:
        Tensor: Processed images.
    """
    if isinstance(min_size, int):
        min_size = (min_size, min_size)

    _, h, w = tensorA.shape

    # Calculate padding
    pad_top = (min_size[0] - h + 1) // 2 if h < min_size[0] else 0
    pad_bottom = (min_size[0] - h) // 2 if h < min_size[0] else 0
    pad_left = (min_size[1] - w + 1) // 2 if w < min_size[1] else 0
    pad_right = (min_size[1] - w) // 2 if w < min_size[1] else 0

    # Apply padding
    if pad_top != 0 or pad_left != 0:
        tensorA = nnF.pad(
            tensorA, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value
        )
        tensorB = nnF.pad(
            tensorB, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value
        )

    return tensorA, tensorB


def random_crop(tensorA, tensorB, output_size=(128, 128)):
    """
    Crop randomly the image in a sample.

    Args:
        tensor (Tensor): Image to be cropped.
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    Returns:
        Tensor: Cropped image.
    """
    # Handle the case where the output size is an integer
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Ensure the tensor has the correct dimensions
    assert len(tensorA.shape) == 3, "Input tensor A must have 3 dimensions (C, H, W)"
    assert len(tensorB.shape) == 3, "Input tensor B must have 3 dimensions (C, H, W)"

    _, h, w = tensorA.shape

    # Calculate the top left corner of the random crop
    top = (
        torch.randint(0, h - output_size[0] + 1, size=(1,)).item()
        if h > output_size[0]
        else 0
    )
    left = (
        torch.randint(0, w - output_size[1] + 1, size=(1,)).item()
        if w > output_size[1]
        else 0
    )
    # top = torch.randint(0, h - output_size[0] + 1, size=(1,)).item()
    # left = torch.randint(0, w - output_size[1] + 1, size=(1,)).item()

    # Perform the crop
    tensorA = F.crop(tensorA, top, left, output_size[0], output_size[1])
    tensorB = F.crop(tensorB, top, left, output_size[0], output_size[1])

    return tensorA, tensorB


def even_crop(tensorA, tensorB, target_size):
    """
    Crop the image to the target size evenly from all sides.

    Args:
        tensorA (Tensor): Image to be cropped.
        tensorB (Tensor): Second Image to be cropped.
        target_size (tuple): Desired output size (height, width).

    Returns:
        Tensor: Cropped images.
    """
    _, h, w = tensorA.shape
    new_h, new_w = target_size

    # Calculate cropping dimensions
    top = (h - new_h) // 2
    bottom = h - new_h - top
    left = (w - new_w) // 2
    right = w - new_w - left

    # Crop images
    tensorA = F.crop(tensorA, top, left, new_h, new_w)
    tensorB = F.crop(tensorB, top, left, new_h, new_w)

    return tensorA, tensorB


log = utils.get_pylogger(__name__)


class dataset_SynthRAD_MR_CT_Pelvis(Dataset):
    def __init__(
        self,
        data_dir: str,
        flip_prob: float = 0.5,
        rot_prob: float = 0.5,
        rand_crop: bool = False,
        reverse=False,
        padding: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        self.reverse = reverse
        self.padding = padding

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"

        # Each patient has a different number of slices
        self.patient_keys = []
        with h5py.File(self.data_dir, "r") as file:
            self.patient_keys = list(file["MR"].keys())
            self.slice_counts = [file["MR"][key].shape[-1] for key in self.patient_keys]
            self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)

        self.aug_func = Compose(
            [
                RandFlipd(keys=["A", "B"], prob=flip_prob, spatial_axis=[0, 1]),
                RandRotate90d(keys=["A", "B"], prob=rot_prob, spatial_axes=[0, 1]),
            ]
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.cumulative_slice_counts[-1]

    def __getitem__(self, idx):
        """Fetches a sample from the dataset given an index.

        Args:
            idx (int): The index for the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the samples for A and B.
        """
        patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
        slice_idx = idx - self.cumulative_slice_counts[patient_idx]
        patient_key = self.patient_keys[patient_idx]

        with h5py.File(self.data_dir, "r") as file:
            A = file["MR"][patient_key][..., slice_idx]
            B = file["CT"][patient_key][..., slice_idx]

        A = torch.from_numpy(A).unsqueeze(0).float()
        B = torch.from_numpy(B).unsqueeze(0).float()

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}

        # TODO: auf func 원래 위치

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.padding:
            A, B = padding_target_size(A, B)

        # Apply the random flipping
        data_dict = self.aug_func(data_dict)

        if self.rand_crop:
            # A, B = random_crop(A, B, (320,192)) # 이게 지금까지 계속 써왔던 것
            A, B = random_crop(A, B, (96, 96))  # resvit 용
            # A, B = random_crop(A, B, (416,256)) # 이건 weight 맞출때
        else:
            _, h, w = A.shape
            # A, B = even_crop(A, B, (256,256)) # under nearest multiple of 14 # TODO: resvit돌리기위한 코드
            A, B = even_crop(
                A, B, (h // 16 * 16, w // 16 * 16) # 16의 배수로
            )  # under nearest multiple of 16 # adaconv랑 다른것들도 다 이거
            # A, B = even_crop(A, B, (h//4*4,w//4*4)) # under nearest multiple of 16
            # A, B = random_crop(A, B, (h//4*4,w//4*4)) # under nearest multiple of four

        if self.reverse:
            return B, A
        else:
            return A, B

    def get_patient_slice_idx(self, idx):
        """주어진 샘플 인덱스에 대한 환자 인덱스와 슬라이스 인덱스를 반환합니다."""
        patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
        slice_idx = idx - self.cumulative_slice_counts[patient_idx]
        return patient_idx, slice_idx

from typing import Any, Dict, Optional, Tuple


import os
import torch
import h5py
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.data.components.transforms import (
    dataset_SynthRAD
)

class SynthRAD_MR_CT_Pelvis_DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        data_group_1: str,
        data_group_2: str,
        data_group_3: str,
        data_group_4: str,
        data_group_5: str,
        data_group_6: str = None,  # For triple outputs (MRA)
        data_group_7: str = None,  # For triple outputs (MRA_moved)
        is_3d: bool = False,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = False,
        use_split_inference: bool = False,
        train_file: str = "",
        val_file: str = "",
        test_file: str = "",
        reverse: bool = False,  # Reverse the order of the images (bool)
        flip_prob: float = 0.0,  # augmentation for training (flip)
        rot_prob: float = 0.0,  # augmentation for training (rot90)
        padding_size: Optional[Tuple[int, int]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        norm_ZeroToOne: bool = False,
        use_25d_style: bool = False,
        ref_stack_size: int = 3,
        slice_axis: int = 2,
        apply_rigid_registration: bool = False,
        registration_targets: list = None,
        **kwargs: Any
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.data_group_1 = data_group_1
        self.data_group_2 = data_group_2
        self.data_group_3 = data_group_3
        self.data_group_4 = data_group_4
        self.data_group_5 = data_group_5
        self.data_group_6 = data_group_6
        self.data_group_7 = data_group_7
        self.is_3d = is_3d
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_split_inference = use_split_inference
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.reverse = reverse
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.padding_size = padding_size
        self.crop_size = crop_size
        self.norm_ZeroToOne = norm_ZeroToOne
        self.use_25d_style = use_25d_style
        self.ref_stack_size = ref_stack_size
        self.slice_axis = slice_axis
        self.apply_rigid_registration = apply_rigid_registration
        self.registration_targets = registration_targets

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        self.train_dir = os.path.join(self.data_dir, 'train', self.train_file)
        self.val_dir = os.path.join(self.data_dir, 'val', self.val_file)
        self.test_dir = os.path.join(self.data_dir, 'test', self.test_file) # TODO: If you want to change test file, change it here.

        self.data_train = dataset_SynthRAD(
            self.train_dir,
            data_group_1=self.data_group_1,
            data_group_2=self.data_group_2,
            data_group_3=self.data_group_3,
            data_group_4=self.data_group_4,
            data_group_5=self.data_group_5,
            data_group_6=self.data_group_6,
            data_group_7=self.data_group_7,
            is_3d=self.is_3d,
            padding_size=self.padding_size,
            flip_prob=self.flip_prob,
            rot_prob=self.rot_prob,
            crop_size=self.crop_size,
            reverse=self.reverse,
            norm_ZeroToOne=self.norm_ZeroToOne,
            use_25d_style=self.use_25d_style,
            ref_stack_size=self.ref_stack_size,
            slice_axis=self.slice_axis,
            apply_rigid_registration=self.apply_rigid_registration,
            registration_targets=self.registration_targets,
        )  # Use flip and crop augmentation for training data
        self.data_val = dataset_SynthRAD(
            self.val_dir,
            data_group_1=self.data_group_1,
            data_group_2=self.data_group_2,
            data_group_3=self.data_group_3,
            data_group_4=self.data_group_4,
            data_group_5=self.data_group_5,
            data_group_6=self.data_group_6,
            data_group_7=self.data_group_7,
            is_3d=self.is_3d,
            padding_size=self.padding_size,
            # flip_prob=0.0,
            # rot_prob=0.0,
            # crop_size=self.crop_size,
            reverse=self.reverse,
            norm_ZeroToOne=self.norm_ZeroToOne,
            use_25d_style=self.use_25d_style,
            ref_stack_size=self.ref_stack_size,
            slice_axis=self.slice_axis,
            apply_rigid_registration=self.apply_rigid_registration,
            registration_targets=self.registration_targets,
        )
        self.data_test = dataset_SynthRAD(
            self.test_dir,
            data_group_1=self.data_group_1,
            data_group_2=self.data_group_2,
            data_group_3=self.data_group_3,
            data_group_4=self.data_group_4,
            data_group_5=self.data_group_5,
            data_group_6=self.data_group_6,
            data_group_7=self.data_group_7,
            is_3d=self.is_3d,
            padding_size=self.padding_size,
            # flip_prob=0.0,
            # rot_prob=0.0,
            # crop_size=self.crop_size,
            reverse=self.reverse,
            norm_ZeroToOne=self.norm_ZeroToOne,
            use_25d_style=self.use_25d_style,
            ref_stack_size=self.ref_stack_size,
            slice_axis=self.slice_axis,
            apply_rigid_registration=self.apply_rigid_registration,
            registration_targets=self.registration_targets,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = SynthRAD_MR_CT_Pelvis_DataModule() 
    _.setup()

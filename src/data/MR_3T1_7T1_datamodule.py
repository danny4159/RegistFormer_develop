from typing import Any, Dict, Optional, Tuple


import os
import torch
import h5py
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.data.components.transforms import (
    dataset_SynthRAD,
    download_process_MR_3T_7T
)

class MR_3T_7T_DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        data_group_1: str,
        data_group_2: str,
        data_group_3: str,
        is_3d: bool,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        use_split_inference: bool,
        train_file: str = "",
        val_file: str = "",
        test_file: str = "",
        reverse: bool = False,  # Reverse the order of the images (bool)
        flip_prob: float = 0.0,  # augmentation for training (flip)
        rot_prob: float = 0.0,  # augmentation for training (rot90)
        padding_size: Optional[Tuple[int, int]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        misalign_x: float = 0.0, # maximum misalignment in x direction (float)
        misalign_y: float = 0.0, # maximum misalignment in y direction (float)
        degree: float = 0.0, # The rotation range in z axis (float)
        motion_prob: float = 0.0, # The probability of occurrence of motion (float)
        deform_prob: float = 0.0, # Deformation probability (float)
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
        self.misalign_x = misalign_x
        self.misalign_y = misalign_y
        self.degree = degree
        self.motion_prob = motion_prob
        self.deform_prob = deform_prob

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def misalign(self):
        return 'Misalignment x:{}, y:{}, R:{}, M:{}, D:{}'.format(self.misalign_x, self.misalign_y, self.degree, self.motion_prob, self.deform_prob)

    def prepare_data(self):
        """Prepares the data for usage.

        This function is responsible for the misalignment of training data and saving the data to hdf5 format.
        It doesn't assign any state variables.
        """
        
        for phase in ['train','val','test']:
            target_file = os.path.join(self.data_dir, phase, '3T7T_data.mat')
            
            if phase == 'train': # misalign only for training data
                mis_x, mis_y, Rot_z, M_prob, D_prob = self.misalign_x, self.misalign_y, self.degree, self.motion_prob, self.deform_prob
                if self.train_file:
                    write_dir = os.path.join(self.data_dir, phase, self.train_file)
                else:
                    write_dir = os.path.join(self.data_dir, phase, '3T7T_{}_{}_{}_{}_{}.h5'.format(mis_x,mis_y,Rot_z,M_prob,D_prob)) # save to hdf5
                self.train_dir = write_dir

            # no misalignment for validation and test data
            elif phase == 'val':
                mis_x, mis_y, Rot_z, M_prob, D_prob = 0, 0, 0, 0, 0
                if self.val_file:
                    write_dir = os.path.join(self.data_dir, phase, self.val_file)
                else:
                    write_dir = os.path.join(self.data_dir, phase, '3T7T_{}_{}_{}_{}_{}.h5'.format(mis_x, mis_y, Rot_z, M_prob, D_prob))
                self.val_dir = write_dir

            elif phase == 'test':
                mis_x, mis_y, Rot_z, M_prob, D_prob = 0, 0, 0, 0, 0
                if self.test_file:
                    write_dir = os.path.join(self.data_dir, phase, self.test_file)
                else:
                    write_dir = os.path.join(self.data_dir, phase, '3T7T_{}_{}_{}_{}_{}.h5'.format(mis_x, mis_y, Rot_z, M_prob, D_prob))
                self.test_dir = write_dir
                
            if os.path.exists(write_dir):
                print('path exists for {}'.format(write_dir))
            else:
                download_process_MR_3T_7T(target_file, write_dir, mis_x, mis_y, Rot_z, M_prob, D_prob) # call function

    def setup(self, stage: Optional[str] = None):

        self.data_train = dataset_SynthRAD(
            self.train_dir,
            data_group_1=self.data_group_1,
            data_group_2=self.data_group_2,
            data_group_3=self.data_group_3,
            is_3d=self.is_3d,
            padding_size=self.padding_size,
            flip_prob=self.flip_prob,
            rot_prob=self.rot_prob,
            crop_size=self.crop_size,
            reverse=self.reverse,
        )  # Use flip and crop augmentation for training data
        self.data_val = dataset_SynthRAD(
            self.val_dir,
            data_group_1=self.data_group_1,
            data_group_2=self.data_group_2,
            data_group_3=self.data_group_3,
            is_3d=self.is_3d,
            # padding_size=self.padding_size,
            # flip_prob=0.0,
            # rot_prob=0.0,
            # crop_size=self.crop_size,
            reverse=self.reverse,
        )
        self.data_test = dataset_SynthRAD(
            self.test_dir,
            data_group_1=self.data_group_1,
            data_group_2=self.data_group_2,
            data_group_3=self.data_group_3,
            is_3d=self.is_3d,
            # padding_size=self.padding_size,
            # flip_prob=0.0,
            # rot_prob=0.0,
            # crop_size=self.crop_size,
            reverse=self.reverse,
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
    _ = MR_3T_7T_DataModule()
    _.prepare_data()
    _.setup()

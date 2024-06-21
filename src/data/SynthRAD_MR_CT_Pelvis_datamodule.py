from typing import Any, Dict, Optional

import os
import torch
import h5py
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.data.components.transforms import (
    dataset_SynthRAD_MR_CT_Pelvis, 
    dataset_SynthRAD_MR_CT_Pelvis_3D,
)


class SynthRAD_MR_CT_Pelvis_DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        misalign_x: float = 0.0,  # maximum misalignment in x direction (float)
        misalign_y: float = 0.0,  # maximum misalignment in y direction (float)
        degree: float = 0.0,  # The rotation range in z axis (float)
        motion_prob: float = 0.0,  # The probability of occurrence of motion (float)
        deform_prob: float = 0.0,  # Deformation probability (float)
        train_file: str = "",
        val_file: str = "",
        test_file: str = "",
        reverse: bool = False,  # Reverse the order of the images (bool)
        flip_prob: float = 0.0,  # augmentation for training (flip)
        rot_prob: float = 0.0,  # augmentation for training (rot90)
        rand_crop: bool = False,  # augmentation for training (random crop)
        batch_size: int = 64,
        num_workers: int = 5,
        pin_memory: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.misalign_x = misalign_x
        self.misalign_y = misalign_y
        self.degree = degree
        self.motion_prob = motion_prob
        self.deform_prob = deform_prob
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        self.data_dir = data_dir

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def misalign(self):
        return "Misalignment x:{}, y:{}, R:{}, M:{}, D:{}".format(
            self.misalign_x,
            self.misalign_y,
            self.degree,
            self.motion_prob,
            self.deform_prob,
        )


    def setup(self, stage: Optional[str] = None):
        """Sets up the datasets.

        This function is responsible for loading the data and assigning the datasets.

        Args:
            stage (str, optional): The stage for which to setup the data. Can be None, 'fit' or 'test'. Defaults to None.
        """
        # load and split datasets only if not loaded already
        self.train_dir = os.path.join(self.data_dir, 'train', self.train_file)
        self.val_dir = os.path.join(self.data_dir, 'val', self.val_file)
        self.test_dir = os.path.join(self.data_dir, 'test', self.test_file)

        dataset_class = dataset_SynthRAD_MR_CT_Pelvis_3D if self.hparams.flag_3d else dataset_SynthRAD_MR_CT_Pelvis
        # TODO: Adaconv 쓸 시 수정
        # adaconv 말고 나머지 다 이거
        self.data_train = dataset_class(
            self.train_dir,
            data_group_1=self.hparams.data_group_1,
            data_group_2=self.hparams.data_group_2,
            data_group_3=self.hparams.data_group_3,
            reverse=self.hparams.reverse,
            flip_prob=self.hparams.flip_prob,
            rot_prob=self.hparams.rot_prob,
            padding=self.hparams.padding,
            rand_crop=self.hparams.rand_crop,
        )  # Use flip and crop augmentation for training data
        self.data_val = dataset_class(
            self.val_dir,
            data_group_1=self.hparams.data_group_1,
            data_group_2=self.hparams.data_group_2,
            data_group_3=self.hparams.data_group_3,
            reverse=self.hparams.reverse,
            flip_prob=0.0,
            rot_prob=0.0,
            padding=self.hparams.padding,
            # rand_crop=self.hparams.rand_crop,
        )
        self.data_test = dataset_class(
            self.test_dir,
            data_group_1=self.hparams.data_group_1,
            data_group_2=self.hparams.data_group_2,
            data_group_3=self.hparams.data_group_3,
            reverse=self.hparams.reverse,
            flip_prob=0.0,
            rot_prob=0.0,
            padding=self.hparams.padding,
            # rand_crop=self.hparams.rand_crop,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
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
    # _.prepare_data()
    _.setup()

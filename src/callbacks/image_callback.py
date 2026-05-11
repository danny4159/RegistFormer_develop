from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.utils import make_grid
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import resize
import torch
from typing import Any, List, Optional
from lightning.pytorch import Callback
from src import utils
import numpy as np
import nibabel as nib
import h5py
import os
from PIL import Image
# import itk


log = utils.get_pylogger(__name__)


class ImageLoggingCallback(Callback):
    def __init__(
        self,
        val_batch_idx: List[int] = [10, 20, 30, 40, 50],
        tst_batch_idx: List[int] = [7, 8, 9, 10, 11],
        center_crop: int = 256,
        every_epoch=5,
        log_test: bool = False,
        norm_ZeroToOne: bool = False,
    ):
        """_summary_

        Args:
            batch_idx (List[int], optional): _description_. Defaults to [10,20,30,40,50].
            log_test (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.val_batch_idx = val_batch_idx  # log images on the validation stage
        self.tst_batch_idx = tst_batch_idx  # log images on the testing stage

        self.every_epoch = every_epoch
        self.log_test = log_test  # log images on the testing stage as well
        self.center_crop = center_crop  # center crop the images to this size
        self.norm_ZeroToOne = norm_ZeroToOne

    def normalize_img(self, tensor):
        return tensor if self.norm_ZeroToOne else (tensor + 1) / 2
    
    @staticmethod
    def _to_vis(t):
        """Collapse multi-channel tensors (e.g. 2.5D K-slice stacks) to center channel."""
        if t is not None and isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[1] > 1:
            k = t.shape[1]
            return t[:, k // 2: k // 2 + 1]
        return t

    def saving_to_grid(self, res):
        if not isinstance(res, (list, tuple)):
            print(f"Unexpected input to saving_to_grid: {type(res)}")
            print(f"Input content: {res}")
            return

        # Collapse any multi-channel tensors (2.5D ref stacks) to center channel
        res = tuple(self._to_vis(r) for r in res)
        
        if len(res) == 10: # multi-contrast generation with moved references
            a, b, c, d, preds_b, preds_c, preds_d, b_ref, c_ref, d_ref = res
            err_b = torch.abs(b_ref - preds_b)
            err_c = torch.abs(c_ref - preds_c)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            if d is None or preds_d is None or d_ref is None:
                self.ngrid = 7
                self.img_grid.extend([
                    resize(self.normalize_img(a[0]), self.first_image_size),
                    resize(self.normalize_img(b[0]), self.first_image_size),
                    resize(self.normalize_img(b_ref[0]), self.first_image_size),
                    resize(self.normalize_img(preds_b[0]), self.first_image_size),
                    resize(self.normalize_img(c[0]), self.first_image_size),
                    resize(self.normalize_img(c_ref[0]), self.first_image_size),
                    resize(self.normalize_img(preds_c[0]), self.first_image_size),
                ])
                self.err_grid.extend([
                    resize(err_b[0], self.first_image_size),
                    resize(err_c[0], self.first_image_size),
                ])
            else:
                self.ngrid = 10
                err_d = torch.abs(d_ref - preds_d)
                self.img_grid.extend([
                    resize(self.normalize_img(a[0]), self.first_image_size),
                    resize(self.normalize_img(b[0]), self.first_image_size),
                    resize(self.normalize_img(b_ref[0]), self.first_image_size),
                    resize(self.normalize_img(preds_b[0]), self.first_image_size),
                    resize(self.normalize_img(c[0]), self.first_image_size),
                    resize(self.normalize_img(c_ref[0]), self.first_image_size),
                    resize(self.normalize_img(preds_c[0]), self.first_image_size),
                    resize(self.normalize_img(d[0]), self.first_image_size),
                    resize(self.normalize_img(d_ref[0]), self.first_image_size),
                    resize(self.normalize_img(preds_d[0]), self.first_image_size),
                ])
                self.err_grid.extend([
                    resize(err_b[0], self.first_image_size),
                    resize(err_c[0], self.first_image_size),
                    resize(err_d[0], self.first_image_size)
                ])

        elif len(res) == 7: # multi-contrast generation
            a, b, c, d, preds_b, preds_c, preds_d = res
            err_b = torch.abs(b - preds_b)
            err_c = torch.abs(c - preds_c)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            if d is None or preds_d is None:
                self.ngrid = 5
                self.img_grid.extend([
                    resize(self.normalize_img(a[0]), self.first_image_size),
                    resize(self.normalize_img(preds_b[0]), self.first_image_size),
                    resize(self.normalize_img(b[0]), self.first_image_size),
                    resize(self.normalize_img(preds_c[0]), self.first_image_size),
                    resize(self.normalize_img(c[0]), self.first_image_size),
                ])
                self.err_grid.extend([
                    resize(err_b[0], self.first_image_size),
                    resize(err_c[0], self.first_image_size),
                ])
            else:
                self.ngrid = 7
                err_d = torch.abs(d - preds_d)
                self.img_grid.extend([
                    resize(self.normalize_img(a[0]), self.first_image_size),
                    resize(self.normalize_img(b[0]), self.first_image_size),
                    resize(self.normalize_img(preds_b[0]), self.first_image_size),
                    resize(self.normalize_img(c[0]), self.first_image_size),
                    resize(self.normalize_img(preds_c[0]), self.first_image_size),
                    resize(self.normalize_img(d[0]), self.first_image_size),
                    resize(self.normalize_img(preds_d[0]), self.first_image_size),
                ])
                self.err_grid.extend([
                    resize(err_b[0], self.first_image_size),
                    resize(err_c[0], self.first_image_size),
                    resize(err_d[0], self.first_image_size)
                ])

        elif len(res) == 6:
            self.ngrid = 6
            a, b, a2, b2, preds_a, preds_b = res
            err_a = torch.abs(a2 - preds_a)
            err_b = torch.abs(b2 - preds_b)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            self.img_grid.extend([
            resize(self.normalize_img(a[0]), self.first_image_size),
            resize(self.normalize_img(a2[0]), self.first_image_size),
            resize(self.normalize_img(preds_a[0]), self.first_image_size),
            resize(self.normalize_img(b[0]), self.first_image_size),
            resize(self.normalize_img(b2[0]), self.first_image_size),
            resize(self.normalize_img(preds_b[0]), self.first_image_size),
            ])
            self.err_grid.extend([
                resize(err_a[0], self.first_image_size),
                resize(err_b[0], self.first_image_size),
            ])

        elif len(res) == 5:
            self.ngrid = 5
            a, b, c, preds_b, preds_c = res
            err_a = torch.abs(b - preds_b)
            err_b = torch.abs(c - preds_c)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            self.img_grid.extend([
                resize(self.normalize_img(a[0]), self.first_image_size),
                resize(self.normalize_img(preds_b[0]), self.first_image_size),
                resize(self.normalize_img(b[0]), self.first_image_size),
                resize(self.normalize_img(preds_c[0]), self.first_image_size),
                resize(self.normalize_img(c[0]), self.first_image_size),
            ])
            self.err_grid.extend([
                resize(err_a[0], self.first_image_size),
                resize(err_b[0], self.first_image_size),
            ])

        elif len(res) == 4:
            self.ngrid = 4
            a, b, preds_a, preds_b = res
            err_a = torch.abs(a - preds_a)
            err_b = torch.abs(b - preds_b)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            self.img_grid.extend([
                resize(self.normalize_img(a[0]), self.first_image_size),
                resize(self.normalize_img(preds_a[0]), self.first_image_size),
                resize(self.normalize_img(b[0]), self.first_image_size),
                resize(self.normalize_img(preds_b[0]), self.first_image_size),
            ])
            self.err_grid.extend([
                resize(err_a[0], self.first_image_size),
                resize(err_b[0], self.first_image_size),
            ])
            
        elif len(res) == 3:
            self.ngrid = 3
            a, b, preds_b = res
            err_b = torch.abs(b - preds_b)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            self.img_grid.extend([
                resize(self.normalize_img(a[0]), self.first_image_size),
                resize(self.normalize_img(preds_b[0]), self.first_image_size),
                resize(self.normalize_img(b[0]), self.first_image_size),
            ])
            self.err_grid.extend([
                resize(err_b[0], self.first_image_size),
            ])
            
    def on_validation_start(self, trainer, pl_module):
        self.img_grid = []
        self.err_grid = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        
        if len(batch[0].size()) == 5: # 3D Image
            self.val_batch_idx = [0, 1, 2, 3, 4]

        if (
            batch_idx in self.val_batch_idx
            and trainer.current_epoch % self.every_epoch == 0
        ):  
            
            if len(batch[0].size()) == 5 and pl_module.params.is_registration == True: # 3D Image Registration
                d_index = 30  # 5번째 D 차원의 30번째 데이터 선택
                evaluation_img, moving_img, fixed_img, warped_img = pl_module.model_step(batch, is_3d=True)
                evaluation_img = evaluation_img[:, :, :, :, d_index].squeeze(-1)
                moving_img = moving_img[:, :, :, :, d_index].squeeze(-1)
                fixed_img = fixed_img[:, :, :, :, d_index].squeeze(-1)
                warped_img = warped_img[:, :, :, :, d_index].squeeze(-1)
                self.saving_to_grid([evaluation_img, moving_img, fixed_img, warped_img])
            elif len(batch[0].size()) == 5 and pl_module.params.is_registration == False:  # 3D Image Generation
                d_index = 4
                real_a, real_b, fake_b, *_ = pl_module.model_step(batch)
                real_a = real_a[:, :, :, :, d_index].squeeze(-1)
                real_b = real_b[:, :, :, :, d_index].squeeze(-1)
                fake_b = fake_b[:, :, :, :, d_index].squeeze(-1)
                self.saving_to_grid([real_a, real_b, fake_b])
            elif len(batch[0].size()) == 4: # 2D
                res = pl_module.model_step(batch)
                self.saving_to_grid(res)
            

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if len(self.img_grid) > 0 and trainer.current_epoch % self.every_epoch == 0:
            log.info(f"Saving validation img_grid shape: <{len(self.img_grid)}>")

            img_grid = make_grid(self.img_grid, nrow=self.ngrid)
            err_grid = make_grid(
                self.err_grid, nrow=self.ngrid // 2, value_range=(0, 1)
            )

            # Log to TensorBoard
            trainer.logger.experiment.add_image(
                f"val/images", img_grid, trainer.current_epoch
            )
            trainer.logger.experiment.add_image(
                f"val/error", err_grid, trainer.current_epoch
            )
            self.img_grid = []
            self.err_grid = []
        else:
            log.debug(f"No images to log for validation")

    def on_test_start(self, trainer, pl_module):
        self.img_grid = []
        self.err_grid = [] 

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if len(batch[0].size()) == 5: # 3D Image
            self.val_batch_idx = [0, 1, 2, 3, 4]
            
        if (
            self.log_test and batch_idx in self.tst_batch_idx
        ):  # log every indexes for slice number in test set
            
            if len(batch[0].size()) == 5 and pl_module.params.is_registration == True: # 3D Image
                d_index = 30  # 5번째 D 차원의 30번째 데이터 선택
                evaluation_img, moving_img, fixed_img, warped_img = pl_module.model_step(batch, is_3d=True)
                evaluation_img = evaluation_img[:, :, :, :, d_index].squeeze(-1)
                moving_img = moving_img[:, :, :, :, d_index].squeeze(-1)
                fixed_img = fixed_img[:, :, :, :, d_index].squeeze(-1)
                warped_img = warped_img[:, :, :, :, d_index].squeeze(-1)
                self.saving_to_grid([evaluation_img, moving_img, fixed_img, warped_img])
            elif len(batch[0].size()) == 5 and pl_module.params.is_registration == False:  # 3D Image Generation
                d_index = 4
                real_a, real_b, fake_b, *_ = pl_module.model_step(batch)
                real_a = real_a[:, :, :, :, d_index].squeeze(-1)
                real_b = real_b[:, :, :, :, d_index].squeeze(-1)
                fake_b = fake_b[:, :, :, :, d_index].squeeze(-1)
                self.saving_to_grid([real_a, real_b, fake_b])
            elif len(batch[0].size()) == 4:
                res = pl_module.model_step(batch)
                self.saving_to_grid(res)

    def on_test_end(self, trainer, pl_module):
        log.info(f"Saving test img_grid shape: <{len(self.img_grid)}>")

        # Create a grid of images
        if len(self.img_grid) > 0:
            img_grid = make_grid(self.img_grid, nrow=self.ngrid)
            # Log to TensorBoard
            trainer.logger.experiment.add_image(f"test/final_image", img_grid)
        else:
            log.warning(f"No images to log for testing")

#####################################################################################################################################

#####################################################################################################################################
class ImageSavingCallback(Callback):
    def __init__(self, 
                 center_crop: int = 256, 
                 subject_number_length: int = 3, 
                 test_file: str = None,
                 flag_normalize: bool = True,
                 data_dir: str = None,
                 data_type:str = None,
                 norm_ZeroToOne: bool = False,
                 ):
        """_summary_
        Image saving callback : Save images in nii format for each subject

        """
        super().__init__()
        self.center_crop = center_crop  # center crop the images to this size
        self.subject_number_length = subject_number_length
        self.test_file = test_file
        self.flag_normalize = flag_normalize
        self.data_dir = data_dir
        self.data_type = data_type
        self.norm_ZeroToOne = norm_ZeroToOne
        # print("test_file: ", test_file)
        # print("flag_normalize: ", self.flag_normalize)

    def normalize_np(self, arr):
        return arr if self.norm_ZeroToOne else (arr + 1) / 2
    
    @staticmethod
    def change_torch_numpy(a, b, c, d, e=None, f=None):
        assert (
            a.ndim == b.ndim == c.ndim == d.ndim
        ), "Arrays a, b, c, and d must have the same number of dimensions"
        
        if e is not None or f is not None:
            assert (e is None or e.ndim == a.ndim) and (f is None or f.ndim == a.ndim), "Arrays e and f must have the same number of dimensions as a, b, c, and d if they are provided"

        if a.ndim == 4 or a.ndim == 5:
            a_np = a.cpu().detach().numpy()[0, 0]
            b_np = b.cpu().detach().numpy()[0, 0]
            c_np = c.cpu().detach().numpy()[0, 0]
            d_np = d.cpu().detach().numpy()[0, 0]
            e_np = e.cpu().detach().numpy()[0, 0] if e is not None else None
            f_np = f.cpu().detach().numpy()[0, 0] if f is not None else None
        else:
            raise NotImplementedError("This function has not been implemented yet.")
        
        return a_np, b_np, c_np, d_np, e_np, f_np

    def change_numpy_nii(self, a, b, c, d, e, flag_normalize=True):
        assert (
            a.ndim == b.ndim == c.ndim == d.ndim == 3
        ), "All input arrays must have the same number of dimensions (3)"

        if flag_normalize:
            # scale to [0, 1] and [0, 255]
            a, b, c, d, e = (
                self.normalize_np(a) * 255,
                self.normalize_np(b) * 255,
                self.normalize_np(c) * 255,
                self.normalize_np(d) * 255,
                self.normalize_np(e) * 255,
            )

            # type to np.int16
            a, b, c, d, e= (
                a.astype(np.int16),
                b.astype(np.int16),
                c.astype(np.int16),
                d.astype(np.int16),
                e.astype(np.int16),
            )

        # transpose 1, 2 dim (for viewing on ITK-SNAP)
        # a, b, c, d = (
        #     np.transpose(a, axes=(1, 0, 2))[:, ::-1],
        #     np.transpose(b, axes=(1, 0, 2))[:, ::-1],
        #     np.transpose(c, axes=(1, 0, 2))[:, ::-1],
        #     np.transpose(d, axes=(1, 0, 2))[:, ::-1],
        # )

        # flip rows and columns (행과 열 반전) -> Align with original image
        a, b, c, d, e = (
            a[::-1, ::-1, :],
            b[::-1, ::-1, :],
            c[::-1, ::-1, :],
            d[::-1, ::-1, :] if d is not None else None,
            e[::-1, ::-1, :] if e is not None else None,
        )

        # Create Nifti1Image for each
        a_nii, b_nii, c_nii, d_nii, e_nii = (
            nib.Nifti1Image(a, np.eye(4)),
            nib.Nifti1Image(b, np.eye(4)),
            nib.Nifti1Image(c, np.eye(4)),
            nib.Nifti1Image(d, np.eye(4)),
            nib.Nifti1Image(e, np.eye(4)),
        )

        return a_nii, b_nii, c_nii, d_nii, e_nii
    
    def change_numpy_tif(self, a, b, a2, b2, preds_a, preds_b):
        a = (self.normalize_np(a) * 255).astype(np.uint8)
        b = (self.normalize_np(a) * 255).astype(np.uint8)
        a2 = (self.normalize_np(a) * 255).astype(np.uint8)
        b2 = (self.normalize_np(a) * 255).astype(np.uint8)
        preds_a = (self.normalize_np(a) * 255).astype(np.uint8)
        preds_b = (self.normalize_np(a) * 255).astype(np.uint8)
        return a, b, a2, b2, preds_a, preds_b

    def save_nii(self, a_nii, b_nii, c_nii, d_nii, e_nii, subject_number, folder_path, f_nii=None):
        nib.save(a_nii, os.path.join(folder_path, f"a_{subject_number}.nii.gz"))
        nib.save(b_nii, os.path.join(folder_path, f"b_{subject_number}.nii.gz"))
        nib.save(c_nii, os.path.join(folder_path, f"preds_a_{subject_number}.nii.gz"))
        if d_nii is not None:
            nib.save(
                d_nii, os.path.join(folder_path, f"preds_b_{subject_number}.nii.gz")
            )
        if e_nii is not None:
            nib.save(
                e_nii, os.path.join(folder_path, f"preds_c_{subject_number}.nii.gz")
            )
        if f_nii is not None:
            nib.save(
                f_nii, os.path.join(folder_path, f"b_moved_{subject_number}.nii.gz")
            )
        return
    
    @staticmethod
    def save_nii_registration(a_nii, b_nii, c_nii, d_nii, subject_number, folder_path):
        nib.save(a_nii, os.path.join(folder_path, f"evaluation_img_{subject_number}.nii.gz"))
        nib.save(b_nii, os.path.join(folder_path, f"moving_img_{subject_number}.nii.gz"))
        nib.save(c_nii, os.path.join(folder_path, f"fixed_img_{subject_number}.nii.gz"))
        if d_nii is not None:
            nib.save(
                d_nii, os.path.join(folder_path, f"warped_img_{subject_number}.nii.gz")
            )
        return

    @staticmethod
    def save_tif(a, b, a2, b2, preds_a, preds_b, subject_number, folder_path):
        a_tif = Image.fromarray(a)
        b_tif = Image.fromarray(b)
        a2_tif = Image.fromarray(a2)
        b2_tif = Image.fromarray(b2)
        preds_a_tif = Image.fromarray(preds_a)
        preds_b_tif = Image.fromarray(preds_b)
        
        a_tif.save(os.path.join(folder_path, f"{subject_number}_a.tif"))
        b_tif.save(os.path.join(folder_path, f"{subject_number}_b.tif"))
        a2_tif.save(os.path.join(folder_path, f"{subject_number}_a2.tif"))
        b2_tif.save(os.path.join(folder_path, f"{subject_number}_b2.tif"))
        preds_a_tif.save(os.path.join(folder_path, f"{subject_number}_preds_a.tif"))
        preds_b_tif.save(os.path.join(folder_path, f"{subject_number}_preds_b.tif"))
        return
    
    def saving_to_nii(self, a, b, preds_a, preds_b=None, preds_c=None, b_moved=None):
        if preds_a is None:
            preds_a = torch.zeros_like(a)
        if preds_b is None:
            preds_b = torch.zeros_like(a)
        if preds_c is None:
            preds_c = torch.zeros_like(a)

        a, b, preds_a, preds_b, preds_c, _= self.change_torch_numpy(a, b, preds_a, preds_b, preds_c)
        b_moved_np = b_moved.cpu().detach().numpy()[0, 0] if b_moved is not None else None
        if a.ndim == 2: # 2D image
            self.img_a.append(a)
            self.img_b.append(b)
            self.img_preds_a.append(preds_a)
            if preds_b is not None:
                self.img_preds_b.append(preds_b)
            if preds_c is not None:
                self.img_preds_c.append(preds_c)
            if b_moved_np is not None:
                self.img_b_moved.append(b_moved_np)

            if len(self.img_a) == self.subject_slice_num[0]:
                a_nii = np.stack(self.img_a, -1)
                b_nii = np.stack(self.img_b, -1)
                preds_a_nii = np.stack(self.img_preds_a, -1)
                if preds_b is not None:
                    preds_b_nii = np.stack(self.img_preds_b, -1)
                else:
                    preds_b_nii = a_nii * 0  # Placeholder if preds_b is None
                if preds_c is not None:
                    preds_c_nii = np.stack(self.img_preds_c, -1)
                else:
                    preds_c_nii = a_nii * 0
                b_moved_nii = np.stack(self.img_b_moved, -1) if self.img_b_moved else None

                # convert numpy to nii
                a_nii, b_nii, preds_a_nii, preds_b_nii, preds_c_nii = self.change_numpy_nii(
                    a_nii, b_nii, preds_a_nii, preds_b_nii, preds_c_nii, flag_normalize=self.flag_normalize
                )
                if b_moved_nii is not None:
                    _, b_moved_nii, _, _, _ = self.change_numpy_nii(
                        b_moved_nii, b_moved_nii, b_moved_nii, b_moved_nii, b_moved_nii, flag_normalize=self.flag_normalize
                    )
                # save nii image to (.nii) file
                self.save_nii(
                    a_nii,
                    b_nii,
                    preds_a_nii,
                    preds_b_nii if preds_b is not None else None,
                    preds_c_nii if preds_c is not None else None,
                    subject_number=self.dataset_list[0], # 환자이름으로저장
                    folder_path=self.save_folder_name,
                    f_nii=b_moved_nii,
                )

                # empty list
                self.img_a = []
                self.img_b = []
                self.img_preds_a = []
                if preds_b is not None:
                    self.img_preds_b = []
                if preds_c is not None:
                    self.img_preds_c = []
                self.img_b_moved = []
                self.dataset_list.pop(0)
                self.subject_slice_num.pop(0)

            if self.subject_number > self.subject_number_length:
                log.info(f"Saving test images up to {self.subject_number_length}")
                return
            
        elif a.ndim == 3: # 3D image            
            preds_b = a * 0 if preds_b is None else preds_b
            preds_c = a * 0 if preds_c is None else preds_c
            a_nii, b_nii, preds_a_nii, preds_b_nii, preds_c_nii = self.change_numpy_nii(
                a, b, preds_a, preds_b, preds_c,
                flag_normalize=self.flag_normalize
            )
            
            # save nii image to (.nii) file
            self.save_nii_registration(
                a_nii,
                b_nii,
                preds_a_nii,
                preds_b_nii if preds_b is not None else None,
                subject_number=self.dataset_list[0], # 환자이름으로저장
                folder_path=self.save_folder_name,
            )
            self.dataset_list.pop(0)

    @staticmethod
    def change_torch_numpy_multi(*tensors):
        valid_tensors = [tensor for tensor in tensors if tensor is not None]
        if not valid_tensors:
            raise ValueError("At least one tensor must be provided.")

        ndim = valid_tensors[0].ndim
        assert all(tensor.ndim == ndim for tensor in valid_tensors), "All tensors must have the same number of dimensions"

        if ndim not in [4, 5]:
            raise NotImplementedError("This function has not been implemented yet.")

        arrays = []
        for tensor in tensors:
            if tensor is None:
                arrays.append(None)
            else:
                arrays.append(tensor.cpu().detach().numpy()[0, 0])
        return arrays

    def change_numpy_nii_multi(self, arrays, flag_normalize=True):
        valid_arrays = [arr for arr in arrays if arr is not None]
        if not valid_arrays:
            raise ValueError("At least one array must be provided.")

        assert all(arr.ndim == 3 for arr in valid_arrays), "All input arrays must be 3D"

        nii_list = []
        for arr in arrays:
            if arr is None:
                nii_list.append(None)
                continue

            if flag_normalize:
                arr = (self.normalize_np(arr) * 255).astype(np.int16)

            arr = arr[::-1, ::-1, :]
            nii_list.append(nib.Nifti1Image(arr, np.eye(4)))

        return nii_list

    @staticmethod
    def save_nii_multi(a_nii, b_nii, c_nii, d_nii, preds_b_nii, preds_c_nii, preds_d_nii, b_ref_nii=None, c_ref_nii=None, d_ref_nii=None, subject_number=None, folder_path=None):
        nib.save(a_nii, os.path.join(folder_path, f"a_{subject_number}.nii.gz"))
        nib.save(b_nii, os.path.join(folder_path, f"b_{subject_number}.nii.gz"))
        if c_nii is not None:
            nib.save(c_nii, os.path.join(folder_path, f"c_{subject_number}.nii.gz"))
        if d_nii is not None:
            nib.save(d_nii, os.path.join(folder_path, f"d_{subject_number}.nii.gz"))
        if b_ref_nii is not None:
            nib.save(b_ref_nii, os.path.join(folder_path, f"b_ref_{subject_number}.nii.gz"))
        if c_ref_nii is not None:
            nib.save(c_ref_nii, os.path.join(folder_path, f"c_ref_{subject_number}.nii.gz"))
        if d_ref_nii is not None:
            nib.save(d_ref_nii, os.path.join(folder_path, f"d_ref_{subject_number}.nii.gz"))
        if preds_b_nii is not None:
            nib.save(preds_b_nii, os.path.join(folder_path, f"preds_b_{subject_number}.nii.gz"))
        if preds_c_nii is not None:
            nib.save(preds_c_nii, os.path.join(folder_path, f"preds_c_{subject_number}.nii.gz"))
        if preds_d_nii is not None:
            nib.save(preds_d_nii, os.path.join(folder_path, f"preds_d_{subject_number}.nii.gz"))
        return

    def saving_to_nii_multi(self, a, b, c=None, d=None, preds_b=None, preds_c=None, preds_d=None, b_ref=None, c_ref=None, d_ref=None):
        if c is None:
            c = torch.zeros_like(a)
        if d is None:
            d = torch.zeros_like(a)
        if preds_b is None:
            preds_b = torch.zeros_like(a)
        if preds_c is None:
            preds_c = torch.zeros_like(a)
        if preds_d is None:
            preds_d = torch.zeros_like(a)

        a, b, c, d, preds_b, preds_c, preds_d, b_ref, c_ref, d_ref = self.change_torch_numpy_multi(
            a, b, c, d, preds_b, preds_c, preds_d, b_ref, c_ref, d_ref
        )

        if a.ndim == 2:
            self.img_a.append(a)
            self.img_b.append(b)
            self.img_c.append(c)
            self.img_d.append(d)
            self.img_b_ref.append(b_ref)
            self.img_c_ref.append(c_ref)
            self.img_d_ref.append(d_ref)
            self.img_preds_b.append(preds_b)
            self.img_preds_c.append(preds_c)
            self.img_preds_d.append(preds_d)

            if len(self.img_a) == self.subject_slice_num[0]:
                a_nii = np.stack(self.img_a, -1)
                b_nii = np.stack(self.img_b, -1)
                c_nii = np.stack(self.img_c, -1) if self.img_c and self.img_c[0] is not None else None
                d_nii = np.stack(self.img_d, -1) if self.img_d and self.img_d[0] is not None else None
                b_ref_nii = np.stack(self.img_b_ref, -1) if self.img_b_ref and self.img_b_ref[0] is not None else None
                c_ref_nii = np.stack(self.img_c_ref, -1) if self.img_c_ref and self.img_c_ref[0] is not None else None
                d_ref_nii = np.stack(self.img_d_ref, -1) if self.img_d_ref and self.img_d_ref[0] is not None else None
                preds_b_nii = np.stack(self.img_preds_b, -1)
                preds_c_nii = np.stack(self.img_preds_c, -1) if self.img_preds_c and self.img_preds_c[0] is not None else None
                preds_d_nii = np.stack(self.img_preds_d, -1) if self.img_preds_d and self.img_preds_d[0] is not None else None

                a_nii, b_nii, c_nii, d_nii, preds_b_nii, preds_c_nii, preds_d_nii, b_ref_nii, c_ref_nii, d_ref_nii = self.change_numpy_nii_multi(
                    [a_nii, b_nii, c_nii, d_nii, preds_b_nii, preds_c_nii, preds_d_nii, b_ref_nii, c_ref_nii, d_ref_nii],
                    flag_normalize=self.flag_normalize,
                )
                self.save_nii_multi(
                    a_nii,
                    b_nii,
                    c_nii,
                    d_nii,
                    preds_b_nii,
                    preds_c_nii,
                    preds_d_nii,
                    b_ref_nii=b_ref_nii,
                    c_ref_nii=c_ref_nii,
                    d_ref_nii=d_ref_nii,
                    subject_number=self.dataset_list[0],
                    folder_path=self.save_folder_name,
                )

                self.img_a = []
                self.img_b = []
                self.img_c = []
                self.img_d = []
                self.img_b_ref = []
                self.img_c_ref = []
                self.img_d_ref = []
                self.img_preds_b = []
                self.img_preds_c = []
                self.img_preds_d = []
                self.dataset_list.pop(0)
                self.subject_slice_num.pop(0)

            if self.subject_number > self.subject_number_length:
                log.info(f"Saving test images up to {self.subject_number_length}")
                return

        elif a.ndim == 3:
            a_nii, b_nii, c_nii, d_nii, preds_b_nii, preds_c_nii, preds_d_nii, b_ref_nii, c_ref_nii, d_ref_nii = self.change_numpy_nii_multi(
                [a, b, c, d, preds_b, preds_c, preds_d, b_ref, c_ref, d_ref],
                flag_normalize=self.flag_normalize,
            )
            self.save_nii_multi(
                a_nii,
                b_nii,
                c_nii,
                d_nii,
                preds_b_nii,
                preds_c_nii,
                preds_d_nii,
                b_ref_nii=b_ref_nii,
                c_ref_nii=c_ref_nii,
                d_ref_nii=d_ref_nii,
                subject_number=self.dataset_list[0],
                folder_path=self.save_folder_name,
            )
            self.dataset_list.pop(0)

    def saving_to_tif(self, a, b, a2, b2, preds_a, preds_b=None):
        if preds_a is None:
            preds_a = torch.zeros_like(a)
        if preds_b is None:
            preds_b = torch.zeros_like(b)
            
        a, b, a2, b2, preds_a, preds_b = self.change_torch_numpy(a, b, a2, b2, preds_a, preds_b)
        a, b, a2, b2, preds_a, preds_b = self.change_numpy_tif(a, b, a2, b2, preds_a, preds_b)
        self.save_tif(
            a,
            b,
            a2,
            b2,
            preds_a,
            preds_b if preds_b is not None else None,
            subject_number=self.dataset_list.pop(0),  # Use the dataset name for saving
            folder_path=self.save_folder_name,
        )
        
    def on_test_start(self, trainer, pl_module):
        # make save folder
        folder_name = os.path.join(trainer.default_root_dir, "results")
        log.info(f"Saving test images to nifti files to {folder_name}")

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.save_folder_name = folder_name

        self.img_a = []
        self.img_b = []
        self.img_b_moved = []
        self.img_c = []
        self.img_d = []
        self.img_b_ref = []
        self.img_c_ref = []
        self.img_d_ref = []
        self.img_preds_a = []
        self.img_preds_b = []
        self.img_preds_c = []
        self.img_preds_d = []
        self.i = 0
        self.subject_slice_num = []
        self.subject_number = 1

        data_path = os.path.join(self.data_dir, "test", self.test_file)  # TODO: If you want to change test file, change it here.

        if self.data_type == 'nifti':
            with h5py.File(data_path, "r") as file:
                first_group = file[list(file.keys())[0]]
                self.dataset_list = [
                    key for key in first_group.keys()
                ]  # 데이터셋 이름을 리스트로 저장
                self.subject_slice_num = [
                    first_group[key].shape[2] for key in self.dataset_list
                ]  # slice number를 리스트로 저장

        elif self.data_type == 'photo':
            with h5py.File(data_path, "r") as file:
                first_group = file[list(file.keys())[0]]
                self.dataset_list = [
                    key for key in first_group.keys()
                ]

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if len(batch[0].size()) == 5:
            res = pl_module.model_step(batch, is_3d=True)
        elif len(batch[0].size()) == 4:
            res = pl_module.model_step(batch)
        
        if len(res) == 10: # Multi-contrast generation with moved references
            a, b, c, d, preds_b, preds_c, preds_d, b_ref, c_ref, d_ref = res
            if self.data_type == 'nifti':
                self.saving_to_nii_multi(a, b, c, d, preds_b, preds_c, preds_d, b_ref, c_ref, d_ref)

        elif len(res) == 7: # Multi-contrast generation
            a, b, c, d, preds_b, preds_c, preds_d = res
            if self.data_type == 'nifti':
                self.saving_to_nii_multi(a, b, c, d, preds_b, preds_c, preds_d)

        elif len(res) == 6:
            a, b, a2, b2, preds_a, preds_b = res
            if self.data_type == 'nifti':
                self.saving_to_nii(a, b, a2, b2, preds_a, preds_b)
            elif self.data_type == 'photo':
                self.saving_to_tif(a, b, a2, b2, preds_a, preds_b)
        
        elif len(res) == 5:
            a, b, c, preds_b, preds_c = res
            if self.data_type == 'nifti':
                self.saving_to_nii_multi(a, b, c, None, preds_b, preds_c, None)
            elif self.data_type == 'photo':
                self.saving_to_tif(a, b, preds_a, preds_b)

        elif len(res) == 4:
            a, b, preds_a, preds_b = res
            if self.data_type == 'nifti':
                b_moved = batch[2] if (len(batch) == 3 and getattr(pl_module.params, 'use_misalign_simul', False)) else None
                self.saving_to_nii(a, b, preds_a, preds_b, b_moved=b_moved)
            # elif self.data_type == 'photo':
            #     self.saving_to_tif(a, b, preds_a, preds_b)

        elif len(res) == 3:
            a, b, preds_b = res
            if self.data_type == 'nifti':
                self.saving_to_nii(a, b, preds_b)
            elif self.data_type == 'photo':
                self.saving_to_tif(a, b, preds_b)

        else:
            log.error(f"Unexpected res length: {len(res)}. This case has not been implemented.")
            raise NotImplementedError("This function has not been implemented yet.")
        return
    
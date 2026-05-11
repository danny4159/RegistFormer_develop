#!/usr/bin/env python3
"""
IXI Test Set Preprocessing - Ver3
Output: IXI_Testset_Ver3_T1_T2_PD_MRA_IOPdataset_30Patient_MisalignSimul_80Slice.h5

Pipeline per modality per patient:
  1. Load NIfTI (256, 256, 100)
  2. Clip [0, 99.9th pct] → normalize to [-1, 1]
  3. RandomAffine (degrees=5, translation=5) independently per modality → _moved
  4. Rand3DElastic (sigma=(10,15), magnitude=(600,650)) applied on top of affine → _moved
  5. Trim 10 slices each side → (256, 256, 80)
  6. Background removal: Otsu + Gaussian feathering (sigma=1.5) on both original & _moved
  7. Save to H5

H5 Groups (each: 30 patients, shape=(256, 256, 80), float32):
  T1, T2, PD, MRA           — original (affine+nonlinear NOT applied)
  T1_moved, T2_moved,
  PD_moved, MRA_moved       — affine + non-linear misaligned
"""

import os
import re
import random
import numpy as np
import nibabel as nib
import h5py
import torch
from monai.transforms import Rand3DElastic
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.filters import threshold_otsu
import torchio as tio

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = "/SSD5_8TB/Daniel/Daniel_ssd2/Dataset/IXI/IXI-PreprocessVer1(Resample)"
OUTPUT_DIR = "/SSD5_8TB/Daniel/Daniel_ssd2/RegistFormer/data/IXI/test"
OUTPUT_FILENAME = "IXI_Testset_Ver3_T1_T2_PD_MRA_IOPdataset_30Patient_MisalignSimul_80Slice.h5"

MODALITIES = ['T1', 'T2', 'PD', 'MRA']
N_PATIENTS = 30
RANDOM_SEED = 42

# Affine params (matching notebook: degrees=5, translation=5)
AFFINE_DEGREES = (5, 5, 5)
AFFINE_TRANSLATION = (5, 5, 5)

# Non-linear elastic params (matching existing scripts)
ELASTIC_SIGMA_RANGE = (10, 15)
ELASTIC_MAGNITUDE_RANGE = (600, 650)

TRIM_SLICES = 10   # slices removed from each side: 100 - 20 = 80
BG_SIGMA = 1.5     # Gaussian feathering sigma


# ── Helper Functions ───────────────────────────────────────────────────────────

def get_iop_patients(base_dir, modalities):
    """Return sorted list of IOP patient IDs present in ALL modalities, and file map."""
    mod_maps = {}
    for mod in modalities:
        folder = os.path.join(base_dir, mod)
        id_to_path = {}
        for fname in os.listdir(folder):
            if 'IOP' not in fname:
                continue
            m = re.match(rf'(IXI\d+)-IOP-\d+-{mod}\.nii\.gz', fname)
            if m:
                id_to_path[m.group(1)] = os.path.join(folder, fname)
        mod_maps[mod] = id_to_path

    common = set(mod_maps[modalities[0]].keys())
    for mod in modalities[1:]:
        common &= set(mod_maps[mod].keys())

    file_map = {pid: {mod: mod_maps[mod][pid] for mod in modalities}
                for pid in common}
    return sorted(common), file_map


def load_and_cast(data: np.ndarray) -> np.ndarray:
    """Cast to float32. Data is already in [-1, 1] from IXI-PreprocessVer1."""
    return np.clip(data.astype(np.float32), -1.0, 1.0)


def apply_affine(data_hwz: np.ndarray, seed: int) -> np.ndarray:
    """
    Apply RandomAffine matching notebook convention.
    Input/output shape: (H, W, Z).
    Internally uses (1, Z, H, W) for torchio.
    """
    # (H, W, Z) → (Z, H, W) → (1, Z, H, W)
    data_t = np.transpose(data_hwz, (2, 0, 1))
    tensor = torch.from_numpy(data_t).unsqueeze(0).float()

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    transform = tio.RandomAffine(
        scales=0,
        degrees=AFFINE_DEGREES,
        translation=AFFINE_TRANSLATION,
        isotropic=True,
        default_pad_value='otsu',
        center='image',
    )

    transformed = transform(tensor)
    if isinstance(transformed, torch.Tensor):
        transformed = transformed.numpy()

    # (1, Z, H, W) → (Z, H, W) → (H, W, Z)
    return transformed.squeeze(0).transpose(1, 2, 0).astype(np.float32)


def apply_nonlinear(data_hwz: np.ndarray, seed: int) -> np.ndarray:
    """
    Apply Rand3DElastic following existing apply_nonlinear_misalignment.py convention.
    Input/output shape: (H, W, Z).
    Internally uses (1, H, W, Z) for MONAI.
    """
    # (H, W, Z) → (1, H, W, Z)
    tensor = torch.from_numpy(data_hwz[np.newaxis]).float()

    elastic = Rand3DElastic(
        prob=1.0,
        sigma_range=ELASTIC_SIGMA_RANGE,
        magnitude_range=ELASTIC_MAGNITUDE_RANGE,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        mode="bilinear",
        padding_mode="border",
    )
    elastic.set_random_state(seed=seed)
    transformed = elastic(tensor)

    if isinstance(transformed, torch.Tensor):
        transformed = transformed.numpy()

    # (1, H, W, Z) → (H, W, Z)
    return transformed[0].astype(np.float32)


def trim_slices(data_hwz: np.ndarray, n: int = 10) -> np.ndarray:
    """Remove n slices from each end of the Z axis."""
    return data_hwz[:, :, n:-n]


def remove_background(vol: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Remove background from [-1, 1] normalized volume.
    Otsu threshold → fill holes slice-wise → Gaussian feathering.
    Background pixels fade to -1.
    """
    img_01 = (vol + 1.0) / 2.0
    thresh = threshold_otsu(img_01)
    mask = img_01 > thresh

    mask_filled = np.stack(
        [binary_fill_holes(mask[:, :, i]) for i in range(mask.shape[2])],
        axis=2,
    )

    mask_feather = gaussian_filter(mask_filled.astype(float), sigma=sigma)
    result = vol * mask_feather + (-1.0) * (1.0 - mask_feather)
    return result.astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("IXI Testset Preprocessing  Ver3  (IOP · 4 modalities · 80 slices)")
    print("=" * 70)

    # ── Step 1: Patient selection ─────────────────────────────────────────────
    all_ids, file_map = get_iop_patients(BASE_DIR, MODALITIES)
    print(f"Total IOP patients with all 4 modalities: {len(all_ids)}")

    selected_ids = sorted(random.sample(all_ids, N_PATIENTS))
    print(f"Randomly selected {N_PATIENTS} patients (seed={RANDOM_SEED}):")
    print(f"  {selected_ids}\n")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Buffers: group_name → {patient_id: ndarray}
    buffers = {mod: {} for mod in MODALITIES}
    buffers.update({f"{mod}_moved": {} for mod in MODALITIES})

    # ── Steps 2-6: Process each patient ──────────────────────────────────────
    for p_idx, pid in enumerate(selected_ids):
        print(f"[{p_idx+1:2d}/{N_PATIENTS}] {pid}")

        for m_idx, mod in enumerate(MODALITIES):
            # Unique seed per (patient, modality) combination
            seed = RANDOM_SEED + p_idx * 1000 + m_idx * 100

            # 1. Load
            nii = nib.load(file_map[pid][mod])
            data = nii.get_fdata().astype(np.float32)  # (256, 256, 100)

            # 2. Data is already [-1, 1] from IXI-PreprocessVer1; just cast
            data_norm = load_and_cast(data)

            # 3. Random Affine → _moved
            data_moved = apply_affine(data_norm, seed=seed)

            # 4. Non-linear Elastic → applied on top of affine
            data_moved = apply_nonlinear(data_moved, seed=seed + 1)

            # 5. Trim 10 slices each side → (256, 256, 80)
            data_norm_trim = trim_slices(data_norm, TRIM_SLICES)
            data_moved_trim = trim_slices(data_moved, TRIM_SLICES)

            # 6. Background removal
            data_norm_bg = remove_background(data_norm_trim, sigma=BG_SIGMA)
            data_moved_bg = remove_background(data_moved_trim, sigma=BG_SIGMA)

            buffers[mod][pid] = data_norm_bg
            buffers[f"{mod}_moved"][pid] = data_moved_bg

        print(f"       shape(orig/moved): {data_norm_bg.shape}")

    # ── Step 7: Save to H5 ────────────────────────────────────────────────────
    print(f"\nSaving → {output_path}")
    group_order = MODALITIES + [f"{m}_moved" for m in MODALITIES]

    with h5py.File(output_path, 'w') as f:
        for grp_name in group_order:
            grp = f.create_group(grp_name)
            for pid, arr in sorted(buffers[grp_name].items()):
                grp.create_dataset(
                    pid, data=arr, compression='gzip', compression_opts=4
                )

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"File size: {size_mb:.1f} MB")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("H5 Structure Summary")
    print("=" * 70)
    with h5py.File(output_path, 'r') as f:
        for grp_name in f.keys():
            patients = sorted(f[grp_name].keys())
            shape = f[grp_name][patients[0]].shape
            dtype = f[grp_name][patients[0]].dtype
            print(f"  {grp_name:<20s}: {len(patients)} patients, shape={shape}, dtype={dtype}")

    print("\nDone!")
    print("=" * 70)


if __name__ == "__main__":
    main()

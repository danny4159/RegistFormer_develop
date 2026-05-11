#!/usr/bin/env python3
"""
IXI Ver4 Preprocessing Pipeline — Full Reference Implementation
================================================================
This script documents the complete preprocessing pipeline used to produce
the Ver4 H5 datasets. It is intended as a single-file reference that
consolidates all steps across the individual scripts that were actually run.

Output files (per split):
  *_80Slice.h5  — all 24 groups, shape (256, 256, 80)
  *_20Slice.h5  — all 24 groups, shape (256, 256, 20)  ← trimmed final dataset

24 groups per file:
  T1, T2, PD, MRA
  T1_moved, T2_moved, PD_moved, MRA_moved
  T1_3mm,   T2_3mm,   PD_3mm,   MRA_3mm
  T1_5mm,   T2_5mm,   PD_5mm,   MRA_5mm
  T1_moved_3mm, T2_moved_3mm, PD_moved_3mm, MRA_moved_3mm
  T1_moved_5mm, T2_moved_5mm, PD_moved_5mm, MRA_moved_5mm

Pipeline overview:
  STAGE 0  — Upstream (IXI-PreprocessVer1, done before this script)
  STAGE 1  — Original modalities from Ver3 H5 (copied as-is)
  STAGE 2  — Ver4 _moved: improved rigid + nonlinear misalignment
  STAGE 3  — Slice thickness simulation (3 mm, 5 mm)
  STAGE 4  — Slice trimming 80 → 20

See Ver4_Preprocess.txt for detailed explanation.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import re
import random
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from monai.transforms import Rand3DElastic
from scipy.ndimage import binary_fill_holes, gaussian_filter, zoom
from skimage.filters import threshold_otsu


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
MODALITIES  = ["T1", "T2", "PD", "MRA"]

# ── STAGE 0: Upstream normalization (IXI-PreprocessVer1) ─────────────────────
# Applied to raw NIfTI before anything below. Not re-run here.
#   - Resample to 1×1×1 mm isotropic, shape (256, 256, 100)
#   - Percentile clip [0, 99.9th percentile]
#   - Normalize to [-1, 1]  (stored in IXI-PreprocessVer1 NIfTI files)

# ── STAGE 1: Ver3 original modalities ────────────────────────────────────────
# T1/T2/PD/MRA are taken from Ver3 H5 files (already processed below).
# Ver3 pipeline (ixi_preprocess_ver3_iop_testset.py):
#   1. Load [-1,1] normalized NIfTI (256,256,100)
#   2. Trim 10 slices each side → (256,256,80)
#   3. Background removal (Otsu + Gaussian feather, σ=1.5)
# The original modalities are copied verbatim into Ver4.

# ── STAGE 2: Ver4 _moved parameters ──────────────────────────────────────────
# Ver3 used:  bilinear interp, border padding, sigma~U(10,15), mag~U(600,650)
# Ver4 uses:  cubic spline (order=3), reflect padding, sigma=13, mag=650 (fixed)
V4_AFFINE_DEGREES     = (5, 5, 5)    # RandomAffine rotation range (degrees)
V4_AFFINE_TRANSLATION = (5, 5, 5)    # RandomAffine translation range (voxels)
V4_ELASTIC_SIGMA      = 13           # Rand3DElastic sigma (fixed)
V4_ELASTIC_MAG        = 650          # Rand3DElastic magnitude (fixed)
V4_BG_SIGMA           = 1.5          # Gaussian feathering sigma

# ── STAGE 3: Slice thickness simulation ──────────────────────────────────────
ORIG_SPACING     = np.array([1.0, 1.0, 1.0])  # mm (isotropic after ANTs)
TARGET_THICKNESS = [3, 5]                       # mm

# ── STAGE 4: Final slice trim ─────────────────────────────────────────────────
TRIM_SLICES = 30   # remove from each side: 80 - 60 = 20 remaining slices

# ── File paths ────────────────────────────────────────────────────────────────
DATA_ROOT = "/SSD5_8TB/Daniel/Daniel_ssd2/RegistFormer/data/IXI"

VER3_SRC = {
    "train": (DATA_ROOT + "/train/IXI_Trainset_Ver3_T1_T2_PD_MRA_Registered_30Patient_"
              "MisalignSimul_Rotate5Translate5_NonLinearSig10_15_Mag600_650_"
              "BackGroundOutFeather_SliceThicknessSimul_20SliceOut.h5"),
    "val":   (DATA_ROOT + "/val/IXI_Valset_Ver3_T1_T2_PD_MRA_Registered_3Patient_"
              "MisalignSimul_Rotate5Translate5_NonLinearSig10_15_Mag600_650_"
              "BackGroundOutFeather_SliceThicknessSimul_20SliceOut.h5"),
    "test":  (DATA_ROOT + "/test/IXI_Testset_Ver3_T1_T2_PD_MRA_Registered_30Patient_"
              "MisalignSimul_Rotate5Translate5_NonLinearSig10_15_Mag600_650_"
              "BackGroundOutFeather_SliceThicknessSimul_20SliceOut.h5"),
}

VER4_80SLICE = {
    "train": (DATA_ROOT + "/train/IXI_Trainset_Ver4_T1_T2_PD_MRA_Registered_30Patient_"
              "MisalignSimul_Rotate5Translate5_NonLinearSig13_Mag650_CubicInterpolate_"
              "BackGroundOutFeather_SliceThicknessSimul_80Slice.h5"),
    "val":   (DATA_ROOT + "/val/IXI_Valset_Ver4_T1_T2_PD_MRA_Registered_3Patient_"
              "MisalignSimul_Rotate5Translate5_NonLinearSig13_Mag650_CubicInterpolate_"
              "BackGroundOutFeather_SliceThicknessSimul_80Slice.h5"),
    "test":  (DATA_ROOT + "/test/IXI_Testset_Ver4_T1_T2_PD_MRA_Registered_30Patient_"
              "MisalignSimul_Rotate5Translate5_NonLinearSig13_Mag650_CubicInterpolate_"
              "BackGroundOutFeather_SliceThicknessSimul_80Slice.h5"),
}

VER4_20SLICE = {k: v.replace("_80Slice.h5", "_20Slice.h5") for k, v in VER4_80SLICE.items()}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 HELPERS — Misalignment transforms
# ══════════════════════════════════════════════════════════════════════════════

def apply_rigid(vol_hwz: np.ndarray, seed: int) -> np.ndarray:
    """
    Rigid misalignment via torchio RandomAffine.
    Input/output: (H, W, Z) float32, values in [-1, 1].

    Internally torchio expects (1, Z, H, W):
      (H,W,Z) → transpose → (Z,H,W) → unsqueeze → (1,Z,H,W)
    default_pad_value='otsu': pads with Otsu-threshold value (≈tissue boundary)
    to avoid introducing artificial black borders during rotation.
    """
    data_t = np.transpose(vol_hwz, (2, 0, 1))           # (Z, H, W)
    tensor = torch.from_numpy(data_t).unsqueeze(0).float()  # (1, Z, H, W)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    transform = tio.RandomAffine(
        scales=0,
        degrees=V4_AFFINE_DEGREES,
        translation=V4_AFFINE_TRANSLATION,
        isotropic=True,
        default_pad_value='otsu',
        center='image',
    )
    out = transform(tensor)
    if isinstance(out, torch.Tensor):
        out = out.numpy()
    return out.squeeze(0).transpose(1, 2, 0).astype(np.float32)  # back to (H,W,Z)


def apply_nonlinear_v4(vol_hwz: np.ndarray, seed: int) -> np.ndarray:
    """
    Nonlinear misalignment via MONAI Rand3DElastic — Ver4 improved settings.
    Input/output: (H, W, Z) float32.

    Key improvements over Ver3:
      mode=3  (cubic spline via scipy.ndimage.map_coordinates, order=3)
              → preserves intensity peaks; bilinear blurs by weighted averaging
      padding_mode='reflect'
              → mirrors the volume at boundaries instead of repeating edge values,
                preventing stripe artifacts at the brain boundary
      sigma=13 (fixed, not random)
      mag=650  (fixed, not random)

    Internally MONAI expects (1, H, W, Z):
      rand_offset ~ U(-1,1) shape (3,H,W,Z)
      gaussian(rand_offset, sigma) * magnitude  → smooth displacement field
      grid_sample with the displacement field applied to the image
    """
    tensor = torch.from_numpy(vol_hwz[np.newaxis]).float()   # (1, H, W, Z)
    elastic = Rand3DElastic(
        prob=1.0,
        sigma_range=(V4_ELASTIC_SIGMA, V4_ELASTIC_SIGMA),
        magnitude_range=(V4_ELASTIC_MAG, V4_ELASTIC_MAG),
        mode=3,                  # cubic spline (scipy backend, order=3)
        padding_mode="reflect",  # reflect boundary (not border/zeros)
    )
    elastic.set_random_state(seed=seed)
    out = elastic(tensor)
    if isinstance(out, torch.Tensor):
        out = out.numpy()
    return out[0].astype(np.float32)  # (H, W, Z)


def remove_background(vol: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Background removal for [-1, 1] normalized MRI volumes.

    Steps per volume:
      1. Map [-1,1] → [0,1] for Otsu thresholding
      2. Otsu threshold → binary foreground mask
      3. Slice-wise binary_fill_holes (fills enclosed cavities per 2D slice)
      4. Gaussian blur on mask (sigma=1.5) → soft feathering at brain boundary
      5. Blend: out = vol * mask_soft + (-1.0) * (1 - mask_soft)
         → background fades to -1 instead of hard cutoff

    The feathering prevents checkerboard/aliasing artifacts at the skull edge.
    """
    img_01 = (vol + 1.0) / 2.0
    thresh = threshold_otsu(img_01)
    mask = img_01 > thresh
    mask_filled = np.stack(
        [binary_fill_holes(mask[:, :, i]) for i in range(mask.shape[2])], axis=2
    )
    mask_feather = gaussian_filter(mask_filled.astype(float), sigma=sigma)
    return (vol * mask_feather + (-1.0) * (1.0 - mask_feather)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 HELPERS — Slice thickness simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_slice_thickness(vol: np.ndarray, thick_mm: int) -> np.ndarray:
    """
    Simulate MRI slice thickness by degrading z-axis resolution.

    Pipeline:
      1. Downsample z: zoom factor = orig_spacing_z / target_spacing_z = 1/thick_mm
         e.g. thick_mm=3 → zoom_z = 1/3  (80 slices → ~27 slices)
      2. Upsample back to original shape
      Both steps use scipy.ndimage.zoom with cubic spline (order=3).

    This emulates the partial-volume effect that occurs when a scanner acquires
    thick slices: adjacent tissues blur together along the slice direction.

    Input/output shape: (H, W, Z) float32, same shape.
    """
    orig_shape = vol.shape
    zoom_down = ORIG_SPACING / np.array([ORIG_SPACING[0], ORIG_SPACING[1], float(thick_mm)])
    vol_down = zoom(vol, zoom_down, order=3, prefilter=True, mode="nearest")
    zoom_up = np.array(orig_shape, dtype=float) / np.array(vol_down.shape, dtype=float)
    vol_back = zoom(vol_down, zoom_up, order=3, prefilter=True, mode="nearest")
    return vol_back.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 HELPERS — Slice trimming
# ══════════════════════════════════════════════════════════════════════════════

def trim_z(vol: np.ndarray, n: int = 30) -> np.ndarray:
    """Remove n slices from each end of the Z axis: (H,W,80) → (H,W,20)."""
    return vol[:, :, n:-n]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_ver4_80slice(split: str):
    """
    STAGE 1 + STAGE 2 + STAGE 3 → *_80Slice.h5

    Reads Ver3 H5 (contains original T1/T2/PD/MRA already background-removed).
    Copies originals, re-generates _moved with improved settings, adds _3mm/_5mm.
    """
    src_path = VER3_SRC[split]
    dst_path = VER4_80SLICE[split]

    print(f"\n{'='*70}")
    print(f"[{split.upper()}] Building Ver4 80Slice")
    print(f"  src: {Path(src_path).name}")
    print(f"  dst: {Path(dst_path).name}")
    print(f"{'='*70}")

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        patients = sorted(src["T1"].keys())
        print(f"  Patients: {len(patients)}")

        # ── STAGE 1: Copy original modalities from Ver3 ───────────────────────
        # T1/T2/PD/MRA in Ver3 are already:
        #   - Clipped [0, 99.9th pct] → normalized [-1,1]  (IXI-PreprocessVer1)
        #   - Trimmed 10 slices each side (100→80)
        #   - Background removed (Otsu + Gaussian feather)
        print("\n  [STAGE 1] Copying original modalities from Ver3...")
        for mod in MODALITIES:
            grp = dst.create_group(mod)
            for pid in patients:
                grp.create_dataset(pid, data=src[mod][pid][:],
                                   compression="gzip", compression_opts=4)
        print(f"    Copied: {MODALITIES}")

        # ── STAGE 2: Generate _moved with Ver4 improved misalignment ──────────
        # Rigid (torchio RandomAffine) → Nonlinear (cubic spline Rand3DElastic)
        # → Background removal
        # Seed: RANDOM_SEED + patient_idx * 1000 + modality_idx * 100
        print("\n  [STAGE 2] Generating _moved groups (Ver4 improved)...")
        for m_idx, mod in enumerate(MODALITIES):
            grp = dst.create_group(f"{mod}_moved")
            print(f"    {mod}_moved ...", end="", flush=True)
            for p_idx, pid in enumerate(patients):
                seed = RANDOM_SEED + p_idx * 1000 + m_idx * 100
                vol = src[mod][pid][:]
                vol = apply_rigid(vol, seed=seed)
                vol = apply_nonlinear_v4(vol, seed=seed + 1)
                vol = remove_background(vol, sigma=V4_BG_SIGMA)
                grp.create_dataset(pid, data=vol, compression="gzip", compression_opts=4)
            print(" done")

        # ── STAGE 3: Slice thickness simulation ───────────────────────────────
        # Applied to BOTH original and _moved groups.
        # For each modality × thickness: downsample z → upsample back (cubic).
        print("\n  [STAGE 3] Slice thickness simulation (3mm, 5mm)...")
        for mod in MODALITIES:
            for suffix in [mod, f"{mod}_moved"]:
                for thick in TARGET_THICKNESS:
                    new_grp_name = f"{suffix}_{thick}mm"
                    grp = dst.create_group(new_grp_name)
                    print(f"    {suffix} → {new_grp_name} ...", end="", flush=True)
                    for pid in patients:
                        vol = dst[suffix][pid][:]
                        grp.create_dataset(pid, data=simulate_slice_thickness(vol, thick),
                                           compression="gzip", compression_opts=1)
                    print(" done")

    print(f"\n  Saved: {dst_path}")


def build_ver4_20slice(split: str):
    """
    STAGE 4 → *_20Slice.h5

    Trims 30 slices from each end of all groups: (256,256,80) → (256,256,20).
    This removes the lower-quality peripheral slices that suffer from:
      - Partial volume effects at the top/bottom of the brain
      - Reduced tissue coverage in z
    """
    src_path = VER4_80SLICE[split]
    dst_path = VER4_20SLICE[split]

    print(f"\n{'='*70}")
    print(f"[{split.upper()}] Building Ver4 20Slice (trimming 30 each side)")
    print(f"  src: {Path(src_path).name}")
    print(f"  dst: {Path(dst_path).name}")
    print(f"{'='*70}")

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        groups = sorted(src.keys())
        for grp_name in groups:
            out_grp = dst.create_group(grp_name)
            for pid in sorted(src[grp_name].keys()):
                vol = src[grp_name][pid][:]
                out_grp.create_dataset(pid, data=trim_z(vol, TRIM_SLICES),
                                       compression="gzip", compression_opts=4)
            sample_shape = trim_z(src[grp_name][sorted(src[grp_name].keys())[0]][:]).shape
            print(f"  {grp_name}: → {sample_shape}")

    print(f"\n  Saved: {dst_path}")


def verify(split: str):
    """Verify all 24 groups are present with correct shapes and no NaN."""
    path = VER4_20SLICE[split]
    expected = []
    for mod in MODALITIES:
        for suf in ["", "_moved", "_3mm", "_moved_3mm", "_5mm", "_moved_5mm"]:
            expected.append(f"{mod}{suf}")

    issues = []
    with h5py.File(path, "r") as f:
        groups = set(f.keys())
        pids = sorted(f["T1"].keys())
        for grp in expected:
            if grp not in groups:
                issues.append(f"MISSING group: {grp}")
                continue
            for pid in pids:
                d = f[grp][pid][:]
                if d.shape != (256, 256, 20):
                    issues.append(f"SHAPE {grp}/{pid}: {d.shape}")
                if np.isnan(d).any():
                    issues.append(f"NaN in {grp}/{pid}")

    status = "OK" if not issues else "FAIL"
    print(f"  [{status}] {split}: {len(pids)} patients, {len(expected)} groups")
    for iss in issues:
        print(f"    !! {iss}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for split in ["train", "val", "test"]:
        build_ver4_80slice(split)

    for split in ["train", "val", "test"]:
        build_ver4_20slice(split)

    print("\n\n" + "="*70)
    print("Verification — *_20Slice.h5")
    print("="*70)
    for split in ["train", "val", "test"]:
        verify(split)

    print("\nAll done.")


if __name__ == "__main__":
    main()

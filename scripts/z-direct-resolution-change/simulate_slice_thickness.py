"""
Simulate slice thickness change on T1/PD from IXI HDF5 test set.

Pipeline per contrast (T1, PD):
  1. Original  → save as NIfTI  (original spacing)
  2. Downsample z to 2 mm slice thickness
  3. Resample back to T2 reference space (original spacing / shape)
  4. Downsample z to 3 mm slice thickness
  5. Resample back to T2 reference space

T2 is kept as-is and used as the resampling target.

Assumed voxel spacing: 1.0 x 1.0 x 1.5 mm  (IXI registered/cropped data)
"""

import os
import h5py
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# ──────────────────────────────────────────────────────────────
H5_PATH = (
    "/SSD1_1TB/home/milab/daniel/03_Registformer_develop/RegistFormer_develop"
    "/data/IXI/test/"
    "IXI_Testset_T1_T2_PD_Registered_30Patient_MisalignSimul_Rotate5Translate5_20SliceOut.h5"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PATIENT_ID = "IXI033-HH-1259"      # first patient — change as needed

# Original voxel spacing: 1mm isotropic (ANTs from_numpy default, confirmed from NIfTI headers)
ORIG_SPACING = np.array([1.0, 1.0, 1.0])   # mm  (x, y, z)

TARGET_SLICE_THICKNESSES = [2.0, 3.0, 4.0, 5.0]  # mm
# ──────────────────────────────────────────────────────────────


def make_affine(spacing):
    """Simple diagonal affine (no rotation, origin at 0)."""
    aff = np.diag([*spacing, 1.0])
    return aff


def resample_volume(vol, src_spacing, tgt_spacing, order=3):
    """
    Resample `vol` from src_spacing to tgt_spacing using spline interpolation.
    Returns (resampled_vol, tgt_spacing).
    """
    zoom_factors = np.array(src_spacing) / np.array(tgt_spacing)
    resampled = zoom(vol, zoom_factors, order=order, prefilter=True)
    return resampled


def resample_to_target(vol, src_spacing, tgt_shape, tgt_spacing, order=3):
    """
    Resample `vol` (with src_spacing) back to tgt_shape / tgt_spacing.
    zoom factor = src_shape / tgt_shape  (equivalent to tgt_spacing/src_spacing along each axis)
    """
    src_shape = np.array(vol.shape, dtype=float)
    tgt_shape_arr = np.array(tgt_shape, dtype=float)
    zoom_factors = tgt_shape_arr / src_shape
    resampled = zoom(vol, zoom_factors, order=order, prefilter=True)
    # Ensure exact target shape (zoom can be off by 1 voxel due to rounding)
    slices = tuple(slice(0, s) for s in tgt_shape)
    out = np.zeros(tgt_shape, dtype=resampled.dtype)
    insert = tuple(slice(0, min(resampled.shape[i], tgt_shape[i])) for i in range(3))
    out[insert] = resampled[insert]
    return out


def save_nifti(vol, affine, path):
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    nib.save(img, path)
    print(f"  saved → {os.path.basename(path)}  shape={vol.shape}")


# ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with h5py.File(H5_PATH, "r") as f:
        t2 = f["T2"][PATIENT_ID][:]      # (256, 256, 20)
        t1 = f["T1"][PATIENT_ID][:]
        pd = f["PD"][PATIENT_ID][:]

    print(f"Patient : {PATIENT_ID}")
    print(f"Shape   : {t2.shape}  (assumed spacing {ORIG_SPACING} mm)")

    orig_shape   = t2.shape
    orig_affine  = make_affine(ORIG_SPACING)
    orig_spacing = ORIG_SPACING.copy()

    # ── 1. Save originals ────────────────────────────────────
    print("\n[1] Saving originals …")
    save_nifti(t2, orig_affine, os.path.join(OUT_DIR, f"{PATIENT_ID}_T2_orig.nii.gz"))
    save_nifti(t1, orig_affine, os.path.join(OUT_DIR, f"{PATIENT_ID}_T1_orig.nii.gz"))
    save_nifti(pd, orig_affine, os.path.join(OUT_DIR, f"{PATIENT_ID}_PD_orig.nii.gz"))

    # ── 2. Simulate thicker slice thickness for T1 and PD ───
    for thick_mm in TARGET_SLICE_THICKNESSES:
        thick_spacing = np.array([orig_spacing[0], orig_spacing[1], thick_mm])
        tag = f"{int(thick_mm)}mm"

        print(f"\n[{tag}] Downsampling T1 / PD to {thick_mm} mm slice thickness …")

        for name, vol in [("T1", t1), ("PD", pd)]:
            # Step A: downsample z  →  simulate acquisition at thicker slice
            vol_down = resample_volume(vol, orig_spacing, thick_spacing, order=3)
            down_affine = make_affine(thick_spacing)
            save_nifti(
                vol_down, down_affine,
                os.path.join(OUT_DIR, f"{PATIENT_ID}_{name}_slicethick{tag}.nii.gz"),
            )

            # Step B: resample back to T2 reference space
            vol_resampled = resample_to_target(
                vol_down, thick_spacing, orig_shape, orig_spacing, order=3
            )
            save_nifti(
                vol_resampled, orig_affine,
                os.path.join(
                    OUT_DIR,
                    f"{PATIENT_ID}_{name}_slicethick{tag}_resampled_to_T2space.nii.gz",
                ),
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

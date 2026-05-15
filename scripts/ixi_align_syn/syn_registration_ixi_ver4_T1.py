#!/usr/bin/env python3
"""
SyN-based registration of T1_moved, T1_moved_5mm to T2 (fixed)
on IXI Ver4 test dataset. Metrics (SSIM, PSNR, LPIPS, Sharpness) computed
with GT=T1, matching the base_module_AtoB.py convention exactly.

Data: [-1, 1] normalized float32 H5 file, shape (256, 256, 80) per patient.
Pipeline:
  1. Parallel SyN registration (8 workers) — saves warped + GT arrays (.npy)
  2. Per-slice metric computation on cuda:1
"""

import os
import sys
import h5py
import numpy as np
import ants
import torch
import cv2
from pathlib import Path
from multiprocessing import Pool
import tempfile
import json

# ── Paths & Config ────────────────────────────────────────────────────────────
H5_PATH = (
    "/SSD5_8TB/Daniel/Daniel_ssd2/RegistFormer/data/IXI/test"
    "/IXI_Testset_Ver4_T1_T2_PD_MRA_Registered_30Patient_MisalignSimul_Rotate5Translate5"
    "_NonLinearSig13_Mag650_CubicInterpolate_BackGroundOutFeather_SliceThicknessSimul_80Slice.h5"
)
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results_ver4_T1"
FIXED_MOD = "T2"
MODALITY_PAIRS = [
    ("T1_moved",     "T1"),   # (moving_key_in_h5, gt_key_in_h5)
    ("T1_moved_5mm", "T1"),
]
NUM_WORKERS  = 8
ITK_THREADS  = 4              # 8 workers × 4 = 32 threads total
SYN_TYPE     = "SyN"
DEVICE       = "cuda:1" if torch.cuda.is_available() else "cpu"

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_robust(arr: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [pmin, pmax])
    arr = np.clip(arr, lo, hi)
    if hi - lo < 1e-8:
        return arr - lo
    return (arr - lo) / (hi - lo)


def register_one(args):
    """Worker: register one (patient, modality) pair and save .npy results."""
    (patient_id, h5_path, fixed_mod, moving_mod, gt_mod,
     results_dir, idx, total, itk_threads) = args

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)

    results_dir  = Path(results_dir)
    warped_path  = results_dir / f"{patient_id}_{moving_mod}_warped.npy"
    gt_path      = results_dir / f"{patient_id}_{gt_mod}_gt.npy"
    fixed_path   = results_dir / f"{patient_id}_{fixed_mod}.npy"

    if warped_path.exists() and gt_path.exists():
        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:14s} -- skip (already done)")
        return True, patient_id, moving_mod, gt_mod

    try:
        with h5py.File(h5_path, "r") as hf:
            fixed_arr  = hf[fixed_mod][patient_id][:]   # (H, W, D) float32 in [-1, 1]
            moving_arr = hf[moving_mod][patient_id][:]
            gt_arr     = hf[gt_mod][patient_id][:]

        fixed_ants  = ants.from_numpy(fixed_arr.astype(np.float32))
        moving_ants = ants.from_numpy(moving_arr.astype(np.float32))

        fixed_norm  = ants.from_numpy(normalize_robust(fixed_arr))
        moving_norm = ants.from_numpy(normalize_robust(moving_arr))

        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = ants.registration(
                fixed=fixed_norm,
                moving=moving_norm,
                type_of_transform=SYN_TYPE,
                outprefix=os.path.join(tmp_dir, "ants_"),
                verbose=False,
            )
            warped = ants.apply_transforms(
                fixed=fixed_ants,
                moving=moving_ants,
                transformlist=reg["fwdtransforms"],
                interpolator="linear",
            )
            warped_arr = warped.numpy()   # (H, W, D), still in [-1, 1]

        np.save(str(warped_path), warped_arr)
        np.save(str(gt_path),     gt_arr)
        if not fixed_path.exists():
            np.save(str(fixed_path), fixed_arr)

        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:14s} -- done")
        return True, patient_id, moving_mod, gt_mod

    except Exception as exc:
        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:14s} -- FAILED: {exc}")
        return False, patient_id, moving_mod, gt_mod


# ── Metric computation ────────────────────────────────────────────────────────
# Matches base_module_AtoB.py exactly:
#   gray2rgb      = lambda x: cat((x,x,x), dim=1)
#   norm_to_uint8 = lambda x: ((x+1)/2*255).to(uint8)
#   SSIM  : StructuralSimilarityIndexMeasure(reduction="none"), update(gt, pred)
#   PSNR  : PeakSignalNoiseRatio(data_range=2.0), compute+reset per sample
#   LPIPS : LearnedPerceptualImagePatchSimilarity(), gray2rgb, compute+reset per sample
#   Sharp : Laplacian variance of norm_to_uint8(pred).float()  [on prediction only]

def compute_metrics_for_patient(warped_arr, gt_arr, device):
    """
    Returns dict with lists of per-slice scalar values.
    warped_arr, gt_arr: (H, W, D) float32 in [-1, 1]
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    ssim_fn  = StructuralSimilarityIndexMeasure(reduction="none").to(device)
    psnr_fn  = PeakSignalNoiseRatio(data_range=2.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity().to(device)

    n_slices = warped_arr.shape[-1]
    ssim_vals, psnr_vals, lpips_vals, sharp_vals = [], [], [], []

    for s in range(n_slices):
        pred_np = warped_arr[:, :, s].astype(np.float32)
        gt_np   = gt_arr[:, :, s].astype(np.float32)

        # (1, 1, H, W) tensors
        pred_t = torch.tensor(pred_np).unsqueeze(0).unsqueeze(0).to(device)
        gt_t   = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0).to(device)

        # SSIM — update(gt, pred), accumulate across slices, compute later
        ssim_fn.update(gt_t, pred_t)

        # PSNR — update(gt, pred), compute+reset per sample
        psnr_fn.update(gt_t, pred_t)
        psnr_vals.append(psnr_fn.compute().item())
        psnr_fn.reset()

        # LPIPS — gray2rgb, update(gt, pred), compute+reset per sample
        pred_rgb = pred_t.clamp(-1, 1).repeat(1, 3, 1, 1)
        gt_rgb   = gt_t.clamp(-1, 1).repeat(1, 3, 1, 1)
        lpips_fn.update(gt_rgb, pred_rgb)
        lpips_vals.append(lpips_fn.compute().item())
        lpips_fn.reset()

        # Sharpness — Laplacian variance of norm_to_uint8(pred) [on prediction]
        pred_uint8 = ((pred_np + 1) / 2 * 255).clip(0, 255).astype(np.float32)
        blur_map   = cv2.Laplacian(pred_uint8, cv2.CV_32F)
        sharp_vals.append(float(np.var(blur_map)))

    # SSIM — compute once over all accumulated slices, then split per slice
    ssim_all = ssim_fn.compute()   # tensor of shape (n_slices,) due to reduction="none"
    ssim_vals = ssim_all.cpu().numpy().tolist()
    ssim_fn.reset()

    return {
        "ssim":      ssim_vals,
        "psnr":      psnr_vals,
        "lpips":     lpips_vals,
        "sharpness": sharp_vals,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_PATH, "r") as hf:
        patients = sorted(hf[FIXED_MOD].keys())
    n_patients = len(patients)

    print(f"\n{'='*70}")
    print(f"IXI Ver4 SyN Registration  |  {n_patients} patients  |  {len(MODALITY_PAIRS)} modalities")
    print(f"Fixed: {FIXED_MOD}   |   Moving: {[m for m, _ in MODALITY_PAIRS]}")
    print(f"Workers: {NUM_WORKERS}   |   ITK threads/worker: {ITK_THREADS}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"{'='*70}\n")

    # ── Phase 1: Registration ─────────────────────────────────────────────────
    total = n_patients * len(MODALITY_PAIRS)
    tasks, idx = [], 1
    for patient_id in patients:
        for moving_mod, gt_mod in MODALITY_PAIRS:
            tasks.append((
                patient_id, H5_PATH, FIXED_MOD, moving_mod, gt_mod,
                str(RESULTS_DIR), idx, total, ITK_THREADS,
            ))
            idx += 1

    print(f"Phase 1: Running {total} SyN registrations ({NUM_WORKERS} workers)...")
    with Pool(NUM_WORKERS) as pool:
        reg_results = pool.map(register_one, tasks)

    n_done = sum(1 for r in reg_results if r[0])
    print(f"\nRegistration complete: {n_done}/{total} succeeded, {total - n_done} failed\n")

    # ── Phase 2: Metric computation ───────────────────────────────────────────
    print(f"Phase 2: Computing metrics on {DEVICE}...")
    print(f"{'='*70}")

    # keyed by moving_mod (T1_moved / T1_moved_5mm)
    all_metrics = {moving_mod: {"ssim": [], "psnr": [], "lpips": [], "sharpness": []}
                   for moving_mod, _ in MODALITY_PAIRS}
    per_patient = {}

    for success, patient_id, moving_mod, gt_mod in reg_results:
        if not success:
            continue
        warped_path = RESULTS_DIR / f"{patient_id}_{moving_mod}_warped.npy"
        gt_path     = RESULTS_DIR / f"{patient_id}_{gt_mod}_gt.npy"

        if not warped_path.exists() or not gt_path.exists():
            print(f"  Warning: missing files for {patient_id} {moving_mod}, skipping")
            continue

        warped_arr = np.load(str(warped_path))
        gt_arr     = np.load(str(gt_path))

        m = compute_metrics_for_patient(warped_arr, gt_arr, DEVICE)

        for k in m:
            all_metrics[moving_mod][k].extend(m[k])

        key = f"{patient_id}_{moving_mod}"
        per_patient[key] = {k: float(np.mean(v)) for k, v in m.items()}
        print(f"  {patient_id} | {moving_mod:14s}  "
              f"SSIM={np.mean(m['ssim']):.4f}  "
              f"PSNR={np.mean(m['psnr']):.4f}  "
              f"LPIPS={np.mean(m['lpips']):.4f}  "
              f"Sharp={np.mean(m['sharpness']):.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY  (SyN, IXI Ver4 test set, T2 fixed, GT=T1)")
    print(f"{'='*70}")

    summary = {}
    for moving_mod, mdict in all_metrics.items():
        if not mdict["ssim"]:
            continue
        ssim_a  = np.array(mdict["ssim"])
        psnr_a  = np.array(mdict["psnr"])
        lpips_a = np.array(mdict["lpips"])
        sharp_a = np.array(mdict["sharpness"])
        n = len(ssim_a)
        print(f"\n  {moving_mod}  ({n} slices, GT=T1)")
        print(f"    SSIM:      {ssim_a.mean():.4f}  ± {ssim_a.std():.4f}")
        print(f"    PSNR:      {psnr_a.mean():.4f}  ± {psnr_a.std():.4f}")
        print(f"    LPIPS:     {lpips_a.mean():.4f}  ± {lpips_a.std():.4f}")
        print(f"    Sharpness: {sharp_a.mean():.4f}  ± {sharp_a.std():.4f}")
        summary[moving_mod] = {
            "n_slices":       n,
            "ssim_mean":      float(ssim_a.mean()),  "ssim_std":      float(ssim_a.std()),
            "psnr_mean":      float(psnr_a.mean()),  "psnr_std":      float(psnr_a.std()),
            "lpips_mean":     float(lpips_a.mean()), "lpips_std":     float(lpips_a.std()),
            "sharpness_mean": float(sharp_a.mean()), "sharpness_std": float(sharp_a.std()),
        }
    print(f"\n{'='*70}")

    # ── Save ──────────────────────────────────────────────────────────────────
    metrics_txt  = SCRIPT_DIR / "metrics_syn_ver4_T1.txt"
    metrics_json = SCRIPT_DIR / "metrics_syn_ver4_T1.json"

    with open(metrics_txt, "w") as f:
        f.write("SyN Registration Metrics  (IXI Ver4 test set, T2 fixed, GT=T1)\n")
        f.write(f"{'='*60}\n")
        for moving_mod, s in summary.items():
            f.write(f"\n{moving_mod}  ({s['n_slices']} slices)\n")
            f.write(f"  SSIM:      {s['ssim_mean']:.4f}  +/- {s['ssim_std']:.4f}\n")
            f.write(f"  PSNR:      {s['psnr_mean']:.4f}  +/- {s['psnr_std']:.4f}\n")
            f.write(f"  LPIPS:     {s['lpips_mean']:.4f}  +/- {s['lpips_std']:.4f}\n")
            f.write(f"  Sharpness: {s['sharpness_mean']:.4f}  +/- {s['sharpness_std']:.4f}\n")

    with open(metrics_json, "w") as f:
        json.dump({"summary": summary, "per_patient": per_patient}, f, indent=2)

    print(f"\nSaved: {metrics_txt}")
    print(f"Saved: {metrics_json}")
    print(f"Saved per-patient .npy arrays: {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()

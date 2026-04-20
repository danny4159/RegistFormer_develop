#!/usr/bin/env python3
"""
SyN registration: T1_moved, PD_moved → T2 (fixed)
Dataset: IXI_Testset_T1_T2_PD_Registered_30Patient_MisalignSimul_Rotate5Translate5_20SliceOut.h5
Shape: (256, 256, 80) per patient, [-1, 1] normalized.
Saves warped + GT + fixed as .nii.gz. Computes SSIM, PSNR, LPIPS, Sharpness.
"""

import os, sys, json, h5py, tempfile, cv2
import numpy as np
import ants
import torch
from pathlib import Path
from multiprocessing import Pool

SCRIPT_DIR = Path(__file__).parent
H5_PATH = (
    "/SSD1_1TB/home/milab/daniel/03_Registformer_develop/RegistFormer_develop"
    "/data/IXI/test"
    "/IXI_Testset_T1_T2_PD_Registered_30Patient_MisalignSimul_Rotate5Translate5_20SliceOut.h5"
)
RESULTS_DIR = SCRIPT_DIR / "results_20sliceout"
FIXED_MOD = "T2"
MODALITY_PAIRS = [("T1_moved", "T1"), ("PD_moved", "PD")]
NUM_WORKERS = 8
ITK_THREADS = 4
SYN_TYPE = "SyN"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def normalize_robust(arr, pmin=1.0, pmax=99.0):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [pmin, pmax])
    arr = np.clip(arr, lo, hi)
    return arr - lo if hi - lo < 1e-8 else (arr - lo) / (hi - lo)


def register_one(args):
    patient_id, h5_path, fixed_mod, moving_mod, gt_mod, results_dir, idx, total, itk_threads = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)
    results_dir = Path(results_dir)

    warped_path = results_dir / f"{patient_id}_{moving_mod}_warped.nii.gz"
    gt_path     = results_dir / f"{patient_id}_{gt_mod}_gt.nii.gz"
    fixed_path  = results_dir / f"{patient_id}_{fixed_mod}.nii.gz"

    if warped_path.exists() and gt_path.exists():
        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:9s} -- skip")
        return True, patient_id, moving_mod, gt_mod

    try:
        with h5py.File(h5_path, "r") as hf:
            fixed_arr  = hf[fixed_mod][patient_id][:].astype(np.float32)
            moving_arr = hf[moving_mod][patient_id][:].astype(np.float32)
            gt_arr     = hf[gt_mod][patient_id][:].astype(np.float32)

        fixed_ants  = ants.from_numpy(fixed_arr)
        moving_ants = ants.from_numpy(moving_arr)
        fixed_norm  = ants.from_numpy(normalize_robust(fixed_arr))
        moving_norm = ants.from_numpy(normalize_robust(moving_arr))

        with tempfile.TemporaryDirectory() as tmp:
            reg = ants.registration(
                fixed=fixed_norm, moving=moving_norm,
                type_of_transform=SYN_TYPE,
                outprefix=os.path.join(tmp, "ants_"),
                verbose=False,
            )
            warped = ants.apply_transforms(
                fixed=fixed_ants, moving=moving_ants,
                transformlist=reg["fwdtransforms"], interpolator="linear",
            )

        ants.image_write(warped, str(warped_path))
        ants.image_write(ants.from_numpy(gt_arr), str(gt_path))
        if not fixed_path.exists():
            ants.image_write(fixed_ants, str(fixed_path))

        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:9s} -- done")
        return True, patient_id, moving_mod, gt_mod

    except Exception as e:
        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:9s} -- FAILED: {e}")
        return False, patient_id, moving_mod, gt_mod


def slice_metrics(pred_arr, gt_arr, device):
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    ssim_fn  = StructuralSimilarityIndexMeasure(reduction="none").to(device)
    psnr_fn  = PeakSignalNoiseRatio(data_range=2.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity().to(device)

    ssim_vals, psnr_vals, lpips_vals, sharp_vals = [], [], [], []
    for s in range(pred_arr.shape[-1]):
        p = torch.tensor(pred_arr[:, :, s], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        g = torch.tensor(gt_arr[:, :, s],   dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        ssim_fn.reset(); ssim_fn.update(g, p)
        ssim_vals.append(ssim_fn.compute().item())

        psnr_fn.reset(); psnr_fn.update(g, p)
        psnr_vals.append(psnr_fn.compute().item())

        lpips_fn.reset(); lpips_fn.update(g.repeat(1,3,1,1), p.repeat(1,3,1,1))
        lpips_vals.append(lpips_fn.compute().item())

        pred_u8 = ((pred_arr[:, :, s] + 1) / 2 * 255).clip(0, 255).astype(np.float32)
        sharp_vals.append(float(np.var(cv2.Laplacian(pred_u8, cv2.CV_32F))))

    return dict(ssim=ssim_vals, psnr=psnr_vals, lpips=lpips_vals, sharpness=sharp_vals)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_PATH, "r") as hf:
        patients = sorted(hf[FIXED_MOD].keys())

    total = len(patients) * len(MODALITY_PAIRS)
    print(f"\n{'='*70}")
    print(f"IXI SyN (20SliceOut)  |  {len(patients)} patients  |  {[m for m,_ in MODALITY_PAIRS]}")
    print(f"Fixed: {FIXED_MOD}   |   Workers: {NUM_WORKERS}   |   ITK threads/worker: {ITK_THREADS}")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*70}\n")

    # ── Phase 1: Registration ─────────────────────────────────────────────────
    tasks = [
        (patient_id, H5_PATH, FIXED_MOD, moving_mod, gt_mod,
         str(RESULTS_DIR), i * len(MODALITY_PAIRS) + j + 1, total, ITK_THREADS)
        for i, patient_id in enumerate(patients)
        for j, (moving_mod, gt_mod) in enumerate(MODALITY_PAIRS)
    ]

    print(f"Phase 1: {total} SyN registrations ({NUM_WORKERS} workers)...")
    with Pool(NUM_WORKERS) as pool:
        reg_results = pool.map(register_one, tasks)

    n_done = sum(1 for r in reg_results if r[0])
    print(f"\nRegistration: {n_done}/{total} succeeded\n")

    # ── Phase 2: Metrics ──────────────────────────────────────────────────────
    print(f"Phase 2: Computing metrics on {DEVICE}...")
    all_metrics = {gt_mod: dict(ssim=[], psnr=[], lpips=[], sharpness=[])
                   for _, gt_mod in MODALITY_PAIRS}
    per_patient = {}

    with h5py.File(H5_PATH, "r") as hf:
        for success, patient_id, moving_mod, gt_mod in reg_results:
            if not success:
                continue
            warped_path = RESULTS_DIR / f"{patient_id}_{moving_mod}_warped.nii.gz"
            gt_path     = RESULTS_DIR / f"{patient_id}_{gt_mod}_gt.nii.gz"
            if not warped_path.exists() or not gt_path.exists():
                continue

            warped_arr = ants.image_read(str(warped_path)).numpy()
            gt_arr     = ants.image_read(str(gt_path)).numpy()

            m = slice_metrics(warped_arr, gt_arr, DEVICE)
            for k in m:
                all_metrics[gt_mod][k].extend(m[k])

            per_patient[f"{patient_id}_{gt_mod}"] = {k: float(np.mean(v)) for k, v in m.items()}
            print(f"  {patient_id} | {gt_mod:3s}  SSIM={np.mean(m['ssim']):.4f}  "
                  f"PSNR={np.mean(m['psnr']):.4f}  LPIPS={np.mean(m['lpips']):.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY  (SyN, 20SliceOut, T2 fixed)")
    print(f"{'='*70}")

    summary = {}
    for gt_mod, mdict in all_metrics.items():
        if not mdict["ssim"]:
            continue
        s = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in mdict.items()}
        summary[gt_mod] = s
        n = len(mdict["ssim"])
        print(f"\n  {gt_mod}  ({n} slices)")
        for metric, stats in s.items():
            print(f"    {metric:10s}: {stats['mean']:.4f}  ± {stats['std']:.4f}")

    print(f"\n{'='*70}")

    txt_path  = SCRIPT_DIR / "metrics_syn_20sliceout.txt"
    json_path = SCRIPT_DIR / "metrics_syn_20sliceout.json"

    with open(txt_path, "w") as f:
        f.write("SyN Registration Metrics (IXI 20SliceOut, T2 fixed)\n")
        f.write(f"{'='*60}\n")
        for gt_mod, s in summary.items():
            f.write(f"\n{gt_mod}  ({len(all_metrics[gt_mod]['ssim'])} slices)\n")
            for metric, stats in s.items():
                f.write(f"  {metric:10s}: {stats['mean']:.4f}  +/- {stats['std']:.4f}\n")

    with open(json_path, "w") as f:
        json.dump({"summary": summary, "per_patient": per_patient}, f, indent=2)

    print(f"\nSaved: {txt_path}")
    print(f"Saved: {json_path}")
    print(f"NIfTI files: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()

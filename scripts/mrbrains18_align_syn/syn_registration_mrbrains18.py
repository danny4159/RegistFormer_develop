#!/usr/bin/env python3
"""
SyN registration: T1, IR → FLAIR (fixed)
Dataset: MRBrainS18 testset.h5, shape (256,256,48), [-1,1] normalized.

Metrics match base_module_AtoB.py (eval_on_align=False), 2D slice-wise:
  - GC       : norm_to_uint8(FLAIR) vs norm_to_uint8(warped)   [uint8, CUDA]
  - NMI      : flatten(norm_to_uint8(FLAIR)) vs flatten(norm_to_uint8(warped))  per-slice → mean
  - FID      : gray2rgb(norm_to_uint8(GT)) real=True / gray2rgb(norm_to_uint8(warped)) real=False
  - KID      : same as FID
  - Sharpness: norm_to_uint8(warped)  [uint8, CUDA]

norm_to_uint8 = lambda x: ((x+1)/2*255).to(torch.uint8)
gray2rgb      = lambda x: x.repeat(1,3,1,1)   (or torch.cat)
flatten_to_1d = lambda x: x.contiguous().view(-1)
"""

import os, sys, json, h5py, tempfile, cv2
import numpy as np
import ants
import torch
from pathlib import Path
from multiprocessing import Pool

# ── add project root to sys.path so src.metrics can be imported ──────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR  = Path(__file__).parent
H5_PATH     = "/SSD1_1TB/home/milab/daniel/03_Registformer_develop/RegistFormer_develop/data/MRBrainS18/test/testset.h5"
RESULTS_DIR = SCRIPT_DIR / "results"
FIXED_MOD   = "FLAIR"
MODALITY_PAIRS = [("T1", "T1"), ("IR", "IR")]   # (moving_key, gt_key)
NUM_WORKERS = 8
ITK_THREADS = 4
SYN_TYPE    = "SyN"
DEVICE      = "cuda:1" if torch.cuda.is_available() else "cpu"

# ── preprocessing lambdas (identical to base_module_AtoB.py) ─────────────────
norm_to_uint8 = lambda x: ((x + 1) / 2 * 255).to(torch.uint8)
gray2rgb      = lambda x: x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
flatten_to_1d = lambda x: x.contiguous().view(-1)


# ─── Registration worker ──────────────────────────────────────────────────────

def normalize_robust(arr, pmin=1.0, pmax=99.0):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [pmin, pmax])
    arr = np.clip(arr, lo, hi)
    return arr - lo if hi - lo < 1e-8 else (arr - lo) / (hi - lo)


def register_one(args):
    patient_id, h5_path, fixed_mod, moving_mod, results_dir, idx, total, itk_threads = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)
    results_dir = Path(results_dir)

    warped_path = results_dir / f"{patient_id}_{moving_mod}_warped.nii.gz"
    fixed_path  = results_dir / f"{patient_id}_{fixed_mod}.nii.gz"
    gt_path     = results_dir / f"{patient_id}_{moving_mod}_gt.nii.gz"

    if warped_path.exists():
        print(f"[{idx:2d}/{total}] {patient_id} | {moving_mod} -- skip")
        return True, patient_id, moving_mod

    try:
        with h5py.File(h5_path, "r") as hf:
            fixed_arr  = hf[fixed_mod][patient_id][:].astype(np.float32)
            moving_arr = hf[moving_mod][patient_id][:].astype(np.float32)

        fixed_ants  = ants.from_numpy(fixed_arr)
        moving_ants = ants.from_numpy(moving_arr)

        with tempfile.TemporaryDirectory() as tmp:
            reg = ants.registration(
                fixed=ants.from_numpy(normalize_robust(fixed_arr)),
                moving=ants.from_numpy(normalize_robust(moving_arr)),
                type_of_transform=SYN_TYPE,
                outprefix=os.path.join(tmp, "ants_"),
                verbose=False,
            )
            warped = ants.apply_transforms(
                fixed=fixed_ants, moving=moving_ants,
                transformlist=reg["fwdtransforms"], interpolator="linear",
            )

        ants.image_write(warped, str(warped_path))
        ants.image_write(ants.from_numpy(moving_arr), str(gt_path))
        if not fixed_path.exists():
            ants.image_write(fixed_ants, str(fixed_path))

        print(f"[{idx:2d}/{total}] {patient_id} | {moving_mod} -- done")
        return True, patient_id, moving_mod

    except Exception as e:
        print(f"[{idx:2d}/{total}] {patient_id} | {moving_mod} -- FAILED: {e}")
        return False, patient_id, moving_mod


# ─── Metric computation (2D slice-wise) ───────────────────────────────────────

def compute_all_metrics(warped_arr, gt_arr, fixed_arr, device,
                        gc_fn, nmi_fn, fid_fn, kid_fn, sharp_fn):
    """
    Process all slices of one patient's volume through the metrics.
    warped_arr, gt_arr, fixed_arr: (H, W, D) float32 [-1,1]
    Updates metric accumulators in-place; returns per-slice nmi_scores.
    """
    nmi_scores = []
    n_slices = warped_arr.shape[-1]

    for s in range(n_slices):
        # (1, 1, H, W) float32 tensors on device
        warped_s = torch.tensor(warped_arr[:, :, s], dtype=torch.float32,
                                device=device).unsqueeze(0).unsqueeze(0)
        gt_s     = torch.tensor(gt_arr[:, :, s],     dtype=torch.float32,
                                device=device).unsqueeze(0).unsqueeze(0)
        fixed_s  = torch.tensor(fixed_arr[:, :, s],  dtype=torch.float32,
                                device=device).unsqueeze(0).unsqueeze(0)

        # Preprocess → uint8 (same as base_module)
        warped_u8 = norm_to_uint8(warped_s)   # (1,1,H,W) uint8
        gt_u8     = norm_to_uint8(gt_s)
        fixed_u8  = norm_to_uint8(fixed_s)

        # GC: FLAIR vs warped
        gc_fn.update(fixed_u8, warped_u8)

        # NMI: FLAIR vs warped, per slice
        nmi_score = nmi_fn(flatten_to_1d(fixed_u8), flatten_to_1d(warped_u8))
        nmi_scores.append(nmi_score)

        # FID / KID: GT distribution vs warped distribution
        fid_fn.update(gray2rgb(gt_u8),     real=True)
        fid_fn.update(gray2rgb(warped_u8), real=False)
        kid_fn.update(gray2rgb(gt_u8),     real=True)
        kid_fn.update(gray2rgb(warped_u8), real=False)

        # Sharpness: warped image
        sharp_fn.update(warped_u8)

    return nmi_scores


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from src.metrics.gradient_correlation import GradientCorrelationMetric
    from src.metrics.sharpness import SharpnessMetric
    from torchmetrics.clustering import NormalizedMutualInfoScore
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_PATH, "r") as hf:
        patients = sorted(hf[FIXED_MOD].keys())

    total = len(patients) * len(MODALITY_PAIRS)
    print(f"\n{'='*70}")
    print(f"MRBrainS18 SyN  |  {len(patients)} patients  |  {[m for m,_ in MODALITY_PAIRS]}")
    print(f"Fixed: {FIXED_MOD}   |   Workers: {NUM_WORKERS}   |   Device: {DEVICE}")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*70}\n")

    # ── Phase 1: Registration ──────────────────────────────────────────────────
    tasks = [
        (patient_id, H5_PATH, FIXED_MOD, moving_mod,
         str(RESULTS_DIR), i * len(MODALITY_PAIRS) + j + 1, total, ITK_THREADS)
        for i, patient_id in enumerate(patients)
        for j, (moving_mod, _) in enumerate(MODALITY_PAIRS)
    ]

    print(f"Phase 1: {total} SyN registrations ({NUM_WORKERS} workers)...")
    with Pool(NUM_WORKERS) as pool:
        reg_results = pool.map(register_one, tasks)

    n_done = sum(1 for r in reg_results if r[0])
    print(f"\nRegistration: {n_done}/{total} succeeded\n")

    # ── Phase 2: Metrics (2D slice-wise, identical to base_module_AtoB) ───────
    print(f"Phase 2: Computing metrics on {DEVICE}...")

    summary = {}
    for moving_mod, gt_mod in MODALITY_PAIRS:
        print(f"\n  --- {moving_mod} ---")

        # Instantiate fresh metrics per modality (same as define_metrics())
        gc_fn    = GradientCorrelationMetric().to(DEVICE)
        nmi_fn   = NormalizedMutualInfoScore().to(DEVICE)
        fid_fn   = FrechetInceptionDistance().to(DEVICE)
        kid_fn   = KernelInceptionDistance(subset_size=2).to(DEVICE)
        sharp_fn = SharpnessMetric().to(DEVICE)

        all_nmi_scores = []
        per_patient_gc = []

        with h5py.File(H5_PATH, "r") as hf:
            for success, patient_id, mod in reg_results:
                if not success or mod != moving_mod:
                    continue

                warped_path = RESULTS_DIR / f"{patient_id}_{moving_mod}_warped.nii.gz"
                if not warped_path.exists():
                    print(f"  Warning: missing {warped_path.name}, skipping")
                    continue

                warped_arr = ants.image_read(str(warped_path)).numpy()
                gt_arr     = hf[gt_mod][patient_id][:].astype(np.float32)
                fixed_arr  = hf[FIXED_MOD][patient_id][:].astype(np.float32)

                nmi_scores = compute_all_metrics(
                    warped_arr, gt_arr, fixed_arr, DEVICE,
                    gc_fn, nmi_fn, fid_fn, kid_fn, sharp_fn,
                )
                all_nmi_scores.extend(nmi_scores)

                # Snapshot per-patient GC for reporting
                gc_now = gc_fn.correlations.mean().item() if gc_fn.correlations.numel() > 0 else float("nan")
                nmi_now = torch.mean(torch.stack(nmi_scores)).item() if nmi_scores else float("nan")
                print(f"    {patient_id}: GC={gc_now:.4f}  NMI={nmi_now:.4f}")

        # ── Compute final metrics ─────────────────────────────────────────────
        gc_val       = gc_fn.compute()
        nmi_val      = torch.mean(torch.stack(all_nmi_scores))
        fid_val      = fid_fn.compute()
        kid_mean, kid_std = kid_fn.compute()
        sharp_val    = sharp_fn.compute()

        # std for GC and NMI
        gc_std   = gc_fn.correlations.std() if gc_fn.correlations.numel() > 1 else torch.tensor(0.0)
        nmi_arr  = torch.stack(all_nmi_scores)
        nmi_std  = nmi_arr.std()
        sharp_std = sharp_fn.scores.std() if sharp_fn.scores.numel() > 1 else torch.tensor(0.0)

        n = len(all_nmi_scores)
        print(f"\n  {moving_mod}  ({n} slices)")
        print(f"    GC:        {gc_val.item():.4f}  ± {gc_std.item():.4f}")
        print(f"    NMI:       {nmi_val.item():.4f}  ± {nmi_std.item():.4f}")
        print(f"    FID:       {fid_val.item():.4f}")
        print(f"    KID:       {kid_mean.item():.4f}  ± {kid_std.item():.4f}")
        print(f"    Sharpness: {sharp_val.item():.4f}  ± {sharp_std.item():.4f}")

        summary[moving_mod] = {
            "n_slices": n,
            "GC":        {"mean": gc_val.item(),    "std": gc_std.item()},
            "NMI":       {"mean": nmi_val.item(),   "std": nmi_std.item()},
            "FID":       {"mean": fid_val.item()},
            "KID":       {"mean": kid_mean.item(),  "std": kid_std.item()},
            "Sharpness": {"mean": sharp_val.item(), "std": sharp_std.item()},
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    txt_path  = SCRIPT_DIR / "metrics_syn_mrbrains18.txt"
    json_path = SCRIPT_DIR / "metrics_syn_mrbrains18.json"

    with open(txt_path, "w") as f:
        f.write("SyN Registration Metrics  (MRBrainS18, FLAIR fixed)\n")
        f.write(f"{'='*60}\n")
        for mod, s in summary.items():
            f.write(f"\n{mod}  ({s['n_slices']} slices)\n")
            for metric, stats in s.items():
                if metric == "n_slices":
                    continue
                if "std" in stats:
                    f.write(f"  {metric:10s}: {stats['mean']:.4f}  +/- {stats['std']:.4f}\n")
                else:
                    f.write(f"  {metric:10s}: {stats['mean']:.4f}\n")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")
    print(f"NIfTI files: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()

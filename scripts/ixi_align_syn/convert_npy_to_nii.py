#!/usr/bin/env python3
"""
Convert .npy files in results/ to .nii.gz.
Also compute baseline (no-registration) metrics and Rigid-only registration metrics
for comparison with SyN, to diagnose performance gap.
"""

import os
import sys
import json
import h5py
import numpy as np
import ants
import torch
import cv2
from pathlib import Path
from multiprocessing import Pool
import tempfile

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
H5_PATH = (
    "/SSD1_1TB/home/milab/daniel/03_Registformer_develop/RegistFormer_develop"
    "/data/IXI/test"
    "/IXI_Testset_T1_T2_PD_MRA_Registered_30Patient_MisalignSimul_Rotate5Translate5_80SliceOut.h5"
)
FIXED_MOD = "T2"
MODALITY_PAIRS = [("T1_moved", "T1"), ("PD_moved", "PD"), ("MRA_moved", "MRA")]
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
ITK_THREADS = 4


# ─── npy → nii.gz ─────────────────────────────────────────────────────────────

def convert_npy_to_nii():
    npy_files = sorted(RESULTS_DIR.glob("*.npy"))
    print(f"Converting {len(npy_files)} .npy files to .nii.gz ...")
    for p in npy_files:
        out = p.with_suffix("").with_suffix(".nii.gz")
        if out.exists():
            continue
        arr = np.load(str(p)).astype(np.float32)
        img = ants.from_numpy(arr)
        ants.image_write(img, str(out))
    print(f"Done. Files in: {RESULTS_DIR}")


# ─── Per-slice metric helpers ─────────────────────────────────────────────────

def slice_metrics(pred_arr, gt_arr, device):
    """pred_arr, gt_arr: (H, W, D) float32 in [-1,1]. Returns per-slice dict."""
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


def agg(vals):
    a = np.array(vals)
    return f"{a.mean():.4f} ± {a.std():.4f}"


# ─── Rigid registration worker ────────────────────────────────────────────────

def rigid_register_one(args):
    patient_id, h5_path, fixed_mod, moving_mod, gt_mod, results_dir, idx, total = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(ITK_THREADS)

    results_dir = Path(results_dir)
    out_path = results_dir / f"{patient_id}_{moving_mod}_rigid_warped.npy"
    if out_path.exists():
        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:9s} rigid -- skip")
        return True, patient_id, moving_mod, gt_mod

    try:
        with h5py.File(h5_path, "r") as hf:
            fixed_arr  = hf[fixed_mod][patient_id][:]
            moving_arr = hf[moving_mod][patient_id][:]

        fixed_ants  = ants.from_numpy(fixed_arr.astype(np.float32))
        moving_ants = ants.from_numpy(moving_arr.astype(np.float32))

        def norm01(arr):
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip(arr, lo, hi)
            return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            reg = ants.registration(
                fixed=ants.from_numpy(norm01(fixed_arr)),
                moving=ants.from_numpy(norm01(moving_arr)),
                type_of_transform="Rigid",
                outprefix=os.path.join(tmp, "ants_"),
                verbose=False,
            )
            warped = ants.apply_transforms(
                fixed=fixed_ants, moving=moving_ants,
                transformlist=reg["fwdtransforms"], interpolator="linear",
            )
        np.save(str(out_path), warped.numpy())
        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:9s} rigid -- done")
        return True, patient_id, moving_mod, gt_mod
    except Exception as e:
        print(f"[{idx:3d}/{total}] {patient_id} | {moving_mod:9s} rigid -- FAILED: {e}")
        return False, patient_id, moving_mod, gt_mod


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Convert npy → nii.gz
    print(f"\n{'='*70}\nStep 1: Converting .npy → .nii.gz\n{'='*70}")
    convert_npy_to_nii()

    # 2. Run Rigid registrations for comparison
    print(f"\n{'='*70}\nStep 2: Rigid registration (for comparison with SyN)\n{'='*70}")
    with h5py.File(H5_PATH, "r") as hf:
        patients = sorted(hf[FIXED_MOD].keys())

    total = len(patients) * len(MODALITY_PAIRS)
    tasks = []
    for i, patient_id in enumerate(patients):
        for j, (moving_mod, gt_mod) in enumerate(MODALITY_PAIRS):
            idx = i * len(MODALITY_PAIRS) + j + 1
            tasks.append((patient_id, H5_PATH, FIXED_MOD, moving_mod, gt_mod,
                          str(RESULTS_DIR), idx, total))

    with Pool(NUM_WORKERS) as pool:
        rigid_results = pool.map(rigid_register_one, tasks)

    # 3. Compute metrics: Baseline(no-reg) | Rigid | SyN
    print(f"\n{'='*70}\nStep 3: Computing metrics on {DEVICE}\n{'='*70}")

    all_metrics = {
        gt_mod: {
            "baseline": dict(ssim=[], psnr=[], lpips=[], sharpness=[]),
            "rigid":    dict(ssim=[], psnr=[], lpips=[], sharpness=[]),
            "syn":      dict(ssim=[], psnr=[], lpips=[], sharpness=[]),
        }
        for _, gt_mod in MODALITY_PAIRS
    }

    with h5py.File(H5_PATH, "r") as hf:
        for patient_id in patients:
            for moving_mod, gt_mod in MODALITY_PAIRS:
                gt_arr     = np.load(str(RESULTS_DIR / f"{patient_id}_{gt_mod}_gt.npy"))
                syn_arr    = np.load(str(RESULTS_DIR / f"{patient_id}_{moving_mod}_warped.npy"))
                rigid_path = RESULTS_DIR / f"{patient_id}_{moving_mod}_rigid_warped.npy"

                # baseline: original misaligned (no registration)
                moved_arr = hf[moving_mod][patient_id][:]

                m_base  = slice_metrics(moved_arr, gt_arr, DEVICE)
                m_syn   = slice_metrics(syn_arr, gt_arr, DEVICE)

                for k in m_base:
                    all_metrics[gt_mod]["baseline"][k].extend(m_base[k])
                    all_metrics[gt_mod]["syn"][k].extend(m_syn[k])

                if rigid_path.exists():
                    rigid_arr = np.load(str(rigid_path))
                    m_rigid = slice_metrics(rigid_arr, gt_arr, DEVICE)
                    for k in m_rigid:
                        all_metrics[gt_mod]["rigid"][k].extend(m_rigid[k])

                print(f"  {patient_id} | {gt_mod}: "
                      f"baseline SSIM={np.mean(m_base['ssim']):.4f}  "
                      f"syn SSIM={np.mean(m_syn['ssim']):.4f}")

    # 4. Print & save summary
    print(f"\n{'='*70}")
    print("COMPARISON: No-Reg Baseline  |  Rigid  |  SyN")
    print(f"{'='*70}")

    summary = {}
    for gt_mod, methods in all_metrics.items():
        print(f"\n  {gt_mod}")
        summary[gt_mod] = {}
        for method, mdict in methods.items():
            if not mdict["ssim"]:
                continue
            n = len(mdict["ssim"])
            summary[gt_mod][method] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in mdict.items()
            }
            print(f"    [{method:8s}] ({n} slices)  "
                  f"SSIM={agg(mdict['ssim'])}  "
                  f"PSNR={agg(mdict['psnr'])}  "
                  f"LPIPS={agg(mdict['lpips'])}")

    # Save
    out_txt  = SCRIPT_DIR / "metrics_comparison.txt"
    out_json = SCRIPT_DIR / "metrics_comparison.json"

    with open(out_txt, "w") as f:
        f.write("Comparison: No-Reg Baseline | Rigid | SyN\n")
        f.write("(IXI test set, T2 fixed, 30 patients x 20 slices)\n")
        f.write(f"{'='*70}\n")
        for gt_mod, methods in summary.items():
            f.write(f"\n{gt_mod}\n")
            for method, mdict in methods.items():
                n_slices = len(all_metrics[gt_mod][method]["ssim"])
                f.write(f"  [{method:8s}] ({n_slices} slices)\n")
                for metric, stats in mdict.items():
                    f.write(f"    {metric:10s}: {stats['mean']:.4f} +/- {stats['std']:.4f}\n")

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_json}")
    print(f"NIfTI files in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

"""
Reference Sensitivity Test (검증 1)

같은 real_a에 대해 4가지 reference로 inference해서
모델이 reference를 얼마나 실제로 활용하는지 측정.

Usage:
    python scripts/ref_sensitivity_test.py --ckpt <checkpoint_path> --n_samples 20

결과 해석:
    diff_wrong도 작다     → 모델이 reference를 별로 안 씀. SIRM 효과 기대 낮음
    diff_wrong은 크고 diff_aug만 작다  → 이미 robust함. SIRM 필요성 작음
    diff_aug가 크고 크게 다름          → SIRM 방향 맞음. 구조/lambda 조정 필요
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.proposed_synthesis_module import ProposedSynthesisModule, scanner_like_aug

DATA_PATH = "/SSD1_1TB/home/milab/daniel/03_Registformer_develop/RegistFormer_develop/data/IXI/test/IXI_Testset_Ver3_T1_T2_PD_MRA_IOPdataset_30Patient_MisalignSimul_20Slice.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_batch(data_path, n_samples=20, ref_stack_size=5, slice_axis=2):
    """IOP test set에서 n_samples 슬라이스를 로드."""
    samples = []
    with h5py.File(data_path, "r") as f:
        patient_keys = list(f["T1"].keys())
        count = 0
        for key in patient_keys:
            vol_a = f["T1"][key][...]   # [H, W, S]
            vol_c = f["T2_moved"][key][...]  # [H, W, S] - misaligned ref
            n_slices = vol_a.shape[slice_axis]
            for sl in range(ref_stack_size // 2, n_slices - ref_stack_size // 2):
                # Source (real_a): single slice [1, H, W]
                real_a = torch.from_numpy(vol_a[:, :, sl]).unsqueeze(0).float()

                # Reference stack (real_b_ref): K-stack [K, H, W]
                half = ref_stack_size // 2
                stack = []
                for offset in range(-half, half + 1):
                    stack.append(vol_c[:, :, sl + offset])
                real_b_ref = torch.from_numpy(np.stack(stack, axis=0)).float()

                samples.append((real_a, real_b_ref))
                count += 1
                if count >= n_samples:
                    return samples
    return samples


@torch.no_grad()
def run_sensitivity(model, samples, device):
    """4가지 reference 조건에서 output diff 계산."""
    results = {
        "diff_aug":   [],
        "diff_wrong": [],
        "diff_zero":  [],
    }

    n = len(samples)
    real_a_list = [s[0] for s in samples]
    ref_list    = [s[1] for s in samples]

    for i, (real_a, ref) in enumerate(samples):
        real_a = real_a.unsqueeze(0).to(device)   # [1,1,H,W]
        ref    = ref.unsqueeze(0).to(device)        # [1,K,H,W]

        # (1) Original ref
        merged_orig  = torch.cat([real_a, ref], dim=1)
        fake_orig    = model.netG_A(merged_orig)

        # (2) Augmented ref
        ref_aug      = scanner_like_aug(ref)
        merged_aug   = torch.cat([real_a, ref_aug], dim=1)
        fake_aug_out = model.netG_A(merged_aug)

        # (3) Wrong (shuffled) ref — 다른 샘플의 ref 사용
        wrong_idx     = (i + 1) % n
        ref_wrong     = ref_list[wrong_idx].unsqueeze(0).to(device)
        merged_wrong  = torch.cat([real_a, ref_wrong], dim=1)
        fake_wrong    = model.netG_A(merged_wrong)

        # (4) Zero ref
        ref_zero     = torch.zeros_like(ref)
        merged_zero  = torch.cat([real_a, ref_zero], dim=1)
        fake_zero    = model.netG_A(merged_zero)

        diff_aug   = (fake_orig - fake_aug_out).abs().mean().item()
        diff_wrong = (fake_orig - fake_wrong).abs().mean().item()
        diff_zero  = (fake_orig - fake_zero).abs().mean().item()

        results["diff_aug"].append(diff_aug)
        results["diff_wrong"].append(diff_wrong)
        results["diff_zero"].append(diff_zero)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1:3d}/{n}]  aug={diff_aug:.5f}  wrong={diff_wrong:.5f}  zero={diff_zero:.5f}")

    return results


def summarize(results):
    print("\n" + "=" * 70)
    print("REFERENCE SENSITIVITY TEST — SUMMARY")
    print("=" * 70)
    for key, vals in results.items():
        arr = np.array(vals)
        print(f"  {key:14s}:  mean={arr.mean():.5f}  std={arr.std():.5f}  "
              f"min={arr.min():.5f}  max={arr.max():.5f}")

    aug   = np.mean(results["diff_aug"])
    wrong = np.mean(results["diff_wrong"])
    zero  = np.mean(results["diff_zero"])

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if wrong < 0.005:
        conclusion = "❌ 모델이 reference를 거의 무시함 → SIRM으로 개선 불가"
    elif aug < wrong * 0.3:
        conclusion = "⚠️  이미 augmentation에는 robust함 → SIRM 필요성 낮음"
    elif aug > wrong * 0.7:
        conclusion = "✅ aug에 민감 → SIRM 방향 맞음. lambda/structure 조정 여지 있음"
    else:
        conclusion = "⚠️  부분적으로 sensitive → SIRM 효과 제한적일 수 있음"

    print(f"  diff_aug/diff_wrong ratio: {aug/wrong:.3f}")
    print(f"  {conclusion}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--ref_stack_size", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)

    model = ProposedSynthesisModule.load_from_checkpoint(
        args.ckpt, map_location=DEVICE, strict=False
    )
    model.eval().to(DEVICE)
    print(f"Model loaded on {DEVICE}")

    print(f"\nLoading {args.n_samples} samples from IOP test set...")
    samples = load_batch(DATA_PATH, n_samples=args.n_samples, ref_stack_size=args.ref_stack_size)
    print(f"Loaded {len(samples)} samples\n")

    print("Running sensitivity test...")
    results = run_sensitivity(model, samples, DEVICE)
    summarize(results)


if __name__ == "__main__":
    main()

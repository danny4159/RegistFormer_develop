"""
V33 Protocol A+B — Penalty Distribution & Oracle-Low/High Analysis
No model needed. Pure data statistics.

Usage:
  source .venv/bin/activate && python scripts/v33_preflight_A.py
"""
import sys, random
sys.path.insert(0, ".")

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_H5 = (
    "data/IXI/train/"
    "IXI_Trainset_Ver4_T1_T2_PD_MRA_Registered_30Patient_MisalignSimul_"
    "Rotate5Translate5_NonLinearSig13_Mag650_CubicInterpolate_BackGroundOut"
    "Feather_SliceThicknessSimul_20Slice.h5"
)
REF_STACK_K       = 3
CROP              = 128
N_SAMPLES         = 500
ORACLE_DOWNSAMPLE = 4
MISMATCH_TAU      = 0.08
P_MIN             = 0.01
P_MAX             = 0.05
SOFTMIN_TAU       = 0.3
FIXED_PENALTY     = 0.05   # original baseline
SEED              = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED); torch.manual_seed(SEED)

def lowpass(x, factor=4):
    h, w = x.shape[-2:]
    xs = F.interpolate(x, size=(max(1,h//factor), max(1,w//factor)), mode="bilinear", align_corners=False)
    return F.interpolate(xs, size=(h,w), mode="bilinear", align_corners=False)

def to_t(arr, crop=CROP):
    t = torch.from_numpy(arr.astype(np.float32))
    mn, mx = t.min(), t.max()
    if mx > mn: t = (t - mn) / (mx - mn) * 2 - 1
    t = t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    H, W = t.shape[-2:]
    y = random.randint(0, H - crop); x = random.randint(0, W - crop)
    return t[..., y:y+crop, x:x+crop]

def ssim_simple(a, b):
    mu_a,mu_b = a.mean(),b.mean()
    s_a,s_b   = a.std(),b.std()
    cov = ((a-mu_a)*(b-mu_b)).mean()
    C1,C2 = 0.01**2, 0.03**2
    return float(((2*mu_a*mu_b+C1)*(2*cov+C2)) / ((mu_a**2+mu_b**2+C1)*(s_a**2+s_b**2+C2)))

def softmin_center_prob(cx_losses_k, penalty_base, tau=SOFTMIN_TAU, K=REF_STACK_K):
    """cx_losses_k: list of K scalars; return center probability"""
    center = K // 2
    losses = torch.tensor([cx + abs(i-center)*penalty_base for i,cx in enumerate(cx_losses_k)])
    probs  = F.softmax(-losses / tau, dim=0)
    return float(probs[center])

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
samples = []
with h5py.File(TRAIN_H5, "r") as f:
    patients = list(f["T1"].keys()); random.shuffle(patients)
    for pid in patients:
        t2v = f["T2"][pid][()]; t1v = f["T1"][pid][()]
        if t2v.ndim == 3 and t2v.shape[2] < t2v.shape[0]:
            t2v = t2v.transpose(2,0,1); t1v = t1v.transpose(2,0,1)
        S = t2v.shape[0]; c = REF_STACK_K // 2
        for s in range(c, S-c):
            samples.append((t2v[s], t1v[s], np.stack([t1v[s+i-c] for i in range(REF_STACK_K)], axis=0)))
            if len(samples) >= N_SAMPLES: break
        if len(samples) >= N_SAMPLES: break
print(f"  {len(samples)} samples loaded")

# ── compute per-sample stats ──────────────────────────────────────────────────
records = []  # dict per sample

for idx, (t2_sl, t1_c, t1_stack) in enumerate(samples):
    gt   = to_t(t1_c)                                          # [1,1,C,C]
    refs = torch.cat([to_t(t1_stack[k]) for k in range(REF_STACK_K)], dim=1)  # [1,K,C,C]

    center = REF_STACK_K // 2

    # V33 penalty
    ref_lp  = lowpass(refs[:, center:center+1], ORACLE_DOWNSAMPLE)
    gt_lp   = lowpass(gt, ORACLE_DOWNSAMPLE)
    mismatch    = float((ref_lp - gt_lp).abs().flatten(1).mean())
    reliability = float(np.clip(1.0 - mismatch / max(MISMATCH_TAU, 1e-6), 0.0, 1.0))
    penalty_v33 = P_MIN + (P_MAX - P_MIN) * reliability

    # ref_center vs GT SSIM (for oracle-low/high split)
    ref_center_np = refs[0, center].numpy()
    gt_np         = gt[0, 0].numpy()
    ref_gt_ssim   = ssim_simple(torch.tensor(ref_center_np), torch.tensor(gt_np))

    # soft-min center_prob: we use uniform CX (can't compute real CX without model)
    # Instead show theoretical center_prob at uniform CX losses = 0
    cp_original = softmin_center_prob([0.0]*REF_STACK_K, FIXED_PENALTY)
    cp_v33      = softmin_center_prob([0.0]*REF_STACK_K, penalty_v33)

    records.append({
        "mismatch":      mismatch,
        "reliability":   reliability,
        "penalty_v33":   penalty_v33,
        "ref_gt_ssim":   ref_gt_ssim,
        "cp_original":   cp_original,
        "cp_v33":        cp_v33,
        "penalty_delta": penalty_v33 - FIXED_PENALTY,
    })

    if (idx+1) % 100 == 0:
        mis  = np.mean([r["mismatch"]    for r in records])
        rel  = np.mean([r["reliability"] for r in records])
        pen  = np.mean([r["penalty_v33"] for r in records])
        print(f"  [{idx+1:4d}/{len(samples)}]  mismatch={mis:.4f}  reliability={rel:.3f}  penalty={pen:.4f}")

# ── split oracle-low / high ───────────────────────────────────────────────────
ssims  = [r["ref_gt_ssim"] for r in records]
thresh_lo = np.percentile(ssims, 20)
thresh_hi = np.percentile(ssims, 80)

low  = [r for r in records if r["ref_gt_ssim"] <= thresh_lo]
high = [r for r in records if r["ref_gt_ssim"] >= thresh_hi]
all_ = records

def stats(lst, key):
    vals = [r[key] for r in lst]
    return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

# ── report ────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("V33 Protocol A — Penalty Distribution Audit")
print("=" * 65)
print(f"  N={len(all_)}  K={REF_STACK_K}  tau={MISMATCH_TAU}  p_min={P_MIN}  p_max={P_MAX}")
print(f"  oracle-low  threshold (SSIM ≤{thresh_lo:.3f}): N={len(low)}")
print(f"  oracle-high threshold (SSIM ≥{thresh_hi:.3f}): N={len(high)}")
print()

print("  ── Overall penalty distribution ──────────────────────────")
for key, label in [("mismatch","mismatch"), ("reliability","reliability"),
                   ("penalty_v33","penalty_V33"), ("penalty_delta","penalty delta vs orig")]:
    mn, sd, mi, mx = stats(all_, key)
    print(f"  {label:28s}: mean={mn:.4f}  std={sd:.4f}  [{mi:.4f}, {mx:.4f}]")
print()

print("  ── Oracle-low vs Oracle-high comparison ──────────────────")
header = f"  {'metric':26s}  {'oracle-low':>12}  {'oracle-high':>12}  {'diff':>10}  판정"
print(header)
print("  " + "-"*63)

def cmp(key, label, want_low_smaller=True):
    lo_m = np.mean([r[key] for r in low])
    hi_m = np.mean([r[key] for r in high])
    diff = lo_m - hi_m
    ok   = (diff < 0) if want_low_smaller else (diff > 0)
    mark = "✓" if ok else "✗"
    print(f"  {label:26s}  {lo_m:>12.4f}  {hi_m:>12.4f}  {diff:>+10.4f}  {mark}")

cmp("mismatch",    "mismatch",        want_low_smaller=False)  # low should have higher mismatch
cmp("reliability", "reliability",     want_low_smaller=True)
cmp("penalty_v33", "penalty_V33",     want_low_smaller=True)
cmp("cp_v33",      "center_prob V33", want_low_smaller=True)
print()

# ── judgement ─────────────────────────────────────────────────────────────────
print("  ── Judgement ──────────────────────────────────────────────")
all_pen_mn = np.mean([r["penalty_v33"] for r in all_])
lo_pen_mn  = np.mean([r["penalty_v33"] for r in low])
hi_pen_mn  = np.mean([r["penalty_v33"] for r in high])
pen_diff   = hi_pen_mn - lo_pen_mn
all_mis_mn = np.mean([r["mismatch"]    for r in all_])
pen_range  = np.max([r["penalty_v33"] for r in all_]) - np.min([r["penalty_v33"] for r in all_])

viable = True

if all_pen_mn > P_MAX - 0.005:
    print(f"  ✗ penalty 평균 {all_pen_mn:.4f} ≈ p_max → adaptive 거의 작동 안 함. tau 낮추거나 p_range 넓혀야")
    viable = False
elif all_pen_mn < P_MIN + 0.005:
    print(f"  ✗ penalty 평균 {all_pen_mn:.4f} ≈ p_min → center prior 거의 없음. tau 높여야")
    viable = False
else:
    print(f"  ✓ penalty 평균 {all_pen_mn:.4f} → 0.01~0.05 범위 안")

if pen_range < 0.005:
    print(f"  ✗ penalty range {pen_range:.4f} → 거의 고정됨. adaptive 효과 없음")
    viable = False
else:
    print(f"  ✓ penalty range {pen_range:.4f} → 실제로 움직임")

if pen_diff < 0.005:
    print(f"  ✗ oracle-low vs high penalty diff {pen_diff:.4f} < 0.005 → 분리 안 됨")
    viable = False
else:
    print(f"  ✓ oracle-low vs high penalty diff {pen_diff:+.4f} → 분리됨")

print()
if viable:
    print("  → PROCEED to Protocol D (micro-training)")
else:
    print("  → CAUTION: tau/penalty 파라미터 조정 필요")
    if all_mis_mn < MISMATCH_TAU * 0.5:
        print(f"     mismatch 평균({all_mis_mn:.4f}) << tau({MISMATCH_TAU}) → tau를 {all_mis_mn*1.5:.3f} 근처로 낮추세요")
    elif all_mis_mn > MISMATCH_TAU * 2.0:
        print(f"     mismatch 평균({all_mis_mn:.4f}) >> tau({MISMATCH_TAU}) → tau를 {all_mis_mn*0.7:.3f} 근처로 높이세요")
print("=" * 65)

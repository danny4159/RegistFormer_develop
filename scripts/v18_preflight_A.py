"""
V18 Protocol A — Oracle Posterior Audit
No training. Checks whether p* has meaningful non-center selection.

Usage:
  source .venv/bin/activate && python scripts/v18_preflight_A.py
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
REF_STACK_K   = 3
CROP          = 128
N_SAMPLES     = 600
ORACLE_FACTOR = 4
ORACLE_TAU    = 0.05
ORACLE_EDGE_W = 0.20
CENTER_PEN    = 0.02
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED); torch.manual_seed(SEED)


# ── helpers ───────────────────────────────────────────────────────────────────
def sobel_mag(x):
    B, C, H, W = x.shape
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3).repeat(C,1,1,1)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3).repeat(C,1,1,1)
    xp = F.pad(x, (1,1,1,1), mode="reflect")
    return torch.sqrt(F.conv2d(xp, kx, groups=C)**2 + F.conv2d(xp, ky, groups=C)**2 + 1e-6)

def lowpass(x, factor=4):
    h, w = x.shape[-2:]
    xs = F.interpolate(x, size=(max(1,h//factor), max(1,w//factor)), mode="bilinear", align_corners=False)
    return F.interpolate(xs, size=(h,w), mode="bilinear", align_corners=False)

def oracle_posterior(ref_stack, gt, G=8):
    B, K, H, W = ref_stack.shape
    center = K // 2
    with torch.no_grad():
        gt_lp   = lowpass(gt, ORACLE_FACTOR)
        gt_edge = sobel_mag(gt_lp)
        dist_list = []
        for i in range(K):
            r_lp   = lowpass(ref_stack[:, i:i+1], ORACLE_FACTOR)
            r_edge = sobel_mag(r_lp)
            d = (r_lp - gt_lp).abs() + ORACLE_EDGE_W * (r_edge - gt_edge).abs()
            d_tile = F.adaptive_avg_pool2d(d, (G, G))
            if CENTER_PEN > 0:
                d_tile = d_tile + CENTER_PEN * abs(i - center)
            dist_list.append(d_tile)
        dist   = torch.cat(dist_list, dim=1)
        p_star = F.softmax(-dist / max(ORACLE_TAU, 1e-6), dim=1)
    return p_star, dist

def ssim_simple(a, b):
    mu_a = a.mean(); mu_b = b.mean()
    s_a  = a.std();  s_b  = b.std()
    cov  = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return float(((2*mu_a*mu_b + C1)*(2*cov + C2)) / ((mu_a**2 + mu_b**2 + C1)*(s_a**2 + s_b**2 + C2)))

def rand_crop(arr, crop=CROP):
    H, W = arr.shape[-2:]
    y = random.randint(0, H - crop)
    x = random.randint(0, W - crop)
    return arr[..., y:y+crop, x:x+crop]

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
samples = []  # list of (T2_crop, T1_center_crop, T1_stack_crop)

with h5py.File(TRAIN_H5, "r") as f:
    t2_key  = "T2"
    t1_key  = "T1"
    patients = list(f[t1_key].keys())
    random.shuffle(patients)

    for pid in patients:
        t2_vol = f[t2_key][pid][()]  # (H, W, slices) or (slices, H, W)
        t1_vol = f[t1_key][pid][()]

        # normalise shape to (S, H, W)
        if t2_vol.ndim == 3 and t2_vol.shape[2] < t2_vol.shape[0]:
            t2_vol = t2_vol.transpose(2, 0, 1)
            t1_vol = t1_vol.transpose(2, 0, 1)

        S = t2_vol.shape[0]
        center = REF_STACK_K // 2

        for s in range(center, S - center):
            t2_sl   = t2_vol[s]                              # source
            t1_c    = t1_vol[s]                              # GT center
            t1_stack = np.stack([t1_vol[s + i - center] for i in range(REF_STACK_K)], axis=0)  # (K,H,W)

            samples.append((t2_sl, t1_c, t1_stack))
            if len(samples) >= N_SAMPLES:
                break
        if len(samples) >= N_SAMPLES:
            break

print(f"  Loaded {len(samples)} samples")

# ── normalise to [-1,1] ───────────────────────────────────────────────────────
def to_tensor_norm(arr):
    t = torch.from_numpy(arr.astype(np.float32))
    mn, mx = t.min(), t.max()
    if mx > mn:
        t = (t - mn) / (mx - mn) * 2 - 1
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,H,W]

# ── statistics accumulators ───────────────────────────────────────────────────
center_top1_cnt   = 0
neighbor_top1_cnt = 0
entropy_list      = []
oracle_ssim_list  = []
center_ssim_list  = []
oracle_l1_list    = []
center_l1_list    = []
total             = 0

print(f"\nRunning oracle posterior on {len(samples)} samples...")
print(f"  K={REF_STACK_K}  tau={ORACLE_TAU}  edge_w={ORACLE_EDGE_W}  center_pen={CENTER_PEN}")
print()

K      = REF_STACK_K
center = K // 2
G      = 8

for idx, (t2_sl, t1_c, t1_stack) in enumerate(samples):
    # crop
    src   = rand_crop(to_tensor_norm(t2_sl))              # [1,1,C,C]
    gt    = rand_crop(to_tensor_norm(t1_c))               # [1,1,C,C]
    refs  = torch.cat([rand_crop(to_tensor_norm(t1_stack[k])) for k in range(K)], dim=1)  # [1,K,C,C]

    p_star, dist = oracle_posterior(refs, gt, G=G)        # [1,K,G,G]

    # tile-wise top1
    top1 = torch.argmax(p_star, dim=1)  # [1,G,G]
    is_center = (top1 == center).float()
    c_rate    = is_center.mean().item()
    center_top1_cnt   += c_rate
    neighbor_top1_cnt += (1 - c_rate)

    # entropy
    ent = -(p_star * torch.log(p_star + 1e-8)).sum(dim=1).mean().item()
    entropy_list.append(ent)

    # oracle-selected ref: weighted sum of ref slices
    p_up = F.interpolate(p_star, size=(CROP, CROP), mode="bilinear", align_corners=False)  # [1,K,C,C]
    oracle_ref = (refs * p_up).sum(dim=1, keepdim=True)       # [1,1,C,C]
    center_ref = refs[:, center:center+1]                     # [1,1,C,C]

    # metrics vs GT
    gt_np          = gt.squeeze().cpu().numpy()
    oracle_ref_np  = oracle_ref.squeeze().cpu().detach().numpy()
    center_ref_np  = center_ref.squeeze().cpu().numpy()

    oracle_ssim_list.append(ssim_simple(torch.tensor(oracle_ref_np), torch.tensor(gt_np)))
    center_ssim_list.append(ssim_simple(torch.tensor(center_ref_np), torch.tensor(gt_np)))
    oracle_l1_list.append(float(np.abs(oracle_ref_np - gt_np).mean()))
    center_l1_list.append(float(np.abs(center_ref_np - gt_np).mean()))

    total += 1
    if (idx + 1) % 100 == 0:
        print(f"  [{idx+1:4d}/{len(samples)}]  "
              f"center_top1={center_top1_cnt/total:.3f}  "
              f"entropy={np.mean(entropy_list):.4f}  "
              f"oracle_ssim={np.mean(oracle_ssim_list):.4f}  "
              f"center_ssim={np.mean(center_ssim_list):.4f}")

# ── report ────────────────────────────────────────────────────────────────────
center_rate   = center_top1_cnt   / total
neighbor_rate = neighbor_top1_cnt / total
mean_entropy  = np.mean(entropy_list)
max_entropy   = float(np.log(K))

oracle_ssim   = np.mean(oracle_ssim_list)
center_ssim   = np.mean(center_ssim_list)
oracle_l1     = np.mean(oracle_l1_list)
center_l1     = np.mean(center_l1_list)
ssim_gain     = oracle_ssim - center_ssim
l1_gain       = center_l1 - oracle_l1  # positive = oracle better

print()
print("=" * 60)
print("V18 Protocol A — Oracle Posterior Audit")
print("=" * 60)
print(f"  N samples         : {total}")
print(f"  K (ref slices)    : {K}  (center={center})")
print()
print("  ── p* statistics ──────────────────────────────────────")
print(f"  center_top1 rate  : {center_rate:.3f}  ({center_rate*100:.1f}%)")
print(f"  neighbor_top1 rate: {neighbor_rate:.3f}  ({neighbor_rate*100:.1f}%)")
print(f"  p* entropy (mean) : {mean_entropy:.4f}  (max={max_entropy:.4f})")
print(f"  entropy ratio     : {mean_entropy/max_entropy:.3f}")
print()
print("  ── oracle-selected ref vs center ref ───────────────────")
print(f"  center ref SSIM   : {center_ssim:.4f}")
print(f"  oracle ref SSIM   : {oracle_ssim:.4f}  (gain={ssim_gain:+.4f})")
print(f"  center ref L1     : {center_l1:.4f}")
print(f"  oracle ref L1     : {oracle_l1:.4f}  (gain={l1_gain:+.4f})")
print()
print("  ── Judgement ───────────────────────────────────────────")

viable = True
if neighbor_rate < 0.05:
    print(f"  ✗ neighbor_top1 {neighbor_rate*100:.1f}% < 5%  → V18 effect likely weak")
    viable = False
elif neighbor_rate < 0.10:
    print(f"  △ neighbor_top1 {neighbor_rate*100:.1f}%  → marginal")
else:
    print(f"  ✓ neighbor_top1 {neighbor_rate*100:.1f}%  → meaningful variation in p*")

if ssim_gain > 0.002:
    print(f"  ✓ oracle SSIM gain {ssim_gain:+.4f}  → strong signal for V18")
elif ssim_gain > 0:
    print(f"  △ oracle SSIM gain {ssim_gain:+.4f}  → weak signal")
else:
    print(f"  ✗ oracle SSIM gain {ssim_gain:+.4f}  → no benefit from neighbor slices")
    viable = False

if mean_entropy / max_entropy > 0.3:
    print(f"  ✓ p* entropy ratio {mean_entropy/max_entropy:.3f}  → p* is not too hard")
else:
    print(f"  △ p* entropy ratio {mean_entropy/max_entropy:.3f}  → p* is very peaked, consider raising tau")

print()
if viable:
    print("  → PROCEED to Protocol B (selector training)")
else:
    print("  → CAUTION: V18 may have limited effect on this data")
print("=" * 60)

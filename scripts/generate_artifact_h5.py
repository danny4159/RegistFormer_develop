#!/usr/bin/env python3
"""
Apply MRI artifact simulation to _moved modalities for 10 randomly selected test patients.

Input  : IXI test 80-slice H5 (30 patients)
Output : IXI_Testset_Ver4_T1_T2_PD_MRA_ArtifactSimul_to_moved_10Patient_80Slice.h5

Groups saved per patient:
  T1, T2, PD, MRA                                  (originals, copied as-is)
  T1_moved, T2_moved, PD_moved, MRA_moved          (moved, copied as-is)
  {mod}_moved_gibbs_{weak|medium|strong}            (3 levels)
  {mod}_moved_kspike_{weak|medium|strong}           (3 levels)
  {mod}_moved_motion_v2d_{weak|medium|strong}       (3 levels)
  {mod}_moved_ghost_{weak|medium|strong}            (3 levels)
  → 8 + 4×12 = 56 groups total
"""

import h5py
import numpy as np
import torch
import torchio as tio
from monai.transforms import RandGibbsNoise, RandKSpaceSpikeNoise
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import time

IN_H5  = Path(
    "/SSD5_8TB/Daniel/Daniel_ssd2/RegistFormer/data/IXI/test/"
    "IXI_Testset_Ver4_T1_T2_PD_MRA_Registered_30Patient_MisalignSimul_"
    "Rotate5Translate5_NonLinearSig13_Mag650_CubicInterpolate_"
    "BackGroundOutFeather_SliceThicknessSimul_80Slice.h5"
)
OUT_H5 = Path(
    "/SSD5_8TB/Daniel/Daniel_ssd2/RegistFormer/data/IXI/test/"
    "IXI_Testset_Ver4_T1_T2_PD_MRA_ArtifactSimul_to_moved_10Patient_80Slice.h5"
)

SEED       = 42
MODALITIES = ["T1", "T2", "PD", "MRA"]
N_SELECT   = 10
SPACING    = 1.0   # 1×1×1 mm isotropic after preprocessing

# ── select 10 patients (reproducible) ────────────────────────────────────────
with h5py.File(IN_H5, "r") as f:
    all_patients = sorted(f["T1"].keys())

rng_sel  = np.random.default_rng(SEED)
selected = sorted(rng_sel.choice(all_patients, size=N_SELECT, replace=False).tolist())
print(f"Selected patients ({N_SELECT}):")
for p in selected:
    print(f"  {p}")
print()

# ── artifact helpers ──────────────────────────────────────────────────────────

def apply_gibbs(vol, alpha):
    t = RandGibbsNoise(prob=1.0, alpha=(alpha, alpha))
    t.set_random_state(seed=SEED)
    return t(torch.from_numpy(vol[np.newaxis])).numpy()[0].astype(np.float32)

def apply_kspike(vol_01, irange):
    t = RandKSpaceSpikeNoise(prob=1.0, intensity_range=irange, channel_wise=False)
    t.set_random_state(seed=SEED)
    out_01 = t(torch.from_numpy(vol_01[np.newaxis])).numpy()[0]
    return (out_01 * 2.0 - 1.0).astype(np.float32)

def apply_ghost(vol_01, num_ghosts, intensity):
    torch.manual_seed(SEED)
    img    = tio.ScalarImage(tensor=torch.from_numpy(vol_01[np.newaxis]).float())
    t      = tio.RandomGhosting(num_ghosts=num_ghosts, axes=(0, 1, 2),
                                intensity=intensity, restore=0.02, p=1.0)
    out_01 = t(img).numpy()[0].astype(np.float32)
    return (out_01 * 2.0 - 1.0).astype(np.float32)

# view2D motion
def fft2c(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))

def phase_ramp_2d(shape, dx_pix, dy_pix):
    h, w = shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    return np.exp(-2j * np.pi * (fy[:, None] * dy_pix + fx[None, :] * dx_pix))

def normalize_curve(x):
    x = x - np.mean(x)
    m = np.max(np.abs(x))
    return x / m if m > 1e-8 else np.zeros_like(x)

def make_trajectory(n, amplitude, motion_type, rng):
    if amplitude == 0:
        return np.zeros(n, dtype=np.float32)
    if motion_type == "sudden":
        traj = np.zeros(n, dtype=np.float32)
        n_events = rng.integers(2, max(3, n // 30) + 1)
        for _ in range(n_events):
            c = rng.integers(0, n)
            w = rng.integers(1, max(2, n // 60 + 1))
            v = rng.uniform(-amplitude, amplitude)
            s, e = max(0, c - w), min(n, c + w + 1)
            win = np.hanning(e - s) if (e - s) > 1 else np.ones(1)
            traj[s:e] += v * win
        return np.clip(traj, -amplitude, amplitude).astype(np.float32)
    elif motion_type == "periodic":
        t = np.linspace(0, n * 0.06, n)
        return (amplitude * np.sin(2 * np.pi * rng.uniform(0.2, 0.7) * t)).astype(np.float32)
    else:
        walk = np.cumsum(rng.normal(0, 1, n))
        return (amplitude * normalize_curve(
            gaussian_filter1d(walk, sigma=max(1, n // 35))
        )).astype(np.float32)

def corrupt_slice(img2d, tx_pix, ty_pix, rot_deg, bg=0.0):
    h, w    = img2d.shape
    clean_k = fft2c(img2d)
    corr_k  = np.zeros_like(clean_k, dtype=np.complex64)
    for i in range(h):
        ang, dx, dy = float(rot_deg[i]), float(tx_pix[i]), float(ty_pix[i])
        k = fft2c(ndi_rotate(img2d, ang, reshape=False, order=1,
                              mode="constant", cval=bg,
                              prefilter=False)) if abs(ang) > 1e-6 else clean_k
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            k = k * phase_ramp_2d((h, w), dx, dy)
        corr_k[i, :] = k[i, :]
    return np.abs(ifft2c(corr_k)).astype(np.float32)

def apply_motion_v2d(vol, trans_mm, rot_deg_val, motion_type):
    rng    = np.random.default_rng(SEED)
    H      = vol.shape[0]
    tx_pix = make_trajectory(H, trans_mm / SPACING, motion_type, rng)
    ty_pix = make_trajectory(H, trans_mm / SPACING, motion_type, rng)
    rot    = make_trajectory(H, rot_deg_val,         motion_type, rng)
    out    = np.zeros_like(vol)
    for z in range(vol.shape[2]):
        sl = vol[:, :, z]
        if np.max(np.abs(sl)) < 1e-8:
            out[:, :, z] = sl
            continue
        sl_01        = (sl + 1.0) / 2.0
        bg_01        = float(np.percentile(sl_01, 1))
        out[:, :, z] = corrupt_slice(sl_01, tx_pix, ty_pix, rot, bg_01) * 2.0 - 1.0
    return out

# ── artifact presets ──────────────────────────────────────────────────────────
GIBBS_PRESETS  = [("weak", 0.70), ("medium", 0.73), ("strong", 0.75)]
KSPIKE_PRESETS = [("weak", (9.3, 9.8)), ("medium", (10, 11)), ("strong", (11, 12))]
MOTION_PRESETS = [
    ("weak",   dict(trans_mm=3.5,  rot_deg=3.0, motion_type="sudden")),
    ("medium", dict(trans_mm=11.0, rot_deg=8.0, motion_type="sudden")),
    ("strong", dict(trans_mm=1.5,  rot_deg=1.2, motion_type="periodic")),
]
GHOST_PRESETS  = [
    ("weak",   (4, (0.30, 0.40))),
    ("medium", (6, (0.50, 0.60))),
    ("strong", (8, (0.65, 0.75))),
]

# ── main ──────────────────────────────────────────────────────────────────────
t_start = time.time()

with h5py.File(IN_H5, "r") as f_in, h5py.File(OUT_H5, "w") as f_out:

    for p_idx, pid in enumerate(selected):
        t_p = time.time()
        print(f"[{p_idx+1}/{N_SELECT}] {pid}")

        # copy originals & moved
        for mod in MODALITIES:
            for key in [mod, f"{mod}_moved"]:
                vol = f_in[key][pid][:]
                f_out.require_group(key).create_dataset(
                    pid, data=vol, compression="gzip", compression_opts=4)

        # apply artifacts to each _moved
        for mod in MODALITIES:
            vol    = f_in[f"{mod}_moved"][pid][:]
            vol_01 = (vol + 1.0) / 2.0

            for level, alpha in GIBBS_PRESETS:
                out = apply_gibbs(vol, alpha)
                f_out.require_group(f"{mod}_moved_gibbs_{level}").create_dataset(
                    pid, data=out, compression="gzip", compression_opts=4)

            for level, irange in KSPIKE_PRESETS:
                out = apply_kspike(vol_01, irange)
                f_out.require_group(f"{mod}_moved_kspike_{level}").create_dataset(
                    pid, data=out, compression="gzip", compression_opts=4)

            for level, p in MOTION_PRESETS:
                t1  = time.time()
                out = apply_motion_v2d(vol, p["trans_mm"], p["rot_deg"], p["motion_type"])
                f_out.require_group(f"{mod}_moved_motion_v2d_{level}").create_dataset(
                    pid, data=out, compression="gzip", compression_opts=4)
                print(f"  {mod}_moved_motion_v2d_{level}: {time.time()-t1:.1f}s")

            for level, (ng, intensity) in GHOST_PRESETS:
                out = apply_ghost(vol_01, ng, intensity)
                f_out.require_group(f"{mod}_moved_ghost_{level}").create_dataset(
                    pid, data=out, compression="gzip", compression_opts=4)

        elapsed = time.time() - t_p
        total_so_far = time.time() - t_start
        remaining = total_so_far / (p_idx + 1) * (N_SELECT - p_idx - 1)
        print(f"  done in {elapsed:.0f}s  |  ETA {remaining/60:.1f} min\n")

total_min = (time.time() - t_start) / 60
print(f"Finished in {total_min:.1f} min")
print(f"Saved: {OUT_H5}")

# ── verify ────────────────────────────────────────────────────────────────────
with h5py.File(OUT_H5, "r") as f:
    groups = sorted(f.keys())
    print(f"\nGroups ({len(groups)}): {groups[:8]} ...")
    print(f"Patients per group: {len(f[groups[0]])}")
    sample_key = f"{MODALITIES[0]}_moved_gibbs_weak"
    pid0 = selected[0]
    vol  = f[sample_key][pid0][:]
    print(f"Sample [{sample_key}][{pid0}]: shape={vol.shape}  range=[{vol.min():.3f}, {vol.max():.3f}]")

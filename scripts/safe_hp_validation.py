"""
Safe HP Residual Injection — Post-hoc Validation
Steps 1-5 from the validation plan (Step 6 partial: IOP only).

Data (IOP unseen, 30 subjects):
  b_*.nii.gz        = real_B (T2 GT)
  preds_b_*.nii.gz  = fake_base (Original model output)
  preds_a_*.nii.gz  = ref_center (T2_moved registered ref)

Formula tested:
  diff        = ref_center - fake_base
  diff_hp     = HP(diff)                         [highpass with reflect-pad Gaussian]
  diff_hp_clip = clip * tanh(diff_hp / clip)     [soft clipping]
  low_fake    = blur(fake_base)
  low_ref     = blur(ref_center)
  mask        = exp(-|low_ref - low_fake| / tau)  [reliability mask]
  fake_test   = fake_base + alpha * mask * diff_hp_clip
"""

import os, glob, warnings
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import itertools

warnings.filterwarnings('ignore')

IOP_DIR = ("/SSD1_1TB/home/milab/daniel/03_Registformer_develop/RegistFormer_develop/"
           "logs/Model_ProposedSynthesis_Data_IXI_Align_T2_T1_PD/"
           "ProposeSynth_IXI_25d_T1toT2_DataVer3_80SliceOut_Ver0_Original_InferUnseenIOP/"
           "runs/2026-05-01_15-45-40/results")

VIS_DIR = os.path.join(os.path.dirname(IOP_DIR), "safe_hp_vis")
os.makedirs(VIS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ── utilities ─────────────────────────────────────────────────────────────────

def load_nii(path):
    vol = nib.load(path).get_fdata().astype(np.float32)
    mn, mx = vol.min(), vol.max()
    if mx - mn > 1e-6:
        vol = (vol - mn) / (mx - mn)
    return vol  # (H, W, Z), [0,1]


def gaussian_blur_reflect(x, sigma=2.0, kernel_size=9):
    """(1,1,H,W) tensor → blurred. reflect padding."""
    ax = torch.arange(kernel_size, device=x.device, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-ax**2 / (2 * sigma**2)); g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(xp, kernel, padding=0)


def highpass(x, sigma=2.0):
    return x - gaussian_blur_reflect(x, sigma=sigma)


def soft_clip(x, clip):
    return clip * torch.tanh(x / clip)


def metrics_2d(pred, gt):
    """pred, gt: numpy (H,W) in [0,1]. returns ssim, psnr."""
    s = ssim(pred, gt, data_range=1.0)
    p = psnr(gt, pred, data_range=1.0)
    return s, p


def cosine_sim(a, b, eps=1e-8):
    return float((a * b).sum() / (a.norm() * b.norm() + eps))


def get_ids(d):
    return sorted(p[2:-7] for p in
                  [os.path.basename(f) for f in glob.glob(os.path.join(d, "b_*.nii.gz"))])


# ── per-slice injection ───────────────────────────────────────────────────────

def inject_slice(fake_t, ref_t, gt_t, alpha, tau, clip, sigma_hp=1.5, sigma_mask=3.0):
    """
    All tensors: (1,1,H,W), float32.
    Returns dict with metrics and diagnostic scalars.
    """
    diff      = ref_t - fake_t
    diff_hp   = highpass(diff, sigma=sigma_hp)
    diff_clip = soft_clip(diff_hp, clip)

    low_fake = gaussian_blur_reflect(fake_t, sigma=sigma_mask)
    low_ref  = gaussian_blur_reflect(ref_t,  sigma=sigma_mask)
    mask     = torch.exp(-torch.abs(low_ref - low_fake) / tau)

    delta    = alpha * mask * diff_clip
    fake_out = torch.clamp(fake_t + delta, 0.0, 1.0)

    gt_np   = gt_t[0, 0].cpu().numpy()
    out_np  = fake_out[0, 0].cpu().numpy()

    s, p = metrics_2d(out_np, gt_np)
    return {
        'ssim': s, 'psnr': p,
        'mask_mean': mask.mean().item(),
        'delta_abs':  delta.abs().mean().item(),
        'diffhp_abs': diff_hp.abs().mean().item(),
    }, fake_out, diff_hp, mask, delta


# ── baseline metrics ─────────────────────────────────────────────────────────

def baseline_metrics_slice(fake_t, gt_t):
    gt_np = gt_t[0, 0].cpu().numpy()
    fb_np = fake_t[0, 0].cpu().numpy()
    s, p = metrics_2d(fb_np, gt_np)
    sharp = float(np.std(np.diff(fb_np, axis=1)))
    return s, p, sharp


# ── cosine residual correlation ───────────────────────────────────────────────

def residual_corr_slice(fake_t, gt_t, ref_t, alpha, tau, clip,
                        sigma_hp=1.5, sigma_mask=3.0):
    """Cosine sim between HP(GT residual) and HP(ref residual after mask+clip)."""
    gt_resid_hp  = highpass(gt_t - fake_t, sigma=sigma_hp)

    diff_hp   = highpass(ref_t - fake_t, sigma=sigma_hp)
    diff_clip = soft_clip(diff_hp, clip)
    low_fake  = gaussian_blur_reflect(fake_t, sigma=sigma_mask)
    low_ref   = gaussian_blur_reflect(ref_t,  sigma=sigma_mask)
    mask      = torch.exp(-torch.abs(low_ref - low_fake) / tau)
    ref_resid = mask * diff_clip

    # naive (no mask/clip) for comparison
    naive_hp  = highpass(ref_t - fake_t, sigma=sigma_hp)

    return {
        'masked_corr': cosine_sim(ref_resid, gt_resid_hp),
        'naive_corr':  cosine_sim(naive_hp,  gt_resid_hp),
    }


# ── Step A: naive HP (no mask, no clip) ──────────────────────────────────────

def naive_inject_slice(fake_t, ref_t, gt_t, alpha, sigma_hp=1.5):
    diff_hp  = highpass(ref_t - fake_t, sigma=sigma_hp)
    fake_out = torch.clamp(fake_t + alpha * diff_hp, 0.0, 1.0)
    gt_np    = gt_t[0, 0].cpu().numpy()
    out_np   = fake_out[0, 0].cpu().numpy()
    s, p = metrics_2d(out_np, gt_np)
    return s, p


# ── Step B: clipped only (no mask) ───────────────────────────────────────────

def clip_only_inject_slice(fake_t, ref_t, gt_t, alpha, clip, sigma_hp=1.5):
    diff_hp  = highpass(ref_t - fake_t, sigma=sigma_hp)
    diff_clip = soft_clip(diff_hp, clip)
    fake_out = torch.clamp(fake_t + alpha * diff_clip, 0.0, 1.0)
    gt_np    = gt_t[0, 0].cpu().numpy()
    out_np   = fake_out[0, 0].cpu().numpy()
    s, p = metrics_2d(out_np, gt_np)
    return s, p


# ── main loop ─────────────────────────────────────────────────────────────────

def process_subject(sid):
    gt_vol   = load_nii(os.path.join(IOP_DIR, f"b_{sid}.nii.gz"))
    fake_vol = load_nii(os.path.join(IOP_DIR, f"preds_b_{sid}.nii.gz"))
    ref_vol  = load_nii(os.path.join(IOP_DIR, f"preds_a_{sid}.nii.gz"))
    H, W, Z  = gt_vol.shape

    # Configs to sweep
    ALPHAS = [0.10, 0.15, 0.20]
    TAUS   = [0.10, 0.15, 0.20]
    CLIPS  = [0.10, 0.20]

    out = {'baseline': {'ssim':[], 'psnr':[], 'sharp':[]}}
    # naive A
    for a in ALPHAS:
        out[f'naive_a{a}'] = {'ssim':[], 'psnr':[]}
    # clip-only B
    for a in ALPHAS:
        for c in CLIPS:
            out[f'clip_a{a}_c{c}'] = {'ssim':[], 'psnr':[]}
    # masked+clip C
    for a, tau, c in itertools.product(ALPHAS, TAUS, CLIPS):
        out[f'masked_a{a}_t{tau}_c{c}'] = {
            'ssim':[], 'psnr':[], 'mask_mean':[], 'delta_abs':[], 'diffhp_abs':[]}
    # residual corr (use one representative config)
    out['corr_masked'] = []
    out['corr_naive']  = []

    for z in range(Z):
        gt_t   = torch.tensor(gt_vol[:,:,z],   device=device).unsqueeze(0).unsqueeze(0)
        fake_t = torch.tensor(fake_vol[:,:,z],  device=device).unsqueeze(0).unsqueeze(0)
        ref_t  = torch.tensor(ref_vol[:,:,z],   device=device).unsqueeze(0).unsqueeze(0)

        # Baseline
        s, p, sh = baseline_metrics_slice(fake_t, gt_t)
        out['baseline']['ssim'].append(s)
        out['baseline']['psnr'].append(p)
        out['baseline']['sharp'].append(sh)

        # Naive
        for a in ALPHAS:
            s, p = naive_inject_slice(fake_t, ref_t, gt_t, a)
            out[f'naive_a{a}']['ssim'].append(s)
            out[f'naive_a{a}']['psnr'].append(p)

        # Clip only
        for a in ALPHAS:
            for c in CLIPS:
                s, p = clip_only_inject_slice(fake_t, ref_t, gt_t, a, c)
                out[f'clip_a{a}_c{c}']['ssim'].append(s)
                out[f'clip_a{a}_c{c}']['psnr'].append(p)

        # Masked + clip
        for a, tau, c in itertools.product(ALPHAS, TAUS, CLIPS):
            info, _, _, _, _ = inject_slice(fake_t, ref_t, gt_t, a, tau, c)
            key = f'masked_a{a}_t{tau}_c{c}'
            for k in ['ssim','psnr','mask_mean','delta_abs','diffhp_abs']:
                out[key][k].append(info[k])

        # Residual corr (representative: a=0.15, tau=0.15, clip=0.20)
        corr = residual_corr_slice(fake_t, gt_t, ref_t, alpha=0.15, tau=0.15, clip=0.20)
        out['corr_masked'].append(corr['masked_corr'])
        out['corr_naive'].append(corr['naive_corr'])

    # Average over slices
    result = {}
    for k, v in out.items():
        if isinstance(v, dict):
            result[k] = {m: float(np.mean(vals)) for m, vals in v.items()}
        else:
            result[k] = float(np.mean(v))
    return result


def mean_metric(all_results, key, metric=None):
    vals = []
    for r in all_results:
        if metric:
            vals.append(r[key][metric])
        else:
            vals.append(r[key])
    return float(np.mean(vals))


def main():
    sids = get_ids(IOP_DIR)
    print(f"Subjects: {len(sids)}  (all IOP unseen)")

    all_results = []
    for i, sid in enumerate(sids):
        print(f"  [{i+1:2d}/{len(sids)}] {sid}", end='', flush=True)
        try:
            r = process_subject(sid)
            r['sid'] = sid
            all_results.append(r)
            print(f"  baseline SSIM={r['baseline']['ssim']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*70)
    print("IOP UNSEEN — POST-HOC SAFE HP INJECTION VALIDATION")
    print("="*70)

    # ── Step 1/2: Baseline ────────────────────────────────────────────────────
    print("\n[BASELINE — Original fake_base]")
    bl_ssim  = mean_metric(all_results, 'baseline', 'ssim')
    bl_psnr  = mean_metric(all_results, 'baseline', 'psnr')
    bl_sharp = mean_metric(all_results, 'baseline', 'sharp')
    print(f"  SSIM={bl_ssim:.4f}  PSNR={bl_psnr:.2f}dB  Sharp={bl_sharp:.2f}")

    # ── Step A: Naive ─────────────────────────────────────────────────────────
    print("\n[Step A: Naive HP injection  (no mask, no clip)]")
    for a in [0.10, 0.15, 0.20]:
        s = mean_metric(all_results, f'naive_a{a}', 'ssim')
        p = mean_metric(all_results, f'naive_a{a}', 'psnr')
        ds = s - bl_ssim; dp = p - bl_psnr
        print(f"  alpha={a:.2f}  SSIM={s:.4f} ({ds:+.4f})  PSNR={p:.2f} ({dp:+.2f})")

    # ── Step B: Clip only ─────────────────────────────────────────────────────
    print("\n[Step B: Clipped HP injection  (no mask)]")
    for a in [0.10, 0.15, 0.20]:
        for c in [0.10, 0.20]:
            s = mean_metric(all_results, f'clip_a{a}_c{c}', 'ssim')
            p = mean_metric(all_results, f'clip_a{a}_c{c}', 'psnr')
            ds = s - bl_ssim; dp = p - bl_psnr
            print(f"  alpha={a:.2f} clip={c:.2f}  SSIM={s:.4f} ({ds:+.4f})  PSNR={p:.2f} ({dp:+.2f})")

    # ── Step C: Masked + clip — table ─────────────────────────────────────────
    print("\n[Step C: Clipped + Reliability Mask]")
    print(f"  {'config':<32} {'SSIM':>7} {'dSSIM':>7} {'PSNR':>7} {'dPSNR':>7} "
          f"{'mask':>6} {'delta':>7} {'dhp':>7}")
    best_ssim_key = None; best_ssim_val = -1
    for a, tau, c in itertools.product([0.10, 0.15, 0.20], [0.10, 0.15, 0.20], [0.10, 0.20]):
        key = f'masked_a{a}_t{tau}_c{c}'
        s   = mean_metric(all_results, key, 'ssim')
        p   = mean_metric(all_results, key, 'psnr')
        mm  = mean_metric(all_results, key, 'mask_mean')
        da  = mean_metric(all_results, key, 'delta_abs')
        dh  = mean_metric(all_results, key, 'diffhp_abs')
        ds  = s - bl_ssim; dp = p - bl_psnr
        label = f"a={a} tau={tau} c={c}"
        print(f"  {label:<32} {s:.4f} {ds:+.4f} {p:.2f} {dp:+.2f} {mm:.3f} {da:.5f} {dh:.5f}")
        if s > best_ssim_val:
            best_ssim_val = s; best_ssim_key = (a, tau, c)

    # ── Step 5: Residual direction ────────────────────────────────────────────
    print("\n[Step 5: Residual Direction — cosine(HP(ref_resid), HP(GT_resid))]")
    mc = mean_metric(all_results, 'corr_masked')
    nc = mean_metric(all_results, 'corr_naive')
    print(f"  Naive HP corr  (no mask/clip) : {nc:.4f}")
    print(f"  Masked+clip corr (a=0.15,t=0.15,c=0.20): {mc:.4f}")
    print(f"  {'Mask+clip maintains signal direction' if mc > 0.0 else 'Signal lost after masking'}")

    # ── Best config summary ───────────────────────────────────────────────────
    if best_ssim_key:
        a, tau, c = best_ssim_key
        key = f'masked_a{a}_t{tau}_c{c}'
        s   = mean_metric(all_results, key, 'ssim')
        p   = mean_metric(all_results, key, 'psnr')
        mm  = mean_metric(all_results, key, 'mask_mean')
        da  = mean_metric(all_results, key, 'delta_abs')
        print(f"\n[Best config by SSIM]  alpha={a}, tau={tau}, clip={c}")
        print(f"  SSIM={s:.4f} ({s-bl_ssim:+.4f})  PSNR={p:.2f} ({p-bl_psnr:+.2f})")
        print(f"  mask_mean={mm:.3f}  delta_abs={da:.5f}")

    # ── Go/No-Go ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("GO / NO-GO JUDGMENT")
    print("="*70)
    best_s = mean_metric(all_results, f'masked_a{best_ssim_key[0]}_t{best_ssim_key[1]}_c{best_ssim_key[2]}', 'ssim')
    best_p = mean_metric(all_results, f'masked_a{best_ssim_key[0]}_t{best_ssim_key[1]}_c{best_ssim_key[2]}', 'psnr')
    criteria = {
        'SSIM +0.003': (best_s - bl_ssim) >= 0.003,
        'PSNR +0.1dB': (best_p - bl_psnr) >= 0.1,
        'Corr > 0.3':  mc > 0.3,
        'mask 0.1~0.9': 0.1 < mean_metric(all_results, f'masked_a{best_ssim_key[0]}_t{best_ssim_key[1]}_c{best_ssim_key[2]}', 'mask_mean') < 0.9,
    }
    for crit, ok in criteria.items():
        print(f"  {'✓' if ok else '✗'} {crit}")
    n_pass = sum(criteria.values())
    print(f"\n  Passed {n_pass}/{len(criteria)} criteria → {'GO: proceed to training' if n_pass >= 3 else 'NO-GO: revisit parameters'}")

    # ── Step 6: Note ──────────────────────────────────────────────────────────
    print("\n[Step 6: Site split]")
    print("  All 30 subjects are IOP unseen. HH/Guys seen-site split NOT available in this results dir.")
    print("  To get seen-site comparison, need to run inference on train/val set separately.")


if __name__ == '__main__':
    main()

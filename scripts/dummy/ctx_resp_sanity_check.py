"""
CtxResp Sanity Check
--------------------
1. Fixed one-hot target (offset +2) 로 alpha가 움직이는지 확인
2. q_proj / k_proj gradient norm 확인

모든 다른 loss off, CtxResp만 on.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import hydra

DEVICE = "cuda:0"
K = 5
N_STEPS = 200
LR = 1e-4

# ── 1. Build model directly ────────────────────────────────────────────────
from src.models.components.network_proposed_synthesis import PatchwiseSliceFusionConditioner25D

cond = PatchwiseSliceFusionConditioner25D(
    window=3,
    coarse=False,
    dim=16,
    use_confidence_gate=True,
    qk_input_mode='image',
    use_qk_norm=True,
    fixed_temperature=False,
).to(DEVICE)

# Dummy inputs: B=1, C=1, H=16, W=16  (simulating post-downsample feature)
B, C, H, W = 1, 1, 16, 16
source    = torch.randn(B, C, H, W, device=DEVICE)
ref_stack = torch.randn(B, K, H, W, device=DEVICE)  # [1,K,H,W]

# Fixed one-hot target: offset +2 = index 4
CENTER = K // 2  # 2
TARGET_IDX = CENTER + 2  # 4 = offset +2
target_resp = torch.zeros(K, device=DEVICE)
target_resp[TARGET_IDX] = 1.0

optimizer = torch.optim.Adam(cond.parameters(), lr=LR)

eps = 1e-8

print(f"{'step':>5}  {'loss_resp':>10}  {'alpha_eff_k':>11}  "
      f"{'alpha_+2':>9}  {'alpha_+0':>9}  "
      f"{'grad_q_proj':>12}  {'grad_k_proj':>12}  {'grad_log_T':>11}")
print("-" * 100)

for step in range(N_STEPS + 1):
    optimizer.zero_grad()

    # Forward
    style, stats, aux = cond(source=source, ref_stack=ref_stack, out_size=(H, W))

    alpha_mean = aux["alpha_mean_per_slice"]  # [K], differentiable

    # KL(target || alpha_mean)
    loss_resp = (
        target_resp * (target_resp.clamp_min(eps).log() - alpha_mean.clamp_min(eps).log())
    ).sum()

    loss_resp.backward()

    # Gradient norms
    def gnorm(p):
        if p is None or p.grad is None:
            return 0.0
        return p.grad.norm().item()

    g_q = gnorm(cond.q_proj.weight)
    g_k = gnorm(cond.k_proj.weight)
    g_t = gnorm(cond.log_temperature) if hasattr(cond, 'log_temperature') else 0.0

    if step % 10 == 0:
        a = alpha_mean.detach()
        eff_k = torch.exp(-(a * a.clamp_min(eps).log()).sum()).item()
        a2 = a[TARGET_IDX].item()   # offset +2
        a0 = a[CENTER].item()       # offset +0 (center)
        print(f"{step:>5}  {loss_resp.item():>10.4f}  {eff_k:>11.4f}  "
              f"{a2:>9.4f}  {a0:>9.4f}  "
              f"{g_q:>12.2e}  {g_k:>12.2e}  {g_t:>11.2e}")

    optimizer.step()

print("\n--- Final alpha distribution ---")
a = alpha_mean.detach()
for i in range(K):
    off = i - CENTER
    bar = "█" * int(a[i].item() * 50)
    print(f"  offset {off:+d}: {a[i].item():.4f}  {bar}")

print(f"\nFinal alpha_eff_k = {torch.exp(-(a * a.clamp_min(eps).log()).sum()).item():.4f}")
print(f"Target was offset +2 (index {TARGET_IDX})")

if a[TARGET_IDX].item() > 0.5:
    print("✅ PASS: alpha moved toward target")
else:
    print("❌ FAIL: alpha did NOT move toward target")
    print("   → gradient path broken or loss too weak")

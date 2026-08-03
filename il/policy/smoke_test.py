"""Shape/gradient smoke test for the 4 policy presets — no data/robot needed.
Run: python il/policy/smoke_test.py   (needs torch; use the curobo2 env)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from policy import build_policy, PRESETS


def rand_obs(B, To, cams, P, H=96):
    o = {c: torch.randn(B, To, 3, H, H) for c in cams}
    o["proprio"] = torch.randn(B, To, P)
    return o


def main():
    B, To, Tp, A, P = 2, 2, 16, 7, 7
    cams = ("wrist_rgb", "fixed_rgb")
    print(f"{'preset':10} {'params(M)':>9} {'loss':>8} {'pred_shape':>14} {'sample_ms':>10}")
    for name in PRESETS:
        pol = build_policy(name, action_dim=A, proprio_dim=P, Tp=Tp, To=To, cams=cams)
        pol.set_norm(torch.zeros(A), torch.ones(A))
        obs = rand_obs(B, To, cams, P); act = torch.randn(B, Tp, A)
        # train step
        loss = pol.loss(obs, act); loss.backward()
        gnorm = sum(p.grad.abs().sum() for p in pol.parameters() if p.grad is not None)
        assert torch.isfinite(loss) and gnorm > 0, f"{name}: bad loss/grad"
        # inference
        t0 = time.time(); a = pol.predict(obs); dt = (time.time() - t0) * 1000
        assert a.shape == (B, Tp, A), f"{name}: pred shape {a.shape}"
        assert torch.isfinite(a).all(), f"{name}: non-finite pred"
        npar = sum(p.numel() for p in pol.parameters()) / 1e6
        print(f"{name:10} {npar:9.2f} {float(loss):8.3f} {str(tuple(a.shape)):>14} {dt:10.1f}")
    print("OK — all 4 presets: loss backprops, sampling returns finite (B,Tp,7) joint+suction chunks.")


if __name__ == "__main__":
    main()

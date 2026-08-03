"""Train any of the four policy presets on the canonical episode dataset (sim + real).
One loop for all heads (diffusion / umi / cfm / drift).

  python3 il/train.py --policy umi --data outputs/il_episodes --steps 20000 --out outputs/il_ckpt
"""
import argparse
import copy
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "policy"))
sys.path.insert(0, os.path.join(HERE, "data"))
import torch
from torch.utils.data import DataLoader
from policy import build_policy, PRESETS
from dataset import EpisodeDataset


def to_dev(obs, dev):
    return {k: v.to(dev) for k, v in obs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="umi", choices=list(PRESETS))
    ap.add_argument("--data", default=os.path.join(HERE, "..", "outputs", "il_episodes"))
    ap.add_argument("--sources", default="", help="comma list to filter (sim,real); empty=all")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--To", type=int, default=2)
    ap.add_argument("--Tp", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "outputs", "il_ckpt"))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    srcs = [s for s in args.sources.split(",") if s] or None

    ds = EpisodeDataset(args.data, To=args.To, Tp=args.Tp, img_size=args.img_size, sources=srcs)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=(dev == "cuda"))
    print(f"[data] {len(ds.eps)} episodes, {len(ds)} windows  (To={args.To} Tp={args.Tp})")

    pol = build_policy(args.policy, action_dim=7, proprio_dim=7, Tp=args.Tp, To=args.To).to(dev)
    pol.fit_norm(loader, n_batches=30, dev=dev)                 # rep-aware normalization
    ema = copy.deepcopy(pol).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr, weight_decay=1e-6)
    os.makedirs(args.out, exist_ok=True)

    def save(tag):
        path = os.path.join(args.out, f"{args.policy}_{tag}.pt")
        torch.save({"policy": args.policy, "ema": ema.state_dict(), "model": pol.state_dict(),
                    "norm": [pol.a_mean.cpu(), pol.a_std.cpu()], "cfg": vars(args)}, path)
        return path

    step = 0; t0 = time.time(); run = None
    while step < args.steps:
        for obs, action in loader:
            obs, action = to_dev(obs, dev), action.to(dev)
            loss = pol.loss(obs, action)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), pol.parameters()):
                    pe.mul_(args.ema).add_(pm, alpha=1 - args.ema)
            run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"step {step:6d}/{args.steps}  loss {run:.4f}  ({step/(time.time()-t0):.1f} it/s)")
            if step % args.save_every == 0:
                save(f"step{step}")
            if step >= args.steps:
                break
    print("saved ->", save("final"))


if __name__ == "__main__":
    main()

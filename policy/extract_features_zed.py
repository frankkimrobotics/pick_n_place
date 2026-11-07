#!/usr/bin/env python3
"""extract_features_zed :: frozen lingbot-vision (or clip / dinov2) features for BOTH cameras of a
Gen-B scene directory (wrist `d405_rgb.mp4` + fixed `d435_rgb.mp4`, which the ZED generators render
from the calibrated ZED pose), i.e. the files `policy/train_policy_zed.py` loads:

    <enc>_pool.npy       [T, D] fp16  mean of the patch tokens, wrist camera
    <enc>_zed_pool.npy   [T, D] fp16  same, ZED camera
    <enc>_tokens.npy / <enc>_zed_tokens.npy   [T, P, D] fp16, only with --tokens (1.3 GB per scene)

Preprocessing is exactly `extract_features.frames_224` (432x240 -> 224x404 bilinear -> centre crop
224x224), which `policy/real_rollout_zed.py` reproduces on the live frames (verified cos 1.0).

    PNP_DS=~/pnp_dataset_dp ~/miniconda3/envs/ilpolicy/bin/python policy/extract_features_zed.py --enc lingbot
"""
import argparse
import glob
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
sys.path.insert(0, HERE)
import extract_features as EF  # noqa: E402

DS = os.path.expanduser(os.environ.get("PNP_DS", "~/pnp_dataset_zed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="lingbot", choices=["clip", "dinov2", "lingbot", "cradio"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tokens", action="store_true", help="also save the full patch tokens")
    ap.add_argument("--scenes", default=None, help="glob under PNP_DS (default scene_*)")
    a = ap.parse_args()
    fwd, mean, std = EF.load_encoder(a.enc)
    mean, std = mean.to(EF.DEV), std.to(EF.DEV)
    cams = [("d405_rgb.mp4", ""), ("d435_rgb.mp4", "_zed")]
    for sdir in sorted(glob.glob(os.path.join(DS, a.scenes or "scene_*"))):
        for mp4, suffix in cams:
            out_pool = os.path.join(sdir, f"{a.enc}{suffix}_pool.npy")
            out_tok = os.path.join(sdir, f"{a.enc}{suffix}_tokens.npy")
            if os.path.exists(out_pool) and (not a.tokens or os.path.exists(out_tok)):
                continue
            src = os.path.join(sdir, mp4)
            if not os.path.exists(src):
                print(os.path.basename(sdir), "missing", mp4)
                continue
            imgs = EF.frames_224(src)
            pools, toks = [], []
            with torch.no_grad():
                for i in range(0, len(imgs), a.batch):
                    px = ((imgs[i:i + a.batch].to(EF.DEV)) - mean) / std
                    t = fwd(px)                                  # [B, P, D]
                    pools.append(t.mean(1).half().cpu())
                    if a.tokens:
                        toks.append(t.half().cpu())
            np.save(out_pool, torch.cat(pools).numpy())
            if a.tokens:
                np.save(out_tok, torch.cat(toks).numpy())
            print(os.path.basename(sdir), suffix or "_wrist", tuple(torch.cat(pools).shape), flush=True)
    print("done", a.enc)


if __name__ == "__main__":
    main()

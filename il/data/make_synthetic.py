"""Generate synthetic episodes in the canonical layout — for testing the dataset/training path
without the robot, and as a smoke fixture. Images encode the joint state so obs→action is learnable.

  python3 il/data/make_synthetic.py --root outputs/il_synth --episodes 8 --frames 40
"""
import argparse
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode_writer import EpisodeWriter

Q0 = np.array([0.0, -1.9, 1.4, -1.4, -1.57, 0.0])   # BASE_Q-ish (rad); synthetic, exact value irrelevant


def blob(hw, cx, cy, color):
    im = np.zeros((*hw, 3), np.uint8)
    cv2.circle(im, (int(cx) % hw[1], int(cy) % hw[0]), 22, color, -1)   # color is RGB
    return im


def gen(root, n_eps, n_frames, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    for e in range(n_eps):
        w = EpisodeWriter(root, f"{e:03d}", source="sim",
                          meta_extra={"object": f"obj{e % 3}", "synthetic": True, "seed": int(seed + e)})
        tgt = Q0 + rng.uniform(-0.4, 0.4, 6)
        for k in range(n_frames):
            a = k / (n_frames - 1); q = Q0 * (1 - a) + tgt * a
            suc = 1 if a > 0.6 else 0
            wr = blob((240, 320), 160 + 80 * np.sin(q[0]), 120 + 80 * np.sin(q[1]),
                      (0, 200, 0) if suc else (200, 60, 60))
            fx = blob((240, 320), 160 + 80 * np.sin(q[0]), 120 + 80 * np.sin(q[2]), (60, 120, 220))
            w.add(k / 10.0, q, suc, wr, fx, phase="grasp" if suc else "reach")
        w.close(success=True)
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "il_synth"))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("wrote", gen(a.root, a.episodes, a.frames, a.seed))

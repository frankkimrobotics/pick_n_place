"""Export a canonical-episode collection (sim + real) → a LeRobot dataset for community trainers.
Needs `pip install lerobot`. The canonical format is the source of truth; this is a one-way export.

  python3 il/data/to_lerobot.py --root outputs/il_episodes --repo-id mycobot/pnp --fps 10
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import PROPRIO_DIM, ACTION_DIM


def _proprio(joints, suction):
    return np.concatenate([joints, suction[:, None].astype(np.float32)], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="canonical episode collection dir")
    ap.add_argument("--repo-id", default="mycobot/pnp")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--img", type=int, default=224)
    args = ap.parse_args()
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        print("lerobot not installed. `pip install lerobot`. (Canonical episodes remain usable via "
              "il/data/dataset.py without this export.)"); sys.exit(1)

    hw = (args.img, args.img)
    feats = {
        "observation.images.wrist": {"dtype": "video", "shape": (hw[0], hw[1], 3), "names": ["h", "w", "c"]},
        "observation.images.fixed": {"dtype": "video", "shape": (hw[0], hw[1], 3), "names": ["h", "w", "c"]},
        "observation.state": {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=feats,
                               root=os.path.join(args.root, "..", "lerobot_" + args.repo_id.split("/")[-1]))

    def load_img(d, sub, k):
        im = cv2.imread(os.path.join(d, sub, f"{k:06d}.jpg"))
        return cv2.cvtColor(cv2.resize(im, hw[::-1]), cv2.COLOR_BGR2RGB)

    n = 0
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isfile(os.path.join(d, "arrays.npz")):
            continue
        arr = np.load(os.path.join(d, "arrays.npz"), allow_pickle=True)
        prop = _proprio(arr["joints"].astype(np.float32), arr["suction"].astype(np.float32))
        N = len(prop)
        for k in range(N - 1):                         # action = next state (DP joint-target target)
            ds.add_frame({
                "observation.images.wrist": load_img(d, "wrist", k),
                "observation.images.fixed": load_img(d, "fixed", k),
                "observation.state": prop[k], "action": prop[k + 1],
            })
        ds.save_episode(task="pick and place"); n += 1
        print(f"  + {os.path.basename(d)} ({N} frames)")
    ds.consolidate()
    print(f"done: {n} episodes -> LeRobot dataset '{args.repo_id}'")


if __name__ == "__main__":
    main()

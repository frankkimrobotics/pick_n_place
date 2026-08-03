"""Canonical episode schema shared by SIM capture, REAL rosbag conversion, dataset loading, and
inference. Kept format-neutral: episodes are written as a simple self-contained folder (below), and
`to_lerobot.py` converts a whole collection into a LeRobot dataset for community trainers.

On-disk episode layout (one folder per episode):
  <root>/<source>_<id>/
    meta.json                 episode metadata (below)
    wrist/000000.jpg ...      D405 wrist RGB
    fixed/000000.jpg ...      D435 fixed RGB
    arrays.npz                t(N), joints(N,6 rad), suction(N,), tcp_pose(N,7 aux), phase(N,) str

Observation (history To) = { wrist_rgb, fixed_rgb, proprio = joints(6)+suction(1) = 7 }.
Action (horizon Tp)      = future joints(6)+suction(1) = 7  (DP-style: absolute-joint targets are the
                           future achieved states; the umi preset re-expresses them as Δjoint).
So the writer only stores per-frame state; the dataset DERIVES the action horizon from future frames.
"""
FPS = 10                     # obs/action logging rate
CAMS = ["wrist_rgb", "fixed_rgb"]
STORE_HW = (240, 320)        # stored jpg size (H,W); loader resizes to the model input
NJ = 6                       # joint dims
PROPRIO_DIM = 7              # joints(6) + suction(1)
ACTION_DIM = 7               # joints(6) + suction(1)
PHASES = ["start", "reach", "descend", "grasp", "lift", "carry", "place", "release", "home", "end"]


def lerobot_features():
    """Feature spec for LeRobot dataset creation (used by to_lerobot.py)."""
    import numpy as np  # noqa
    return {
        "observation.images.wrist": {"dtype": "video", "shape": (STORE_HW[0], STORE_HW[1], 3), "names": ["h", "w", "c"]},
        "observation.images.fixed": {"dtype": "video", "shape": (STORE_HW[0], STORE_HW[1], 3), "names": ["h", "w", "c"]},
        "observation.state": {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": ["j1", "j2", "j3", "j4", "j5", "j6", "suction"]},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["j1", "j2", "j3", "j4", "j5", "j6", "suction"]},
    }

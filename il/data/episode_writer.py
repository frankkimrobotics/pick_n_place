"""EpisodeWriter — writes one episode in the canonical layout (schema.py). Used by both the sim
capture and the rosbag→episode converter, so sim and real data are identical on disk."""
import json
import os
import numpy as np
import cv2

from schema import STORE_HW


class EpisodeWriter:
    def __init__(self, root, episode_id, source="sim", fps=10, meta_extra=None, store_hw=STORE_HW):
        self.dir = os.path.join(root, f"{source}_{episode_id}")
        os.makedirs(os.path.join(self.dir, "wrist"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "fixed"), exist_ok=True)
        self.hw = store_hw; self.n = 0
        self.t, self.joints, self.suction, self.tcp, self.phase = [], [], [], [], []
        self.meta = {"episode_id": str(episode_id), "source": source, "fps": fps,
                     "cams": ["wrist_rgb", "fixed_rgb"], **(meta_extra or {})}

    def _img(self, sub, arr):
        if arr is None:
            arr = np.zeros((*self.hw, 3), np.uint8)
        if arr.shape[:2] != tuple(self.hw):
            arr = cv2.resize(arr, (self.hw[1], self.hw[0]))
        cv2.imwrite(os.path.join(self.dir, sub, f"{self.n:06d}.jpg"),
                    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    def add(self, t, joints, suction, wrist_rgb, fixed_rgb, tcp_pose=None, phase=""):
        """joints: 6 rad; suction: 0/1; *_rgb: HxWx3 uint8 RGB; tcp_pose: 7 (xyz+wxyz) optional."""
        self._img("wrist", wrist_rgb); self._img("fixed", fixed_rgb)
        self.t.append(float(t)); self.joints.append([float(x) for x in joints])
        self.suction.append(int(bool(suction)))
        self.tcp.append([float(x) for x in (tcp_pose if tcp_pose is not None else [0] * 7)])
        self.phase.append(str(phase)); self.n += 1

    def close(self, success=None):
        np.savez(os.path.join(self.dir, "arrays.npz"),
                 t=np.array(self.t, np.float64), joints=np.array(self.joints, np.float32),
                 suction=np.array(self.suction, np.int8), tcp_pose=np.array(self.tcp, np.float32),
                 phase=np.array(self.phase))
        self.meta["n_frames"] = self.n
        if success is not None:
            self.meta["success"] = bool(success)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)
        return self.dir

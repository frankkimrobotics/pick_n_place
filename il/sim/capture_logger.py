"""In-process capture for the MuJoCo sim: renders the wrist+fixed cameras from the live MjData and
writes canonical episodes (same schema as real). Must be driven from the sim thread that owns MjData
(no cross-thread rendering). Plus object randomization for episode resets."""
import os
import sys
import time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "data")))
from episode_writer import EpisodeWriter


class CaptureLogger:
    def __init__(self, model, out_root, hw=(240, 320)):
        self.r = mujoco.Renderer(model, hw[0], hw[1])
        self.root = out_root; os.makedirs(out_root, exist_ok=True)
        self.ep = None; self._t0 = 0.0

    def start(self, episode_id, meta=None):
        self.ep = EpisodeWriter(self.root, episode_id, source="sim", meta_extra=meta or {})
        self._t0 = time.time()

    def tick(self, data, q_rad, suction, phase=""):
        """Render both cams from `data` and log one obs frame. Call on the sim thread at ~10 Hz."""
        if self.ep is None:
            return
        self.r.update_scene(data, camera="wrist"); wr = self.r.render().copy()
        self.r.update_scene(data, camera="fixed"); fx = self.r.render().copy()
        self.ep.add(time.time() - self._t0, q_rad, int(bool(suction)), wr, fx, phase=phase)

    def stop(self, success=True):
        if self.ep is None:
            return None
        d = self.ep.close(success=success); self.ep = None
        return d

    @property
    def active(self):
        return self.ep is not None


def _yaw_quat(yaw):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def randomize_objects(data, objs, n_keep, rng, table_z=-0.10,
                      xr=(0.30, 0.46), yr=(-0.12, 0.12), min_sep=0.06):
    """Place n_keep objects at random non-overlapping table poses; park the rest out of view.
    `objs` = list of (name, qposadr, half_height) as built by sim_mujoco_node. Returns kept names."""
    idx = list(range(len(objs))); rng.shuffle(idx)
    keep = idx[:n_keep]; placed = []
    kept_names = []
    for j, (name, adr, hh) in enumerate(objs):
        if j in keep:
            for _ in range(50):
                x = rng.uniform(*xr); y = rng.uniform(*yr)
                if all(np.hypot(x - px, y - py) > min_sep for px, py in placed):
                    break
            placed.append((x, y))
            data.qpos[adr:adr + 3] = [x, y, table_z + hh]
            data.qpos[adr + 3:adr + 7] = _yaw_quat(rng.uniform(-np.pi, np.pi))
            kept_names.append(name)
        else:                                        # park far below the floor, out of both views
            data.qpos[adr:adr + 3] = [0.9, 0.9, -1.0]
            data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    if hasattr(data, "qvel"):
        data.qvel[:] = 0.0
    return kept_names

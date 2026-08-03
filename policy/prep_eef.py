"""prep_eef :: per-scene FK of recorded q -> tcp pose (pos + quat wxyz),
aligned with the 10 Hz dataset frames. Run in the mjwarp env (has mujoco)."""
import glob
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "outputs", "mujoco_sim", "robot_warp.xml")
DS = os.path.expanduser("~/pnp_dataset")

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")

for sdir in sorted(glob.glob(os.path.join(DS, "scene_*"))):
    if os.path.exists(os.path.join(sdir, "eef.npz")):
        continue
    t = np.load(os.path.join(sdir, "traj.npz"))
    q = t["q"]
    pos = np.zeros((len(q), 3))
    quat = np.zeros((len(q), 4))
    for i in range(len(q)):
        d.qpos[:6] = q[i]
        mujoco.mj_kinematics(m, d)
        pos[i] = d.site_xpos[sid]
        mujoco.mju_mat2Quat(quat[i], d.site_xmat[sid])
    np.savez_compressed(os.path.join(sdir, "eef.npz"), pos=pos, quat=quat,
                        suction=t["suction"])
    print(os.path.basename(sdir), len(q))
print("done")

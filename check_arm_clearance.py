#!/usr/bin/env python3
"""check_arm_clearance :: does the EXECUTED trajectory keep the arm out of the box and the table?

The twin's arm links are contactless (the only robot collision geom is `cup_tip`), so MuJoCo never
reports a link striking the box; a link could sweep straight through a wall and nothing would notice.
This replays the recorded joints and measures the cuRobo robot model's own collision spheres
(mycobot_pro_630_v2.yml) against every static world box (box walls, table) analytically.

    ~/miniconda3/envs/mjwarp/bin/python check_arm_clearance.py ~/pnp_dataset_curobo/scene_002
"""
import json
import os
import sys

import mujoco
import numpy as np
import yaml

YML = ("/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/"
       "mycobot_description/curobo/mycobot_pro_630_v2.yml")


def load_spheres():
    cfg = yaml.safe_load(open(YML))["robot_cfg"]["kinematics"]["collision_spheres"]
    return {k: [(np.array(s["center"], float), float(s["radius"])) for s in v] for k, v in cfg.items()}


def world_boxes(m):
    """Static collidable boxes/planes of the scene (box walls, table, floor)."""
    out = []
    for i in range(m.ngeom):
        if m.geom_bodyid[i] != 0 or not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            out.append((name, "box", m.geom_pos[i].copy(), m.geom_size[i].copy()))
        elif m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            out.append((name, "plane", m.geom_pos[i].copy(), None))
    return out


def clearance(p, r, boxes):
    """Smallest (surface distance - radius) over the static geometry; < 0 means penetration."""
    worst, who = np.inf, None
    for name, kind, pos, size in boxes:
        if kind == "plane":
            dist = p[2] - pos[2]
        else:
            dq = np.abs(p - pos) - size
            outside = np.linalg.norm(np.maximum(dq, 0.0))
            dist = outside if outside > 0 else float(dq.max())
        if dist - r < worst:
            worst, who = dist - r, name
    return worst, who


def main():
    sdir = os.path.expanduser(sys.argv[1])
    m = mujoco.MjModel.from_xml_path(os.path.join(sdir, "scene.xml"))
    d = mujoco.MjData(m)
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    tr = np.load(os.path.join(sdir, "traj.npz"))
    q, suction = tr["q"], tr["suction"]
    spheres = load_spheres()
    boxes = world_boxes(m)
    links = {}
    for lk in spheres:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, lk)
        if bid >= 0:
            links[lk] = bid
    missing = [lk for lk in spheres if lk not in links]
    nsp = sum(len(spheres[lk]) for lk in links)
    print(f"{os.path.basename(sdir)}: {len(q)} frames, {nsp} collision spheres on {len(links)} links"
          f"{' (no body for: ' + ','.join(missing) + ')' if missing else ''}")
    print(f"  static geometry: {[b[0] for b in boxes]}")

    def phase(k):
        for pk in meta["picks"]:
            if pk["frame_start"] <= k < max(pk["frame_end"], pk["frame_start"] + 1):
                if pk["frame_latch"] < 0:
                    return f"{pk['object']} approach/retry"
                if k < pk["frame_latch"]:
                    return f"{pk['object']} approach"
                if pk["frame_release"] < 0 or k < pk["frame_release"]:
                    return f"{pk['object']} carry"
                return f"{pk['object']} release/retreat"
        return "home"

    worst = (np.inf, None, None, None)
    hits = []
    per_phase = {}
    for k in range(len(q)):
        d.qpos[:6] = q[k]
        mujoco.mj_forward(m, d)
        fw, fwho, fwlink = np.inf, None, None
        for lk, bid in links.items():
            R, t = d.xmat[bid].reshape(3, 3), d.xpos[bid]
            for c, r in spheres[lk]:
                c_w = t + R @ c
                cl, who = clearance(c_w, r, boxes)
                if cl < fw:
                    fw, fwho, fwlink = cl, who, lk
        ph = phase(k)
        per_phase[ph] = min(per_phase.get(ph, np.inf), fw)
        if fw < worst[0]:
            worst = (fw, k, fwlink, fwho)
        if fw < 0:
            hits.append((k, ph, fwlink, fwho, fw))
    print(f"  worst clearance {1000 * worst[0]:+.1f} mm at frame {worst[1]} ({worst[2]} vs {worst[3]}, "
          f"phase '{phase(worst[1])}')")
    print(f"  frames with penetration: {len(hits)} / {len(q)}")
    for k, ph, lk, who, cl in hits[:12]:
        print(f"     frame {k:4d} {ph:28s} {lk:12s} vs {who:10s} {1000 * cl:+.1f} mm")
    print("  min clearance by phase (mm):")
    for ph, v in sorted(per_phase.items(), key=lambda x: x[1]):
        print(f"     {ph:30s} {1000 * v:+8.1f}")


if __name__ == "__main__":
    main()

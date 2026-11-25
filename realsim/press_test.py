#!/usr/bin/env python3
"""realsim.press_test :: scripted vertical press on a scene object through the realistic twin,
rigid tip vs plunger. Reports contact force, plunger compression, drive following-error
(stall) fraction and the arm's tracking lag -- what the 15 mm demo press does to the drive.

    conda activate mj4 && python realsim/press_test.py [--scene 90] [--over 0.025]
"""
import argparse
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "policy")]
from realsim.config import make_cfg  # noqa: E402
from realsim.plunger import load_with_plunger, compression  # noqa: E402
from realsim.twin import Pro630Twin  # noqa: E402
from realsim.validate import Streamer  # noqa: E402
import real_rollout_zed as RZ  # noqa: E402
import dataset_gen_zed as dg  # noqa: E402


def run(scene, plunger, over, dt=0.2, step=0.002):
    xml = os.path.expanduser(f"~/pnp_dataset_zed/scene_{scene:03d}/scene.xml")
    m = load_with_plunger(xml)[0] if plunger else mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    objs = dg.sample_scene(scene)
    o = max(objs, key=lambda o: o["half_extents"][2])
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    cup = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")
    RZ.LC_LO[:], RZ.LC_HI[:] = -10.0, 10.0
    mujoco.mj_forward(m, d)
    top = d.xpos[bid].copy()
    top[2] += o["half_extents"][2] + 0.008               # grasp pose = object top + cup radius (demo convention)
    dik = mujoco.MjData(m)
    q0, pe, _ = RZ.ik(m, dik, sid, top + [0, 0, 0.05], RZ.R_DOWN, np.array([0, -0.35, 1.4, 0.17, -1.57, 0]))
    d.qpos[:6] = q0
    mujoco.mj_forward(m, d)
    tw = Pro630Twin(m, d, make_cfg("real"))
    tw.advance(0.4)
    st = Streamer(tw, dt=dt)
    q_cmd = q0.copy()
    rows = []
    z_goal = top[2] - over
    z = top[2] + 0.05
    for k in range(int((0.05 + over) / step) + 15):
        z = max(z - step, z_goal)
        q_t, pe, _ = RZ.ik(m, dik, sid, np.array([top[0], top[1], z]), RZ.R_DOWN, q_cmd)
        st.send(q_cmd, q_t, tw.t, t_anchor_k=tw.t)
        q_cmd = q_t
        f0 = tw.stats["ferror"]
        fmax = 0.0
        for _ in range(int(round(dt / 0.01))):
            tw.tick()
            fb = np.zeros(6)
            for ci in range(d.ncon):
                c = d.contact[ci]
                if cup in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(m, d, ci, fb)
                    fmax = max(fmax, abs(fb[0]))
        rows.append(dict(z_cmd=z - top[2], z_tip=d.site_xpos[sid][2] - top[2], F=fmax,
                         x=compression(m, d) if plunger else 0.0,
                         stall=(tw.stats["ferror"] - f0) / (tw.sub * dt / 0.01),
                         lag=np.degrees(np.abs(q_cmd - d.qpos[:6]).max()),
                         obj_dz=d.xpos[bid][2] - (top[2] - o["half_extents"][2] - 0.008)))
    run.last = (m, d, bid)
    return o["name"], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=90)
    ap.add_argument("--over", type=float, default=0.025, help="commanded press past the grasp pose (m)")
    a = ap.parse_args()
    for plunger in (False, True):
        name, rows = run(a.scene, plunger, a.over)
        print(f"== {'PLUNGER' if plunger else 'RIGID  '} tip, {name}, press {1000 * a.over:.0f} mm past grasp")
        print("  z_cmd(mm) tip(mm)  F(N)  comp(mm) stall%  lag(deg)")
        for r in rows:
            if r["z_cmd"] < 0.004:
                print(f"  {1000 * r['z_cmd']:8.1f} {1000 * r['z_tip']:7.1f} {r['F']:6.1f} {1000 * r['x']:8.1f} "
                      f"{100 * r['stall']:6.0f} {r['lag']:8.2f}")


if __name__ == "__main__":
    main()

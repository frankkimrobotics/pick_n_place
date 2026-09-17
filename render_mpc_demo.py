#!/usr/bin/env python3
"""render_mpc_demo :: render example task-MPC pick-and-place runs to mp4.

Reuses pick_place_compare.run_trial (clutter4 scene, QP task controller +
LQR inner loop) with a frame callback; two camera views (iso | front),
phase + SEALED overlay, green ring marks the place target.

    python render_mpc_demo.py --trials 2 --out ~/pnp_rl/mpc_demo.mp4
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

import pick_place_compare as ppc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--ctrl", default="taskmpc")
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/mpc_demo.mp4"))
    a = ap.parse_args()

    raw = a.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 25, (864, 360))
    state = {}

    def frame_cb(sim, t, phase, held):
        k = state.get("k", 0)
        state["k"] = k + 1
        if k % 2:                                   # 50 Hz sim -> 25 fps
            return
        if state.get("ren_model") is not sim.m:
            state["ren"] = mujoco.Renderer(sim.m, height=300, width=432)
            state["ren_model"] = sim.m
            state["vopt"] = mujoco.MjvOption()
            state["vopt"].sitegroup[:] = 0
        ren, vopt = state["ren"], state["vopt"]
        frames = []
        for cam in ("iso", "front"):
            ren.update_scene(sim.d, camera=cam, scene_option=vopt)
            # green ring at the place target (post-hoc scene decoration)
            scn = ren.scene
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                    np.array([0.035, 0.035, 0.001]),
                    np.array([sim.place[0], sim.place[1], sim.tgt_h + 0.002]),
                    np.eye(3).flatten(), np.array([0.1, 0.9, 0.2, 0.5],
                                                  dtype=np.float32))
                scn.ngeom += 1
            frames.append(cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR))
        canvas = np.zeros((360, 864, 3), np.uint8)
        canvas[60:] = np.hstack(frames)
        label = (f"{a.ctrl} trial {state['trial']}  t={t:5.1f}s  "
                 f"phase={phase}  target=({sim.place[0]:.2f},{sim.place[1]:.2f})")
        cv2.putText(canvas, label, (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if held:
            cv2.rectangle(canvas, (0, 60), (863, 359), (0, 180, 255), 3)
        vw.write(canvas)
        state["last"] = canvas

    for i in range(a.trials):
        state["trial"] = f"{i + 1}/{a.trials}"
        r = ppc.run_trial(a.ctrl, a.seed0 + i, T=a.T, frame_cb=frame_cb)
        print(f"[render] trial {i+1}: success={r['success']} "
              f"t={r['t_total']:.1f}s place_err={r['place_err']*100:.1f}cm "
              f"rmse={r['rmse_deg']:.2f}deg")
        for _ in range(12):
            vw.write(state["last"])
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 "
              f"-pix_fmt yuv420p {a.out}")
    os.remove(raw)
    print(f"[render] wrote {a.out}")


if __name__ == "__main__":
    main()

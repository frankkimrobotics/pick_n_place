"""render_demo :: render example spline-expert episodes to mp4.

Rolls out the collect.py SplineExpert in PickEnv, records qpos + pedestal
mocap per tick, replays through the MuJoCo renderer (iso | fixed_d435,
sealed = orange border, green disc = place target).

    python bc_mystery/render_demo.py --episodes 6
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rl"))
sys.path.insert(0, HERE)
import warp as wp                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/bc_expert_demo.mp4"))
    ap.add_argument("--scene", default=os.path.join(
        os.path.dirname(HERE), "rl", "scenes", "box_med_ped.xml"))
    a = ap.parse_args()
    wp.init()
    from env_warp import PickEnv
    from collect import SplineExpert, TermTracker, T_EP

    N = a.episodes
    rng = np.random.default_rng(a.seed)
    env = PickEnv(nworld=N, mode="pnp", dr=False, xml=a.scene, lift_req=0.30)
    env.auto_reset = False
    ex = SplineExpert(env, rng)
    env.reset(torch.ones(N, dtype=torch.bool, device=env.device))
    ex.plan()
    trk = TermTracker(N, env.device)

    traj = [[] for _ in range(N)]
    for t in range(T_EP):
        for i in range(N):
            traj[i].append((env.qpos[i].detach().cpu().numpy().copy(),
                            bool(env.sealed[i]),
                            env.place_target[i].detach().cpu().numpy().copy(),
                            float(env.target_h[i]),
                            env.mocap_pos[i].detach().cpu().numpy().copy()
                            if env.mocap_pos is not None else None))
        act = ex.act(t)
        obs, r, done, info = env.step(act)
        trk.update(done, info)
    met = trk.table()
    v3 = met[:, 4].bool().cpu()
    print("[render] episode V3:", ["OK" if bool(x) else "fail" for x in v3])

    import cv2
    xml_txt = open(a.scene).read()
    marker = ('<body name="place_marker" mocap="true" pos="0 0 -1">'
              '<geom type="cylinder" size="0.035 0.0015" rgba="0.1 0.9 0.2 0.6" '
              'contype="0" conaffinity="0"/></body>')
    xml_txt = xml_txt.replace("</worldbody>", marker + "</worldbody>", 1)
    mpath = os.path.join(os.path.dirname(a.scene), "_render_marked.xml")
    open(mpath, "w").write(xml_txt)
    m = mujoco.MjModel.from_xml_path(mpath)
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=300, width=432)
    vopt = mujoco.MjvOption()
    vopt.sitegroup[:] = 0
    vw = cv2.VideoWriter(a.out.replace(".mp4", "_raw.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (864, 360))
    for i in range(N):
        for qpos, sealed, tgt, th, mocap in traj[i]:
            d.qpos[:len(qpos)] = qpos
            if mocap is not None:
                d.mocap_pos[0] = mocap[0]              # pedestal
            d.mocap_pos[-1] = [tgt[0], tgt[1], th + 0.002]  # target marker
            mujoco.mj_forward(m, d)
            frames = []
            for cam in ("iso", "fixed_d435"):
                ren.update_scene(d, camera=cam, scene_option=vopt)
                frames.append(cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR))
            row = np.hstack(frames)
            canvas = np.zeros((360, 864, 3), np.uint8)
            canvas[60:] = row
            label = (f"spline expert ep {i+1}/{N}  "
                     f"({'V3 SUCCESS' if v3[i] else 'fail'})  "
                     f"target=({tgt[0]:.2f},{tgt[1]:.2f},h={th:.2f})")
            cv2.putText(canvas, label, (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if sealed:
                cv2.rectangle(canvas, (0, 60), (863, 359), (0, 180, 255), 3)
            vw.write(canvas)
        for _ in range(5):
            vw.write(canvas)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {a.out.replace('.mp4', '_raw.mp4')} "
              f"-c:v libx264 -pix_fmt yuv420p {a.out}")
    os.remove(a.out.replace(".mp4", "_raw.mp4"))
    print(f"[render] wrote {a.out}")


if __name__ == "__main__":
    main()

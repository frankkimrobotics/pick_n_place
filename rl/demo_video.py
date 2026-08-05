"""demo_video :: roll out the trained attach actor, render episodes.

Runs N worlds of PickEnv(mode=attach) with the SAC actor, records full
qpos per decision step, then replays through the standard MuJoCo renderer
(wrist D405 | fixed D435, sealed = orange border) into one mp4.

    python rl/demo_video.py --actor ~/pnp_rl/h100_attach1/actor.pt --episodes 6
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import torch
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default=os.path.expanduser("~/pnp_rl/h100_attach1/actor.pt"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/attach_demo"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    ap.add_argument("--mode", default="attach", choices=["attach", "pnp"])
    ap.add_argument("--horizon", type=int, default=45)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wp.init()
    from env_warp import PickEnv
    from sac import Actor

    N = a.episodes
    env = PickEnv(nworld=N, mode=a.mode, xml=a.scene)
    actor = Actor().to(env.device)
    ck = torch.load(a.actor, map_location=env.device, weights_only=False)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    print(f"[demo] actor from step {ck.get('step')}")

    traj = [[] for _ in range(N)]      # per-world (qpos, sealed)
    obs = env.observe()
    done_seen = torch.zeros(N, dtype=torch.bool, device=env.device)
    results = [None] * N
    for t in range(a.horizon):
        for i in range(N):
            if not done_seen[i]:
                traj[i].append((env.qpos[i].detach().cpu().numpy().copy(),
                                bool(env.sealed[i]),
                                env.place_target[i].detach().cpu().numpy().copy()))
        with torch.no_grad():
            mu, _ = actor(obs)
            act = torch.tanh(mu)                     # deterministic policy
        obs, r, done, info = env.step(act)
        for i in range(N):
            if done[i] and not done_seen[i]:
                done_seen[i] = True
                results[i] = bool(info["placed"][i])
        if done_seen.all():
            break
    crit = ("seal+lift 2cm" if a.mode == "attach"
            else "placed at target + released + at rest")
    print(f"[demo] episode results ({crit}):",
          ["OK" if r else "fail" for r in results])

    # replay through renderer
    import cv2
    # inject a visible place-target marker (mocap disc) into the render model
    xml_txt = open(a.scene).read()
    marker = ('<body name="place_marker" mocap="true" pos="0 0 -1">'
              '<geom type="cylinder" size="0.035 0.0015" rgba="0.1 0.9 0.2 0.6" '
              'contype="0" conaffinity="0"/></body>')
    xml_txt = xml_txt.replace("</worldbody>", marker + "</worldbody>", 1)
    mpath = os.path.join(os.path.dirname(a.scene), "_render_marked.xml")
    open(mpath, "w").write(xml_txt)
    m = mujoco.MjModel.from_xml_path(mpath)
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=240, width=424)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0
    vw = cv2.VideoWriter(os.path.join(a.out, "_raw.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (864, 300))
    for i in range(N):
        for qpos, sealed, tgt in traj[i]:
            d.qpos[:len(qpos)] = qpos
            if a.mode == "pnp":
                d.mocap_pos[0] = [tgt[0], tgt[1], 0.002]
            mujoco.mj_forward(m, d)
            frames = []
            for cam in ("wrist_d405", "fixed_d435"):
                ren.update_scene(d, camera=cam, scene_option=vopt)
                frames.append(cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR))
            row = np.hstack([cv2.resize(f, (432, 240)) for f in frames])
            canvas = np.zeros((300, 864, 3), np.uint8)
            canvas[60:] = row
            label = (f"{a.mode} episode {i+1}/{N}  "
                     f"({'SUCCESS' if results[i] else 'fail'})"
                     + (f"  target=({tgt[0]:.2f},{tgt[1]:.2f})" if a.mode == "pnp" else ""))
            cv2.putText(canvas, label, (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if sealed:
                cv2.putText(canvas, "SEALED", (720, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
                cv2.rectangle(canvas, (0, 60), (863, 299), (0, 180, 255), 3)
            vw.write(canvas)
        for _ in range(5):                            # pause between episodes
            vw.write(canvas)
    vw.release()
    print(f"[demo] wrote {a.out}/_raw.mp4")


if __name__ == "__main__":
    main()

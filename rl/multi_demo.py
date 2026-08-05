"""multi_demo :: sequential multi-object clearing with the ppo4 policy.

Composed architecture: for each object -- scripted IK reach to hover,
then the LEARNED pick-and-place policy carries it to its assigned target
and sets it down. Renders the whole sequence (targets = green discs).

    python rl/multi_demo.py --actor ~/pnp_rl/ppo4/final.pt
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
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

OBJECTS = ["object0", "object1", "object2", "object3"]
TARGETS = [(0.28, -0.135), (0.28, -0.045), (0.28, 0.045), (0.28, 0.135)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default=os.path.expanduser("~/pnp_rl/ppo4/final.pt"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "multi4.xml"))
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/multi_demo"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wp.init()
    from env_warp import PickEnv, CUP_R
    from ppo import AC

    env = PickEnv(nworld=1, mode="pnp", xml=a.scene)
    env.auto_reset = False
    # restore ALL objects to their authored scene poses (init reset moved
    # the default object into the attach-style start)
    import mujoco_warp as mjw
    q0 = torch.tensor(env.mjm.qpos0, device=env.device, dtype=torch.float32)
    env.qpos[0, 6:] = q0[6:]
    env.qvel[0, :] = 0.0
    mjw.forward(env.m, env.d)
    net = AC().to(env.device)
    ck = torch.load(a.actor, map_location=env.device, weights_only=False)
    net.load_state_dict(ck["ac"])
    net.eval()

    # CPU IK for the scripted reach
    demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    mj_m = env.mjm
    dik = mujoco.MjData(mj_m)
    name2id = mujoco.mj_name2id

    frames = []          # (full qpos, sealed, cur_target, cur_obj_idx)
    results = []
    for oi, (oname, tgt) in enumerate(zip(OBJECTS, TARGETS)):
        # --- select object: repoint env accessors ---
        env.bid_obj = name2id(mj_m, mujoco.mjtObj.mjOBJ_BODY, oname)
        env.jadr_obj = mj_m.jnt_qposadr[mj_m.body_jntadr[env.bid_obj]]
        env.vadr_obj = mj_m.jnt_dofadr[mj_m.body_jntadr[env.bid_obj]]
        env.obj_mass = float(mj_m.body_mass[env.bid_obj])
        env.place_target[0] = torch.tensor(tgt, device=env.device)
        env.xfrc[:] = 0.0
        env.sealed[:] = False
        env.ever_sealed[:] = False
        env.sat_count[:] = 0
        env.t_step[:] = 0
        env.ep_comp[:] = 0.0
        # --- scripted reach: IK hover 3 cm above the object ---
        op = env._obj_pos()[0].cpu().numpy()
        q_now = env.qpos[0, :6].cpu().numpy()
        q_hov, e = demo["ik"](mj_m, dik, "tcp",
                              [float(op[0]), float(op[1]),
                               float(op[2]) + float(env.half[2]) + CUP_R + 0.03],
                              demo["R_DOWN"], q_now)
        assert e < 0.01, f"reach IK failed for {oname}"
        # animate the reach as a short interpolation (visual continuity)
        for s_ in np.linspace(0, 1, 8):
            q_i = (1 - s_) * q_now + s_ * q_hov
            env.qpos[0, :6] = torch.tensor(q_i, device=env.device, dtype=torch.float32)
            env.qvel[0, :6] = 0.0
            import mujoco_warp as mjw
            mjw.forward(env.m, env.d)
            frames.append((env.qpos[0].cpu().numpy().copy(), False, tgt, oi))
        env.q_target[0] = env.qpos[0, :6]
        env.phi_approach[0] = -torch.norm(env._tcp()[0][0] - env._grasp_point()[0])
        env.phi_transport[0] = -torch.norm(
            env._obj_pos()[0, :2] - env.place_target[0])
        env.phi_lift[0] = 0.0
        # --- learned pick-and-place ---
        obs = env.observe()
        ok = False
        for t in range(110):
            with torch.no_grad():
                act = torch.tanh(net.pi(obs))
            obs, r, done, info = env.step(act)
            frames.append((env.qpos[0].cpu().numpy().copy(),
                           bool(env.sealed[0]), tgt, oi))
            if done[0]:
                ok = bool(info["placed"][0])
                break
        d_final = float(torch.norm(env._obj_pos()[0, :2] - env.place_target[0]))
        results.append((oname, ok, d_final))
        print(f"[multi] {oname}: {'PLACED' if ok else 'fail'} "
              f"(final dist {100*d_final:.1f} cm)", flush=True)

    print("[multi] summary:", [(n, "OK" if ok else "fail") for n, ok, _ in results])

    # ---- render ----
    import cv2
    xml_txt = open(a.scene).read()
    marker = ('<body name="place_marker" mocap="true" pos="0 0 -1">'
              '<geom type="cylinder" size="0.035 0.0015" rgba="0.1 0.9 0.2 0.6" '
              'contype="0" conaffinity="0"/></body>')
    xml_txt = xml_txt.replace("</worldbody>", marker + "</worldbody>", 1)
    mpath = os.path.join(os.path.dirname(a.scene), "_multi_marked.xml")
    open(mpath, "w").write(xml_txt)
    m2 = mujoco.MjModel.from_xml_path(mpath)
    d2 = mujoco.MjData(m2)
    ren = mujoco.Renderer(m2, height=240, width=424)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0
    vw = cv2.VideoWriter(os.path.join(a.out, "_raw.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (864, 300))
    for qpos, sealed, tgt, oi in frames:
        d2.qpos[:len(qpos)] = qpos
        d2.mocap_pos[0] = [tgt[0], tgt[1], 0.002]
        mujoco.mj_forward(m2, d2)
        ims = []
        for cam in ("fixed_d435", "top"):
            ren.update_scene(d2, camera=cam, scene_option=vopt)
            ims.append(cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR))
        row = np.hstack([cv2.resize(f, (432, 240)) for f in ims])
        canvas = np.zeros((300, 864, 3), np.uint8)
        canvas[60:] = row
        nm, ok, dd = results[oi] if oi < len(results) else (OBJECTS[oi], False, 0)
        cv2.putText(canvas, f"object {oi+1}/4 -> target {oi+1}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if sealed:
            cv2.putText(canvas, "SEALED", (720, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
            cv2.rectangle(canvas, (0, 60), (863, 299), (0, 180, 255), 3)
        vw.write(canvas)
    vw.release()
    print(f"[multi] wrote {a.out}/_raw.mp4 ({len(frames)} frames)")


if __name__ == "__main__":
    main()

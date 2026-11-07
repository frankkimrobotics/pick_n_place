#!/usr/bin/env python3
"""rollout_sim_zed :: closed-loop MuJoCo evaluation of a policy/train_policy_zed.py checkpoint
(dp / cfm, unet / dit / mlp) -- the simulator counterpart of policy/real_rollout_zed.py.

Scenes come from dataset_gen_div.sample_scene(seed) (unseen seeds), built by sim_robot_mjcf with the
calibrated wrist D405 and the fixed ZED camera injected exactly as the generators do. Each 10 Hz
decision: render both cameras (424x240, resized to 432x240 like the ffmpeg-written dataset frames),
lingbot-encode, build the UMI observation (2 frames x 2 cameras + relative EE poses + suction),
sample a 16-step relative-EE chunk every --n_exec decisions, convert the current action to an
absolute target, IK (same routine and guards as the real executor: yaw-locked tilt, step / joint
clips, z floor), then track a 0.1 s linear joint segment with the generators' computed-torque PD
(mjwarp_pick_demo._gains_and_ff / pd_tau, 4 ms ticks, 2 ms physics).

Suction follows the generators' physics: a predicted suction-on latches the weld only when the cup
tip presses an object face-on (contact force > SEAL_N and normal tilt < SEAL_DEG); predicted-off
for --suction_off_n consecutive decisions releases it.

Per scene: sealed, lifted (object +5 cm while attached), released, placed (released inside the bin
box), press centring error (tip xy vs the nearest object top centre when the tip first comes within
3 cm of an object top), time to seal. Writes metrics.json (+ mp4 with --render).

    PY=~/miniconda3/envs/ilpolicy/bin/python
    $PY policy/rollout_sim_zed.py --ckpt ~/pnp_policies/dp_lingbot_all1050/ckpt_final.pt --scenes 10 --render
"""
import argparse
import json
import os
import sys
import time
from collections import deque

os.environ.setdefault("MUJOCO_GL", "osmesa")
import torch  # noqa: E402
# triton + diffusers MUST be imported before mujoco: with MUJOCO_GL=osmesa, loading triton's C extension
# after mujoco segfaults (library clash); the real executor imports diffusers lazily, so do it here
try:
    import triton  # noqa: E402,F401
    import diffusers  # noqa: E402,F401
except ImportError:
    pass
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import mujoco  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import real_rollout_zed as Z  # noqa: E402  (ZedPolicy, ik, rot6d_to_R, rel_pose, R_DOWN, tilt_deg)
import dataset_gen_div as G  # noqa: E402  (sample_scene, build_scene_xml, LIGHTS, W, H, BIN_XY, BIN_HALF)

ZED_CAM = ('<camera name="zed" pos="0.6014 -0.0870 0.6772" '
           'xyaxes="0.0723 0.9968 0.0349 -0.9062 0.0510 0.4197" fovy="69.1"/>')
DEMO = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
exec(open(DEMO["__file__"]).read().split("if __name__")[0], DEMO)
# the sim demos' IK only had the +-170 deg wrist range; the real executor's LinuxCNC bounds (j4 >= -85 deg)
# make some demo configurations unreachable here -> widen them for the simulator
Z.LC_LO[:] = -2.0 * np.pi
Z.LC_HI[:] = 2.0 * np.pi
CTRL_DT, NSUB = DEMO["CTRL_DT"], DEMO["NSUB"]           # 4 ms ticks, 2 physics substeps
TICKS = int(round(0.10 / CTRL_DT))                      # 25 ticks per 10 Hz decision


def build_scene(seed, out_xml):
    objs = G.sample_scene(seed)
    G.build_scene_xml(objs, G.LIGHTS[seed % 3], out_xml)
    x = open(out_xml).read()
    if 'name="zed"' not in x:
        x = x.replace('<camera name="fixed_d435"', ZED_CAM + '\n    <camera name="fixed_d435"', 1)
        open(out_xml, "w").write(x)
    return objs


def contacting_object(m, d, cup_gid, obj_bids):
    """Body id of the object whose geom the cup tip touches (None if none)."""
    for i in range(d.ncon):
        c = d.contact[i]
        if cup_gid in (c.geom1, c.geom2):
            g = c.geom2 if c.geom1 == cup_gid else c.geom1
            b = m.geom_bodyid[g]
            if b in obj_bids:
                return b
    return None


def run_episode(pol, seed, a, out_dir):
    xml = os.path.join(out_dir, f"scene_{seed}.xml")
    objs = build_scene(seed, xml)
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    d.qpos[:6] = DEMO["Q_START"]
    mujoco.mj_forward(m, d)
    dik = mujoco.MjData(m)
    tcp_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    tcp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
    cup_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")
    obj = []
    for o in objs:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
        eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{o['name']}")
        obj.append(dict(name=o["name"], bid=bid, eq=eq, top=float(d.xpos[bid][2] + o["half_extents"][2]),
                        p0=d.xpos[bid].copy()))
    obj_bids = {o["bid"]: o for o in obj}
    ren = mujoco.Renderer(m, height=G.H, width=G.W)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0

    def render(cam):
        ren.update_scene(d, camera=cam, scene_option=vopt)
        return cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)

    hist = deque(maxlen=2)
    state = dict(suction=0, attached=None, off=0)
    met = dict(seed=seed, n_objects=len(objs), sealed=False, seal_k=None, lifted=False, released=False,
               placed=False, placed_k=None, press_err_cm=None, press_obj=None, min_tip_z=1.0, holds=0,
               attached_obj=None, release_pos=None)
    q_cmd = np.asarray(DEMO["Q_START"], float).copy()
    plan = None
    frames = []
    t0 = time.time()
    for k in range(a.steps):
        p = d.site_xpos[tcp_sid].copy()
        R = d.site_xmat[tcp_sid].reshape(3, 3).copy()
        wrist, zed = render("wrist_d405"), render("zed")
        hist.append(dict(p=p, R=R, suc=state["suction"], wrist_tok=pol.encode(wrist), zed_tok=pol.encode(zed)))
        if a.render:
            fr = zed.copy()
            fr[4:4 + 80, 4:4 + 141] = cv2.resize(wrist, (141, 80), interpolation=cv2.INTER_AREA)
            cv2.putText(fr, f"k{k} suc{state['suction']}", (150, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frames.append(fr)
        # ---- plan
        if plan is None or (k - plan["k_obs"]) >= a.n_exec:
            obs = [hist[-2] if len(hist) >= 2 else hist[-1], hist[-1]]
            tok4 = torch.stack([obs[0]["wrist_tok"], obs[1]["wrist_tok"], obs[0]["zed_tok"], obs[1]["zed_tok"]])
            p_t, R_t = obs[1]["p"], obs[1]["R"]
            prop = np.stack([Z.rel_pose(p_t, R_t, obs[0]["p"], obs[0]["R"], obs[0]["suc"]),
                             Z.rel_pose(p_t, R_t, p_t, R_t, obs[1]["suc"])])
            act = pol(tok4, prop)
            plan = dict(k_obs=k, p_abs=np.stack([p_t + R_t @ act[i, :3] for i in range(act.shape[0])]),
                        R_abs=np.stack([R_t @ Z.rot6d_to_R(act[i, 3:9]) for i in range(act.shape[0])]),
                        suc=act[:, 9].copy())
        i = min(k - plan["k_obs"], len(plan["suc"]) - 1)
        tgt_p, tgt_R, suc_pred = plan["p_abs"][i].copy(), plan["R_abs"][i].copy(), float(plan["suc"][i])
        z = tgt_R[:, 2] / (np.linalg.norm(tgt_R[:, 2]) + 1e-9)          # yaw lock (as the real executor)
        x = Z.R_DOWN[:, 0] - z * (z @ Z.R_DOWN[:, 0])
        if np.linalg.norm(x) > 1e-3:
            x /= np.linalg.norm(x)
            tgt_R = np.stack([x, np.cross(z, x), z], axis=1)
        # ---- suction (physics seal check instead of the robot's height gate)
        if suc_pred > 0.5:
            state["off"] = 0
            if state["suction"] == 0:
                f, tilt = DEMO["_cup_contact"](m, d, cup_gid)
                b = contacting_object(m, d, cup_gid, obj_bids)
                if b is not None and f > DEMO["SEAL_N"] and tilt < DEMO["SEAL_DEG"]:
                    o = obj_bids[b]
                    DEMO["_latch_weld"](m, d, o["eq"], tcp_bid, o["bid"])
                    state["suction"], state["attached"] = 1, o
                    met["sealed"], met["seal_k"], met["attached_obj"] = True, k, o["name"]
        else:
            state["off"] += 1
            if state["suction"] == 1 and state["off"] >= a.suction_off_n:
                o = state["attached"]
                d.eq_active[o["eq"]] = 0
                state["suction"], state["attached"] = 0, None
                met["released"], met["release_pos"] = True, np.round(d.xpos[o["bid"]], 4).tolist()
        # ---- press centring error: first time the tip comes within 3 cm of an object top
        if met["press_err_cm"] is None:
            for o in obj:
                if abs(p[2] - o["top"]) < 0.03 and np.linalg.norm(p[:2] - d.xpos[o["bid"]][:2]) < 0.08:
                    errs = [(np.linalg.norm(p[:2] - d.xpos[oo["bid"]][:2]), oo["name"]) for oo in obj]
                    e, nm = min(errs)
                    met["press_err_cm"], met["press_obj"] = round(100 * e, 2), nm
                    break
        if state["attached"] is not None and d.xpos[state["attached"]["bid"]][2] > state["attached"]["p0"][2] + 0.05:
            met["lifted"] = True
        met["min_tip_z"] = min(met["min_tip_z"], float(p[2]))
        # ---- guards + IK (mirrors real_rollout_zed)
        p_cmd = d.site_xpos[tcp_sid].copy() if k == 0 else p_cmd_prev
        step = tgt_p - p_cmd
        n = float(np.linalg.norm(step))
        if n > a.step_max:
            tgt_p = p_cmd + step * (a.step_max / n)
        tgt_p[2] = max(tgt_p[2], a.z_floor)
        q_ik, pe, re = Z.ik(m, dik, tcp_sid, tgt_p, tgt_R, q_cmd)
        q_tgt = q_cmd.copy()
        if pe <= 0.003 and re <= 0.05:
            dq = q_ik - q_cmd
            mx = float(np.degrees(np.abs(dq)).max())
            if mx > a.dq_max:
                dq *= a.dq_max / mx
            q_tgt = q_cmd + dq
        else:
            met["holds"] += 1
        # ---- track the segment with the generators' computed-torque PD
        qref = np.linspace(q_cmd, q_tgt, TICKS + 1)[1:]
        qdref, tau_ff, kp, kd = DEMO["_gains_and_ff"](m, d, qref)
        sched = dict(qref=qref, qdref=qdref, tau_ff=tau_ff, kp=kp, kd=kd)
        for t in range(TICKS):
            for _ in range(NSUB):
                d.ctrl[:6] = DEMO["pd_tau"](sched, t, d.qpos[:6], d.qvel[:6])
                mujoco.mj_step(m, d)
        q_cmd = q_tgt
        dik.qpos[:6] = q_cmd
        mujoco.mj_kinematics(m, dik)
        p_cmd_prev = dik.site_xpos[tcp_sid].copy()
        # ---- placed?
        if met["released"] and not met["placed"]:
            o = obj_bids[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, met["attached_obj"])]
            po = d.xpos[o["bid"]]
            if np.all(np.abs(po[:2] - G.BIN_XY) < G.BIN_HALF) and po[2] < 0.0:
                met["placed"], met["placed_k"] = True, k
        if met["placed"] and k > met["placed_k"] + 10:
            break
    met["steps"] = k + 1
    met["wall_s"] = round(time.time() - t0, 1)
    met["objects_final"] = {o["name"]: np.round(d.xpos[o["bid"]], 3).tolist() for o in obj}
    if a.render and frames:
        vw = cv2.VideoWriter(os.path.join(out_dir, f"scene_{seed}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (G.W, G.H))
        for fr in frames:
            vw.write(fr)
        vw.release()
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=5000, help="dataset_gen_div.sample_scene seeds (training used 0-199)")
    ap.add_argument("--steps", type=int, default=300, help="decisions per scene (0.1 s each)")
    ap.add_argument("--n_exec", type=int, default=8)
    ap.add_argument("--euler", type=int, default=16)
    ap.add_argument("--step_max", type=float, default=0.02)
    ap.add_argument("--dq_max", type=float, default=3.0)
    ap.add_argument("--z_floor", type=float, default=0.005)
    ap.add_argument("--suction_off_n", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seeds (overrides --seed0/--scenes)")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    out_dir = a.out or os.path.expanduser(f"~/pnp_rollouts_zed/{run}")
    os.makedirs(out_dir, exist_ok=True)
    pol = Z.ZedPolicy(a.ckpt, euler=a.euler, frozen_noise=True)
    mets = []
    seeds = a.seeds if a.seeds else [a.seed0 + s for s in range(a.scenes)]
    for seed in seeds:
        met = run_episode(pol, seed, a, out_dir)
        mets.append(met)
        print(f"[{run}] seed {met['seed']}: objs {met['n_objects']} sealed {met['sealed']} (k {met['seal_k']}) "
              f"lifted {met['lifted']} released {met['released']} placed {met['placed']} press_err "
              f"{met['press_err_cm']} cm on {met['press_obj']} min_tip_z {met['min_tip_z']:.3f} holds {met['holds']} "
              f"({met['wall_s']} s)", flush=True)
        json.dump(dict(ckpt=a.ckpt, args=vars(a), scenes=mets), open(os.path.join(out_dir, "metrics%s.json" % ("_seeds" if a.seeds else "")), "w"), indent=1)
    n = len(mets)
    errs = [m_["press_err_cm"] for m_ in mets if m_["press_err_cm"] is not None]
    print(f"[{run}] {n} scenes: sealed {sum(m_['sealed'] for m_ in mets)}/{n}  lifted {sum(m_['lifted'] for m_ in mets)}/{n}  "
          f"placed {sum(m_['placed'] for m_ in mets)}/{n}  reached an object {len(errs)}/{n}  "
          f"press centring err median {np.median(errs) if errs else float('nan'):.1f} cm", flush=True)


if __name__ == "__main__":
    main()

"""rollout_sim :: closed-loop policy-in-the-loop evaluation in MuJoCo.

Feeds train_lit policy inference back into the simulation: at 10 Hz the
policy sees exactly the training observation (dual-cam JPEG-domain images
x2 steps, proprio, goal grasp pose relative to current tcp) and returns 16
B-spline control points (+ per-step suction). The spline is evaluated at
a 1 kHz control tick (physics timestep dropped to 1 ms, NSUB=1) through
the same PD + RNEA-feedforward controller shape used for dataset
generation, receding horizon: the first --exec_steps x 0.1 s are
executed, then re-infer.

Chunk splices are CONTINUOUS by construction: each chunk is evaluated at
its true phase (spline s = clip(tau - 0.1, 0, 1.5), since s=0 encodes the
action for t+0.1 s) and the whole window carries a positional alignment
offset (q_now - chunk_start) decaying min-jerk over --blend_s (0.3 s), so
the commanded reference never steps at a re-inference boundary.

Suction is policy-commanded: when suction_cmd >= 0.5 and the cup contact
passes the seal check (same SEAL_N / SEAL_DEG as the dataset), the weld
latches; when < 0.5 it releases.

Scenes are generated with dataset_gen.sample_scene(idx) -- pick --scene
indices >= 600 for configurations unseen in training.

Run in the sam3 env (torch + mujoco + open_clip):
    MUJOCO_GL=osmesa python rollout_sim.py --ckpt ~/pnp_runs/dp_q/ckpt_final.pt \
        --scene 700 --out ~/pnp_rollouts
    # mechanics smoke test without a trained model:
    MUJOCO_GL=osmesa python rollout_sim.py --untrained dp --scene 700 --steps_max 30
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import train_lit as tl                                     # noqa: E402
import convert_litdata as cl                               # noqa: E402
import dataset_gen as dg                                   # noqa: E402
import mujoco                                              # noqa: E402

W, H = 424, 240
REC_HZ = 10


def jpeg_domain(rgb):
    """Match the dataset pixel pipeline: 424x240 render -> 432x240 (ffmpeg
    macroblock resize) -> JPEG q90 roundtrip -> bytes."""
    img = cv2.resize(rgb, (432, 240))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


class Policy:
    def __init__(self, a, dev):
        self.dev = dev
        if a.ckpt:
            ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
            self.model, self.action = ck["model"], ck["action"]
            self.spec = ck["spec"]
            self.lo = ck["lo"].to(dev)
            self.hi = ck["hi"].to(dev)
            self.ctx = tl.Context(512).to(dev)
            self.net = (tl.DiT(512) if self.model in ("dp", "cfm")
                        else tl.ACT(512)).to(dev)
            src = ck["ema"] if ck.get("ema") is not None else None
            if src is not None:
                self.net.load_state_dict({k: v for k, v in src[0].items()})
                self.ctx.load_state_dict({k: v for k, v in src[1].items()})
            else:
                self.net.load_state_dict(ck["net"])
                self.ctx.load_state_dict(ck["ctx"])
        else:                                             # --untrained smoke
            self.model, self.action = a.untrained, "q"
            self.spec = json.load(open(os.path.join(
                os.path.expanduser(a.data), "dataset_spec.json")))
            self.lo = torch.full((7,), -1.0, device=dev)
            self.hi = torch.full((7,), 1.0, device=dev)
            self.ctx = tl.Context(512).to(dev)
            self.net = (tl.DiT(512) if self.model in ("dp", "cfm")
                        else tl.ACT(512)).to(dev)
        self.ctx.eval()
        self.net.eval()
        if self.model == "dp":
            from diffusers import DDIMScheduler
            self.isched = DDIMScheduler(num_train_timesteps=50,
                                        beta_schedule="squaredcos_cap_v2",
                                        prediction_type="epsilon")
            self.isched.set_timesteps(16)

    @torch.no_grad()
    def __call__(self, jpgs, prop, goal):
        imgs = np.stack([tl.decode(j) for j in jpgs])
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)[None].to(self.dev)
        prop = torch.from_numpy(prop)[None].to(self.dev).float()
        goal = torch.from_numpy(goal)[None].to(self.dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.dev == "cuda"):
            ctx = self.ctx(imgs, prop, goal)
            if self.model == "dp":
                x = torch.randn(1, tl.TA, tl.ADIM, device=self.dev)
                for t in self.isched.timesteps:
                    tb = t.expand(1).to(self.dev)
                    x = self.isched.step(self.net(x, tb, ctx).float(),
                                         t, x).prev_sample
            elif self.model == "cfm":
                x = torch.randn(1, tl.TA, tl.ADIM, device=self.dev)
                for k in range(16):
                    tb = torch.full((1,), k / 16, device=self.dev)
                    x = x + self.net(x, tb * 50, ctx).float() / 16
            else:
                x, _, _ = self.net(ctx, act=None)
        x = (x.float()[0] + 1) / 2 * (self.hi - self.lo) + self.lo
        ctrl = x[:, :6].cpu().numpy().astype(np.float64)     # (16, 6)
        suction = (x[:, 6] > 0.5).cpu().numpy()              # (16,)
        return ctrl, suction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--untrained", choices=["dp", "cfm", "act"], default=None)
    ap.add_argument("--data", default=os.path.expanduser("~/pnp_litdata"),
                    help="dataset dir (spec for --untrained)")
    ap.add_argument("--scene", type=int, default=700)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rollouts"))
    ap.add_argument("--exec_steps", type=int, default=2,
                    help="0.1 s action steps executed per inference "
                         "(2 = 5 Hz re-planning; TRT-measured inference is "
                         "13 ms, so even 1 is feasible on-robot)")
    ap.add_argument("--blend_s", type=float, default=0.2,
                    help="min-jerk decay horizon of the splice alignment offset")
    ap.add_argument("--steps_max", type=int, default=150,
                    help="max policy steps per pick episode")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    assert a.ckpt or a.untrained, "need --ckpt or --untrained"
    torch.manual_seed(0)
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    pol = Policy(a, dev)
    tag = os.path.splitext(os.path.basename(a.ckpt))[0] if a.ckpt \
        else f"untrained_{a.untrained}"
    odir = os.path.join(a.out, f"scene{a.scene}_{pol.model}_{tag}")
    os.makedirs(odir, exist_ok=True)

    # ---- scene + controller plumbing (same as dataset_gen) ----
    demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    objs = dg.sample_scene(a.scene)
    light = dg.LIGHTS[a.scene % 3]
    xml = os.path.join(odir, "scene.xml")
    dg.build_scene_xml(objs, light, xml)
    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.timestep = 0.001            # 1 kHz control => 1 ms physics, NSUB=1
    d = mujoco.MjData(m)
    d.qpos[:6] = demo["Q_START"]
    mujoco.mj_forward(m, d)
    ren = mujoco.Renderer(m, height=H, width=W)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0
    rf_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,
                                            "tip_range")]
    tcp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
    tcp_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    cup_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")
    CTRL_DT, NSUB = 0.001, 1
    demo["CTRL_DT"] = CTRL_DT         # _gains_and_ff differentiates with this
    ticks_per_step = int(round(0.1 / CTRL_DT))               # 100

    spec = pol.spec
    # true phase: at elapsed tau the intended action is spline(tau - 0.1)
    # (s=0 encodes t+0.1); clamp holds the first/last knot at the edges
    exec_tau = np.arange(0, a.exec_steps * 0.1, CTRL_DT)
    dense_s = np.clip(exec_tau - 0.1, 0.0, None)
    DENSE = np.stack([cl.bspline_basis(s) for s in dense_s])
    # min-jerk decay weight for the splice alignment offset
    u = np.clip(exec_tau / max(a.blend_s, 1e-6), 0.0, 1.0)
    BLEND_W = 1.0 - (10 * u**3 - 15 * u**4 + 6 * u**5)       # 1 -> 0, smooth

    def render_cam(cam):
        ren.update_scene(d, camera=cam, scene_option=vopt)
        return ren.render()

    def tcp_pose():
        p = d.site_xpos[tcp_sid].copy()
        R = d.site_xmat[tcp_sid].reshape(3, 3).copy()
        return p, R

    quat_down = np.zeros(4)
    mujoco.mju_mat2Quat(quat_down, np.asarray(demo["R_DOWN"], float).ravel())

    video = cv2.VideoWriter(os.path.join(odir, "rollout.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), REC_HZ, (2 * 432, 240))
    metrics = dict(scene=a.scene, model=pol.model, action=pol.action,
                   ckpt=a.ckpt, picks=[])
    # per-4ms-tick trajectory log: time, actual q, commanded qref, suction,
    # plus the wall of every policy-inference splice point
    tlog = dict(t=[], q=[], qref=[], suction=[], infer_t=[], sim_t=0.0)

    order = sorted(range(len(objs)),
                   key=lambda i: np.hypot(*np.array(objs[i]["pos"][:2])))
    for oi in order:
        o = objs[oi]
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
        eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                  f"suction_{o['name']}")
        latched = was_latched = False
        prev = None
        t0 = time.time()
        for step in range(a.steps_max):
            # ---- observation (training domain) ----
            j405 = jpeg_domain(render_cam("wrist_d405"))
            j435 = jpeg_domain(render_cam("fixed_d435"))
            if prev is None:
                prev = (j405, j435, d.qpos[:6].copy(), d.qvel[:6].copy())
            q_now, qd_now = d.qpos[:6].copy(), d.qvel[:6].copy()
            prop = np.stack([
                np.concatenate([prev[2], prev[3], [float(latched)],
                                [d.sensordata[rf_adr]]]),
                np.concatenate([q_now, qd_now, [float(latched)],
                                [d.sensordata[rf_adr]]])]).astype(np.float32)
            p_obj = d.xpos[bid].copy()
            g_pos = np.array([p_obj[0], p_obj[1],
                              p_obj[2] + o["half_extents"][2] + demo["CUP_R"]])
            p_t, R_t = tcp_pose()
            R_g = tl.quat_to_R(quat_down.astype(np.float32))
            dR = R_t.T @ R_g
            goal = np.concatenate([R_t.T @ (g_pos - p_t),
                                   dR[:, 0], dR[:, 1]]).astype(np.float32)
            jpgs = [prev[0], j405, prev[1], j435]
            ctrl, suc_cmd = pol(jpgs, prop, goal)

            # ---- execute first exec_steps of the chunk at 4 ms ----
            qref = DENSE @ ctrl                              # (n_ticks, 6)
            # splice alignment: shift the whole window so it starts exactly
            # at the current joint state, decaying to the raw chunk
            qref = qref + BLEND_W[:, None] * (d.qpos[:6] - qref[0])
            qdref, tau_ff, kp, kd = demo["_gains_and_ff"](m, d, qref)
            sched = dict(qref=qref, qdref=qdref, tau_ff=tau_ff, kp=kp, kd=kd)
            prev = (j405, j435, q_now, qd_now)
            tlog["infer_t"].append(tlog["sim_t"])
            for k in range(len(qref)):
                cmd = bool(suc_cmd[min(k // ticks_per_step, 15)])
                if cmd and not latched:
                    f, tilt = demo["_cup_contact"](m, d, cup_gid)
                    if f > demo["SEAL_N"] and tilt < demo["SEAL_DEG"]:
                        demo["_latch_weld"](m, d, eq_id, tcp_bid, bid)
                        latched = was_latched = True
                if not cmd and latched:
                    d.eq_active[eq_id] = 0
                    latched = False
                for _ in range(NSUB):
                    d.ctrl[:6] = demo["pd_tau"](sched, k, d.qpos[:6],
                                                d.qvel[:6])
                    mujoco.mj_step(m, d)
                tlog["t"].append(tlog["sim_t"])
                tlog["q"].append(d.qpos[:6].copy())
                tlog["qref"].append(qref[k].copy())
                tlog["suction"].append(int(latched))
                tlog["sim_t"] += CTRL_DT
                if k % ticks_per_step == 0:
                    fa = cv2.resize(cv2.cvtColor(render_cam("wrist_d405"),
                                                 cv2.COLOR_RGB2BGR), (432, 240))
                    fb = cv2.resize(cv2.cvtColor(render_cam("fixed_d435"),
                                                 cv2.COLOR_RGB2BGR), (432, 240))
                    video.write(np.hstack([fa, fb]))
            if was_latched and not latched:
                break                                       # placed (or dropped)
        if latched:                                         # never released
            d.eq_active[eq_id] = 0
            latched = False
        p = d.xpos[bid]
        ok = bool(abs(p[0] - dg.BIN_XY[0]) < dg.BIN_HALF
                  and abs(p[1] - dg.BIN_XY[1]) < dg.BIN_HALF and p[2] < 0.12)
        metrics["picks"].append(dict(
            object=oi, kind=o["kind"], success=ok, sealed=bool(was_latched),
            policy_steps=step + 1, wall_s=round(time.time() - t0, 1)))
        print(f"[rollout] object{oi} ({o['kind']}): sealed={was_latched} "
              f"in_bin={ok} steps={step+1}", flush=True)

    video.release()
    np.savez_compressed(os.path.join(odir, "rollout_traj.npz"),
                        t=np.array(tlog["t"]), q=np.array(tlog["q"]),
                        qref=np.array(tlog["qref"]),
                        suction=np.array(tlog["suction"]),
                        infer_t=np.array(tlog["infer_t"]))
    n_ok = sum(p["success"] for p in metrics["picks"])
    metrics["n_success"] = n_ok
    json.dump(metrics, open(os.path.join(odir, "metrics.json"), "w"), indent=1)
    print(f"[rollout] {n_ok}/{len(objs)} in bin -> {odir}", flush=True)


if __name__ == "__main__":
    main()

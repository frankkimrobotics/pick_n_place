#!/usr/bin/env python3
"""diffog_pnp :: DiffOG layer on mujoco_warp pick-and-place with uncertain
object dimensions and a 0.25 m cubic placement box at (0.08, 0.40).

Scene: 3 objects with randomized half-extents; the BELIEVED dims (what the
teacher/layer see) carry Gaussian error (~1 cm), corrected at seal time from
the tcp height (the twin's stand-in for touch calibration). Teacher = IK
waypoints + cosine easing, planned on BELIEVED geometry, so its raw chunks
are exactly as flawed as real perception would make them. The DiffOG layer
refines each chunk under hard velocity/accel/joint constraints + linearized
collision rows (box walls, distractors, table) built from believed geometry
with margins inflated by the current dimension uncertainty.

    python3 diffog_pnp.py --make_data --episodes 40      # teacher chunks
    python3 diffog_pnp.py --train                        # fit Q-net
    python3 diffog_pnp.py --eval --episodes 8 --render   # closed loop + video
"""
import argparse
import os as _os
_os.environ.setdefault("MUJOCO_GL", "egl")
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import warp as wp                                          # noqa: E402
from diffog_layer import (DT, VMAX, DiffOGLayer, collision_rows,  # noqa: E402
                          qp_solve)

BOX_C = np.array([0.08, 0.40])          # placement box centre (user spec)
BOX_IN = 0.25                            # interior side + wall height
WALL_T = 0.006
H = 12                                   # chunk horizon (1.2 s)
OUT = os.path.expanduser("~/pnp_rl/diffog")


# ------------------------------------------------------------------ scene
SCENE_TMPL = """
<mujoco model="diffog_pnp">
  <option timestep="0.002" integrator="implicitfast"/>
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
  <include file="{incfile}"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.12" rgba="0.35 0.37 0.40 1"/>
    <geom name="table" type="box" pos="0.38 0.00 -0.01" size="0.20 0.22 0.01" rgba="0.55 0.44 0.35 1"/>
    <geom name="pbx_floor" type="box" pos="{bx} {by} 0.025" size="{hi} {hi} 0.025" rgba="0.2 0.55 0.3 1"/>
    <geom name="pbx_xm" type="box" pos="{xm} {by} {hz}" size="{wt} {ho} {hz}" rgba="0.2 0.55 0.3 0.6"/>
    <geom name="pbx_xp" type="box" pos="{xp} {by} {hz}" size="{wt} {ho} {hz}" rgba="0.2 0.55 0.3 0.6"/>
    <geom name="pbx_ym" type="box" pos="{bx} {ym} {hz}" size="{ho} {wt} {hz}" rgba="0.2 0.55 0.3 0.6"/>
    <geom name="pbx_yp" type="box" pos="{bx} {yp} {hz}" size="{ho} {wt} {hz}" rgba="0.2 0.55 0.3 0.6"/>
    {objects}
    <camera name="iso" pos="1.05 -0.55 0.78" xyaxes="0.45 0.89 0 -0.5 0.25 0.83"/>
    <camera name="front" pos="0.38 -0.85 0.45" xyaxes="1 0 0 0 0.45 0.89"/>
  </worldbody>
  <equality>
    <weld name="suction_object0" body1="tcp" body2="object0" active="false" solref="0.005 1"/>
  </equality>
</mujoco>
"""
OBJ_TMPL = ('<body name="object{i}" pos="{x} {y} {z}"><freejoint name="obj{i}_free"/>'
            '<geom name="g_object{i}" type="box" size="{hx} {hy} {hz}" solref="0.004 1" '
            'rgba="{rgba}" mass="0.06"/></body>')


def make_scene(rng, out_xml):
    """3 objects, randomized dims/poses; returns true dims + poses."""
    colors = ["0.85 0.15 0.15 1", "0.15 0.35 0.85 1", "0.9 0.7 0.1 1"]
    spots = [(0.34, -0.03), (0.42, 0.10), (0.30, 0.12)]
    objs, meta = [], []
    for i in range(3):
        hx, hy = rng.uniform(0.018, 0.032, 2)
        hz = rng.uniform(0.015, 0.035)
        x = spots[i][0] + rng.uniform(-0.02, 0.02)
        y = spots[i][1] + rng.uniform(-0.02, 0.02)
        objs.append(OBJ_TMPL.format(i=i, x=x, y=y, z=hz + 0.001,
                                    hx=hx, hy=hy, hz=hz, rgba=colors[i]))
        meta.append(dict(pos=[x, y], half=[hx, hy, hz]))
    hi = BOX_IN / 2
    xml = SCENE_TMPL.format(
        meshdir=os.path.join(HERE, "scenes", "meshes"),
        incfile=os.path.join(OUT, "_arm_include.xml"),
        bx=BOX_C[0], by=BOX_C[1], hi=hi, ho=hi + 2 * WALL_T, wt=WALL_T, hz=BOX_IN / 2,
        xm=BOX_C[0] - hi - WALL_T, xp=BOX_C[0] + hi + WALL_T,
        ym=BOX_C[1] - hi - WALL_T, yp=BOX_C[1] + hi + WALL_T,
        objects="\n    ".join(objs))
    open(out_xml, "w").write(xml)
    return meta


def extract_arm_include():
    """Pull the arm subtree out of an existing scene into _arm_include.xml."""
    src = open(os.path.join(HERE, "scenes", "box_med_ped.xml")).read()
    a0 = src.index("<asset>")
    a1 = src.index("</asset>") + len("</asset>")
    b0 = src.index('<body name="base"')
    depth, i = 0, b0
    while True:
        if src.startswith("<body", i):
            depth += 1
            i = src.index(">", i) + 1
        elif src.startswith("</body>", i):
            depth -= 1
            i += len("</body>")
            if depth == 0:
                break
        else:
            i += 1
    arm = src[b0:i]
    ac0 = src.index("<actuator>")
    ac1 = src.index("</actuator>") + len("</actuator>")
    inc = ("<mujocoinclude>\n" + src[a0:a1] +
           "\n<worldbody>\n" + arm + "\n</worldbody>\n" + src[ac0:ac1] +
           "\n</mujocoinclude>")
    open(os.path.join(OUT, "_arm_include.xml"), "w").write(inc)


# ------------------------------------------------------------------ obstacles
def obstacles_for(env, believed, sigma, held=False, skip_obj0=False, dev="cuda"):
    """AABB list [(center, half+margin)] from BELIEVED geometry, margins
    inflated by the dimension uncertainty sigma."""
    obs = []
    hi = BOX_IN / 2
    wz = BOX_IN / 2
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)
    for cx, cy, hx, hy in [
            (BOX_C[0] - hi - WALL_T, BOX_C[1], WALL_T, hi + 2 * WALL_T),
            (BOX_C[0] + hi + WALL_T, BOX_C[1], WALL_T, hi + 2 * WALL_T),
            (BOX_C[0], BOX_C[1] - hi - WALL_T, hi + 2 * WALL_T, WALL_T),
            (BOX_C[0], BOX_C[1] + hi + WALL_T, hi + 2 * WALL_T, WALL_T)]:
        obs.append((t([cx, cy, wz]), t([hx, hy, wz])))
    for i, m in enumerate(believed):
        if skip_obj0 and i == 0:
            continue
        h = np.array(m["half"]) + sigma[i]
        obs.append((t([*m["pos"], h[2]]), t(h)))
    return obs


def arm_points(env):
    """Collision points + jacobians: cup tip, wrist, camera-mount offset,
    and (when sealed) the believed held-object bottom corners."""
    tcp, R = env._tcp()
    xax = wp.to_torch(env.d.xaxis)[:, :6]
    xan = wp.to_torch(env.d.xanchor)[:, :6]

    def jac_at(p):
        return torch.cross(xax, (p[:, None, :] - xan), dim=-1).transpose(1, 2)
    # R[:,:,2] points along the tool axis TOWARD the workpiece (down when
    # grasping): tip is +z_tool from tcp, wrist/camera are -z_tool (above)
    pts = [tcp + 0.019 * R[:, :, 2],                 # cup tip
           tcp - 0.09 * R[:, :, 2],                  # wrist centre (above)
           tcp - 0.13 * R[:, :, 2] + 0.05 * R[:, :, 1]]  # camera mount lobe
    P = torch.stack(pts, 1)
    J = torch.stack([jac_at(p) for p in pts], 1)
    return P, J


# ------------------------------------------------------------------ teacher
class WaypointTeacher:
    """Full-episode decision-target plan from IK waypoints (believed dims),
    cosine-eased; provides analytic H-step chunks and replans (z shift) when
    the belief is corrected at seal."""

    def __init__(self, env, believed, obj_i=0):
        import mujoco
        from mjwarp_pick_demo import ik, R_DOWN
        self.env = env
        self.mujoco = mujoco
        self.ik = ik
        self.R_DOWN = R_DOWN
        self.dik = mujoco.MjData(env.mjm)
        self.believed = believed
        self.obj_i = obj_i
        self.plan = None
        self.t0 = 0

    def build(self, q_now, t_now, otop_bel, from_carry=False):
        env = self.env
        m = self.believed[self.obj_i]
        ox, oy = m["pos"]
        CUP = 0.019
        hover = [ox, oy, otop_bel + CUP + 0.06]
        press = [ox, oy, otop_bel + CUP - 0.045]
        lift = [ox, oy, 0.42]
        over = [BOX_C[0], BOX_C[1], 0.44]
        # descend INTO the box: release 2 cm above its floor (top 0.024)
        drop = [BOX_C[0], BOX_C[1], 0.05 + 2 * m["half"][2] + CUP + 0.02]
        legs = [(hover, 1.6), (press, 3.2), (press, 0.8),
                (lift, 1.6), (over, 2.2), (drop, 1.6), (drop, 0.5)]
        if from_carry:                      # post-seal replan: NO re-press
            legs = legs[3:]
        wps, durs = [], []
        q_prev = np.asarray(q_now, float)
        for xyz, dur in legs:
            q, err = self.ik(env.mjm, self.dik, "tcp", xyz, self.R_DOWN, q_prev)
            wps.append(np.asarray(q, float))
            durs.append(dur)
            q_prev = wps[-1]
        # release happens at the drop dwell; retreat:
        q, _ = self.ik(env.mjm, self.dik, "tcp", over, self.R_DOWN, q_prev)
        wps.append(q)
        durs.append(1.2)
        # assemble decision-tick plan with cosine easing per leg
        plan = [np.asarray(q_now, float)]
        events = {}
        marks = [("release", 2)] if from_carry else \
            [("suction_on", 1), ("release", 6)]
        if from_carry:
            events["suction_on"] = -1       # already on
        for li, (qw, dur) in enumerate(zip(wps, durs)):
            for nm, idx in marks:
                if idx == li:
                    events[nm] = len(plan)          # first tick of this leg
            n = max(2, int(round(dur / DT)))
            u = (1 - np.cos(np.pi * np.linspace(1.0 / n, 1, n))) / 2
            seg = plan[-1][None] + u[:, None] * (qw - plan[-1])[None]
            plan.extend(list(seg))
        self.plan = np.asarray(plan)
        self.t0 = t_now
        self.events = events

    def chunk(self, t_now):
        k = t_now - self.t0
        idx = np.clip(np.arange(k + 1, k + 1 + H), 0, len(self.plan) - 1)
        return self.plan[idx]

    def want_suction(self, t_now):
        k = t_now - self.t0
        return self.events.get("suction_on", 1e9) <= k < self.events.get("release", 1e9)

    def done(self, t_now):
        return (t_now - self.t0) >= len(self.plan) + 5


# ------------------------------------------------------------------ episode
def make_env(seed, render=False):
    from env_warp import PickEnv
    rng = np.random.default_rng(seed)
    os.makedirs(OUT, exist_ok=True)
    xml = os.path.join(OUT, f"scene_{seed}.xml")
    extract_arm_include()
    meta = make_scene(rng, xml)
    env = PickEnv(nworld=1, mode="pnp", dr=False, xml=xml)
    env.auto_reset = False
    return env, meta, rng


def restore_objects(env, meta):
    """PickEnv.__init__ randomizes object0; put every object back at its
    XML pose so sim matches the scene meta (and the teacher's belief)."""
    qa = env.jadr_obj
    env.qpos[0, qa:qa + 3] = torch.tensor(
        [*meta[0]["pos"], meta[0]["half"][2] + 0.001], device=env.device)
    env.qpos[0, qa + 3:qa + 7] = torch.tensor([1.0, 0, 0, 0], device=env.device)
    env.qvel[0, env.vadr_obj:env.vadr_obj + 6] = 0.0
    env.sealed[:] = False
    env.ever_sealed[:] = False
    import mujoco_warp as mjw
    mjw.step(env.m, env.d)


def reset_arm(env, q):
    qt = torch.as_tensor(q, dtype=torch.float32, device=env.device)
    env.qpos[0, :6] = qt
    env.qvel[0, :6] = 0.0
    env.q_target[0] = qt
    env.q_target_prev[0] = qt
    env.q_hist[0] = qt[None]
    env.q_drive[0] = qt
    env.v_drive[0] = 0.0
    env.q_meas_lag[0] = qt
    env.qd_meas_lag[0] = 0.0
    import mujoco_warp as mjw
    mjw.step(env.m, env.d)


def run_episode(env, meta, rng, layer=None, use_layer=True, record=None,
                log_rows=None):
    """One pick-and-place: teacher chunks -> (DiffOG | clip) -> measured drive.
    Belief: half-extent error ~N(0, 8mm) per axis, corrected in z at seal."""
    import config as C
    dev = env.device
    believed = [dict(pos=list(m["pos"]),
                     half=list(np.abs(np.array(m["half"]) +
                                      rng.normal(0, 0.008, 3))))
                for m in meta]
    sigma = np.full(3, 0.010)
    q_home = np.asarray(C.START_Q, float)
    restore_objects(env, meta)
    reset_arm(env, q_home)
    teacher = WaypointTeacher(env, believed)
    otop_bel = believed[0]["half"][2] * 2
    teacher.build(q_home, 0, otop_bel)
    sealed_before = False
    stats = dict(col_viol=0.0, vel_viol=0.0, chunks=0, corrected=False)
    for t in range(220):
        a_chunk = torch.as_tensor(teacher.chunk(t), dtype=torch.float32,
                                  device=dev)[None]
        q0 = env.qpos[:, :6].clone()
        if use_layer:
            P, J = arm_points(env)
            # object0 is never an obstacle: pre-seal we must touch it,
            # post-seal it is attached to the cup (held-object clearance is
            # handled by the drop altitude in the teacher plan)
            obs_boxes = obstacles_for(env, believed, sigma,
                                      skip_obj0=True, dev=dev)
            G, h = collision_rows(P, J, obs_boxes, margin=0.015, q0=q0, H=H)
            obs_vec = env.observe()[:, :40]
            obj_vec = torch.tensor([[*believed[0]["pos"], otop_bel,
                                     *believed[0]["half"], sigma[0]]],
                                   dtype=torch.float32, device=dev)
            if layer is not None:
                with torch.no_grad():
                    y, info = layer(a_chunk, q0, obs_vec, obj_vec, G=G, h=h)
            else:
                Qm = torch.eye(6, device=dev)[None, None].expand(1, H, 6, 6)
                y, info = qp_solve(a_chunk, q0, Qm, alpha=1.0, G=G, h=h)
            stats["col_viol"] = max(stats["col_viol"], info.get("col_viol", 0))
            stats["vel_viol"] = max(stats["vel_viol"], info["vel_viol"])
        else:
            # greedy clip baseline (the deployed DQ_MAX clamp)
            step1 = (a_chunk[:, 0] - q0).clamp(-VMAX * DT, VMAX * DT)
            y = (q0 + step1)[:, None]
        stats["chunks"] += 1
        if log_rows is not None:
            log_rows.append(dict(
                a=teacher.chunk(t).astype(np.float32),
                q0=q0[0].cpu().numpy().astype(np.float32),
                obs=env.observe()[0, :40].cpu().numpy().astype(np.float32),
                obj=np.array([*believed[0]["pos"], otop_bel,
                              *believed[0]["half"], sigma[0]], np.float32)))
        # execute first refined step through the measured drive
        env.q_target = y[:, 0]
        want = teacher.want_suction(t)
        env._step_real_drive()
        # touch-based latch (the env's 3mm-penetration gate is a knife-edge
        # against the force-limited measured drive): within 8 mm of the grasp
        # point, at/below touch height, slow, suction commanded
        latched = torch.zeros(1, dtype=torch.bool, device=dev)
        # press EVENT-HOLD: at first touch, pin the plan clock so the
        # reference stops descending while the latch conditions settle
        if want and not env.sealed.any():
            tcp_h, _ = env._tcp()
            gp_h = env._grasp_point()
            if (float((tcp_h - gp_h)[0, :2].norm()) < 0.025 and
                    float(tcp_h[0, 2]) < float(gp_h[0, 2]) + 0.003):
                teacher.t0 += 1                       # hold the reference
        if want and not env.sealed.any():
            tcp, R = env._tcp()
            gp = env._grasp_point()
            close = float((tcp - gp).norm()) < 0.014
            touching = float(tcp[0, 2]) < float(gp[0, 2]) + 0.001
            slow = float(env.qvel[0, :6].abs().max()) < 0.5
            tilt_ok = float(R[0, 2, 2]) < -0.9
            if close and touching and slow and tilt_ok:
                idx = torch.tensor([0], device=dev)
                op = env._obj_pos()[idx]
                env.anchor[idx] = torch.einsum(
                    "nij,nj->ni", R[idx].transpose(1, 2), op - tcp[idx])
                env.anchor_R[idx] = torch.einsum(
                    "nij,njk->nik", R[idx].transpose(1, 2), env._obj_R()[idx])
                env.sat_count[idx] = 0
                env.sealed[idx] = True
                env.ever_sealed[idx] = True
                latched = torch.ones(1, dtype=torch.bool, device=dev)
        if latched.any() or (env.sealed.any() and not sealed_before):
            sealed_before = True
            # BELIEF CORRECTION: true top from tcp height at seal
            tcp, _ = env._tcp()
            otop_true = float(tcp[0, 2]) - 0.019
            if abs(otop_true - otop_bel) > 0.004:
                stats["corrected"] = True
                otop_bel = otop_true
                believed[0]["half"][2] = otop_true / 2
                sigma[0] = 0.003
                teacher.build(env.qpos[0, :6].cpu().numpy(), t + 1, otop_bel,
                              from_carry=True)
        if not want and env.sealed.any():
            idx = torch.nonzero(env.sealed).squeeze(-1)
            env.sealed[idx] = False
            env.xfrc[idx, env.bid_obj] = 0.0
        if record is not None:
            record.append((env.qpos[0].cpu().numpy().copy(),
                           bool(env.sealed[0])))
        if teacher.done(t):
            break
    op = env._obj_pos()[0].cpu().numpy()
    in_box = (abs(op[0] - BOX_C[0]) < BOX_IN / 2 and
              abs(op[1] - BOX_C[1]) < BOX_IN / 2 and 0.04 < op[2] < BOX_IN)
    stats["in_box"] = bool(in_box)
    stats["obj_final"] = op.tolist()
    stats["sealed_ever"] = bool(sealed_before)
    return stats


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_data", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="greedy clip, no layer")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    wp.init()

    if a.make_data:
        rows = []
        for e in range(a.episodes):
            env, meta, rng = make_env(a.seed0 + e)
            st = run_episode(env, meta, rng, layer=None, use_layer=True,
                             log_rows=rows)
            print(f"[data ep {e}] in_box={st['in_box']} sealed={st['sealed_ever']} "
                  f"corrected={st['corrected']} col_viol={st['col_viol']:.4f}",
                  flush=True)
            del env
            torch.cuda.empty_cache()
        torch.save(rows, os.path.join(OUT, "chunks.pt"))
        print(f"[data] {len(rows)} chunks -> {OUT}/chunks.pt")

    if a.train:
        rows = torch.load(os.path.join(OUT, "chunks.pt"), weights_only=False)
        dev = "cuda"
        A = torch.tensor(np.stack([r["a"] for r in rows]), device=dev)
        Q0 = torch.tensor(np.stack([r["q0"] for r in rows]), device=dev)
        OB = torch.tensor(np.stack([r["obs"] for r in rows]), device=dev)
        OJ = torch.tensor(np.stack([r["obj"] for r in rows]), device=dev)
        layer = DiffOGLayer(H=H, obs_dim=40, obj_dim=7).to(dev)
        opt = torch.optim.AdamW(layer.parameters(), lr=3e-4, weight_decay=1e-4)
        N = len(A)
        print(f"[train] {N} chunks")
        for ep in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            tot = 0.0
            for i in range(0, N, 256):
                idx = perm[i:i + 256]
                y, _ = layer(A[idx], Q0[idx], OB[idx], OJ[idx])
                loss = ((y - A[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
                opt.step()
                tot += float(loss) * len(idx)
            print(f"  epoch {ep}: loss {tot/N:.6f}", flush=True)
        torch.save(layer.state_dict(), os.path.join(OUT, "layer.pt"))
        print(f"[train] saved {OUT}/layer.pt")

    if a.eval:
        dev = "cuda"
        layer = None
        if not a.baseline and os.path.exists(os.path.join(OUT, "layer.pt")):
            layer = DiffOGLayer(H=H, obs_dim=40, obj_dim=7).to(dev)
            layer.load_state_dict(torch.load(os.path.join(OUT, "layer.pt")))
            layer.eval()
            print("[eval] trained layer loaded")
        results, videos = [], []
        for e in range(a.episodes):
            env, meta, rng = make_env(a.seed0 + 50 + e)
            rec = [] if a.render else None
            st = run_episode(env, meta, rng, layer=layer,
                             use_layer=not a.baseline, record=rec)
            results.append(st)
            print(f"[eval ep {e}] in_box={st['in_box']} sealed={st['sealed_ever']} "
                  f"corrected={st['corrected']} col_viol={st['col_viol']:.4f} "
                  f"obj_final={np.round(st['obj_final'],3)}", flush=True)
            if a.render and rec:
                videos.append((a.seed0 + 50 + e, rec, st))
            del env
            torch.cuda.empty_cache()
        n_ok = sum(r["in_box"] for r in results)
        print(f"\n[eval] in-box {n_ok}/{len(results)}  "
              f"sealed {sum(r['sealed_ever'] for r in results)}/{len(results)}  "
              f"corrected {sum(r['corrected'] for r in results)}")
        json.dump(results, open(os.path.join(
            OUT, "eval_baseline.json" if a.baseline else "eval_layer.json"), "w"),
            indent=1)
        if a.render and videos:
            render_videos(videos, tag="baseline" if a.baseline else "diffog")


def render_videos(videos, tag):
    import cv2
    import mujoco
    frames_out = os.path.join(OUT, f"video_{tag}.mp4")
    vw = None
    for seed, rec, st in videos:
        xml = os.path.join(OUT, f"scene_{seed}.xml")
        m = mujoco.MjModel.from_xml_path(xml)
        d = mujoco.MjData(m)
        ren = mujoco.Renderer(m, height=360, width=480)
        for k, (qpos, sealed) in enumerate(rec):
            if k % 2:
                continue
            d.qpos[:len(qpos)] = qpos
            mujoco.mj_forward(m, d)
            ims = []
            for cam in ("iso", "front"):
                ren.update_scene(d, camera=cam)
                ims.append(cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR))
            row = np.hstack(ims)
            canvas = np.zeros((420, 960, 3), np.uint8)
            canvas[60:] = row
            cv2.putText(canvas, f"{tag}  ep(seed {seed})  "
                        f"{'IN BOX' if st['in_box'] else 'miss'}",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if sealed:
                cv2.rectangle(canvas, (0, 60), (959, 419), (0, 180, 255), 3)
            if vw is None:
                vw = cv2.VideoWriter(frames_out.replace(".mp4", "_raw.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                     (960, 420))
            vw.write(canvas)
        for _ in range(5):
            vw.write(canvas)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {frames_out.replace('.mp4','_raw.mp4')} "
              f"-c:v libx264 -pix_fmt yuv420p {frames_out}")
    os.remove(frames_out.replace(".mp4", "_raw.mp4"))
    print(f"[video] {frames_out}")


if __name__ == "__main__":
    main()

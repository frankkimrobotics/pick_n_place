#!/usr/bin/env python3
"""planner_touch :: cuRobo-planned "touch the object" run on the real Pro 630.

The planner-side half of the A/B test against the RL policy (rl/real_policy_ctrl.py
--touch_only): same robot link, same guards, same contact metric, same log format --
only the motion generator differs (cuRobo trajopt + TOPPRA retiming here, a 10 Hz
neural policy there).

Motion
------
  A  approach : cuRobo plan_pose  START -> (cx, cy, top + CUP_R + hover)
  B  descend  : cuRobo plan_pose  hover -> (cx, cy, top + CUP_R)
                (falls back to a straight-line Cartesian descent solved in 1 cm
                 plan_pose hops when the single short vertical plan fails or bows
                 more than DESCEND_TOL off the vertical)
  A+B are stitched geometrically (curobo_pick_traj._stitch), retimed as ONE motion
  with TOPPRA (curobo_pick_traj._toppra_retime, DT = 4 ms) so the hover pose is
  passed at non-zero velocity, and quintic-blended to v=a=0 at both ends
  (curobo_pick_traj._c2_end_blends).  The 4 ms profile is then decimated to the
  100 ms knots the Pi's B-spline welder consumes.

Simulation A/B (--sim)
----------------------
  The same plan is rolled out in the mujoco_warp twin (env_paper.PaperPickEnv, measured
  drive, 1 world) by turning each 100 ms knot into the env's 10 Hz action:
  a[:6] = clip((q_knot - env.q_target) / dq_max, -1, 1), a[6] = -1 (suction never on).
  The same env/seed then rolls the RL checkpoints (--sim_rl / --sim_residual) so the
  planner and the policy are compared on an identical scene.

Execution (only with --exec)
----------------------------
  1 s torque baseline -> send_path(knots, 0.1) in spline mode -> poll :9999 at 10 Hz.
  STOP when  tau >= TAU_FIRM and |tip - grasp| < 3 cm   ("TOUCH", the success case)
        or   tau >= TAU_HARD anywhere                   (hard contact backstop)
        or   feedback age > 0.5 s                       (stale)
  Stopping OVERWRITES the streamed reference (the Pi follows it open loop): a chunk
  anchored at now holding the measured q.  Then retract 5 cm (straight-up IK) and
  go home at 6 deg/s.  Suction is NEVER touched by this script.

Environments
------------
  The planner client + TOPPRA live in `curobo2` (toppra, no mujoco); the robot link,
  MuJoCo FK/IK and the guards live in `mjwarp` (mujoco, no toppra).  This file is the
  mjwarp-side driver and re-invokes itself under the curobo2 interpreter for the plan
  stage (--_plan_stage, JSON in / JSON out).  One executable, two interpreters.

Usage
-----
  PY=/home/lisc-frank/miniconda3/envs/mjwarp/bin/python
  $PY rl/planner_touch.py --obj 0.44 0.0 0.045                 # dry run (plans only)
  $PY rl/planner_touch.py --vlim 30 --alim 200                 # dry run, tracker estimate
  $PY rl/planner_touch.py --obj 0.44 0.0 0.045 --exec --log ~/pnp_rl/real_planner_1.json
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUROBO_DIR = "/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo"
CUROBO_PY = "/home/lisc-frank/miniconda3/envs/curobo2/bin/python"

DT = 0.004                 # curobo_pick_traj.DT (TOPPRA sampling)
KNOT_DT = 0.10             # Pi spline-welder knot spacing
CUP_R = 0.008
# tcp z down with +90 deg yaw == curobo_pick_traj._yawed(_pose(...), 90):
# R_DOWN_QUAT [0,1,0,0] rotated about world z by +90 deg.
R_DOWN_YAW90 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]   # wxyz
DESCEND_TOL = 0.02         # max lateral bow of the descend plan off the vertical (m)
DESCEND_STEP = 0.01        # fallback straight-line descent step (m)
# real table top in the base frame.  The planner server's own "ground" cuboid spans all
# x/y and cannot be raised to 0 -- the base_link collision spheres reach z = -0.063, so a
# ground top at 0 puts the robot permanently in self-collision and EVERY plan fails
# ("no solution", verified 2026-09-20).  The table is added as a separate cuboid that
# starts clear of the base instead; run the server with its default --ground-z -0.1.
TABLE = {"name": "table", "dims": [1.00, 1.60, 0.10],
         "pose": [0.72, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0]}     # top at z = 0, x in [0.22, 1.22]
# the server's own camera_mount_d435 cuboid is deliberately generous (0.30 x 0.30, front face at
# x = 0.49) and makes EVERY down-pointing goal with x >= 0.40 unplannable -- the "~40 % of hover
# goals rejected near the camera mount" in rl/FINDINGS.md.  --tight_mount overrides it (set_world
# replaces a base cuboid of the same name) with a box around the head + column only.
MOUNT_TIGHT = {"name": "camera_mount_d435", "dims": [0.18, 0.18, 1.20],
               "pose": [0.645, -0.05, 0.45, 1.0, 0.0, 0.0, 0.0]}   # front face x = 0.555


# ======================================================================== plan stage (curobo2)
class PlannerClient:
    def __init__(self, host="127.0.0.1", port=9997, timeout=300.0):
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.s.makefile("rwb")

    def rpc(self, obj):
        self.f.write((json.dumps(obj) + "\n").encode())
        self.f.flush()
        line = self.f.readline()
        if not line:
            raise RuntimeError("planner server closed the connection")
        return json.loads(line)

    def ping(self):
        return self.rpc({"type": "ping"})

    def plan_pose(self, q, pose, max_attempts=10):
        return self.rpc({"type": "plan_pose", "start_q": list(map(float, q)),
                         "goal_pose": list(map(float, pose)), "max_attempts": int(max_attempts)})

    def fk(self, q):
        r = self.rpc({"type": "fk", "q": [float(x) for x in q]})
        return np.asarray(r["pos"][0], float)

    def set_world(self, cuboids):
        return self.rpc({"type": "set_world", "cuboids": cuboids})


def quat_to_R(q):
    """wxyz -> 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def measure_tcp_corr(P, ob, q):
    """World-frame vector cuRobo-tcp -> MuJoCo-tcp at the canonical (down, yaw +90) tool pose.

    The planner's URDF tool frame `tcp` and the MuJoCo scene's `tcp` site (the cup tip the
    guards, the RL observation and every log use) are NOT the same point: a fixed 20 mm
    translation along the tool axis (measured 2026-09-20, constant over 7 random configs).
    Planning straight to the grasp point therefore parks the cup tip 2 cm short of it.
    Goals are corrected by this vector so the MUJOCO tip lands on the grasp point.
    """
    p_c = P.fk(q)                                        # cuRobo tool frame
    p_m, R_m = ob.fk(q)                                  # MuJoCo cup tip + its orientation
    off_tool = R_m.T @ (p_m - p_c)                       # constant offset in the tool frame
    return quat_to_R(R_DOWN_YAW90) @ off_tool, float(np.linalg.norm(off_tool))


def _descend_fallback(P, q_hover, p_hover, p_grasp):
    """Straight-line Cartesian descent solved as DESCEND_STEP hops; returns [N,6] rad."""
    n = max(1, int(round(np.linalg.norm(p_grasp - p_hover) / DESCEND_STEP)))
    way, q = [np.asarray(q_hover, float)], list(q_hover)
    for i in range(1, n + 1):
        p = p_hover + (p_grasp - p_hover) * (i / n)
        r = P.plan_pose(q, list(p) + R_DOWN_YAW90, max_attempts=6)
        if not r["success"]:
            raise RuntimeError(f"descend fallback failed at step {i}/{n} (z {p[2]:.3f}): {r['status']}")
        q = list(np.asarray(r["trajectory"], float)[-1])
        way.append(np.asarray(q, float))
    return np.asarray(way)


def plan_stage(cfg):
    """Plan + retime; runs under the curobo2 interpreter.  cfg/ret are plain JSON."""
    sys.path.insert(0, CUROBO_DIR)
    import curobo_pick_traj as cpt                      # _stitch/_toppra_retime/_c2_end_blends

    t_wall = time.time()
    P = PlannerClient()
    info = {"ping": P.ping()}
    world = [TABLE] + ([MOUNT_TIGHT] if cfg.get("tight_mount") else [])
    info["set_world"] = P.set_world(world)["names"]
    info["tight_mount"] = bool(cfg.get("tight_mount"))

    start_q = np.asarray(cfg["start_q"], float)
    cx, cy, top = cfg["obj"]
    # where the MUJOCO cup tip must end up ...
    p_hover = np.array([cx, cy, top + CUP_R + cfg["hover"]])
    p_grasp = np.array([cx, cy, top + CUP_R])
    # ... and the cuRobo tool-frame goals that put it there
    corr = np.asarray(cfg.get("tcp_corr", [0.0, 0.0, 0.0]), float)
    g_hover, g_grasp = p_hover - corr, p_grasp - corr
    info["tcp_corr"] = corr.round(5).tolist()

    t0 = time.time()
    ra = P.plan_pose(start_q, list(g_hover) + R_DOWN_YAW90, max_attempts=10)
    if not ra["success"] and not cfg.get("tight_mount"):
        # almost always the oversized camera-mount cuboid; retry once with the tight one
        info["set_world"] = P.set_world([TABLE, MOUNT_TIGHT])["names"]
        info["tight_mount"] = "auto (approach failed with the default mount)"
        ra = P.plan_pose(start_q, list(g_hover) + R_DOWN_YAW90, max_attempts=10)
    if not ra["success"]:
        raise RuntimeError(f"approach plan_pose failed: {ra['status']}")
    way_a = np.asarray(ra["trajectory"], float)
    q_hover = way_a[-1]

    rb = P.plan_pose(q_hover, list(g_grasp) + R_DOWN_YAW90, max_attempts=10)
    descend_mode = "plan_pose"
    way_b = None
    if rb["success"]:
        way_b = np.asarray(rb["trajectory"], float)
        # a short vertical move that bows sideways is the classic short-motion trajopt
        # artefact: check the Cartesian path before trusting it
        pts = np.asarray([P.fk(q) for q in way_b[::max(1, len(way_b) // 12)]])
        bow = float(np.abs(pts[:, :2] - g_grasp[:2]).max())
        info["descend_bow_m"] = round(bow, 4)
        if bow > DESCEND_TOL:
            descend_mode = f"straight-line 1 cm hops (plan_pose bowed {100 * bow:.1f} cm)"
            way_b = None
    else:
        descend_mode = f"straight-line 1 cm hops (plan_pose {rb['status']})"
    if way_b is None:
        way_b = _descend_fallback(P, q_hover, P.fk(q_hover), g_grasp)
    t_plan = time.time() - t0

    t0 = time.time()
    way = cpt._stitch([way_a, way_b])
    vlim = np.full(6, np.radians(cfg["vlim"]))
    alim = np.full(6, np.radians(cfg["alim"]))
    t, q, qd, qdd = cpt._toppra_retime(way, vlim, alim)
    # the quintic end blends force v=a=0 at both ends over BLEND_S; the same distance in less
    # time means the middle of each blend window OVERSHOOTS vlim -- report both peaks.
    v_pre = float(np.degrees(np.abs(np.gradient(q, DT, axis=0))).max())
    q = cpt._c2_end_blends(q, qd, qdd)
    v_post = float(np.degrees(np.abs(np.gradient(q, DT, axis=0))).max())
    t_retime = time.time() - t0

    q_end_fk = (P.fk(q[-1]) + corr).tolist()      # predicted MUJOCO cup tip at the final knot
    return {"dt": DT, "t": t.tolist(), "q": np.round(q, 6).tolist(),
            "p_hover": p_hover.tolist(), "p_grasp": p_grasp.tolist(),
            "q_end_fk": q_end_fk, "descend_mode": descend_mode,
            "n_way": [int(len(way_a)), int(len(way_b)), int(len(way))],
            "vpk_pre_blend": v_pre, "vpk_post_blend": v_post,
            "solve_time_a": float(ra.get("solve_time", 0.0)),
            "solve_time_b": float(rb.get("solve_time", 0.0)) if rb["success"] else None,
            "t_plan": t_plan, "t_retime": t_retime, "t_stage": time.time() - t_wall,
            "info": info}


def _run_plan_stage(cfg):
    """Spawn the curobo2 interpreter on this file and get the retimed profile back."""
    if os.path.abspath(sys.executable) == os.path.abspath(CUROBO_PY):
        return plan_stage(cfg)                          # already in curobo2
    p = subprocess.run([CUROBO_PY, os.path.abspath(__file__), "--_plan_stage"],
                       input=json.dumps(cfg).encode(), stdout=subprocess.PIPE, timeout=600)
    out = p.stdout.decode()
    tag = "@@PLAN@@"
    if p.returncode != 0 or tag not in out:
        sys.stderr.write(out)
        raise SystemExit(f"[abort] plan stage failed (rc {p.returncode})")
    head, payload = out.split(tag, 1)
    if head.strip():
        print(head.rstrip())
    return json.loads(payload)


# ======================================================================== knots + report
def to_knots(t, q, knot_dt=KNOT_DT):
    """Decimate the DT profile to knot_dt knots, always keeping the exact final point."""
    step = int(round(knot_dt / DT))
    idx = list(range(0, len(q), step))
    if idx[-1] != len(q) - 1:
        idx.append(len(q) - 1)
    return np.asarray(t)[idx], np.asarray(q)[idx]


# ======================================================================== simulation A/B
def _make_sim_env(scene, ep_len, dq_max_deg, seed=0):
    import torch
    import warp as wp
    wp.init()
    sys.path.insert(0, HERE)
    from env_paper import PaperPickEnv
    torch.manual_seed(seed)
    env = PaperPickEnv(nworld=1, device="cuda:0", xml=scene, dr=False, drive="real",
                       ep_len=ep_len, grasp_shaping=True, obs_ee=True, reach_target="grasp",
                       lift_dense=True, w_reach=0.5, w_track_c=4, w_track_f=8, seed=seed,
                       dq_max_deg=dq_max_deg)
    env.auto_reset = False
    env.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env.reset(torch.ones(1, dtype=torch.bool, device=env.device))
    return env, torch


def _sim_metrics(rows, grasp, t_touch_tol=0.01):
    d = np.asarray([r["d_grasp"] for r in rows])
    hit = np.nonzero(d < t_touch_tol)[0]
    return (float(rows[hit[0]]["t"]) if len(hit) else None,
            float(d.min()), float(d[-1]))


def sim_rollout_planner(env, torch, q_knots, ep_len, hold_steps, verbose=True):
    """Feed each 100 ms knot to the env as a 10 Hz joint-delta action."""
    grasp = env._grasp_point()[0].cpu().numpy()
    targets = list(q_knots) + [q_knots[-1]] * hold_steps
    rows, n_clip, worst = [], 0, 0.0
    for k, q_k in enumerate(targets[1:][:ep_len]):
        need = torch.as_tensor(q_k, dtype=torch.float32, device=env.device)[None] - env.q_target
        a6 = need / env.dq_max
        worst = max(worst, float(a6.abs().max()))
        if float(a6.abs().max()) > 1.0 + 1e-6:
            n_clip += 1
        a = torch.cat([a6.clamp(-1, 1), -torch.ones(1, 1, device=env.device)], dim=-1)
        env.step(a)
        tip = env._tcp()[0][0].cpu().numpy()
        q = env.qpos[0, :6].cpu().numpy()
        rows.append(dict(k=k, t=round((k + 1) * 0.1, 3), q=np.round(q, 4).tolist(),
                         tcp=np.round(tip, 4).tolist(),
                         d_grasp=round(float(np.linalg.norm(tip - grasp)), 4)))
    t1, dmin, dend = _sim_metrics(rows, grasp)
    if verbose:
        print(f"[sim] planner rollout: {len(rows)} decisions, dq_max {np.degrees(env.dq_max):.1f} deg, "
              f"action clipped on {n_clip}/{len(rows)} decisions (worst |a| {worst:.2f})")
        print(f"[sim] time to |tip-grasp| < 1 cm: {('%.1f s' % t1) if t1 else 'never'}  "
              f"min {100 * dmin:.2f} cm  final {100 * dend:.2f} cm")
    return rows, dict(t_1cm=t1, d_min=dmin, d_final=dend, n_clip=n_clip,
                      n_dec=len(rows), worst_a=worst)


def sim_rollout_policy(env, torch, act_fn, ep_len, obs_noise=0.005, label="policy"):
    grasp = env._grasp_point()[0].cpu().numpy()
    rows = []
    for k in range(ep_len):
        with torch.no_grad():
            o = env.observe()
            if obs_noise > 0:
                o = o + torch.randn_like(o) * obs_noise
            a = act_fn(o)
        env.step(a)
        tip = env._tcp()[0][0].cpu().numpy()
        rows.append(dict(k=k, t=round((k + 1) * 0.1, 3),
                         q=np.round(env.qpos[0, :6].cpu().numpy(), 4).tolist(),
                         tcp=np.round(tip, 4).tolist(),
                         d_grasp=round(float(np.linalg.norm(tip - grasp)), 4)))
    t1, dmin, dend = _sim_metrics(rows, grasp)
    print(f"[sim] {label}: time to 1 cm {('%.1f s' % t1) if t1 else 'never'}  "
          f"min {100 * dmin:.2f} cm  final {100 * dend:.2f} cm")
    return rows, dict(t_1cm=t1, d_min=dmin, d_final=dend)


def load_ac(path, obs_dim, device, arch="paper", critic_extra=0):
    import torch
    sys.path.insert(0, HERE)
    from ppo import AC
    ac = AC(obs_dim=obs_dim, arch=arch, critic_extra=critic_extra).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    ac.load_state_dict(ck["ac"] if "ac" in ck else ck)
    ac.eval()
    return ac, ck


def run_sim(a):
    """--sim: plan in the twin's scene and roll the plan (and, optionally, the RL policies)."""
    scene = os.path.join(HERE, "scenes", "box_med.xml")
    env, torch = _make_sim_env(scene, a.sim_steps, a.sim_dq_max)
    obj = env._obj_pos()[0].cpu().numpy()
    grasp = env._grasp_point()[0].cpu().numpy()
    q0 = env.qpos[0, :6].cpu().numpy()
    top = float(obj[2] + env.half[2])
    print(f"[sim] world: object centre {np.round(obj, 4).tolist()} top {top:.4f}  "
          f"grasp point {np.round(grasp, 4).tolist()}")
    print(f"[sim] start q (deg) {np.round(np.degrees(q0), 2).tolist()}  "
          f"start tip {np.round(env._tcp()[0][0].cpu().numpy(), 4).tolist()}")

    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    from real_policy_ctrl import ObsBuilder, Guard
    ob = ObsBuilder(scene, [float(x) for x in env.half])
    corr, off_n = measure_tcp_corr(PlannerClient(), ob, q0)
    print(f"[plan] cuRobo tcp -> MuJoCo cup tip offset {1000 * off_n:.1f} mm; goal correction "
          f"{np.round(corr, 4).tolist()}")

    pl = _run_plan_stage({"start_q": [float(x) for x in q0], "obj": [float(obj[0]), float(obj[1]), top],
                          "hover": a.hover, "vlim": a.vlim, "alim": a.alim,
                          "tight_mount": a.tight_mount, "tcp_corr": corr.tolist()})
    t_dense = np.asarray(pl["t"], float)
    q_dense = np.asarray(pl["q"], float)
    sb = pl["solve_time_b"]
    print(f"[plan] world: {pl['info']['set_world']}  tight_mount={pl['info']['tight_mount']}")
    print(f"[plan] approach {pl['n_way'][0]} pts (solve {pl['solve_time_a']:.3f} s) + descend "
          f"{pl['n_way'][1]} pts ({pl['descend_mode']}" + (f", solve {sb:.3f} s" if sb else "") + ")")
    print(f"[plan] plan {pl['t_plan']:.2f} s + retime {pl['t_retime']:.2f} s -> "
          f"{len(q_dense)} ticks, {t_dense[-1] + DT:.2f} s")
    print(f"[plan] planned end tip {np.round(pl['q_end_fk'], 4).tolist()}  "
          f"err vs grasp {1000 * np.linalg.norm(np.asarray(pl['q_end_fk']) - grasp):.1f} mm")

    t_k, q_k = to_knots(t_dense, q_dense)
    dur = float(t_k[-1])
    vpk = np.degrees(np.abs(np.gradient(q_dense, DT, axis=0))).max(axis=0)
    apk = np.degrees(np.abs(np.gradient(np.gradient(q_dense, DT, axis=0), DT, axis=0))).max()
    print(f"[knots] {len(q_k)} knots at {KNOT_DT * 1000:.0f} ms, planned duration {dur:.2f} s")
    print(f"[knots] peak planned joint speed (deg/s) {np.round(vpk, 1).tolist()} max {vpk.max():.1f}  "
          f"peak accel {apk:.0f} deg/s^2")
    print(f"[knots] TOPPRA peak {pl['vpk_pre_blend']:.1f} deg/s (limit {a.vlim:.0f}) -> "
          f"{pl['vpk_post_blend']:.1f} deg/s after the C2 end blends")

    # the guards are the same ones the robot run uses
    guard = Guard(ob, top)
    viol = [(i, v) for i, qq in enumerate(q_k) for v in [guard.check_q(qq)] if v]
    if viol:
        for i, v in viol[:8]:
            print(f"[guard] knot {i} (t {t_k[i]:.2f} s): " + "; ".join(v))
        raise SystemExit(f"[abort] {len(viol)}/{len(q_k)} knots violate the guards")
    print(f"[guard] all {len(q_k)} knots pass")

    out = {"planned_duration": dur, "n_knots": int(len(q_k)), "peak_speed": float(vpk.max()),
           "vpk_pre_blend": pl["vpk_pre_blend"], "vpk_post_blend": pl["vpk_post_blend"],
           "peak_accel": float(apk), "vlim": a.vlim, "alim": a.alim,
           "t_plan": pl["t_plan"], "t_retime": pl["t_retime"]}
    dq_knot = float(np.degrees(np.abs(np.diff(q_k, axis=0))).max())
    print(f"[knots] largest knot-to-knot joint step {dq_knot:.2f} deg "
          f"(the env integrates q_target by at most dq_max per decision)")
    rows, m = sim_rollout_planner(env, torch, q_k, a.sim_steps, a.sim_hold)
    out["planner"] = m
    out["dq_knot_deg"] = dq_knot
    if m["n_clip"] and a.sim_dq_max is None:
        # the training clamp (2 deg/decision = 20 deg/s) cannot follow a 45 deg/s plan: re-roll with
        # a dq_max that fits the plan, which is what a planner-driven controller would ship.
        fit = float(np.ceil(dq_knot * 10) / 10)
        print(f"[sim] re-rolling with dq_max {fit:.1f} deg (fits the plan)")
        env_f, torch_f = _make_sim_env(scene, a.sim_steps, fit)
        _, mf = sim_rollout_planner(env_f, torch_f, q_k, a.sim_steps, a.sim_hold)
        out["planner_dqfit"] = dict(mf, dq_max_deg=fit)
        del env_f

    # ---- RL baselines in the SAME env / seed ----
    for tag, path, kind in (("dagger", a.sim_rl, "base"), ("residual", a.sim_residual, "res")):
        if not path:
            continue
        env2, torch2 = _make_sim_env(scene, a.sim_steps, None)
        obs_dim = env2.observe().shape[-1]
        if kind == "base":
            ac, _ = load_ac(path, obs_dim, env2.device)
            fn = lambda o, ac=ac: torch2.tanh(ac.pi(o))                     # noqa: E731
            lab = f"DAgger base {os.path.basename(path)}"
        else:
            ck = torch2.load(path, map_location=env2.device, weights_only=False)
            base_path = ck.get("residual_base")
            bound = float(ck.get("residual_bound") or 0.3)
            base, _ = load_ac(base_path, obs_dim, env2.device)
            res, _ = load_ac(path, obs_dim, env2.device,
                             critic_extra=(5 if ck.get("critic_priv") else 0))
            fn = lambda o, b=base, r=res, bd=bound: (                        # noqa: E731
                torch2.tanh(b.pi(o)) + bd * torch2.tanh(r.pi(o))).clamp(-1.0, 1.0)
            lab = f"residual (base {os.path.basename(base_path or '?')}, bound {bound})"
        g2 = env2._grasp_point()[0].cpu().numpy()
        print(f"[sim] {lab}: grasp point {np.round(g2, 4).tolist()} "
              f"({'same scene' if np.allclose(g2, grasp, atol=1e-6) else 'DIFFERENT SCENE'})")
        _, mm = sim_rollout_policy(env2, torch2, fn, a.sim_steps, label=lab)
        out[tag] = mm
        del env2

    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        json.dump({"summary": out, "rows": rows}, open(a.log, "w"))
        print(f"[log] {a.log}")
    print("[sim] summary " + json.dumps(out))
    return out


# ======================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--_plan_stage", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--obj", type=float, nargs=3, default=None,
                    help="object cx cy top (m, base frame); default: one rl/rgb_track.py datagram")
    ap.add_argument("--half", type=float, nargs=3, default=[0.025, 0.025, 0.02],
                    help="object half extents (only used for the MuJoCo scene / grasp point)")
    ap.add_argument("--vlim", type=float, default=45.0, help="joint velocity limit, deg/s")
    ap.add_argument("--alim", type=float, default=400.0, help="joint acceleration limit, deg/s^2")
    ap.add_argument("--hover", type=float, default=0.10, help="hover height above the grasp point (m)")
    ap.add_argument("--pi", default="192.168.50.2")
    ap.add_argument("--start_tol", type=float, default=6.0, help="max deviation from START_Q (deg)")
    ap.add_argument("--tight_mount", action="store_true",
                    help="replace the server's oversized camera_mount_d435 cuboid with a tight one "
                         "(the default box blocks every goal with x >= 0.40)")
    ap.add_argument("--plan_only", action="store_true",
                    help="no robot: plan/retime/guard-check from START_Q (the Pi is not contacted)")
    ap.add_argument("--sim", action="store_true",
                    help="no robot: plan for the mujoco_warp twin's scene and roll the plan there")
    ap.add_argument("--sim_steps", type=int, default=150, help="--sim episode length (decisions)")
    ap.add_argument("--sim_hold", type=int, default=20,
                    help="--sim decisions holding the final knot so the measured drive settles")
    ap.add_argument("--sim_dq_max", type=float, default=None,
                    help="--sim env dq_max in deg/decision (default 2 = the training clamp)")
    ap.add_argument("--sim_rl", default=None, help="--sim: DAgger base checkpoint to roll for comparison")
    ap.add_argument("--sim_residual", default=None, help="--sim: residual checkpoint to roll for comparison")
    ap.add_argument("--exec", action="store_true", help="actually move the robot (default: dry run)")
    ap.add_argument("--force", action="store_true", help="skip the start-pose check")
    ap.add_argument("--log", default=None, help="write the execution log (JSON rows) here")
    a = ap.parse_args()

    if a._plan_stage:                                   # curobo2 side
        cfg = json.loads(sys.stdin.read())
        print("@@PLAN@@" + json.dumps(plan_stage(cfg)))
        return

    if a.sim:                                           # no robot in the loop
        run_sim(a)
        return

    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    from real_policy_ctrl import (PiLink, ObsBuilder, Guard, ObjectTracker, START_Q,
                                  STREAM_GAINS, TAU_FIRM, TAU_HARD, W_J3)

    # ---- object estimate -------------------------------------------------
    if a.obj is not None:
        cx, cy, top = a.obj
        src = "cli"
    else:
        tr = ObjectTracker()
        t_w, det = time.time(), None
        while det is None and time.time() - t_w < 3.0:
            det = tr.poll()
            time.sleep(0.05)
        tr.sock.close()                                 # free :9701 for the RL controller
        if det is None:
            raise SystemExit("[abort] no tracker datagram on udp :9701 -- is rl/rgb_track.py running? "
                             "(or pass --obj cx cy top)")
        cx, cy, top = det["cx"], det["cy"], det["top"]
        src = f"tracker (n={det['n']}, src={det.get('src')})"
    print(f"[obj] cx {cx:.4f} cy {cy:.4f} top {top:.4f}  [{src}]")

    ob = ObsBuilder(os.path.join(HERE, "scenes", "box_med.xml"), a.half)
    guard = Guard(ob, top, force=a.force)
    p_grasp = np.array([cx, cy, top + CUP_R])

    # ---- robot state -----------------------------------------------------
    if a.plan_only:
        link = None
        q0 = np.asarray(START_Q, float)
        print(f"[robot] --plan_only: starting from START_Q (deg) {np.round(np.degrees(q0), 2).tolist()}, "
              f"tip {np.round(ob.fk(q0)[0], 4).tolist()} -- the Pi is not contacted")
    else:
        link = PiLink(a.pi, a.exec, ref_mode="spline")
        link.ref_mode = "spline"
        t_w = time.time()
        while not link.ok() and time.time() - t_w < 6:
            time.sleep(0.1)
        q0, qd0, age = link.state()
        if q0 is None:
            raise SystemExit(f"[abort] no feedback from {a.pi}:9999 -- is robot_hal running on the Pi? "
                             f"(use --plan_only to plan without the robot, or --sim for the twin)")
        tcp0, _ = ob.fk(q0)
        print(f"[robot] q (URDF deg) {np.round(np.degrees(q0), 2).tolist()}  tip {np.round(tcp0, 4).tolist()}  "
              f"fb age {1000 * age:.0f} ms  clock offset {1000 * link.clock_offset:.1f} ms  "
              f"cmd {'ok' if link.cmd_sock else 'DOWN'}")
        dev = float(np.degrees(np.abs(q0 - START_Q)).max())
        if dev > a.start_tol and not a.force:
            link.close()
            raise SystemExit(f"[abort] arm is {dev:.1f} deg from START_Q (limit {a.start_tol:.0f} deg) -- "
                             f"jog it home first (real_policy_ctrl.py --go_home --exec) or pass --force")
        print(f"[robot] {dev:.2f} deg from START_Q (ok)")

        def stop_handler(*_):
            print("\n[ctrl] interrupted -> stop streaming (Pi holds)")
            link.close()
            sys.exit(1)
        signal.signal(signal.SIGINT, stop_handler)

    v0 = guard.check_q(q0)
    if v0:
        raise SystemExit("[abort] start pose violates: " + "; ".join(v0))

    # ---- plan ------------------------------------------------------------
    print(f"[plan] cuRobo :9997  hover {a.hover:.3f} m  vlim {a.vlim:.0f} deg/s  alim {a.alim:.0f} deg/s^2")
    corr, off_n = measure_tcp_corr(PlannerClient(), ob, q0)
    print(f"[plan] cuRobo tcp -> MuJoCo cup tip offset {1000 * off_n:.1f} mm; goal correction "
          f"{np.round(corr, 4).tolist()}")
    pl = _run_plan_stage({"start_q": [float(x) for x in q0], "obj": [cx, cy, top],
                          "hover": a.hover, "vlim": a.vlim, "alim": a.alim,
                          "tight_mount": a.tight_mount, "tcp_corr": corr.tolist()})
    t_dense = np.asarray(pl["t"], float)
    q_dense = np.asarray(pl["q"], float)
    sb = pl["solve_time_b"]
    print(f"[plan] world: {pl['info']['set_world']}  tight_mount={pl['info']['tight_mount']}")
    print(f"[plan] approach {pl['n_way'][0]} pts (solve {pl['solve_time_a']:.3f} s) + descend "
          f"{pl['n_way'][1]} pts ({pl['descend_mode']}" + (f", solve {sb:.3f} s" if sb else "") + f") "
          f"-> {pl['n_way'][2]} stitched waypoints")
    print(f"[plan] plan {pl['t_plan']:.2f} s + retime {pl['t_retime']:.2f} s "
          f"(stage total {pl['t_stage']:.2f} s) -> {len(q_dense)} ticks at {DT * 1000:.0f} ms, "
          f"{t_dense[-1] + DT:.2f} s")
    print(f"[plan] planned end tip {np.round(pl['q_end_fk'], 4).tolist()}  "
          f"grasp point {np.round(p_grasp, 4).tolist()}  "
          f"err {1000 * np.linalg.norm(np.asarray(pl['q_end_fk']) - p_grasp):.1f} mm")

    # ---- knots + guard ---------------------------------------------------
    t_k, q_k = to_knots(t_dense, q_dense)
    dur = float(t_k[-1])
    vpk_dense = np.degrees(np.abs(np.gradient(q_dense, DT, axis=0))).max(axis=0)
    vpk_knot = np.degrees(np.abs(np.diff(q_k, axis=0) / np.diff(t_k)[:, None])).max(axis=0)
    apk = np.degrees(np.abs(np.gradient(np.gradient(q_dense, DT, axis=0), DT, axis=0))).max()
    viol = [(i, v) for i, qq in enumerate(q_k) for v in [guard.check_q(qq)] if v]
    print(f"[knots] {len(q_k)} knots at {KNOT_DT * 1000:.0f} ms, duration {dur:.2f} s")
    print(f"[knots] peak planned joint speed (deg/s): dense {np.round(vpk_dense, 1).tolist()} "
          f"max {vpk_dense.max():.1f}  |  knot-to-knot max {vpk_knot.max():.1f}")
    print(f"[knots] TOPPRA peak {pl['vpk_pre_blend']:.1f} deg/s (limit {a.vlim:.0f}) -> "
          f"{pl['vpk_post_blend']:.1f} deg/s after the C2 end blends"
          + ("  *** above the ~36 deg/s drive following-error ceiling ***"
             if pl['vpk_post_blend'] > 36 else ""))
    print(f"[knots] peak planned joint accel {apk:.0f} deg/s^2")
    if viol:
        for i, v in viol[:8]:
            print(f"[guard] knot {i} (t {t_k[i]:.2f} s): " + "; ".join(v))
        if link is not None:
            link.close()
        raise SystemExit(f"[abort] {len(viol)}/{len(q_k)} knots violate the guards -- nothing sent")
    print(f"[guard] all {len(q_k)} knots pass (elbow box, LinuxCNC soft limits, wall keep-out x>=0.12, "
          f"z floor {guard.z_floor:.3f})")
    tips = np.asarray([ob.fk(qq)[0] for qq in q_k])
    print(f"[knots] tip z {tips[:, 0].size} pts: start {tips[0, 2]:.3f} -> min {tips[:, 2].min():.3f} "
          f"-> end {tips[-1, 2]:.3f} m")

    if a.plan_only or not a.exec:
        print("[dry run] nothing sent to the robot. Re-run with --exec to move.")
        if link is not None:
            link.close()
        return

    # ---- execute ---------------------------------------------------------
    tau_base = link.torque_baseline(1.0)
    print(f"[contact] torque baseline (1 s median) {np.round(tau_base, 3).tolist()}  "
          f"firm {TAU_FIRM} hard {TAU_HARD}  gains {STREAM_GAINS}")
    rows, reason = [], "path completed"
    t_start = time.time() + 0.3
    link.send_path(q_k, KNOT_DT, t_start)
    print(f"[ctrl] EXECUTING: {len(q_k)} knots, {dur:.2f} s, anchored at t+0.3 s")

    t_end = t_start + dur + 0.8
    k = 0
    while True:
        t_dec = t_start + k * 0.1
        d = t_dec - time.time()
        if d > 0:
            time.sleep(d)
        if time.time() > t_end:
            break
        if not link.ok():
            reason = "feedback stale / command socket down"
            break
        q, qd, age = link.state()
        if age > 0.5:
            reason = f"feedback age {1000 * age:.0f} ms > 500 ms"
            link.send_segment(q, q, time.time(), seq=k)
            break
        tip, _ = ob.fk(q)
        tq = link.torque()
        tau = float(abs(tq[1] - tau_base[1]) + W_J3 * abs(tq[2] - tau_base[2]))
        i_ref = min(len(q_k) - 1, int(round((time.time() - t_start) / KNOT_DT)))
        q_ref = q_k[max(0, i_ref)]
        rows.append(dict(k=k, t=round(time.time() - t_start, 3), q=np.round(q, 4).tolist(),
                         qd=np.round(qd, 3).tolist(), tcp=np.round(tip, 4).tolist(),
                         tau=round(tau, 4), q_ref=np.round(q_ref, 4).tolist(),
                         d_grasp=round(float(np.linalg.norm(tip - p_grasp)), 4),
                         age_ms=round(1000 * age, 1)))
        near = np.linalg.norm(tip - p_grasp) < 0.03
        if tau >= TAU_HARD:
            reason = f"HARD contact (tau {tau:.3f} >= {TAU_HARD}) at tip {np.round(tip, 4).tolist()}"
            link.send_segment(q, q, time.time(), seq=k)      # overwrite the streamed reference: hold here
            break
        if tau >= TAU_FIRM and near:
            reason = f"TOUCH (tau {tau:.3f}) at tip {np.round(tip, 4).tolist()}, " \
                     f"{1000 * np.linalg.norm(tip - p_grasp):.0f} mm from the grasp point"
            link.send_segment(q, q, time.time(), seq=k)
            break
        if k % 10 == 0:
            print(f"[ctrl] k={k:3d} t {time.time() - t_start:5.2f} tip {np.round(tip, 3).tolist()} "
                  f"|tip-grasp| {100 * np.linalg.norm(tip - p_grasp):5.1f} cm  tau {tau:.3f}  "
                  f"fb age {1000 * age:.0f} ms")
        k += 1
    print(f"[ctrl] motion end: {reason} after {len(rows)} samples")
    time.sleep(0.4)

    # ---- retract 5 cm, then home ----------------------------------------
    import mujoco
    demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    dik = mujoco.MjData(ob.m)
    q_now, _, _ = link.state()
    tip_now, _ = ob.fk(q_now)
    q_up, err = demo["ik"](ob.m, dik, "tcp",
                           [float(tip_now[0]), float(tip_now[1]), float(tip_now[2]) + 0.05],
                           demo["R_DOWN"], q_now)
    legs = []
    if err < 0.005:
        legs.append((np.asarray(q_up, float), "retract 5 cm"))
    else:
        print(f"[ctrl] retract IK failed (err {err:.4f}) -> skipping that leg")
    legs.append((np.asarray(START_Q, float), "home"))
    for q_to, lab in legs:
        q_from, _, _ = link.state()
        T = max(0.5, float(np.degrees(np.abs(q_to - q_from)).max()) / 6.0)
        n = int(T / KNOT_DT) + 1
        qs = [q_from + (q_to - q_from) * i / (n - 1) for i in range(n)]
        bad = [v for qq in qs for v in [guard.check_q(qq)] if v]
        if bad:
            print(f"[guard] {lab} path violates ({'; '.join(bad[0])}) -> stopping here")
            break
        print(f"[ctrl] {lab}: {T:.1f} s, {n} knots")
        link.send_path(qs, KNOT_DT, time.time() + 0.2)
        time.sleep(T + 0.8)

    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        json.dump(rows, open(a.log, "w"))
        print(f"[log] {len(rows)} rows -> {a.log}")
    q_f, _, _ = link.state()
    print(f"[ctrl] done: {reason}; final tip {np.round(ob.fk(q_f)[0], 4).tolist()}")
    link.close()


if __name__ == "__main__":
    main()

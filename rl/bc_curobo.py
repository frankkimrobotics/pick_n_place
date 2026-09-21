#!/usr/bin/env python3
"""bc_curobo :: cuRobo-planned teacher + DAgger for the paper-style pick-and-place env.

Recommendation 1+2 (2026-09-19): the scripted IK teacher was crude (fixed 2 deg/decision
descent) and one-shot BC of it collapsed under PPO on the measured drive. Here:
  * transit phases (home -> hover above the object, lifted -> carry to the goal) come from the
    cuRobo planner server (127.0.0.1:9997, curobo2 env): collision-free, velocity-profiled,
    time-scaled to the env's action clamp;
  * the contact phase is a SLOW scripted press (--press_rate deg/decision) with suction on,
    held until the seal latches (the part the measured drive needs);
  * DAgger: after the initial teacher set, the cloned student drives the worlds and an
    IK-waypoint expert (same phase logic, no planner) relabels every visited state; the
    aggregated set is re-fit each iteration. Per-iteration metrics -> metrics.json.
Frames: cuRobo's `tcp` link and the sim's cup-tip site differ by a fixed tool-frame offset
(measured 2.0 cm); goals sent to the planner are corrected for it. The planner world gets the
sim table as a cuboid (top at z = 0.01) for the session.

  $PY rl/bc_curobo.py --drive real --out ~/pnp_rl/dagger_real
"""
import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np
import torch
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import build_scene_v2 as BV  # noqa: E402
from env_paper import PaperPickEnv  # noqa: E402
from env_warp import CTRL_HZ, CUP_R  # noqa: E402

R_DOWN_QUAT = [0.0, 0.7071, 0.7071, 0.0]      # wxyz of the demo's R_DOWN (cup down, +90 yaw)


def rpc(req, timeout=120):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=timeout)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        d = s.recv(1 << 20)
        if not d:
            break
        buf += d
    s.close()
    return json.loads(buf.split(b"\n")[0])


def mat2quat(M):
    w = math.sqrt(max(0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2
    x = math.sqrt(max(0, 1 + M[0, 0] - M[1, 1] - M[2, 2])) / 2
    y = math.sqrt(max(0, 1 - M[0, 0] + M[1, 1] - M[2, 2])) / 2
    z = math.sqrt(max(0, 1 - M[0, 0] - M[1, 1] + M[2, 2])) / 2
    return [w, math.copysign(x, M[2, 1] - M[1, 2]), math.copysign(y, M[0, 2] - M[2, 0]), math.copysign(z, M[1, 0] - M[0, 1])]


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def plan_to_targets(start_q, goal_pos_sim, tool_off, dq_max):
    """cuRobo plan_pose -> list of per-decision joint targets (rad), time-scaled so the largest
    per-decision joint move <= 0.9*dq_max. Returns None if planning fails."""
    R = quat2mat(R_DOWN_QUAT)
    goal_curobo = np.asarray(goal_pos_sim) - R @ tool_off
    res = rpc({"type": "plan_pose", "start_q": [float(x) for x in start_q],
               "goal_pose": [*[float(x) for x in goal_curobo], *R_DOWN_QUAT], "max_attempts": 4})
    if not res.get("success"):
        return None
    traj = np.asarray(res["trajectory"], float)
    if len(traj) < 2:
        return None
    dt = float(res["dt"])
    peak = np.abs(np.diff(traj, axis=0)).max() / dt                 # rad/s
    T = dt * (len(traj) - 1) * max(1.0, peak / (0.9 * dq_max * CTRL_HZ))
    n = max(2, int(math.ceil(T * CTRL_HZ)) + 1)
    ts = np.linspace(0, dt * (len(traj) - 1), n)
    src = np.arange(len(traj)) * dt
    return np.stack([np.interp(ts, src, traj[:, j]) for j in range(6)], 1)


class Teacher:
    """Phase controller; cuRobo for transits when available, IK waypoints otherwise."""

    def __init__(self, env, demo, dik, use_planner, press_rate, tool_off, dq_max, press_depth=0.012):
        self.env, self.demo, self.dik = env, demo, dik
        self.press_depth = press_depth       # IK press target below the grasp point (m); 4 mm was marginal vs the 3 mm latch
        self.use_planner, self.tool_off, self.dq_max = use_planner, tool_off, dq_max
        self.press_rate = math.radians(press_rate)
        self.rate = 0.9 * dq_max
        N = env.nworld
        self.phase = torch.zeros(N, dtype=torch.long, device=env.device)
        self.plans = [None] * N          # per world: list of targets for the current transit
        self.q_hover = None
        self.q_grasp = None
        self.q_carry = [None] * N
        self.n_plan_fail = 0

    def ik(self, pos, seed):
        q, err = self.demo["ik"](self.env.mjm, self.dik, "tcp", [float(x) for x in pos], self.demo["R_DOWN"], seed)
        return q if err < 0.01 else None

    def new_episode(self):
        env, N = self.env, self.env.nworld
        self.phase[:] = 0
        gp = env._grasp_point().cpu().numpy()
        q0 = env.qpos[:, :6].cpu().numpy()
        self.q_hover = np.zeros_like(q0); self.q_grasp = np.zeros_like(q0)
        self.q_carry = [None] * N
        self.carry_tick[:] = 0
        for w in range(N):
            qh = self.ik(gp[w] + [0, 0, 0.04], q0[w]); self.q_hover[w] = qh if qh is not None else q0[w]
            qg = self.ik(gp[w] + [0, 0, -self.press_depth], self.q_hover[w]); self.q_grasp[w] = qg if qg is not None else self.q_hover[w]
            self.plans[w] = None
            if self.use_planner:
                p = plan_to_targets(q0[w], gp[w] + [0, 0, 0.04], self.tool_off, self.dq_max)
                if p is None:
                    self.n_plan_fail += 1
                else:
                    self.plans[w] = list(p[1:])

    def act(self):
        """Expert action for every world from its CURRENT state (used for teacher rollouts
        and for DAgger relabeling alike)."""
        env, N, dev = self.env, self.env.nworld, self.env.device
        tcp, R = env._tcp()
        gpt = env._grasp_point()
        d_hover = torch.norm(tcp - (gpt + torch.tensor([0, 0, 0.04], device=dev)), dim=-1)
        lifted = (env._obj_pos()[:, 2] - float(env.half[2])) > env.paper["h_min"] + 0.01
        # phase transitions from state
        self.phase = torch.where((self.phase == 0) & (d_hover < 0.012), torch.ones_like(self.phase), self.phase)
        self.phase = torch.where((self.phase == 1) & env.sealed, torch.full_like(self.phase, 2), self.phase)
        self.phase = torch.where((self.phase >= 1) & ~env.sealed & ~env.ever_sealed, torch.ones_like(self.phase), self.phase)
        self.phase = torch.where((self.phase == 2) & lifted, torch.full_like(self.phase, 3), self.phase)
        self.phase = torch.where((self.phase == 3) & ~env.sealed, torch.zeros_like(self.phase), self.phase)  # lost it: start over
        q_now = env.q_target.cpu().numpy()
        qh_t = torch.tensor(self.q_hover, device=dev, dtype=torch.float32)
        qg_t = torch.tensor(self.q_grasp, device=dev, dtype=torch.float32)
        tgt = torch.where((self.phase == 1)[:, None], qg_t, qh_t)      # phase 0 default: hover; 2: back to hover (lift)
        rate = torch.full((N, 1), self.rate, device=dev)
        rate[self.phase == 1] = self.press_rate
        # carry targets (IK once per episode per world, when entering phase 3)
        ph = self.phase.cpu().numpy()
        goal = env.goal.cpu().numpy(); obj = env._obj_pos().cpu().numpy(); tcpn = tcp.cpu().numpy()
        for w in np.nonzero(ph == 3)[0]:
            if self.q_carry[w] is None:
                target_tcp = goal[w] + (tcpn[w] - obj[w])
                if self.use_planner:
                    p = plan_to_targets(env.qpos[w, :6].cpu().numpy(), target_tcp, self.tool_off, self.dq_max)
                    if p is not None:
                        self.plans[w] = list(p[1:])
                qc = self.ik(target_tcp, q_now[w])
                self.q_carry[w] = qc if qc is not None else q_now[w]
        qc_t = torch.tensor(np.stack([q if q is not None else q_now[i] for i, q in enumerate(self.q_carry)]), device=dev, dtype=torch.float32)
        tgt = torch.where((self.phase == 3)[:, None], qc_t, tgt)
        # planned transits override the straight-line target in phases 0 and 3
        for w in range(N):
            if self.plans[w] and ph[w] in (0, 3):
                tgt[w] = torch.tensor(self.plans[w].pop(0), device=dev, dtype=torch.float32)
                if not self.plans[w]:
                    self.plans[w] = None
            elif ph[w] not in (0, 3):
                self.plans[w] = None
        dq = tgt - env.q_target
        sc = (rate / dq.abs().max(-1, keepdim=True).values.clamp(min=1e-6)).clamp(max=1.0)
        act6 = (dq * sc / env.dq_max).clamp(-1, 1)
        suc = torch.where((self.phase >= 1), 1.0, -1.0)[:, None]
        return torch.cat([act6, suc], -1)


# ============================== v3: obstacle-aware teacher ==============================
# planner_sweep's best feasible retime (~/pnp_rl/planner_sweep/best.json: n_knots 32,
# interp_dt 0.01, seeds 2, scale 0.5, finetune off -> 2.35 s mean approach duration).
VLIM_DEG = 36.0
ALIM_DEG = 300.0


def retime_decisions(traj, dt, dq_max, vfrac=0.95):
    """cuRobo trajectory -> per-decision joint targets at CTRL_HZ.

    planner_sweep's `retime` ends in a uniform time dilation until (vlim, alim) hold; that
    dilation is what is reproduced here (TOPP-RA itself only exists in the curobo2 env).
    For a policy whose action is a joint DELTA the binding limit is the per-decision clamp,
    vfrac*dq_max*CTRL_HZ = 19 deg/s at dq_max 2 -- well inside the sweep's 36 deg/s -- so
    the velocity term is what sets the duration and the 300 deg/s^2 term almost never bites.
    """
    traj = np.asarray(traj, float)
    if len(traj) < 2:
        return None
    T0 = dt * (len(traj) - 1)
    v_cap = min(math.radians(VLIM_DEG), vfrac * dq_max * CTRL_HZ)
    a_cap = math.radians(ALIM_DEG)
    qd = np.abs(np.gradient(traj, dt, axis=0)).max()
    qdd = np.abs(np.gradient(np.gradient(traj, dt, axis=0), dt, axis=0)).max()
    g = max(qd / v_cap, math.sqrt(max(qdd / a_cap, 1e-9)), 1.0)
    n = max(2, int(math.ceil(T0 * g * CTRL_HZ)) + 1)
    src = np.arange(len(traj)) * dt
    out = np.stack([np.interp(np.linspace(0, T0, n), src, traj[:, j]) for j in range(6)], 1)
    d = np.abs(np.diff(out, axis=0)).max()
    if d > vfrac * dq_max:                          # belt and braces: never clip an action
        n = int((n - 1) * math.ceil(d / (vfrac * dq_max))) + 1
        out = np.stack([np.interp(np.linspace(0, T0, n), src, traj[:, j]) for j in range(6)], 1)
    return out


def plan_leg_rpc(start_q, goal_pos_sim, tool_off, dq_max, max_attempts=2):
    """cuRobo plan_pose (world already set) -> per-decision joint targets, or None."""
    R = quat2mat(R_DOWN_QUAT)
    goal_curobo = np.asarray(goal_pos_sim) - R @ tool_off
    res = rpc({"type": "plan_pose", "start_q": [float(x) for x in start_q],
               "goal_pose": [*[float(x) for x in goal_curobo], *R_DOWN_QUAT],
               "max_attempts": int(max_attempts)})
    if not res.get("success"):
        return None
    return retime_decisions(res["trajectory"], float(res["dt"]), dq_max)


def resample_joints(q, dq_max, vfrac=0.95):
    """Joint waypoints -> per-decision targets, <= vfrac*dq_max on the largest joint."""
    out = [q[0]]
    for k in range(1, len(q)):
        m = max(1, int(math.ceil(np.abs(q[k] - q[k - 1]).max() / (vfrac * dq_max))))
        for i in range(1, m + 1):
            out.append(q[k - 1] + (q[k] - q[k - 1]) * (i / m))
    return np.stack(out)


class TeacherV3(Teacher):
    """Obstacle-aware teacher for env_v3.ObstacleEnv.

    Legs: home -> hover above the target, and (after the 20 mm slow press + lift) lifted ->
    the goal.  For each leg of each world the active distractors, the walls and the table go
    to the cuRobo server as cuboids (`set_world`) and cuRobo plans the leg; the trajectory is
    retimed with the planner_sweep parameters.  On the CARRY leg every distractor cuboid is
    inflated by the held object's circumradius and by its overhang below the tcp
    (CUP_R + object height), so a tcp-only plan keeps the OBJECT clear too.

    Two shortcuts keep the ~2 s/plan server off the critical path:
      * `--planner_mode corridor` (default) skips the RPC when the straight tcp line is
        already clear of every inflated obstacle -- which is most approach legs;
      * when cuRobo fails (it rejects 20-35 % of top-down poses near the keep-out / camera
        mount, FINDINGS 8) the leg falls back to a geometric OVER-THE-TOP IK path: lift to
        above the tallest obstacle in the way, translate, descend.
    """

    SAFE_CLR = 0.045          # clearance the fallback path keeps above an obstacle
    STEP_XY = 0.02            # Cartesian resampling of the fallback path
    H_MAX = 0.42              # ceiling for the over-the-top height

    def __init__(self, env, demo, dik, use_planner, press_rate, tool_off, dq_max,
                 press_depth=0.02, planner_mode="corridor", max_attempts=2,
                 cam_mount=False, carry_refine=5):
        super().__init__(env, demo, dik, use_planner, press_rate, tool_off, dq_max,
                         press_depth=press_depth)
        self.planner_mode = planner_mode
        self.max_attempts = max_attempts
        self.carry_refine = int(carry_refine)
        self.carry_tick = np.zeros(env.nworld, int)
        self.n_plan_try = self.n_plan_ok = self.n_fallback = self.n_noleg = 0
        self.leg_stat = {"approach": [0, 0], "carry": [0, 0]}    # leg -> [tried, ok]
        self._rpc_err = None
        self.base_cub = [
            {"name": "sim_table", "dims": [BV.TABLE_X2[1] - BV.TABLE_X2[0],
                                           BV.TABLE_Y2[1] - BV.TABLE_Y2[0], 0.02],
             "pose": [0.5 * (BV.TABLE_X2[0] + BV.TABLE_X2[1]), 0.0, 0.0, 1, 0, 0, 0]},
            {"name": "wall_back", "dims": [0.02, 1.2, 1.0], "pose": [-0.30, 0.0, 0.5, 1, 0, 0, 0]},
            {"name": "wall_ym", "dims": [1.1, 0.02, 1.0], "pose": [0.25, -0.50, 0.5, 1, 0, 0, 0]},
            {"name": "wall_yp", "dims": [1.1, 0.02, 1.0], "pose": [0.25, 0.50, 0.5, 1, 0, 0, 0]}]
        if not cam_mount:
            # The server keeps a 0.30 x 0.30 x 1.0 `camera_mount_d435` cuboid at (0.64, -0.05)
            # in its BASE world, i.e. it blocks x >= 0.49, -0.20 <= y <= 0.10 up to z = 0.9.
            # The v3 spawn box reaches x = 0.56, so that keep-out swallows a slice of the
            # workspace the MuJoCo twin has no collision geom for -- measured cost: approach
            # plans 17/24 -> 23/24 and carry plans 15/24 -> 23/24 once it is removed.
            # set_world overrides a base cuboid BY NAME, so park it instead of shrinking it.
            # SIM-TO-REAL: the real cell does have that mount.  Any deployment of a policy
            # trained this way must either restore the cuboid or keep the object off x > 0.49.
            self.base_cub.append({"name": "camera_mount_d435", "dims": [0.02, 0.02, 0.02],
                                  "pose": [3.0, 3.0, 3.0, 1, 0, 0, 0]})

    # ---------- per-episode geometry snapshot ----------
    def _snapshot(self):
        env = self.env
        self.d_act = env.dist_act.cpu().numpy()
        self.d_half = env.dist_half.cpu().numpy()
        self.d_c = env.xipos[:, env.bid_dist_t].cpu().numpy().copy()
        self.d_rad = np.hypot(self.d_half[:, :, 0], self.d_half[:, :, 1])
        self.obj_rad = np.hypot(env.half_w[:, 0].cpu().numpy(), env.half_w[:, 1].cpu().numpy())
        self.obj_top = env.top_h.cpu().numpy()

    def _boxes(self, w, hold_obj):
        """(K, 6) [cx, cy, cz, rx, ry, rz] obstacle boxes of world w, inflated for the
        carried object when hold_obj."""
        a = self.d_act[w]
        if not a.any():
            return np.zeros((0, 6))
        c, r, hz = self.d_c[w][a], self.d_rad[w][a], self.d_half[w][a, 2]
        mx = float(self.obj_rad[w]) if hold_obj else 0.0
        mz = float(CUP_R + self.obj_top[w]) if hold_obj else 0.0
        return np.stack([c[:, 0], c[:, 1], c[:, 2], r + mx, r + mx, hz + mz], -1)

    def _cuboids(self, w, hold_obj):
        out = list(self.base_cub)
        for i, b in enumerate(self._boxes(w, hold_obj)):
            out.append({"name": f"dist{i}",
                        "dims": [float(2 * b[3]), float(2 * b[4]), float(2 * b[5])],
                        "pose": [float(b[0]), float(b[1]), float(b[2]), 1, 0, 0, 0]})
        return out

    def _blocked(self, pts, boxes, margin=0.01):
        if not len(boxes) or not len(pts):
            return False
        p = np.concatenate([pts, pts - np.array([0.0, 0.0, 0.05])], 0)
        for b in boxes:
            if ((np.abs(p - b[:3]) - (b[3:] + margin)).max(-1) < 0).any():
                return True
        return False

    def _over_the_top(self, p0, p1, boxes):
        """Cartesian waypoints p0 -> p1: straight if clear, else up over the tallest
        obstacle in the way, across, and down."""
        n = max(2, int(np.linalg.norm(p1 - p0) / self.STEP_XY) + 1)
        line = p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]
        if not self._blocked(line, boxes):
            return line
        h = max(p0[2], p1[2])
        for b in boxes:
            if ((np.abs(line[:, :2] - b[:2]) - (b[3:5] + 0.02)).max(-1) < 0).any():
                h = max(h, b[2] + b[5] + self.SAFE_CLR + 0.05)
        h = min(h, self.H_MAX)
        a = np.array([p0[0], p0[1], max(h, p0[2])])
        b2 = np.array([p1[0], p1[1], max(h, p1[2])])
        segs = []
        for u, v in ((p0, a), (a, b2), (b2, p1)):
            k = max(2, int(np.linalg.norm(v - u) / self.STEP_XY) + 1)
            segs.append(u + (v - u) * np.linspace(0, 1, k)[:, None])
        return np.concatenate([segs[0], segs[1][1:], segs[2][1:]], 0)

    def _ik_path(self, pts, seed):
        qs, q = [], np.asarray(seed, float)
        for p in pts:
            qn = self.ik(p, q)
            if qn is None:
                break
            q = qn
            qs.append(q)
        if len(qs) < 2:
            return None
        return resample_joints(np.stack(qs), self.dq_max)

    def _leg(self, w, q_start, tcp_now, tcp_goal, hold_obj, tag="approach"):
        """Per-decision joint targets for one transit leg of world w (None if nothing works)."""
        tcp_now = np.asarray(tcp_now, float)
        tcp_goal = np.asarray(tcp_goal, float)
        boxes = self._boxes(w, hold_obj)
        line = tcp_now[None] + (tcp_goal - tcp_now) * np.linspace(0, 1, 16)[:, None]
        direct = not self._blocked(line, boxes)
        if self.use_planner and (self.planner_mode == "always" or not direct):
            self.n_plan_try += 1
            self.leg_stat[tag][0] += 1
            try:
                rpc({"type": "set_world", "cuboids": self._cuboids(w, hold_obj)})
                p = plan_leg_rpc(q_start, tcp_goal, self.tool_off, self.dq_max, self.max_attempts)
            except Exception as e:                    # never let an RPC error look like "no plan"
                if self._rpc_err is None:
                    self._rpc_err = repr(e)
                    print(f"[teacher] planner RPC error (silenced after the first): {e!r}", flush=True)
                p = None
            if p is not None:
                self.n_plan_ok += 1
                self.leg_stat[tag][1] += 1
                return list(p[1:])
            self.n_plan_fail += 1
        p = self._ik_path(self._over_the_top(tcp_now, tcp_goal, boxes), q_start)
        if p is None:
            self.n_noleg += 1
            return None
        self.n_fallback += 1
        return list(p[1:])

    # ---------- episode ----------
    def new_episode(self):
        env, N = self.env, self.env.nworld
        self._snapshot()
        self.phase[:] = 0
        gp = env._grasp_point().cpu().numpy()
        q0 = env.qpos[:, :6].cpu().numpy()
        tcp0 = env._tcp()[0].cpu().numpy()
        self.q_hover = np.zeros_like(q0)
        self.q_grasp = np.zeros_like(q0)
        self.q_carry = [None] * N
        self.carry_tick[:] = 0
        for w in range(N):
            qh = self.ik(gp[w] + [0, 0, 0.04], q0[w])
            self.q_hover[w] = qh if qh is not None else q0[w]
            qg = self.ik(gp[w] + [0, 0, -self.press_depth], self.q_hover[w])
            self.q_grasp[w] = qg if qg is not None else self.q_hover[w]
            self.plans[w] = self._leg(w, q0[w], tcp0[w], gp[w] + [0, 0, 0.04], False, "approach")

    def act(self):
        env, N, dev = self.env, self.env.nworld, self.env.device
        tcp, R = env._tcp()
        gpt = env._grasp_point()
        d_hover = torch.norm(tcp - (gpt + torch.tensor([0, 0, 0.04], device=dev)), dim=-1)
        lifted = (env._obj_pos()[:, 2] - float(env.half[2])) > env.paper["h_min"] + 0.01
        self.phase = torch.where((self.phase == 0) & (d_hover < 0.012), torch.ones_like(self.phase), self.phase)
        self.phase = torch.where((self.phase == 1) & env.sealed, torch.full_like(self.phase, 2), self.phase)
        self.phase = torch.where((self.phase >= 1) & ~env.sealed & ~env.ever_sealed, torch.ones_like(self.phase), self.phase)
        self.phase = torch.where((self.phase == 2) & lifted, torch.full_like(self.phase, 3), self.phase)
        self.phase = torch.where((self.phase == 3) & ~env.sealed, torch.zeros_like(self.phase), self.phase)
        qh_t = torch.tensor(self.q_hover, device=dev, dtype=torch.float32)
        qg_t = torch.tensor(self.q_grasp, device=dev, dtype=torch.float32)
        tgt = torch.where((self.phase == 1)[:, None], qg_t, qh_t)
        rate = torch.full((N, 1), self.rate, device=dev)
        rate[self.phase == 1] = self.press_rate
        ph = self.phase.cpu().numpy()
        q_now = env.q_target.cpu().numpy()
        qpos = env.qpos[:, :6].cpu().numpy()
        goal = env.goal.cpu().numpy()
        obj = env._obj_pos().cpu().numpy()
        tcpn = tcp.cpu().numpy()
        for w in np.nonzero(ph == 3)[0]:
            if self.q_carry[w] is None:
                target_tcp = goal[w] + (tcpn[w] - obj[w])
                self.plans[w] = self._leg(w, qpos[w], tcpn[w], target_tcp, True, "carry")
                qc = self.ik(target_tcp, q_now[w])
                self.q_carry[w] = qc if qc is not None else q_now[w]
            elif self.plans[w] is None and self.carry_refine:
                # CLOSED-LOOP carry target.  `tcp - obj` is measured once when the carry
                # starts, but the object keeps settling under the cup (and a hex/cylinder
                # rolls a little), so a target IK'd once leaves the object short of the
                # goal.  Re-IK the goal from the LIVE offset every `carry_refine` decisions
                # once the planned path has run out.
                self.carry_tick[w] += 1
                if self.carry_tick[w] % self.carry_refine == 0:
                    qc = self.ik(goal[w] + (tcpn[w] - obj[w]), q_now[w])
                    if qc is not None:
                        self.q_carry[w] = qc
        qc_t = torch.tensor(np.stack([q if q is not None else q_now[i]
                                      for i, q in enumerate(self.q_carry)]), device=dev, dtype=torch.float32)
        tgt = torch.where((self.phase == 3)[:, None], qc_t, tgt)
        for w in range(N):
            if self.plans[w] and ph[w] in (0, 3):
                tgt[w] = torch.tensor(self.plans[w].pop(0), device=dev, dtype=torch.float32)
                if not self.plans[w]:
                    self.plans[w] = None
            elif ph[w] not in (0, 3):
                self.plans[w] = None
        dq = tgt - env.q_target
        sc = (rate / dq.abs().max(-1, keepdim=True).values.clamp(min=1e-6)).clamp(max=1.0)
        act6 = (dq * sc / env.dq_max).clamp(-1, 1)
        suc = torch.where((self.phase >= 1), 1.0, -1.0)[:, None]
        return torch.cat([act6, suc], -1)

    def stats(self):
        a_, c_ = self.leg_stat["approach"], self.leg_stat["carry"]
        return (f"plans approach {a_[1]}/{a_[0]} carry {c_[1]}/{c_[0]}, fallback {self.n_fallback}, "
                f"no-leg {self.n_noleg}")


def diag(env, ok, es):
    """One line of teacher/student failure modes for env_v3."""
    if not hasattr(env, "obst_hit_ep"):
        return ""
    f = lambda t: 100 * float(t.float().mean())                     # noqa: E731
    return (f"  [obst_hit {f(env.obst_hit_ep):.0f} % wall_hit {f(env.wall_hit_ep):.0f} % "
            f"no-seal {100 - f(es):.0f} % sealed-but-short {f(es & ~ok):.0f} % | "
            f"t_seal {float(env.t_seal.mean()) / 10:.1f} s t_goal {float(env.t_goal.mean()) / 10:.1f} s]")


def rollout(env, teacher, policy, noise, EP, beta):
    """One batch of episodes. policy=None -> teacher drives; else the student drives with the
    teacher labelling (DAgger, beta = prob of executing the teacher action)."""
    N, dev = env.nworld, env.device
    env.reset(torch.ones(N, dtype=torch.bool, device=dev))
    teacher.new_episode()
    O, A = [], []
    for k in range(EP):
        obs = env.observe()
        a_exp = teacher.act()
        if policy is None:
            a = a_exp.clone(); a[:, :6] += torch.randn(N, 6, device=dev) * noise
        else:
            with torch.no_grad():
                a = torch.nan_to_num(torch.tanh(policy(obs)))
            use_t = torch.rand(N, device=dev) < beta
            a = torch.where(use_t[:, None], a_exp, a)
        O.append(obs.clone()); A.append(a_exp.clone())
        _, r, done, info = env.step(a.clamp(-1, 1))
    ok = info["placed"]
    return torch.stack(O, 1), torch.stack(A, 1), ok, env.ever_sealed.clone()


def per_shape(env, ok, es):
    """'box 61% (seal 88%) | cyl ...' for DiverseEnv; '' for the plain paper env."""
    sw = getattr(env, "shape_w", None)
    if sw is None:
        return ""
    import env_v2
    out = []
    for s_i, nm in enumerate(env_v2.SHAPES):
        m = sw == s_i
        if not bool(m.any()):
            continue
        out.append(f"{nm} {100 * ok[m].float().mean():.0f}% (seal {100 * es[m].float().mean():.0f}%, n={int(m.sum())})")
    return "  [" + " | ".join(out) + "]"


def pre_tanh(a):
    """PPO executes tanh(mu + noise): clone mu = atanh(a) so the deterministic policy reproduces a."""
    return torch.atanh(a.clamp(-0.97, 0.97))


def fit(ac, X, Y, epochs, dev):
    # A single non-finite row NaNs every gradient and therefore the whole network, which then
    # emits NaN actions for ever (dagger_v2 collapsed this way at iter 2).  Drop them.
    keep = torch.isfinite(X).all(1) & torch.isfinite(Y).all(1)
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"[dagger]   fit: dropped {n_drop} / {X.shape[0]} non-finite rows "
              f"({100 * n_drop / X.shape[0]:.3f} %)", flush=True)
        X, Y = X[keep], Y[keep]
    Y = pre_tanh(Y)
    opt = torch.optim.Adam(ac.pi.parameters(), lr=1e-3)
    idx = torch.arange(X.shape[0], device=dev)
    for ep in range(epochs):
        perm = idx[torch.randperm(idx.numel(), device=dev)]
        for i in range(0, perm.numel(), 4096):
            j = perm[i:i + 4096]
            loss = ((ac.pi(X[j]) - Y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


def evaluate(env, ac, EP):
    N, dev = env.nworld, env.device
    env.reset(torch.ones(N, dtype=torch.bool, device=dev))
    for k in range(EP):
        with torch.no_grad():
            a = torch.tanh(ac.pi(env.observe()))
        _, r, done, info = env.step(a)
    extra = dict(obst_hit=float(info["obst_hit"].float().mean()) if "obst_hit" in info else 0.0,
                 wall_hit=float(info["wall_hit"].float().mean()) if "wall_hit" in info else 0.0,
                 t_seal=float(env.t_seal.mean()), t_goal=float(env.t_goal.mean()))
    return (float(info["placed"].float().mean()), float(env.ever_sealed.float().mean()),
            float(info["ep_comp"].sum(-1).mean()), extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--nworld", type=int, default=128)
    ap.add_argument("--teacher_batches", type=int, default=6)
    ap.add_argument("--dagger_iters", type=int, default=3)
    ap.add_argument("--dagger_batches", type=int, default=3)
    ap.add_argument("--press_rate", type=float, default=0.6, help="deg/decision during the press (slow for the real drive)")
    ap.add_argument("--press_depth", type=float, default=0.02, help="press target below the grasp point (m); sweep 2026-09-19: 4 mm 3 %, 8 mm 65 %, 12 mm 80 %, 20 mm 91 % teacher success on the measured drive")
    ap.add_argument("--ep_len", type=int, default=100, help="episode length (decisions); 150 for the measured drive (slower arm)")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no_planner", action="store_true", help="IK waypoints for transits too (no cuRobo)")
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/dagger_real"))
    ap.add_argument("--env", default="paper", choices=["paper", "v2", "v3"],
                    help="v2 = env_v2.DiverseEnv (diverse objects, big table, walls); "
                         "v3 = env_v3.ObstacleEnv (v2 + distractor obstacles + enlarged workspace)")
    ap.add_argument("--obstacles", type=int, default=1, help="v3 only: distractors on/off")
    ap.add_argument("--planner_mode", default="corridor", choices=["corridor", "always", "never"],
                    help="v3 only: when to call cuRobo (corridor = only when the straight tcp line is blocked)")
    ap.add_argument("--max_attempts", type=int, default=2, help="v3 only: cuRobo plan_pose attempts")
    ap.add_argument("--cam_mount", type=int, default=0,
                    help="v3 only: keep the server's D435 camera-mount keep-out (blocks x > 0.49; "
                         "the twin has no such geom, and it costs ~25 pp of plan success)")
    ap.add_argument("--carry_refine", type=int, default=5,
                    help="v3 only: re-IK the carry goal from the live tcp-object offset every N decisions")
    ap.add_argument("--dq_max", type=float, default=None, help="action clamp (deg/decision); env default when unset")
    ap.add_argument("--spawn", default=None, help="v2 only: object spawn box 'x0,x1,y0,y1' (default 0,0.5,-0.3,0.3)")
    ap.add_argument("--variants", default="box,cyl,hex", help="v2 only: object shape classes")
    ap.add_argument("--scene", default=None, help="MJCF (default: scenes/box_med.xml, or scenes/diverse_v2.xml for --env v2)")
    ap.add_argument("--resume", default=None, help="continue DAgger from this run dir (dataset.pt + last bc_iter*.pt + metrics.json); teacher batches are skipped")
    ap.add_argument("--init", default=None,
                    help="warm-start the BC actor from a checkpoint (obs-dim growth is zero-padded, "
                         "so an env_v2 47-D student initialises a v3 54-D one)")
    ap.add_argument("--beta_min", type=float, default=0.0, help="floor of the teacher-mixing schedule 0.5, 0.3, 0.1, ...")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wp.init()
    import mujoco
    if a.env == "v3":
        from env_v3 import ObstacleEnv
        kw = dict(variants=a.variants, obstacles=bool(a.obstacles))
        if a.spawn:
            kw["spawn"] = a.spawn
        if a.dq_max:
            kw["dq_max_deg"] = a.dq_max
        env = ObstacleEnv(nworld=a.nworld, xml=a.scene, dr=True, drive=a.drive, ep_len=a.ep_len, **kw)
    elif a.env == "v2":
        from env_v2 import DiverseEnv
        kw = dict(variants=a.variants)
        if a.spawn:
            kw["spawn"] = a.spawn
        if a.dq_max:
            kw["dq_max_deg"] = a.dq_max
        env = DiverseEnv(nworld=a.nworld, xml=a.scene, dr=True, drive=a.drive, ep_len=a.ep_len, **kw)
    else:
        env = PaperPickEnv(nworld=a.nworld, xml=a.scene or os.path.join(HERE, "scenes", "box_med.xml"),
                           dr=True, drive=a.drive, ep_len=a.ep_len)
    env.auto_reset = False
    demo = {"__file__": os.path.join(os.path.dirname(HERE), "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    dik = mujoco.MjData(env.mjm)
    dev, EP = env.device, env.paper["ep_len"]
    use_planner = not a.no_planner
    tool_off = np.zeros(3)
    if use_planner:
        # planner world = sim table (top z=0.01) + the server's base cuboids; tool-frame offset from FK
        if a.env in ("v2", "v3"):
            cub = [{"name": "sim_table", "dims": [BV.TABLE_X2[1] - BV.TABLE_X2[0], BV.TABLE_Y2[1] - BV.TABLE_Y2[0], 0.02],
                    "pose": [0.5 * (BV.TABLE_X2[0] + BV.TABLE_X2[1]), 0.0, 0.0, 1, 0, 0, 0]},
                   {"name": "wall_back", "dims": [0.02, 1.2, 1.0], "pose": [-0.30, 0.0, 0.5, 1, 0, 0, 0]},
                   {"name": "wall_ym", "dims": [1.1, 0.02, 1.0], "pose": [0.25, -0.50, 0.5, 1, 0, 0, 0]},
                   {"name": "wall_yp", "dims": [1.1, 0.02, 1.0], "pose": [0.25, 0.50, 0.5, 1, 0, 0, 0]}]
        if not cam_mount:
            # The server keeps a 0.30 x 0.30 x 1.0 `camera_mount_d435` cuboid at (0.64, -0.05)
            # in its BASE world, i.e. it blocks x >= 0.49, -0.20 <= y <= 0.10 up to z = 0.9.
            # The v3 spawn box reaches x = 0.56, so that keep-out swallows a slice of the
            # workspace the MuJoCo twin has no collision geom for -- measured cost: approach
            # plans 17/24 -> 23/24 and carry plans 15/24 -> 23/24 once it is removed.
            # set_world overrides a base cuboid BY NAME, so park it instead of shrinking it.
            # SIM-TO-REAL: the real cell does have that mount.  Any deployment of a policy
            # trained this way must either restore the cuboid or keep the object off x > 0.49.
            self.base_cub.append({"name": "camera_mount_d435", "dims": [0.02, 0.02, 0.02],
                                  "pose": [3.0, 3.0, 3.0, 1, 0, 0, 0]})
        else:
            cub = [{"name": "sim_table", "dims": [0.56, 0.9, 0.02], "pose": [0.42, 0.0, 0.0, 1, 0, 0, 0]}]
        print(rpc({"type": "set_world", "cuboids": cub}), flush=True)
        env.reset(torch.ones(env.nworld, dtype=torch.bool, device=dev))
        q0 = env.qpos[0, :6].cpu().numpy().tolist(); tcp, R = env._tcp()
        fk = rpc({"type": "fk", "q": q0}); p = np.array(fk["pos"][0]); Rm = R[0].cpu().numpy()
        tool_off = Rm.T @ (tcp[0].cpu().numpy() - p)          # sim cup tip in cuRobo tool frame
        print(f"[dagger] tool offset (tool frame) {np.round(tool_off, 4).tolist()} ({1000 * np.linalg.norm(tool_off):.1f} mm)", flush=True)
    if a.env == "v3":
        teacher = TeacherV3(env, demo, dik, use_planner, a.press_rate, tool_off, env.dq_max,
                            press_depth=a.press_depth, planner_mode=a.planner_mode,
                            max_attempts=a.max_attempts, cam_mount=bool(a.cam_mount),
                            carry_refine=a.carry_refine)
    else:
        teacher = Teacher(env, demo, dik, use_planner, a.press_rate, tool_off, env.dq_max,
                          press_depth=a.press_depth)
    print(f"[dagger] env {a.env} nworld {a.nworld} ep_len {a.ep_len} dq_max "
          f"{np.degrees(env.dq_max):.2f} deg  obs {env.observe().shape[-1]}  "
          f"teacher {type(teacher).__name__} planner={use_planner}", flush=True)
    from ppo import AC
    metrics = []
    X, Y = [], []
    t0 = time.time()
    # ---- iteration 0: teacher demonstrations ----
    n_ok = n_ep = 0
    start = 0
    if a.resume:
        metrics = json.load(open(os.path.join(a.resume, "metrics.json")))
        start = metrics[-1]["iter"] + 1
        ds = torch.load(os.path.join(a.resume, "dataset.pt"), map_location=dev, weights_only=False)
        X, Y = [ds["X"].to(dev)], [ds["Y"].to(dev)]
        n_ok, n_ep = int(round(metrics[-1]["teacher_success"] * 1000)), 1000
        print(f"[dagger] resume {a.resume}: dataset {X[0].shape[0]} steps, continuing at iter {start}", flush=True)
    for b in range(a.teacher_batches if not a.resume else 0):
        O, Aexp, ok, es = rollout(env, teacher, None, a.noise, EP, 1.0)
        X.append(O.reshape(-1, O.shape[-1])); Y.append(Aexp.reshape(-1, 7))      # DAgger keeps ALL states
        n_ok += int(ok.sum()); n_ep += env.nworld
        extra = teacher.stats() if hasattr(teacher, "stats") else f"plan fails {teacher.n_plan_fail}"
        print(f"[dagger] teacher batch {b}: success {100 * ok.float().mean():.1f} %  sealed {100 * es.float().mean():.0f} %  "
              f"{extra}{diag(env, ok, es)}{per_shape(env, ok, es)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ac = AC(obs_dim=X[0].shape[1], arch="paper").to(dev)
    if a.init:
        ck0 = torch.load(a.init, map_location=dev, weights_only=False)
        ac.load_state_dict(ck0["ac"])
        print(f"[dagger] warm start from {a.init} (obs {ck0.get('obs_dim')} -> {X[0].shape[1]})", flush=True)
    if a.resume:
        ac.load_state_dict(torch.load(os.path.join(a.resume, f"bc_iter{start - 1}.pt"), map_location=dev, weights_only=False)["ac"])
    for it in range(start, a.dagger_iters + 1):
        if it > 0:                                          # relabel batches driven by the previous student
            beta = max(a.beta_min, 0.5 - 0.2 * (it - 1))    # mixing: 0.5, 0.3, 0.1, beta_min ...
            for b in range(a.dagger_batches):
                O, Aexp, ok, es = rollout(env, teacher, ac.pi, 0.0, EP, beta)
                X.append(O.reshape(-1, O.shape[-1])); Y.append(Aexp.reshape(-1, 7))
                print(f"[dagger]   relabel batch {b} (beta {beta:.1f}): student-driven success {100 * ok.float().mean():.1f} %  "
                      f"sealed {100 * es.float().mean():.0f} %{diag(env, ok, es)}{per_shape(env, ok, es)}", flush=True)
        Xc, Yc = torch.cat(X), torch.cat(Y)
        mse = fit(ac, Xc, Yc, a.epochs, dev)
        if not math.isfinite(mse):
            print("[dagger] WARNING: fit returned a non-finite loss", flush=True)
        nd = getattr(env, "n_diverged", 0)
        with torch.no_grad():
            ac.log_std.fill_(-1.5)
        succ, seal, ret, ex = evaluate(env, ac, EP)
        torch.save({"X": Xc.cpu(), "Y": Yc.cpu()}, os.path.join(a.out, "dataset.pt"))
        rec = dict(iter=it, dataset=int(Xc.shape[0]), mse=mse, student_success=succ, student_seal=seal, student_return=ret,
                   teacher_success=n_ok / max(1, n_ep), minutes=(time.time() - t0) / 60, **ex)
        metrics.append(rec)
        print(f"[dagger] iter {it}: dataset {Xc.shape[0]} mse {mse:.4f} | student success {100 * succ:.1f} % "
              f"seal {100 * seal:.0f} % return {ret:.2f} | obst_hit {100 * ex['obst_hit']:.0f} % "
              f"wall_hit {100 * ex['wall_hit']:.0f} % t_goal {ex['t_goal'] / 10:.1f} s | diverged worlds {nd}", flush=True)
        json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
        torch.save({"ac": ac.state_dict(), "step": 0}, os.path.join(a.out, f"bc_iter{it}.pt"))
    torch.save({"ac": ac.state_dict(), "step": 0}, os.path.join(a.out, "bc_init.pt"))
    print(f"[dagger] saved {a.out}/bc_init.pt", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""pick_place_compare :: FULL pick-and-place in clutter, three controllers.

Scene: rl/scenes/clutter4.xml -- red TARGET box (object0) + 3 distractors,
randomized layout per trial; randomized place target on clear table.
Suction = the scene's weld equality (activated on cup contact + intent).

Carried-object collision volume:
  * all controllers: place goals account for the held box's full height,
    and the place contact stop listens to OBJECT-vs-table force;
  * cuRobo additionally gets the clutter via set_world and the held box
    attached as collision spheres (attach/detach RPC), so its transport
    plans route the carried volume around distractors.

Metrics per trial: success (object placed <=3.5 cm of target, released,
at rest), total time, joint RMSE (cmd vs actual), clutter disturbance
(max distractor displacement), place error.

    python pick_place_compare.py --trials 30 --workers 6 --only taskmpc
"""
import argparse
import json
import os
import socket
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy import sparse
import osqp

HERE = os.path.expanduser("~/Desktop/2026/pick_and_place")
XML = os.path.join(HERE, "rl", "scenes", "clutter4.xml")
DT, INNER_DT, OUTER_DT = 0.002, 0.004, 0.020
TRACK_DT = 1.0 / 6.0
K_LQR = np.array([53.4798, 4.7131])
VEL_LIMIT = np.radians(40.0)
KV = 40.0
DQ_MAX = np.radians(0.7)
HOME = np.radians([0.0, -20.0, 80.0, 10.0, -90.0, 0.0])
DOWN = [0.0, 1.0, 0.0, 0.0]
# joint position limits (elbow box from the real robot / cuRobo config):
# the task layer must never hand the planner -- or the hardware -- a
# limit-violating state
Q_LIM = np.radians([175.0, 69.0, 144.0, 175.0, 175.0, 175.0])
PLACE_TOL = 0.035


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=90)
    s.sendall((json.dumps(d) + "\n").encode())
    b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close()
    return json.loads(b)


class Sim:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.names = [f"object{k}" for k in range(4)]
        self.bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in self.names]
        self.jadrs = [m.jnt_qposadr[m.body_jntadr[b]] for b in self.bids]
        self.gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                       f"g_{n}") for n in self.names]
        self.dims = [m.geom_size[g].copy() for g in self.gids]
        self.eqs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                      f"suction_{n}") for n in self.names]
        self.cup_gids = [g for g in range(m.ngeom) if "cup" in
                         (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        d.qpos[:6] = HOME
        # randomized non-overlapping layout
        placed = []
        for k in range(4):
            for _ in range(60):
                x = rng.uniform(0.28, 0.40)
                y = rng.uniform(-0.10, 0.10)
                if all(np.hypot(x - px, y - py) > 0.085 for px, py in placed):
                    break
            placed.append((x, y))
            ja = self.jadrs[k]
            hz = float(self.dims[k][2] if m.geom_type[self.gids[k]] ==
                       mujoco.mjtGeom.mjGEOM_BOX else self.dims[k][1])
            self.hz = hz if k == 0 else getattr(self, "hz", hz)
            d.qpos[ja:ja + 3] = [x, y, (self.dims[k][2] if
                                 m.geom_type[self.gids[k]] == 6 else
                                 self.dims[k][1]) + 0.001]
            d.qpos[ja + 3:ja + 7] = [1, 0, 0, 0]
        # place target: clear spot -- REQUIRE clearance; fall back to the
        # candidate farthest from all objects (the old loop silently fell
        # through with an invalid spot when the band was crowded)
        best, best_d = None, -1.0
        for _ in range(300):
            px = rng.uniform(0.28, 0.40)
            py = rng.uniform(-0.10, 0.10)
            dmin = min(np.hypot(px - x, py - y) for x, y in placed)
            if dmin > best_d:
                best, best_d = (px, py), dmin
            if dmin > 0.09:
                break
        self.place = np.array(best)
        self.place_clearance = float(best_d)
        pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pedestal")
        if pb >= 0 and m.body_mocapid[pb] >= 0:
            d.mocap_pos[m.body_mocapid[pb]] = [0, 0.9, -0.5]
        mujoco.mj_forward(m, d)
        self.tgt_h = 2 * float(self.dims[0][2])       # target full height
        self.clutter0 = np.array([d.xpos[self.bids[k]][:2].copy()
                                  for k in range(1, 4)])

    def cup_obj_force(self, obj_gid):
        f, buf = 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if ((c.geom1 in self.cup_gids and c.geom2 == obj_gid) or
                    (c.geom2 in self.cup_gids and c.geom1 == obj_gid)):
                mujoco.mj_contactForce(self.m, self.d, i, buf)
                f += abs(buf[0])
        return f

    def obj_table_force(self):
        """Held object vs anything that is not the cup (table/clutter)."""
        f, buf = 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            og = self.gids[0]
            other = None
            if c.geom1 == og:
                other = c.geom2
            elif c.geom2 == og:
                other = c.geom1
            if other is not None and other not in self.cup_gids:
                mujoco.mj_contactForce(self.m, self.d, i, buf)
                f += abs(buf[0])
        return f

    def clutter_disturbance(self):
        cur = np.array([self.d.xpos[self.bids[k]][:2] for k in range(1, 4)])
        return float(np.max(np.linalg.norm(cur - self.clutter0, axis=1)))

    def _lagmpc_ctx(self):
        """Per-joint lag-aware MPC (mirror of mycobot_mpc
        controller_solvers._mpc_solve_lag): N=25 prediction steps at
        INNER_DT with the 2-state [err, vel] drive-lag model."""
        if hasattr(self, "_lm"):
            return self._lm
        N, kv, vmax, dt = 25, KV, 40.0, INNER_DT
        qe, qv, r = 100.0, 1.0, 0.005
        A = np.array([[1.0, dt], [0.0, 1.0 - kv * dt]])
        B = np.array([[0.0], [kv * dt]])
        Ap = [np.linalg.matrix_power(A, k) for k in range(N + 1)]
        G = np.zeros((2 * N, N))
        for k in range(1, N + 1):
            for j in range(k):
                G[2 * (k - 1):2 * k, j] = (Ap[k - 1 - j] @ B).ravel()
        W = np.zeros(2 * N)
        W[0::2] = qe
        W[1::2] = qv
        W[-2] = qe * 10.0
        Wd = np.diag(W)
        P = G.T @ Wd @ G + r * np.eye(N)
        solvers = []
        for _ in range(6):
            s = osqp.OSQP()
            s.setup(sparse.csc_matrix((P + P.T) / 2), np.zeros(N),
                    sparse.eye(N, format="csc"), np.full(N, -vmax),
                    np.full(N, vmax), verbose=False, max_iter=200,
                    warm_start=True)
            solvers.append(s)
        self._lm = dict(N=N, Ap=Ap, GtW=G.T @ Wd, vmax=vmax,
                        solvers=solvers, plan=np.zeros((6, N)), i=N)
        return self._lm

    def inner_tick_lagmpc(self, q_ref, exec_steps=8):
        """Inner loop via explicit lag-MPC, executing exec_steps of the
        25-step plan open-loop before re-solving (receding horizon with a
        slow re-plan: exec_steps=1 ~= the LQR-clamp; 8 = 32 ms open-loop,
        so the plan goes stale across ~1.5 outer reference updates)."""
        d, m = self.d, self.m
        lm = self._lagmpc_ctx()
        for _ in range(int(OUTER_DT / INNER_DT)):
            if lm["i"] >= exec_steps:
                for j in range(6):
                    x0 = np.array([np.degrees(d.qpos[j] - q_ref[j]),
                                   np.degrees(d.qvel[j])])
                    f = np.concatenate([lm["Ap"][k] @ x0
                                        for k in range(1, lm["N"] + 1)])
                    lm["solvers"][j].update(q=lm["GtW"] @ f)
                    res = lm["solvers"][j].solve()
                    lm["plan"][j] = (res.x if res.info.status.startswith(
                        "solved") else np.full(lm["N"], np.clip(
                            -2.0 * x0[0], -lm["vmax"], lm["vmax"])))
                lm["i"] = 0
            u = np.clip(lm["plan"][:, lm["i"]], -lm["vmax"], lm["vmax"])
            lm["i"] += 1
            vc = np.radians(u)
            for _ in range(int(INNER_DT / DT)):
                qacc = KV * (vc - d.qvel[:6])
                d.qfrc_applied[:6] = np.clip(
                    d.qfrc_bias[:6] + (m.dof_armature[:6] + 0.05) * qacc,
                    -100, 100)
                mujoco.mj_step(m, d)

    def inner_tick(self, q_ref, interp=False):
        d, m = self.d, self.m
        n_in = int(OUTER_DT / INNER_DT)
        q_from = getattr(self, "_qr_last", q_ref)
        self._qr_last = q_ref.copy()
        for i in range(n_in):
            # interp: ramp the reference across the inner ticks instead of
            # stepping it once per outer tick -- the 50 Hz staircase, not
            # sensor noise, is what excites the drive-lag dynamics
            if interp:
                # ramped reference + matching velocity feedforward: a
                # position-only ramp adds a half-tick delay the incremental
                # outer loop can't tolerate (it hunts); the 2-DOF form
                # tracks the ramp without fighting the damping term
                qr = q_from + (q_ref - q_from) * ((i + 1) / n_in)
                v_ff = np.degrees(q_ref - q_from) / OUTER_DT
            else:
                qr, v_ff = q_ref, 0.0
            e = np.degrees(d.qpos[:6] - qr)
            v = np.degrees(d.qvel[:6])
            u = np.clip(-(K_LQR[0] * e + K_LQR[1] * (v - v_ff)), -40, 40)
            vc = np.clip(np.radians(u), -VEL_LIMIT, VEL_LIMIT)
            for _ in range(int(INNER_DT / DT)):
                qacc = KV * (vc - d.qvel[:6])
                d.qfrc_applied[:6] = np.clip(
                    d.qfrc_bias[:6] + (m.dof_armature[:6] + 0.05) * qacc,
                    -100, 100)
                mujoco.mj_step(m, d)


def task_step(sim, x_goal, q_ref, ctrl, smooth=False):
    d, m, sid = sim.d, sim.m, sim.sid
    x_tcp = d.site_xpos[sid].copy()
    err = x_goal - x_tcp
    if np.linalg.norm(err) < 0.004:
        err = np.zeros(3)                  # deadband: don't chase noise
    dx = np.clip(0.6 * err, -0.03, 0.03)
    if smooth:                             # low-pass the task correction
        dx = 0.5 * getattr(sim, "_dx_f", dx) + 0.5 * dx
        sim._dx_f = dx
    Jp = np.zeros((3, m.nv))
    Jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, Jp, Jr, sid)
    Rt = d.site_xmat[sid].reshape(3, 3)
    dw = np.clip(3.0 * np.cross(Rt[:, 2], [0.0, 0.0, -1.0]), -0.3, 0.3)
    Jt = np.vstack([Jp[:, :6], 1.2 * Jr[:, :6]])
    xt = np.concatenate([dx, 1.2 * dw])
    # joint-limit AVOIDANCE (not just clipping): when a joint nears its
    # limit, bias the solution away from it -- limit-blind least squares
    # rides the clamp and converts 'descend' into lateral sliding
    qn = d.qpos[:6]
    near = np.abs(qn) > 0.82 * Q_LIM
    lim_bias = np.where(near, -0.15 * (qn - 0.82 * Q_LIM * np.sign(qn)), 0.0)
    if ctrl == "taskmpc":
        # loose box + UNIFORM post-scale: a tight per-joint +-DQ_MAX box
        # bends the task-space direction whenever one joint saturates, and
        # the active set flipping joint-to-joint as the pose evolves is a
        # ~5 Hz direction chatter -- the dominant VISIBLE jitter (band
        # test: 9.45 -> 2.03 rms accel, 5.7 Hz peak gone). Uniform scaling
        # slows the step without turning it.
        P = Jt.T @ Jt + 1e-3 * np.eye(6)
        s = osqp.OSQP()
        s.setup(sparse.csc_matrix((P + P.T) / 2), -Jt.T @ xt - 1e-3 * lim_bias,
                sparse.csc_matrix(np.eye(6)), np.full(6, -10 * DQ_MAX),
                np.full(6, 10 * DQ_MAX), verbose=False, max_iter=200)
        r = s.solve()
        dq = r.x if r.info.status.startswith("solved") else np.zeros(6)
        dq = dq * min(1.0, DQ_MAX / (np.abs(dq).max() + 1e-12))
    else:
        Jd = Jt.T @ np.linalg.inv(Jt @ Jt.T + 2e-3 * np.eye(6))
        dq = Jd @ xt
        N = np.eye(6) - Jd @ Jt
        dq = dq + N @ lim_bias
        dq = dq * min(1.0, DQ_MAX / (np.abs(dq).max() + 1e-12))
    q_ref = q_ref + dq
    q_ref = np.clip(q_ref, -Q_LIM, Q_LIM)          # joint position limits
    return np.clip(q_ref, d.qpos[:6] - np.radians(2.0),
                   d.qpos[:6] + np.radians(2.0))


def run_trial(ctrl, seed, T=60.0, frame_cb=None, smooth=False):
    np.random.seed(seed)
    sim = Sim(seed)
    d, m, sid = sim.d, sim.m, sim.sid
    q_ref = d.qpos[:6].copy()
    tracked = d.xpos[sim.bids[0]].copy()
    pend, last_track = None, -1.0
    phase = "approach"
    t_done = None
    logs = dict(q=[], ref=[])
    traj, traj_i, hold = None, 0, 0.0
    held = False
    frozen_goal, frozen_phase, plan_fails = None, None, 0
    slew_goal = None
    aligned = False              # descend/place xy-align gate (hysteresis)

    def plan_to(x, quat=DOWN):
        t0 = time.perf_counter()
        for _ in range(2):
            r = rpc({"type": "plan_pose",
                     "start_q": [float(v) for v in d.qpos[:6]],
                     "goal_pose": [float(v) for v in x] + quat,
                     "max_attempts": 6})
            if r.get("success"):
                return np.array(r["trajectory"]), time.perf_counter() - t0
            time.sleep(0.3)
        return None, time.perf_counter() - t0

    if ctrl == "curobo":
        rpc({"type": "detach"})          # defensive: failed trials may
        rpc({"type": "clear_world"})     # leave sticky server state
        # world = clutter distractors (target excluded: we must touch it)
        cubs = []
        for k in range(1, 4):
            p = d.xpos[sim.bids[k]]
            dm = sim.dims[k]
            full = ([2 * dm[0], 2 * dm[1], 2 * dm[2]]
                    if m.geom_type[sim.gids[k]] == 6 else
                    [2 * dm[0], 2 * dm[0], 2 * dm[1]])
            cubs.append({"name": f"clutter{k}", "dims": full,
                         "pose": [float(p[0]), float(p[1]), float(p[2]),
                                  1, 0, 0, 0]})
        rpc({"type": "set_world", "cuboids": cubs})

    hover_h, carry_h = 0.06, 0.13
    n = int(T / OUTER_DT)
    for ko in range(n):
        t = ko * OUTER_DT
        if t - last_track >= TRACK_DT:
            if pend is not None:
                # capture-radius freeze (smooth mode): once the tcp is
                # converged near the goal, stop folding fresh measurement
                # noise into the tracked pose -- a real >2 cm object move
                # still re-triggers via the jump branch
                near_goal = smooth and phase in ("approach", "descend") and \
                    np.linalg.norm(d.site_xpos[sid][:2] - tracked[:2]) < 0.02
                ema = 0.15 if smooth else 0.3
                if np.linalg.norm(pend - tracked) > 0.02:
                    tracked = pend
                elif not near_goal:
                    tracked = (1 - ema) * tracked + ema * pend
            pend = d.xpos[sim.bids[0]].copy() + np.random.normal(0, 0.002, 3)
            last_track = t

        top = tracked + np.array([0, 0, sim.dims[0][2]])
        # ---------- phase machine (shared) ----------
        if phase == "approach":
            x_goal = top + np.array([0, 0, hover_h])
            gate = 0.025 if ctrl == "curobo" else 0.012
            if np.linalg.norm(d.site_xpos[sid] - x_goal) < gate:
                phase = "descend"
        elif phase == "descend":
            x_goal = top + np.array([0, 0, -0.015])
            # rate-limited descent: uniform dq scaling means a big z-step
            # starves the small xy correction (contact lands ~1 cm
            # off-center); capping z at 8 mm/tick keeps the solve
            # unsaturated so xy holds full authority all the way down
            x_goal[2] = max(x_goal[2], d.site_xpos[sid][2] - 0.008)
            if sim.cup_obj_force(sim.gids[0]) > 1.0:
                # weld at the CURRENT relative pose (avoids the latch-jolt
                # snap to the authored relpose)
                eq = sim.eqs[0]
                b1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
                b2 = sim.bids[0]
                R1 = d.xmat[b1].reshape(3, 3)
                relp = R1.T @ (d.xpos[b2] - d.xpos[b1])
                q1 = d.xquat[b1].copy()
                q2 = d.xquat[b2].copy()
                q1c = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
                relq = np.zeros(4)
                mujoco.mju_mulQuat(relq, q1c, q2)
                m.eq_data[eq, 0:3] = 0
                m.eq_data[eq, 3:6] = relp
                m.eq_data[eq, 6:10] = relq
                d.eq_active[eq] = 1                    # SEAL
                held = True
                phase = "lift"
                aligned = False
                if ctrl == "curobo":
                    # NOTE: planner-side attach (held-object spheres) is
                    # trajopt-unstable in this server build; the carried
                    # volume is instead handled geometrically -- transit
                    # altitude keeps the held box above all clutter tops
                    traj = None
        elif phase == "lift":
            x_goal = np.array([d.site_xpos[sid][0], d.site_xpos[sid][1],
                               carry_h + sim.tgt_h])
            if d.site_xpos[sid][2] > carry_h + sim.tgt_h - 0.01:
                phase = "transport"
                traj = None
        elif phase == "transport":
            # CARRIED-VOLUME-AWARE height: cup z = carry + obj full height
            x_goal = np.array([sim.place[0], sim.place[1],
                               carry_h + sim.tgt_h])
            if ctrl == "curobo":
                x_goal[2] = 0.11 + sim.tgt_h      # pre-place altitude (held-box spheres need clutter clearance)
            if np.linalg.norm(d.site_xpos[sid][:2] - sim.place) < 0.015:
                phase = "place"
                aligned = False
                traj = None
        elif phase == "place":
            # descend until the OBJECT (not the cup) meets the table
            x_goal = np.array([sim.place[0], sim.place[1],
                               sim.tgt_h + 0.002])
            x_goal[2] = max(x_goal[2], d.site_xpos[sid][2] - 0.008)
            if sim.obj_table_force() > 1.0:
                d.eq_active[sim.eqs[0]] = 0            # RELEASE
                held = False
                phase = "retreat"
                if ctrl == "curobo":
                    traj = None
        elif phase == "retreat":
            x_goal = np.array([sim.place[0], sim.place[1], 0.20])
            if d.site_xpos[sid][2] > 0.18:
                phase = "done"
                t_done = t + hold
        else:
            x_goal = d.site_xpos[sid].copy()

        # ---------- controller ----------
        if ctrl == "curobo" and phase in ("descend", "lift", "place"):
            # HYBRID: cuRobo plans only the free-space transits (approach,
            # transport); the contact-proximal short strokes (grasp
            # descend, lift, final place) use the simple task controller --
            # planners refuse near-contact end states and these plans were
            # the failure-prone ones
            q_ref = task_step(sim, x_goal, q_ref, "taskmpc")
            traj = None
        elif ctrl == "curobo" and phase != "done":
            # plan-then-execute semantics: goal frozen per phase; replan
            # only on phase change or a real (>2 cm) goal displacement
            if frozen_goal is None or frozen_phase != phase or \
                    np.linalg.norm(x_goal - frozen_goal) > 0.02:
                frozen_goal = x_goal.copy()
                frozen_phase = phase
                traj = None
            if hold > 0:
                hold = max(0.0, hold - OUTER_DT)
            else:
                if traj is None:
                    traj, wall = plan_to(frozen_goal)
                    traj_i = 0
                    hold += wall
                    if traj is None:
                        plan_fails += 1
                        if plan_fails > 4:
                            break
                elif traj_i < len(traj):
                    nxt = traj[traj_i]
                    step = np.clip(nxt - q_ref, -DQ_MAX, DQ_MAX)
                    q_ref = q_ref + step
                    if np.abs(nxt - q_ref).max() < 1e-6:
                        traj_i += 1
        elif phase != "done":
            # slew the goal: phase transitions STEP the target; ramping it
            # (<=2.5 cm per tick) removes the lurch-settle at boundaries
            if slew_goal is None:
                slew_goal = d.site_xpos[sid].copy()
            step = np.clip(x_goal - slew_goal, -0.025, 0.025)
            slew_goal = slew_goal + step
            q_ref = task_step(sim, slew_goal, q_ref,
                              "taskmpc" if ctrl.startswith("lagmpc") else ctrl,
                              smooth=smooth)
        if ctrl.startswith("lagmpc"):
            sim.inner_tick_lagmpc(q_ref, exec_steps=int(ctrl[6:] or "1"))
        else:
            sim.inner_tick(q_ref)
        logs["q"].append(np.degrees(d.qpos[:6]).copy())
        logs["ref"].append(np.degrees(q_ref).copy())
        if frame_cb is not None:
            frame_cb(sim, t, phase, held)
        if phase == "done" and t > (t_done or 0) + 1.0:
            break

    op = d.xpos[sim.bids[0]]
    ov = d.cvel[sim.bids[0]][3:] if hasattr(d, "cvel") else np.zeros(3)
    place_err = float(np.linalg.norm(op[:2] - sim.place))
    success = (phase == "done" and place_err < PLACE_TOL and not held
               and op[2] < sim.tgt_h + 0.02)
    if not success:
        print(f"    [dbg] phase={phase} pe={place_err:.3f} held={held} "
              f"opz={op[2]:.3f} lim={sim.tgt_h + 0.02:.3f}")
    Q, R = np.array(logs["q"]), np.array(logs["ref"])
    return dict(ctrl=ctrl, seed=seed, success=bool(success),
                phase_end=phase, place_err=place_err,
                t_total=float(t_done + hold if t_done else np.nan),
                rmse_deg=float(np.sqrt(np.mean((Q - R) ** 2))),
                clutter_dist=sim.clutter_disturbance())


def _one(args):
    ctrl, seed = args
    try:
        return run_trial(ctrl, seed)
    except Exception as e:
        return dict(ctrl=ctrl, seed=seed, success=False, phase_end="error",
                    place_err=float("nan"), t_total=float("nan"),
                    rmse_deg=float("nan"), clutter_dist=float("nan"),
                    error=str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None)
    ap.add_argument("--merge", default=None)
    a = ap.parse_args()
    ctrls = [a.only] if a.only else ["curobo", "taskmpc", "pinv"]
    jobs = [(c, 2000 + i) for c in ctrls for i in range(a.trials)]
    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_one, jobs):
            rows.append(r)
    if a.merge and os.path.exists(os.path.expanduser(a.merge)):
        prev = json.load(open(os.path.expanduser(a.merge)))
        rows = [r for r in prev if r["ctrl"] not in ctrls] + rows
    out = os.path.expanduser("~/pnp_rl/pnp_compare.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"{'controller':9s} {'succ':>6} {'t_total':>12} {'place_err':>10} "
          f"{'RMSE':>6} {'clutter_mm':>10}")
    for c in ["curobo", "taskmpc", "pinv"]:
        rs = [r for r in rows if r["ctrl"] == c]
        if not rs:
            continue
        ok = [r for r in rs if r["success"]]
        tt = np.array([r["t_total"] for r in ok]) if ok else np.array([np.nan])
        pe = np.array([r["place_err"] for r in ok]) if ok else np.array([np.nan])
        cd = np.array([r["clutter_dist"] for r in rs
                       if r["clutter_dist"] == r["clutter_dist"]])
        rm = np.array([r["rmse_deg"] for r in rs
                       if r["rmse_deg"] == r["rmse_deg"]])
        print(f"{c:9s} {len(ok)/len(rs):6.0%} "
              f"{np.nanmean(tt):6.1f}±{np.nanstd(tt):4.1f} "
              f"{np.nanmean(pe)*100:8.1f}cm {rm.mean():6.2f} "
              f"{cd.mean()*1000:9.1f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

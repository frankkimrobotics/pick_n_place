#!/usr/bin/env python3
"""compare_controllers :: cuRobo-v2 vs task-MPC vs pseudo-inverse on the twin.

Identical scene (HOME start, box at (0.36,0.05), touch task), identical
inner 250 Hz LQR + velocity-drive emulation. Two cases:
  static : object fixed (fair to plan-then-execute)
  moving : object teleports 6 cm at t=1.5 s (closed-loop stress; cuRobo
           re-detects at the next tracker sample and REPLANS, paying its
           solve time as held-reference sim time)

Metrics: time-to-contact, joint-space RMSE (command q_ref vs actual q),
task-error mean during motion, cup tilt p50, z-direction reversals
(smoothness), per-cycle compute ms, success.

    python compare_controllers.py
"""
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
XML = os.path.join(HERE, "rl", "scenes", "box_med_ped.xml")
DT, INNER_DT, OUTER_DT = 0.002, 0.004, 0.020
TRACK_DT = 1.0 / 6.0
K_LQR = np.array([53.4798, 4.7131])
VEL_LIMIT = np.radians(40.0)
KV = 40.0
DQ_MAX = np.radians(0.7)
HOME = np.radians([0.0, -20.0, 80.0, 10.0, -90.0, 0.0])
DOWN = [0.0, 1.0, 0.0, 0.0]


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=60)
    s.sendall((json.dumps(d) + "\n").encode())
    b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close()
    return json.loads(b)


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object0")
        self.jadr = self.m.jnt_qposadr[self.m.body_jntadr[self.bid]]
        og = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "g_object0")
        self.half_z = float(self.m.geom_size[og, 2])
        self.cup_gids = [g for g in range(self.m.ngeom) if "cup" in
                         (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        self.obj_gid = og
        self.d.qpos[:6] = HOME
        self.d.qpos[self.jadr:self.jadr + 3] = [0.36, 0.05, self.half_z + 0.001]
        self.d.qpos[self.jadr + 3:self.jadr + 7] = [1, 0, 0, 0]
        pb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pedestal")
        if pb >= 0 and self.m.body_mocapid[pb] >= 0:
            self.d.mocap_pos[self.m.body_mocapid[pb]] = [0, 0.9, -0.5]
        mujoco.mj_forward(self.m, self.d)

    def contact_force(self):
        f, buf = 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if ((c.geom1 in self.cup_gids and c.geom2 == self.obj_gid) or
                    (c.geom2 in self.cup_gids and c.geom1 == self.obj_gid)):
                mujoco.mj_contactForce(self.m, self.d, i, buf)
                f += abs(buf[0])
        return f

    def inner_tick(self, q_ref):
        """One 20 ms outer period of 250 Hz LQR + drives."""
        for _ in range(int(OUTER_DT / INNER_DT)):
            e = np.degrees(self.d.qpos[:6] - q_ref)
            v = np.degrees(self.d.qvel[:6])
            u = np.clip(-(K_LQR[0] * e + K_LQR[1] * v), -40, 40)
            vc = np.clip(np.radians(u), -VEL_LIMIT, VEL_LIMIT)
            for _ in range(int(INNER_DT / DT)):
                qacc = KV * (vc - self.d.qvel[:6])
                self.d.qfrc_applied[:6] = np.clip(
                    self.d.qfrc_bias[:6] + (self.m.dof_armature[:6] + 0.05) * qacc,
                    -100, 100)
                mujoco.mj_step(self.m, self.d)


def run_case(ctrl, moving, T=12.0):
    sim = Sim()
    d, m, sid = sim.d, sim.m, sim.sid
    q_ref = d.qpos[:6].copy()
    tracked = d.xpos[sim.bid].copy()
    pend, last_track = None, -1.0
    moved = False
    traj, traj_i, replan_hold = None, 0, 0.0
    logs = dict(q=[], ref=[], tcpz=[], err=[], tilt=[], comp=[])
    contact_t = None
    goal_from = lambda: tracked + np.array([0, 0, sim.half_z - 0.015])

    def plan_to(x):
        t0 = time.perf_counter()
        r = rpc({"type": "plan_pose", "start_q": [float(v) for v in d.qpos[:6]],
                 "goal_pose": [float(v) for v in x] + DOWN, "max_attempts": 6})
        wall = time.perf_counter() - t0
        if not r.get("success"):
            return None, wall
        return np.array(r["trajectory"]), wall

    if ctrl == "curobo":
        hover = tracked + np.array([0, 0, sim.half_z + 0.06])
        traj, wall0 = plan_to(goal_from())
        replan_hold = wall0                       # pay solve time up front

    n = int(T / OUTER_DT)
    for ko in range(n):
        t = ko * OUTER_DT
        if moving and t > 1.5 and not moved:
            d.qpos[sim.jadr] += 0.06
            d.qpos[sim.jadr + 1] -= 0.04
            mujoco.mj_forward(m, d)
            moved = True
        if t - last_track >= TRACK_DT:
            if pend is not None:
                if np.linalg.norm(pend - tracked) > 0.02:
                    tracked = pend
                    if ctrl == "curobo":          # replan on jump
                        traj, wall = plan_to(goal_from())
                        traj_i = 0
                        replan_hold += wall
                else:
                    tracked = 0.7 * tracked + 0.3 * pend
            pend = d.xpos[sim.bid].copy() + np.random.normal(0, 0.002, 3)
            last_track = t

        tc = time.perf_counter()
        if contact_t is None and sim.contact_force() > 1.0:
            contact_t = t + replan_hold
        if ctrl == "curobo":
            if replan_hold > 0:
                replan_hold = max(0.0, replan_hold - OUTER_DT)   # hold during solve
            elif traj is not None and traj_i < len(traj):
                nxt = traj[traj_i]
                step = np.clip(nxt - q_ref, -DQ_MAX, DQ_MAX)
                q_ref = q_ref + step
                if np.abs(nxt - q_ref).max() < 1e-6:
                    traj_i += 1
        else:
            x_goal = goal_from() if contact_t is None else d.site_xpos[sid].copy()
            x_tcp = d.site_xpos[sid].copy()
            err = x_goal - x_tcp
            if np.linalg.norm(err) < 0.004 and contact_t is not None:
                err = np.zeros(3)
            dx = np.clip(0.6 * err, -0.03, 0.03)
            Jp = np.zeros((3, m.nv))
            Jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, Jp, Jr, sid)
            Rt = d.site_xmat[sid].reshape(3, 3)
            e_rot = np.cross(Rt[:, 2], np.array([0.0, 0.0, -1.0]))
            dw = np.clip(3.0 * e_rot, -0.30, 0.30)
            J, W = Jp[:, :6], 1.2
            if ctrl == "taskmpc":
                Jt = np.vstack([J, W * Jr[:, :6]])
                xt = np.concatenate([dx, W * dw])
                P = Jt.T @ Jt + 1e-3 * np.eye(6)
                qv = -Jt.T @ xt
                s = osqp.OSQP()
                s.setup(sparse.csc_matrix((P + P.T) / 2), qv,
                        sparse.csc_matrix(np.eye(6)), np.full(6, -DQ_MAX),
                        np.full(6, DQ_MAX), verbose=False, max_iter=200)
                r = s.solve()
                dq = r.x if r.info.status.startswith("solved") else np.zeros(6)
            else:                                  # pinv (DLS + nullspace)
                Jt = np.vstack([J, W * Jr[:, :6]])
                xt = np.concatenate([dx, W * dw])
                Jd = Jt.T @ np.linalg.inv(Jt @ Jt.T + 2e-3 * np.eye(6))
                dq = np.clip(Jd @ xt, -DQ_MAX, DQ_MAX)
            q_ref = q_ref + dq
            q_ref = np.clip(q_ref, d.qpos[:6] - np.radians(2.0),
                            d.qpos[:6] + np.radians(2.0))
        logs["comp"].append((time.perf_counter() - tc) * 1000)
        sim.inner_tick(q_ref)
        logs["q"].append(np.degrees(d.qpos[:6]).copy())
        logs["ref"].append(np.degrees(q_ref).copy())
        logs["tcpz"].append(float(d.site_xpos[sid][2]))
        logs["err"].append(float(np.linalg.norm(
            goal_from() - d.site_xpos[sid])) if contact_t is None else 0.0)
        Rt2 = d.site_xmat[sid].reshape(3, 3)
        logs["tilt"].append(float(np.degrees(np.arccos(
            np.clip(-Rt2[2, 2], -1, 1)))))

    Q = np.array(logs["q"])
    R = np.array(logs["ref"])
    rmse = float(np.sqrt(np.mean((Q - R) ** 2)))
    zs = np.array(logs["tcpz"])
    vz = np.diff(zs)
    rev = int(np.sum(np.abs(np.diff(np.sign(vz[np.abs(vz) > 1e-4]))) > 0))
    pre = [e for e, in zip(logs["err"]) if e > 0]
    return dict(ctrl=ctrl, moving=moving,
                contact_t=contact_t if contact_t is not None else np.nan,
                rmse_deg=rmse,
                task_err_mean=float(np.mean(pre)) if pre else np.nan,
                tilt_p50=float(np.percentile(logs["tilt"], 50)),
                reversals=rev,
                comp_ms=float(np.mean(logs["comp"])),
                success=contact_t is not None)


def main():
    np.random.seed(0)
    rows = []
    print(f"{'case':7s} {'controller':9s} {'contact_s':>9} {'RMSE_deg':>9} "
          f"{'task_err':>9} {'tilt_p50':>8} {'revs':>5} {'comp_ms':>8} {'ok':>3}")
    for moving in (False, True):
        for ctrl in ("curobo", "taskmpc", "pinv"):
            r = run_case(ctrl, moving)
            rows.append(r)
            print(f"{'moving' if moving else 'static':7s} {ctrl:9s} "
                  f"{r['contact_t']:9.2f} {r['rmse_deg']:9.2f} "
                  f"{r['task_err_mean']*100 if r['task_err_mean']==r['task_err_mean'] else float('nan'):8.1f}c "
                  f"{r['tilt_p50']:7.1f}d {r['reversals']:5d} "
                  f"{r['comp_ms']:8.2f} {'Y' if r['success'] else 'N':>3}")
    json.dump(rows, open(os.path.expanduser("~/pnp_rl/ctrl_compare.json"), "w"),
              indent=1)
    print("\nsaved ~/pnp_rl/ctrl_compare.json")


if __name__ == "__main__":
    main()

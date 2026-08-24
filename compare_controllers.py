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


def run_case(ctrl, moving, T=12.0, seed=0):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    sim = Sim()
    # per-trial scene randomization: object start + displacement vector
    ox = rng.uniform(0.30, 0.44)
    oy = rng.uniform(-0.10, 0.10)
    sim.d.qpos[sim.jadr:sim.jadr + 2] = [ox, oy]
    mujoco.mj_forward(sim.m, sim.d)
    dxy = rng.uniform(-0.06, 0.06, 2)
    if np.linalg.norm(dxy) < 0.03:
        dxy = np.array([0.05, -0.03])
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
        for attempt in range(2):                  # retry: server can fail
            r = rpc({"type": "plan_pose",         # transiently under load
                     "start_q": [float(v) for v in d.qpos[:6]],
                     "goal_pose": [float(v) for v in x] + DOWN,
                     "max_attempts": 6})
            if r.get("success"):
                return np.array(r["trajectory"]), time.perf_counter() - t0
            time.sleep(0.5)
        return None, time.perf_counter() - t0

    if ctrl == "curobo":
        hover = tracked + np.array([0, 0, sim.half_z + 0.06])
        traj, wall0 = plan_to(goal_from())
        replan_hold = wall0                       # pay solve time up front

    n = int(T / OUTER_DT)
    for ko in range(n):
        t = ko * OUTER_DT
        if moving and t > 1.5 and not moved:
            d.qpos[sim.jadr] += dxy[0]
            d.qpos[sim.jadr + 1] += dxy[1]
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


def _one(args):
    ctrl, moving, seed = args
    try:
        return run_case(ctrl, moving, seed=seed)
    except Exception as e:
        return dict(ctrl=ctrl, moving=moving, success=False,
                    contact_t=float("nan"), rmse_deg=float("nan"),
                    tilt_p50=float("nan"), reversals=0, comp_ms=0.0,
                    task_err_mean=float("nan"), error=str(e)[:60])


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None)
    ap.add_argument("--merge", default=None)
    a = ap.parse_args()
    ctrls = [a.only] if a.only else ["curobo", "taskmpc", "pinv"]
    jobs = [(ctrl, moving, 1000 + i)
            for moving in (False, True)
            for ctrl in ctrls
            for i in range(a.trials)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs)):
            rows.append(r)
            if (k + 1) % 60 == 0:
                print(f"  {k+1}/{len(jobs)} trials done", flush=True)
    if a.merge:
        prev = json.load(open(os.path.expanduser(a.merge)))
        rows = [r for r in prev if r["ctrl"] not in ctrls] + rows
    json.dump(rows, open(os.path.expanduser("~/pnp_rl/ctrl_compare_100.json"),
                         "w"), indent=1)
    print(f"{'case':7s} {'controller':9s} {'succ':>6} {'contact_s':>12} "
          f"{'RMSE_deg':>12} {'tilt_p50':>9} {'revs':>6}")
    for moving in (False, True):
        for ctrl in ("curobo", "taskmpc", "pinv"):
            rs = [r for r in rows if r["ctrl"] == ctrl and r["moving"] == moving]
            ok = [r for r in rs if r["success"]]
            ct = np.array([r["contact_t"] for r in ok]) if ok else np.array([np.nan])
            rm = np.array([r["rmse_deg"] for r in rs if r["rmse_deg"] == r["rmse_deg"]])
            tl = np.array([r["tilt_p50"] for r in rs if r["tilt_p50"] == r["tilt_p50"]])
            rv = np.array([r["reversals"] for r in rs])
            print(f"{'moving' if moving else 'static':7s} {ctrl:9s} "
                  f"{len(ok)/max(1,len(rs)):6.0%} "
                  f"{np.nanmean(ct):6.2f}±{np.nanstd(ct):4.2f} "
                  f"{rm.mean():6.2f}±{rm.std():4.2f} "
                  f"{tl.mean():8.1f} {rv.mean():6.0f}")
    print("\nsaved ~/pnp_rl/ctrl_compare_100.json")


if __name__ == "__main__":
    main()

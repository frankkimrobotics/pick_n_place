#!/usr/bin/env python3
"""planner_sweep :: minimise the duration of the cuRobo pick-APPROACH trajectory.

What it sweeps
--------------
The motion is the one `rl/planner_touch.py` plans and executes:

    A  approach : START_Q            -> hover  = (cx, cy, top + CUP_R + 0.10)
    B  descend  : hover              -> grasp  = (cx, cy, top + CUP_R)

stitched geometrically and retimed as ONE motion.  Unlike planner_touch this
script does NOT talk to the :9997 server (which exposes almost no knobs): it
builds a cuRobo V2 ``MotionPlanner`` in-process so the trajopt knobs can be
swept.  Axes (V2 names; the v0.7 names in the task map onto them):

    n_knots          <- "trajopt_tsteps"  (B-spline control points of the
                        BSPLINE_3 transition model)
    interpolation_dt <- TrajOptSolverCfg.interpolation_dt (dense output dt)
    num_trajopt_seeds
    scale            <- cspace velocity_scale = acceleration_scale = jerk_scale
                        (v0.7's velocity_scale/acceleration_scale/jerk_scale;
                        V2 has no `time_dilation_factor`)
    finetune         <- trajopt time-optimal finetune passes on/off
                        (MotionPlanner hard-codes attempts=1, dt_scale=0.55;
                        this script calls the trajopt solver itself so the
                        pass can be switched off)

and, on top of every cuRobo path, the TOPP-RA retiming limits
vlim in {25,30,36} deg/s x alim in {300,450,600} deg/s^2.

Limit checking (the final 4 ms trajectory)
------------------------------------------
    position   URDF <limit lower/upper> AND the elbow-safe box
               |joint2| <= 70 deg, |joint3| <= 145 deg  (= the yml
               position_limit_clip [0, 1.919862, 0.087266, 0, 0, 0])
    velocity   <= 36 deg/s   (STM32 following-error ceiling)
    accel      <= 600 deg/s^2 (design margin under the measured 800 cap)
    jerk       <= cspace max_jerk from mycobot_pro_630.yml (2000 rad/s^3);
               the 3000 deg/s^3 design figure is also reported (--jerk_design)
    torque     MuJoCo mj_inverse on rl/scenes/box_med.xml, gravity on, no
               payload; limit = the scene's motor ctrlrange (+-100 Nm).  The
               URDF effort="1000.0" on every joint is a placeholder.

The C2 end blends of curobo_pick_traj OVERSHOOT the velocity limit by ~16 %
because they squeeze the same distance into the same window with v=a=0 forced
at the ends.  Here the blend window is STRETCHED in time until the quintic
itself respects (vlim, alim), and a final uniform time dilation repairs any
residual overshoot, so the limits hold on the trajectory that is shipped.

Environments
------------
  curobo2 : cuRobo + toppra (this file's main entry point)
  mjwarp  : mujoco (re-invoked as `--_mj_stage`, a persistent line server)

Usage
-----
  CUDA_VISIBLE_DEVICES=1 /home/lisc-frank/miniconda3/envs/curobo2/bin/python \
      rl/planner_sweep.py --out ~/pnp_rl/planner_sweep
"""
import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUROBO_DIR = "/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo"
CUROBO_PY = "/home/lisc-frank/miniconda3/envs/curobo2/bin/python"
MJ_PY = "/home/lisc-frank/miniconda3/envs/mjwarp/bin/python"
SCENE = os.path.join(HERE, "scenes", "box_med.xml")

DT = 0.004                      # curobo_pick_traj.DT
CUP_R = 0.008
HOVER = 0.10
OBJ_TOP = 0.045
START_Q = [0.0, -0.349066, 1.396263, 0.174533, -1.570796, 0.0]
R_DOWN_YAW90 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]
DESCEND_TOL = 0.02
DESCEND_STEP = 0.01
BLEND_S = 0.35

CASES = [(0.30, 0.00), (0.42, -0.14), (0.44, 0.10), (0.36, 0.20), (0.50, -0.05)]

# ---- world (planner_touch's, minus the server's own defaults) -------------
GROUND = {"dims": [2.0, 2.0, 0.04], "pose": [0.0, 0.0, -0.12, 1, 0, 0, 0]}   # --ground-z -0.1
KEEPOUT = {"dims": [1.0, 2.4, 2.4], "pose": [-0.8, 0.0, 0.2, 1, 0, 0, 0]}    # wall x < -0.30
TABLE = {"dims": [1.00, 1.60, 0.10], "pose": [0.72, 0.0, -0.05, 1, 0, 0, 0]}  # top z = 0
MOUNT_TIGHT = {"dims": [0.18, 0.18, 1.20], "pose": [0.645, -0.05, 0.45, 1, 0, 0, 0]}
# the D435 head sits at (0.626, -0.046, 0.645) on a column; MOUNT_TIGHT is still
# generous enough that (0.50, -0.05) is IK-infeasible at every yaw.  "slim" is a
# column-only box used for that one case (flagged in the results).
MOUNT_SLIM = {"dims": [0.10, 0.10, 1.20], "pose": [0.676, -0.046, 0.45, 1, 0, 0, 0]}


def world(mount="tight"):
    cub = {"ground": GROUND, "keepout_xneg": KEEPOUT, "table": TABLE}
    cub["camera_mount_d435"] = MOUNT_TIGHT if mount == "tight" else MOUNT_SLIM
    return cub


# ---- limits ---------------------------------------------------------------
URDF_LO = np.array([-3.14159, -3.14159, -2.61, -2.9670, -2.93, -3.03])
URDF_HI = np.array([3.14159, 3.14159, 2.618, 2.9670, 2.9321, 3.0368])
ELBOW = np.radians([180.0, 70.0, 145.0, 180.0, 180.0, 180.0])
Q_LO = np.maximum(URDF_LO, -ELBOW)
Q_HI = np.minimum(URDF_HI, ELBOW)
# Joint-side torque limits.  The Pro 630 URDF has effort="1000.0" on EVERY joint --
# a placeholder.  The repo's own datasheet-style figures are in
# mycobot_mpc/controller_params.yaml: "100:1 harmonic drives, joint-side torque limits
# of +-186 Nm (J0-J2) / +-50 Nm (J3-J5)".  (rl/scenes/box_med.xml's motor ctrlrange is
# +-100 Nm, a sim modelling choice, reported alongside.)
TAU_LIM = np.array([186.0, 186.0, 186.0, 50.0, 50.0, 50.0])
TAU_LIM_MJCF = np.full(6, 100.0)


# ==================================================================== mj stage
def mj_stage():
    """mjwarp side: a line server.  One JSON request per line -> one JSON reply."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(SCENE)
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    dinv = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    sys.stdout.write(json.dumps({"ready": True, "nq": int(m.nq), "nv": int(m.nv)}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req["mode"] == "quit":
            return
        if req["mode"] == "fk":
            d0.qpos[:6] = req["q"]
            mujoco.mj_kinematics(m, d0)
            rep = {"pos": d0.site_xpos[sid].tolist(),
                   "R": d0.site_xmat[sid].reshape(3, 3).tolist()}
            mujoco.mj_forward(m, d0)
        elif req["mode"] == "torque_series":
            a = np.asarray(req["a"], float)          # [N, 3, 6]
            tau = np.zeros((len(a), 6))
            for k in range(len(a)):
                dinv.qpos[:] = d0.qpos
                dinv.qvel[:] = 0
                dinv.qacc[:] = 0
                dinv.qpos[:6] = a[k, 0]
                dinv.qvel[:6] = a[k, 1]
                dinv.qacc[:6] = a[k, 2]
                mujoco.mj_inverse(m, dinv)
                tau[k] = dinv.qfrc_inverse[:6]
            rep = {"tau": np.round(tau, 4).tolist()}
        elif req["mode"] == "torque":
            z = np.load(req["npz"])
            out = {}
            for key in z.files:
                a = z[key]                      # [N, 3, 6] = q, qd, qdd (decimated)
                tau = np.zeros((len(a), 6))
                for k in range(len(a)):
                    dinv.qpos[:] = d0.qpos
                    dinv.qvel[:] = 0
                    dinv.qacc[:] = 0
                    dinv.qpos[:6] = a[k, 0]
                    dinv.qvel[:6] = a[k, 1]
                    dinv.qacc[:6] = a[k, 2]
                    mujoco.mj_inverse(m, dinv)
                    tau[k] = dinv.qfrc_inverse[:6]
                out[key] = np.abs(tau).max(axis=0).round(4).tolist()
            rep = {"peak_tau": out}
        else:
            rep = {"error": "unknown mode"}
        sys.stdout.write(json.dumps(rep) + "\n")
        sys.stdout.flush()


class MjLink:
    def __init__(self):
        self.p = subprocess.Popen([MJ_PY, os.path.abspath(__file__), "--_mj_stage"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        hello = json.loads(self.p.stdout.readline())
        assert hello.get("ready"), hello
        self.nq = hello["nq"]

    def rpc(self, req):
        self.p.stdin.write((json.dumps(req) + "\n").encode())
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        try:
            self.rpc({"mode": "quit"})
        except Exception:
            pass
        self.p.terminate()


# ==================================================================== cuRobo
def _yawq(deg):
    """R_DOWN (wxyz [0,1,0,0]) rotated about world z by `deg`."""
    h = np.radians(deg) / 2.0
    qy = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
    qd = np.array([0.0, 1.0, 0.0, 0.0])
    return [qy[0] * qd[0] - qy[1] * qd[1] - qy[2] * qd[2] - qy[3] * qd[3],
            qy[0] * qd[1] + qy[1] * qd[0] + qy[2] * qd[3] - qy[3] * qd[2],
            qy[0] * qd[2] - qy[1] * qd[3] + qy[2] * qd[0] + qy[3] * qd[1],
            qy[0] * qd[3] + qy[1] * qd[2] - qy[2] * qd[1] + qy[3] * qd[0]]


def quat_to_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Curobo:
    """In-process cuRobo V2 planner with the sweep knobs exposed."""

    def __init__(self, n_knots=16, interp_dt=0.025, seeds=4, scale=1.0,
                 ik_seeds=32, base_dt=0.15):
        import yaml
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        self.torch = torch
        v1 = yaml.safe_load(open(os.path.join(CUROBO_DIR, "mycobot_pro_630.yml")))
        kin = v1["robot_cfg"]["kinematics"]
        kin["format_version"] = 2.0
        kin["tool_frames"] = [kin.pop("ee_link")]
        kin.pop("link_names", None)
        cs = kin["cspace"]
        cs["default_joint_position"] = cs.pop("retract_config")
        cs["velocity_scale"] = float(scale)
        cs["acceleration_scale"] = float(scale)
        cs["jerk_scale"] = float(scale)
        self.max_jerk = float(cs["max_jerk"])          # rad/s^3, from the yml
        self.max_accel = float(cs["max_acceleration"])
        tm = {"transition_model_cfg": {
            "control_space": "BSPLINE_3",
            "dt_traj_params": {"base_dt": base_dt, "base_ratio": 1.0, "max_dt": base_dt},
            "horizon": 1, "interpolation_steps": 4, "n_knots": int(n_knots),
            "state_filter_cfg": {"enable": False,
                                 "filter_coeff": {"acceleration": 0.0, "position": 1.0,
                                                  "velocity": 1.0}},
            "teleport_mode": False, "vel_scale": 1.0, "return_full_act_buffer": True}}
        cfg = MotionPlannerCfg.create(
            robot=copy.deepcopy(v1), scene_model=None, use_cuda_graph=False,
            num_trajopt_seeds=int(seeds), num_ik_seeds=int(ik_seeds),
            trajopt_transition_model=tm, collision_cache={"primitive": 16})
        self.mp = MotionPlanner(cfg)
        self.mp.trajopt_solver.config.interpolation_dt = float(interp_dt)
        self.jn = list(self.mp.default_joint_state.joint_names)
        self.dev = self.mp.default_joint_state.position.device

    def set_world(self, cub):
        from curobo._src.geom.types import SceneCfg
        self.mp.update_world(SceneCfg.create({"cuboid": cub}))

    def _state(self, q):
        from curobo._src.state.state_joint import JointState
        t = self.torch.tensor([[float(x) for x in q]], dtype=self.torch.float32,
                              device=self.dev)
        return JointState.from_position(t, joint_names=self.jn)

    def _goal(self, p, quat):
        from curobo._src.types.pose import Pose
        from curobo._src.types.tool_pose import GoalToolPose
        T = self.torch.tensor
        return GoalToolPose.from_poses({"tcp": Pose(
            position=T([[float(v) for v in p]], dtype=self.torch.float32, device=self.dev),
            quaternion=T([[float(v) for v in quat]], dtype=self.torch.float32,
                         device=self.dev))})

    def plan_pose(self, q, p, quat, max_attempts=10, finetune=True):
        """_plan_pose_single with the finetune pass exposed (None on failure)."""
        torch = self.torch
        mp = self.mp
        fa, fs = (1, 0.55) if finetune else (0, 0.55)
        state = self._state(q)
        goal = self._goal(p, quat)
        num_seeds = mp.trajopt_solver.config.num_seeds
        res = None
        for _ in range(max_attempts):
            ik = mp.ik_solver.solve_pose(goal, return_seeds=num_seeds,
                                         current_state=state.clone())
            n_ok = torch.count_nonzero(ik.success)
            if n_ok == 0:
                continue
            seed = ik.solution
            if n_ok < num_seeds:
                good = seed[ik.success][0:1, :].clone()
                seed[~ik.success][:, :] = good
            res = mp.trajopt_solver.solve_pose(goal, state.clone(), seed_config=seed,
                                               use_implicit_goal=True,
                                               finetune_attempts=fa, finetune_dt_scale=fs)
            if torch.count_nonzero(res.success) > 0:
                break
        if res is None or not bool(res.success.any().item()):
            return None
        plan = res.get_interpolated_plan()
        pos = plan.position.squeeze(0).squeeze(0).detach().cpu().numpy().astype(float)
        return {"q": pos, "dt": float(plan.dt),
                "motion_time": float(res.motion_time()),
                "solve_time": float(getattr(res, "solve_time", 0.0) or 0.0)}

    def fk(self, q):
        from curobo._src.state.state_joint import JointState
        a = self.torch.tensor(np.atleast_2d(np.asarray(q, float)),
                              dtype=self.torch.float32, device=self.dev)
        st = self.mp.compute_kinematics(JointState.from_position(a, joint_names=self.jn))
        return st.tool_poses.get_link_pose("tcp").position.detach().cpu().numpy()


# ==================================================================== retiming
def _quintic(q0, v0, a0, q1, v1, a1, T, ts):
    A = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 2, 0, 0],
                  [T**5, T**4, T**3, T**2, T, 1], [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
                  [20*T**3, 12*T**2, 6*T, 2, 0, 0]])
    out = np.zeros((len(ts), len(q0)))
    for j in range(len(q0)):
        c = np.linalg.solve(A, [q0[j], v0[j], a0[j], q1[j], v1[j], a1[j]])
        out[:, j] = np.polyval(c, ts)
    return out


def _stretch_blend(q0, v0, a0, q1, v1, a1, T0, vlim, alim, kmax=4.0):
    """Quintic q0->q1 over the SHORTEST T >= T0 whose own peaks respect the limits.

    curobo_pick_traj._c2_end_blends keeps T = T0, which is exactly why its
    post-blend velocity overshoots the TOPP-RA limit by ~16 %.
    """
    k = 1.0
    while k <= kmax:
        T = k * T0
        n = max(4, int(round(T / DT)) + 1)
        ts = np.linspace(0.0, T, n)
        seg = _quintic(q0, v0, a0, q1, v1, a1, T, ts)
        v = np.gradient(seg, ts[1] - ts[0], axis=0)
        a = np.gradient(v, ts[1] - ts[0], axis=0)
        if np.abs(v).max() <= vlim.max() * 1.001 and np.abs(a).max() <= alim.max() * 1.001:
            return seg, T
        k += 0.05
    return seg, T


def _peaks(q, jerk_smooth=9):
    qd = np.gradient(q, DT, axis=0)
    qdd = np.gradient(qd, DT, axis=0)
    kern = np.ones(jerk_smooth) / jerk_smooth
    qdd_s = np.stack([np.convolve(qdd[:, j], kern, mode="same") for j in range(6)], axis=1)
    jrk = np.gradient(qdd_s, DT, axis=0)
    n = max(3, jerk_smooth)               # drop the convolution edge transient
    return qd, qdd, jrk, (np.abs(qd).max(axis=0), np.abs(qdd).max(axis=0),
                          np.abs(jrk[n:-n]).max(axis=0) if len(jrk) > 2 * n
                          else np.abs(jrk).max(axis=0))


def _resample(q, gamma):
    """Uniform time dilation by gamma >= 1 (slower), resampled back onto DT."""
    from scipy.interpolate import CubicSpline
    T = (len(q) - 1) * DT
    cs = CubicSpline(np.arange(len(q)) * DT, q, axis=0)
    Tn = T * gamma
    tn = np.arange(0.0, Tn + 1e-9, DT)
    tn[-1] = min(tn[-1], Tn)
    return cs(np.clip(tn / gamma, 0.0, T))


def retime(way, vlim_deg, alim_deg, jlim_rad, blend_s=BLEND_S, n_repair=3):
    """TOPP-RA -> stretched C2 end blends -> uniform dilation until limits hold."""
    sys.path.insert(0, CUROBO_DIR)
    import curobo_pick_traj as cpt
    vlim = np.full(6, np.radians(vlim_deg))
    alim = np.full(6, np.radians(alim_deg))
    t, q, qd, qdd = cpt._toppra_retime(way, vlim, alim)
    q = np.asarray(q, float)
    pre = np.degrees(np.abs(np.gradient(q, DT, axis=0))).max()

    nb = min(max(4, int(blend_s / DT)), len(q) // 2 - 1)
    if nb >= 4:
        qd_fd = np.gradient(q, DT, axis=0)
        qdd_fd = np.gradient(qd_fd, DT, axis=0)
        T0 = (nb - 1) * DT
        head, _ = _stretch_blend(q[0], np.zeros(6), np.zeros(6),
                                 q[nb - 1], qd_fd[nb - 1], qdd_fd[nb - 1], T0, vlim, alim)
        tail, _ = _stretch_blend(q[-nb], qd_fd[-nb], qdd_fd[-nb],
                                 q[-1], np.zeros(6), np.zeros(6), T0, vlim, alim)
        q = np.vstack([head[:-1], q[nb - 1:-nb + 1], tail[1:]])

    gamma_tot = 1.0
    for _ in range(n_repair):
        _, _, _, (vpk, apk, jpk) = _peaks(q)
        g = max(vpk.max() / vlim[0], np.sqrt(max(apk.max() / alim[0], 1e-9)),
                (max(jpk.max() / jlim_rad, 1e-9)) ** (1.0 / 3.0), 1.0)
        if g <= 1.0005:
            break
        g = min(g * 1.002, 2.0)
        q = _resample(q, g)
        gamma_tot *= g
    return q, pre, gamma_tot


# ==================================================================== checking
def check(q, jlim_rad, vlim_deg=36.0, alim_deg=600.0):
    qd, qdd, jrk, (vpk, apk, jpk) = _peaks(q)
    out = {"dur": (len(q) - 1) * DT,
           "vpk": np.degrees(vpk), "apk": np.degrees(apk), "jpk": np.degrees(jpk),
           "qmin": q.min(axis=0), "qmax": q.max(axis=0)}
    bad = []
    if (q.min(axis=0) < Q_LO - 1e-6).any() or (q.max(axis=0) > Q_HI + 1e-6).any():
        j = int(np.argmax(np.maximum(Q_LO - q.min(axis=0), q.max(axis=0) - Q_HI)))
        bad.append(f"position(j{j + 1})")
    if np.degrees(vpk).max() > vlim_deg * 1.005:
        bad.append("velocity")
    if np.degrees(apk).max() > alim_deg * 1.005:
        bad.append("acceleration")
    if jpk.max() > jlim_rad * 1.005:
        bad.append("jerk")
    out["viol"] = bad
    return out, qd, qdd


def decimate(q, qd, qdd, every=5):
    kern = np.ones(9) / 9.0
    qdd_s = np.stack([np.convolve(qdd[:, j], kern, mode="same") for j in range(6)], axis=1)
    idx = np.arange(0, len(q), every)
    return np.stack([q[idx], qd[idx], qdd_s[idx]], axis=1).astype(np.float32)


# ==================================================================== pipeline
def plan_case(C, cx, cy, corr, finetune, mount, top=OBJ_TOP):
    """Approach + descend for one object; returns the stitched path + timings."""
    sys.path.insert(0, CUROBO_DIR)
    import curobo_pick_traj as cpt
    C.set_world(world(mount))
    p_hover = np.array([cx, cy, top + CUP_R + HOVER]) - corr
    p_grasp = np.array([cx, cy, top + CUP_R]) - corr
    quat = _yawq(90.0)
    t0 = time.time()
    ra = C.plan_pose(START_Q, p_hover, quat, max_attempts=10, finetune=finetune)
    if ra is None:
        return None, {"fail": "approach", "t_plan": time.time() - t0}
    q_hover = ra["q"][-1]
    rb = C.plan_pose(q_hover, p_grasp, quat, max_attempts=10, finetune=finetune)
    mode = "plan_pose"
    way_b, dt_b = None, None
    if rb is not None:
        pts = C.fk(rb["q"][::max(1, len(rb["q"]) // 12)])
        bow = float(np.abs(pts[:, :2] - p_grasp[:2]).max())
        if bow <= DESCEND_TOL:
            way_b, dt_b = rb["q"], rb["dt"]
        else:
            mode = f"hops(bow {100 * bow:.1f} cm)"
    else:
        mode = "hops(plan failed)"
    if way_b is None:                                    # 1 cm straight-line hops
        n = max(1, int(round(np.linalg.norm(p_grasp - p_hover) / DESCEND_STEP)))
        way, q = [np.asarray(q_hover, float)], list(q_hover)
        for i in range(1, n + 1):
            p = p_hover + (p_grasp - p_hover) * (i / n)
            r = C.plan_pose(q, p, quat, max_attempts=6, finetune=finetune)
            if r is None:
                return None, {"fail": "descend", "t_plan": time.time() - t0}
            q = list(r["q"][-1])
            way.append(np.asarray(q, float))
        way_b = np.asarray(way)
    t_plan = time.time() - t0
    stitched = cpt._stitch([ra["q"], way_b])
    info = {"t_plan": t_plan, "descend_mode": mode, "n_a": len(ra["q"]),
            "n_b": len(way_b), "n_way": len(stitched), "dt_a": ra["dt"], "dt_b": dt_b,
            "curobo_time": (ra["motion_time"] + rb["motion_time"]) if dt_b else None,
            "q_a": ra["q"], "q_b": way_b}
    return stitched, info


def curobo_only(ra_q, dt_a, way_b, dt_b):
    """cuRobo's OWN timing (no TOPP-RA), resampled onto DT."""
    if dt_b is None:
        return None
    from scipy.interpolate import CubicSpline
    t_a = np.arange(len(ra_q)) * dt_a
    t_b = t_a[-1] + dt_b + np.arange(len(way_b)) * dt_b
    t = np.concatenate([t_a, t_b])
    q = np.vstack([ra_q, way_b])
    ok = np.concatenate([[True], np.diff(t) > 1e-9])
    cs = CubicSpline(t[ok], q[ok], axis=0)
    tn = np.arange(0.0, t[-1] + 1e-9, DT)
    return cs(tn)


# ==================================================================== sweep
def run(a):
    os.makedirs(a.out, exist_ok=True)
    mj = MjLink()
    tmp = os.path.join(a.out, "_tau.npz")

    grid = list(itertools.product(a.n_knots, a.interp_dt, a.seeds, a.scale, a.finetune))
    print(f"[sweep] {len(grid)} cuRobo configs x {len(CASES)} cases x "
          f"{len(a.vlim) * len(a.alim)} retimes", flush=True)

    # tcp correction (cuRobo tool frame -> MuJoCo cup tip), measured once
    C = Curobo(n_knots=a.n_knots[0], interp_dt=a.interp_dt[0], seeds=a.seeds[0],
               scale=a.scale[0])
    jlim = C.max_jerk
    p_c = C.fk(START_Q)[0]
    mjfk = mj.rpc({"mode": "fk", "q": list(START_Q)})
    p_m, R_m = np.asarray(mjfk["pos"]), np.asarray(mjfk["R"])
    off_tool = R_m.T @ (p_m - p_c)
    corr = quat_to_R(R_DOWN_YAW90) @ off_tool
    print(f"[sweep] cuRobo tcp -> MuJoCo cup tip {1000 * np.linalg.norm(off_tool):.1f} mm, "
          f"goal correction {np.round(corr, 4).tolist()}", flush=True)
    print(f"[sweep] jerk limit {jlim:.0f} rad/s^3 = {np.degrees(jlim):.0f} deg/s^3 "
          f"(mycobot_pro_630.yml cspace max_jerk); design check "
          f"{a.jerk_design:.0f} deg/s^3", flush=True)
    del C

    rows, t_sweep = [], time.time()
    for ci, (nk, idt, sd, sc, ft) in enumerate(grid):
        t0 = time.time()
        cfg_id = f"nk{nk}_idt{idt}_sd{sd}_sc{sc}_ft{int(ft)}"
        try:
            C = Curobo(n_knots=nk, interp_dt=idt, seeds=sd, scale=sc)
        except Exception as e:                                   # noqa: BLE001
            print(f"[sweep] {cfg_id}: planner build failed ({type(e).__name__}: {e})",
                  flush=True)
            rows.append(dict(cfg=cfg_id, n_knots=nk, interp_dt=idt, seeds=sd, scale=sc,
                             finetune=ft, feasible=False, viol=f"build:{type(e).__name__}"))
            continue
        t_build = time.time() - t0
        batch, meta = {}, {}
        for pi, (cx, cy) in enumerate(CASES):
            mount = "tight" if (cx, cy) != (0.50, -0.05) else a.case5_mount
            try:
                way, info = plan_case(C, cx, cy, corr, ft, mount)
            except Exception as e:                               # noqa: BLE001
                print(f"[sweep] {cfg_id} case {pi}: {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
                rows.append(dict(cfg=cfg_id, n_knots=nk, interp_dt=idt, seeds=sd,
                                 scale=sc, finetune=ft, case=pi, cx=cx, cy=cy,
                                 mount=mount, feasible=False,
                                 viol=f"plan:{type(e).__name__}"))
                way = None
                break
            if way is None:
                rows.append(dict(cfg=cfg_id, n_knots=nk, interp_dt=idt, seeds=sd,
                                 scale=sc, finetune=ft, case=pi, cx=cx, cy=cy,
                                 mount=mount, feasible=False, viol="plan:" + info["fail"],
                                 t_plan=round(info["t_plan"], 3)))
                continue
            for vl, al in itertools.product(a.vlim, a.alim):
                t1 = time.time()
                try:
                    q, pre, gam = retime(way, vl, al, jlim)
                except Exception as e:                       # noqa: BLE001
                    rows.append(dict(cfg=cfg_id, case=pi, feasible=False,
                                     viol=f"retime:{type(e).__name__}"))
                    continue
                t_ret = time.time() - t1
                chk, qd, qdd = check(q, jlim, a.vlim_hard, a.alim_hard)
                key = f"{pi}|{vl}|{al}"
                batch[key] = decimate(q, qd, qdd)
                meta[key] = dict(cfg=cfg_id, n_knots=nk, interp_dt=idt, seeds=sd,
                                 scale=sc, finetune=ft, case=pi, cx=cx, cy=cy,
                                 mount=mount, vlim=vl, alim=al,
                                 t_plan=round(info["t_plan"], 3),
                                 t_retime=round(t_ret, 3),
                                 descend_mode=info["descend_mode"],
                                 n_way=info["n_way"], dilation=round(gam, 4),
                                 vpk_toppra=round(pre, 2),
                                 dur=round(chk["dur"], 4),
                                 vpk=chk["vpk"].round(2).tolist(),
                                 apk=chk["apk"].round(1).tolist(),
                                 jpk=chk["jpk"].round(0).tolist(),
                                 viol=list(chk["viol"]))
            # ---- cuRobo-only reference: cuRobo's OWN timing, no TOPP-RA ----
            qc = curobo_only(info["q_a"], info["dt_a"], info["q_b"], info["dt_b"])
            if qc is not None and len(qc) > 20:
                chk, qd, qdd = check(qc, jlim, a.vlim_hard, a.alim_hard)
                key = f"{pi}|curobo|curobo"
                batch[key] = decimate(qc, qd, qdd)
                meta[key] = dict(cfg=cfg_id, n_knots=nk, interp_dt=idt, seeds=sd,
                                 scale=sc, finetune=ft, case=pi, cx=cx, cy=cy,
                                 mount=mount, vlim="curobo", alim="curobo",
                                 t_plan=round(info["t_plan"], 3), t_retime=0.0,
                                 descend_mode=info["descend_mode"],
                                 n_way=info["n_way"], dilation=1.0, vpk_toppra=None,
                                 dur=round(chk["dur"], 4),
                                 vpk=chk["vpk"].round(2).tolist(),
                                 apk=chk["apk"].round(1).tolist(),
                                 jpk=chk["jpk"].round(0).tolist(),
                                 viol=list(chk["viol"]))
        if batch:
            np.savez(tmp, **batch)
            rep = mj.rpc({"mode": "torque", "npz": tmp})["peak_tau"]
        else:
            rep = {}
        for key, m in meta.items():
            tau = np.asarray(rep.get(key, [np.nan] * 6))
            m["tau"] = np.round(tau, 2).tolist()
            v = list(m["viol"])
            if np.isfinite(tau).all() and (tau > TAU_LIM * 1.005).any():
                v.append("torque")
            jd = max(m["jpk"])
            m["jerk_design_ok"] = bool(jd <= a.jerk_design)
            m["viol"] = ",".join(v)
            m["feasible"] = len(v) == 0
            rows.append(m)
        del C
        done = (ci + 1) / len(grid)
        el = time.time() - t_sweep
        print(f"[sweep] {ci + 1}/{len(grid)} {cfg_id} build {t_build:.1f}s "
              f"({len(meta)} retimes)  elapsed {el / 60:.1f} min  "
              f"eta {el / done * (1 - done) / 60:.1f} min", flush=True)
        json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), default=float)

    mj.close()
    write_outputs(a, rows)
    a.jerk_rad = jlim
    a.corr = corr.tolist()
    analyze(a, rows)
    return rows


def write_outputs(a, rows):
    import csv
    keys = ["cfg", "n_knots", "interp_dt", "seeds", "scale", "finetune", "case", "cx", "cy",
            "mount", "vlim", "alim", "feasible", "viol", "dur", "t_plan", "t_retime",
            "dilation", "vpk_toppra", "descend_mode", "n_way", "jerk_design_ok"]
    perj = ["vpk", "apk", "jpk", "tau"]
    path = os.path.join(a.out, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + [f"{p}{j + 1}" for p in perj for j in range(6)])
        for r in rows:
            base = [r.get(k, "") for k in keys]
            ext = []
            for p in perj:
                v = r.get(p, [""] * 6)
                ext += list(v) if isinstance(v, list) else [""] * 6
            w.writerow(base + ext)
    json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), default=float)
    print(f"[sweep] {len(rows)} rows -> {path}")


# ================================================================ sim validation
def sim_stage():
    """mjwarp side: a PaperPickEnv line server (measured drive, 1 world).

    Reuses planner_touch's own env builder and rollout so `--validate` is the
    `planner_touch.py --sim` rollout with the object parked on a sweep case.
    """
    import torch
    import warp as wp
    wp.init()
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    import env_warp as E
    import planner_touch as PT

    state = {}
    sys.stdout.write("@@SIMREADY@@\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req["mode"] == "quit":
            return
        if req["mode"] == "place":
            env, _ = PT._make_sim_env(SCENE, int(req["steps"]), float(req["dq_max"]))
            qa = env.jadr_obj
            T = lambda v: torch.tensor(v, dtype=torch.float32, device=env.device)  # noqa: E731
            env.qpos[0, qa:qa + 2] = T(list(req["xy"]))
            env.qpos[0, qa + 2] = float(env.half[2]) + 0.001
            env.qpos[0, qa + 3] = 1.0
            env.qpos[0, qa + 4:qa + 7] = 0.0
            env.qpos[0, :6] = T(START_Q)
            env.qvel[0, :] = 0.0
            E.mjw.forward(env.m, env.d)
            env.q_hist[0] = T(START_Q)
            for nm in ("q_target", "q_target_prev", "q_drive", "q_meas_lag"):
                getattr(env, nm)[0] = T(START_Q)
            for nm in ("v_drive", "v_buf", "qd_meas_lag"):
                getattr(env, nm)[0] = 0.0
            obj = env._obj_pos()[0].cpu().numpy()
            grasp = env._grasp_point()[0].cpu().numpy()
            state["env"], state["torch"] = env, torch
            rep = {"obj": obj.tolist(), "top": float(obj[2] + env.half[2]),
                   "grasp": grasp.tolist(),
                   "q0": env.qpos[0, :6].cpu().numpy().tolist(),
                   "tcp": env._tcp()[0][0].cpu().numpy().tolist(),
                   "dq_max_deg": float(np.degrees(env.dq_max))}
        elif req["mode"] == "roll":
            env, torch_ = state["env"], state["torch"]
            knots = np.asarray(req["knots"], float)
            rows, m = PT.sim_rollout_planner(env, torch_, knots, int(req["steps"]),
                                             int(req["hold"]), verbose=False)
            rep = {"rows": rows, "metrics": m}
        else:
            rep = {"error": "unknown mode"}
        sys.stdout.write("@@SIM@@" + json.dumps(rep) + "\n")
        sys.stdout.flush()


class SimLink:
    def __init__(self):
        self.p = subprocess.Popen([MJ_PY, os.path.abspath(__file__), "--_sim_stage"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        while True:
            line = self.p.stdout.readline().decode()
            if not line:
                raise RuntimeError("sim stage died before becoming ready")
            if line.startswith("@@SIMREADY@@"):
                break
            sys.stdout.write("[sim] " + line)

    def rpc(self, req):
        self.p.stdin.write((json.dumps(req) + "\n").encode())
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline().decode()
            if not line:
                raise RuntimeError("sim stage died")
            if line.startswith("@@SIM@@"):
                return json.loads(line[len("@@SIM@@"):])
            sys.stdout.write("[sim] " + line)

    def close(self):
        try:
            self.rpc({"mode": "quit"})
        except Exception:
            pass
        self.p.terminate()


def to_knots(t, q, knot_dt=0.10):
    step = int(round(knot_dt / DT))
    idx = list(range(0, len(q), step))
    if idx[-1] != len(q) - 1:
        idx.append(len(q) - 1)
    return np.asarray(t)[idx], np.asarray(q)[idx]


def validate(a, best, cases=(0, 1)):
    """Roll the best config's plan in the measured-drive twin on `cases`."""
    jlim = a.jerk_rad
    sim = SimLink()
    C = Curobo(n_knots=best["n_knots"], interp_dt=best["interp_dt"],
               seeds=best["seeds"], scale=best["scale"])
    out = []
    for ci in cases:
        cx, cy = CASES[ci]
        mount = "tight" if (cx, cy) != (0.50, -0.05) else a.case5_mount
        pl = sim.rpc({"mode": "place", "xy": [cx, cy], "steps": a.sim_steps,
                      "dq_max": a.sim_dq_max})
        top = float(pl["top"])
        corr = np.asarray(a.corr, float)
        way, info = plan_case(C, cx, cy, corr, best["finetune"], mount, top=top)
        if way is None:
            out.append({"case": ci, "error": "plan failed"})
            continue
        q, _, gam = retime(way, best["vlim"], best["alim"], jlim)
        t = np.arange(len(q)) * DT
        t_k, q_k = to_knots(t, q)
        r = sim.rpc({"mode": "roll", "knots": np.round(q_k, 6).tolist(),
                     "steps": a.sim_steps, "hold": a.sim_hold})
        rows, mtr = r["rows"], r["metrics"]
        # tracking error: measured q at each 10 Hz decision vs the planned knot
        qs = np.asarray([x["q"] for x in rows])
        n = min(len(qs), len(q_k) - 1)
        err = np.degrees(np.abs(qs[:n] - q_k[1:n + 1]))
        grasp = np.asarray(pl["grasp"])
        out.append({"case": ci, "cx": cx, "cy": cy, "obj_top": top,
                    "grasp": grasp.round(4).tolist(),
                    "plan_dur": float(t_k[-1]), "n_knots": int(len(q_k)),
                    "dilation": gam, "t_plan": info["t_plan"],
                    "t_1cm": mtr["t_1cm"], "d_min_cm": 100 * mtr["d_min"],
                    "d_final_cm": 100 * mtr["d_final"], "n_clip": mtr["n_clip"],
                    "dq_max_deg": pl["dq_max_deg"],
                    "track_max_deg": float(err.max()),
                    "track_rms_deg": float(np.sqrt((err ** 2).mean())),
                    "track_final_deg": float(err[-1].max()),
                    "track_per_joint_max": err.max(axis=0).round(2).tolist()})
        print("[validate] " + json.dumps(out[-1]), flush=True)
    sim.close()
    del C
    json.dump(out, open(os.path.join(a.out, "validate.json"), "w"), indent=2, default=float)
    return out


# ==================================================================== analysis
CFG_KEYS = ("n_knots", "interp_dt", "seeds", "scale", "finetune")


def retag_torque(rows):
    """Recompute the torque flag from the stored peaks against the current TAU_LIM."""
    for r in rows:
        tau = r.get("tau")
        if tau is None or r.get("viol") is None:
            continue
        v = [x for x in (r["viol"].split(",") if r["viol"] else []) if x != "torque"]
        t = np.asarray(tau, float)
        if np.isfinite(t).all() and (t > TAU_LIM * 1.005).any():
            v.append("torque")
        r["viol"] = ",".join(v)
        r["feasible"] = len(v) == 0
    return rows


def rank(rows, vlim_hard=36.0):
    """Group rows by (cuRobo config, vlim, alim); keep groups feasible on EVERY case."""
    n_case = len(CASES)
    groups = {}
    for r in rows:
        if r.get("dur") is None or "case" not in r:
            continue
        k = tuple(r.get(x) for x in CFG_KEYS) + (r.get("vlim"), r.get("alim"))
        groups.setdefault(k, {})[r["case"]] = r
    out = []
    for k, per in groups.items():
        if len(per) < n_case:
            continue
        durs = [per[c]["dur"] for c in sorted(per)]
        feas = all(per[c]["feasible"] for c in per)
        out.append(dict(key=k, n_knots=k[0], interp_dt=k[1], seeds=k[2], scale=k[3],
                        finetune=k[4], vlim=k[5], alim=k[6], feasible=feas,
                        mean_dur=float(np.mean(durs)), max_dur=float(max(durs)),
                        durs=durs,
                        viol=sorted({v for c in per for v in
                                     (per[c]["viol"].split(",") if per[c]["viol"] else [])}),
                        t_plan=float(np.mean([per[c]["t_plan"] for c in per])),
                        t_retime=float(np.mean([per[c]["t_retime"] for c in per])),
                        rows=[per[c] for c in sorted(per)]))
    out.sort(key=lambda g: (not g["feasible"], g["mean_dur"]))
    return out


def summary_png(a, rows, best, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g_all = rank(rows)
    g_topp = [g for g in g_all if g["vlim"] != "curobo"]
    g_cur = [g for g in g_all if g["vlim"] == "curobo"]
    fig, ax = plt.subplots(2, 4, figsize=(21, 9))

    # A: every (config, retime) combination, sorted by mean duration
    xs = np.arange(len(g_topp))
    d = [g["mean_dur"] for g in g_topp]
    ok = np.array([g["feasible"] for g in g_topp])
    ax[0, 0].scatter(xs[~ok], np.asarray(d)[~ok], s=12, c="#c8ccd4", label="limit violated")
    ax[0, 0].scatter(xs[ok], np.asarray(d)[ok], s=16, c="#2f6fdb", label="feasible")
    if g_cur:
        b = min(g["mean_dur"] for g in g_cur)
        ax[0, 0].axhline(b, color="#d1495b", ls="--", lw=1,
                         label=f"fastest cuRobo-only, no TOPP-RA ({b:.2f} s, limits violated)")
        cf = [g for g in g_cur if g["feasible"]]
        if cf:
            bf = min(g["mean_dur"] for g in cf)
            ax[0, 0].axhline(bf, color="#2a9d8f", ls="-.", lw=1,
                             label=f"fastest FEASIBLE cuRobo-only ({bf:.2f} s)")
    ax[0, 0].set_xlabel("config x retime combination (sorted)")
    ax[0, 0].set_ylabel("mean duration over the 5 cases (s)")
    ax[0, 0].set_title("duration vs configuration")
    ax[0, 0].legend(fontsize=8)

    # B: duration vs the TOPP-RA limits
    for al in sorted({g["alim"] for g in g_topp}):
        sel = [g for g in g_topp if g["alim"] == al]
        vs = sorted({g["vlim"] for g in sel})
        best_d = [min(g["mean_dur"] for g in sel if g["vlim"] == v) for v in vs]
        ax[0, 1].plot(vs, best_d, "o-", label=f"alim {al:.0f} deg/s^2")
    ax[0, 1].set_xlabel("TOPP-RA vlim (deg/s)"); ax[0, 1].set_ylabel("best mean duration (s)")
    ax[0, 1].set_title("TOPP-RA limits"); ax[0, 1].legend(fontsize=8)

    # C: duration vs the cuRobo knobs (best retime per cuRobo config)
    lbl, val, col = [], [], []
    for keyname in ("n_knots", "scale", "seeds", "finetune", "interp_dt"):
        for v in sorted({g[keyname] for g in g_topp}, key=str):
            sel = [g for g in g_topp if g[keyname] == v and g["feasible"]]
            if not sel:
                continue
            lbl.append(f"{keyname}={v}"); val.append(min(g["mean_dur"] for g in sel))
            col.append("#2f6fdb")
    ax[0, 2].barh(np.arange(len(lbl)), val, color=col)
    ax[0, 2].set_yticks(np.arange(len(lbl))); ax[0, 2].set_yticklabels(lbl, fontsize=8)
    ax[0, 2].set_xlabel("best feasible mean duration (s)")
    ax[0, 2].set_title("cuRobo knobs (best feasible)")
    ax[0, 2].invert_yaxis()

    # D: plan time vs duration
    ax[0, 3].scatter([g["t_plan"] for g in g_topp if not g["feasible"]],
                     [g["mean_dur"] for g in g_topp if not g["feasible"]],
                     s=10, c="#c8ccd4")
    ax[0, 3].scatter([g["t_plan"] for g in g_topp if g["feasible"]],
                     [g["mean_dur"] for g in g_topp if g["feasible"]], s=14, c="#2f6fdb")
    ax[0, 3].set_xlabel("mean plan time (s)"); ax[0, 3].set_ylabel("mean duration (s)")
    ax[0, 3].set_title("plan-time cost")

    # bottom row: the best config's per-joint profiles on its worst case
    if series is not None:
        t = np.arange(len(series["q"])) * DT
        for j in range(6):
            ax[1, 0].plot(t, np.abs(series["qd"][:, j]), lw=1, label=f"j{j + 1}")
            ax[1, 1].plot(t, np.abs(series["qdd"][:, j]), lw=1)
            ax[1, 2].plot(t, np.abs(series["jerk"][:, j]), lw=1)
            ax[1, 3].plot(t, np.abs(series["tau"][:, j]), lw=1)
        ax[1, 0].axhline(a.vlim_hard, color="#d1495b", ls="--", lw=1)
        ax[1, 1].axhline(a.alim_hard, color="#d1495b", ls="--", lw=1)
        ax[1, 2].axhline(np.degrees(series["jlim"]), color="#d1495b", ls="--", lw=1)
        ax[1, 2].axhline(a.jerk_design, color="#e0a458", ls=":", lw=1)
        ax[1, 3].axhline(TAU_LIM[0], color="#d1495b", ls="--", lw=1)
        for k, (ttl, un) in enumerate([("|qd|", "deg/s"), ("|qdd|", "deg/s^2"),
                                       ("|jerk|", "deg/s^3"), ("|tau| (mj_inverse)", "Nm")]):
            ax[1, k].set_xlabel("t (s)"); ax[1, k].set_ylabel(un)
            ax[1, k].set_title(f"best config, case {series['case']}: {ttl}")
        ax[1, 2].set_yscale("log"); ax[1, 3].set_yscale("log")
        ax[1, 0].legend(fontsize=7, ncol=3)
    ttl = "no feasible config" if best is None else (
        f"best feasible: n_knots={best['n_knots']} interp_dt={best['interp_dt']} "
        f"seeds={best['seeds']} scale={best['scale']} finetune={best['finetune']} | "
        f"TOPP-RA vlim={best['vlim']} alim={best['alim']} | "
        f"mean {best['mean_dur']:.2f} s")
    fig.suptitle("planner_sweep: pick-approach duration under the Pro 630 limits\n" + ttl,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(a.out, "summary.png")
    fig.savefig(path, dpi=110)
    print(f"[sweep] {path}")


def best_series(a, best, corr, jlim):
    """Re-plan the best config's worst case and return its dense profiles."""
    if best is None:
        return None
    ci = int(np.argmax(best["durs"]))
    cx, cy = CASES[ci]
    mount = "tight" if (cx, cy) != (0.50, -0.05) else a.case5_mount
    C = Curobo(n_knots=best["n_knots"], interp_dt=best["interp_dt"],
               seeds=best["seeds"], scale=best["scale"])
    way, info = plan_case(C, cx, cy, corr, best["finetune"], mount)
    del C
    if way is None:
        return None
    q, _, _ = retime(way, best["vlim"], best["alim"], jlim)
    qd, qdd, jrk, _ = _peaks(q)
    mj = MjLink()
    kern = np.ones(9) / 9.0
    qdd_s = np.stack([np.convolve(qdd[:, j], kern, mode="same") for j in range(6)], axis=1)
    arr = np.stack([q, qd, qdd_s], axis=1)
    tau = np.asarray(mj.rpc({"mode": "torque_series", "a": arr.tolist()})["tau"])
    mj.close()
    return {"case": ci, "cx": cx, "cy": cy, "q": q, "qd": np.degrees(qd),
            "qdd": np.degrees(qdd), "jerk": np.degrees(jrk), "tau": tau, "jlim": jlim,
            "dur": (len(q) - 1) * DT, "t_plan": info["t_plan"]}


def analyze(a, rows):
    retag_torque(rows)
    groups = rank(rows)
    feas = [g for g in groups if g["feasible"] and g["vlim"] != "curobo"]
    best = feas[0] if feas else None
    cur = [g for g in groups if g["vlim"] == "curobo"]
    cur_feas = [g for g in cur if g["feasible"]]
    jlim = a.jerk_rad
    series = None
    try:
        corr = np.asarray(a.corr, float)
        series = best_series(a, best, corr, jlim)
    except Exception as e:                                       # noqa: BLE001
        print(f"[sweep] best-config re-plan failed: {e}")
    summary_png(a, rows, best, series)
    rep = {"best_feasible": None if best is None else
           {k: best[k] for k in ("n_knots", "interp_dt", "seeds", "scale", "finetune",
                                 "vlim", "alim", "mean_dur", "max_dur", "durs",
                                 "t_plan", "t_retime")},
           "top5": [{k: g[k] for k in ("n_knots", "interp_dt", "seeds", "scale", "finetune",
                                       "vlim", "alim", "mean_dur", "durs", "t_plan")}
                    for g in feas[:5]],
           "curobo_only_best": None if not cur else
           {k: min(cur, key=lambda g: g["mean_dur"])[k] for k in
            ("n_knots", "interp_dt", "seeds", "scale", "finetune", "mean_dur", "durs",
             "feasible", "viol")},
           "curobo_only_best_feasible": None if not cur_feas else
           {k: cur_feas[0][k] for k in ("n_knots", "interp_dt", "seeds", "scale",
                                        "finetune", "mean_dur", "durs")},
           "n_groups": len(groups), "n_feasible": len(feas)}
    if series is not None:
        rep["best_series"] = {"case": series["case"], "dur": series["dur"],
                              "peak_qd": np.abs(series["qd"]).max(axis=0).round(2).tolist(),
                              "peak_qdd": np.abs(series["qdd"]).max(axis=0).round(1).tolist(),
                              "peak_jerk": np.abs(series["jerk"][20:-20]).max(axis=0).round(0).tolist(),
                              "peak_tau": np.abs(series["tau"]).max(axis=0).round(2).tolist()}
        np.savez(os.path.join(a.out, "best_series.npz"),
                 **{k: v for k, v in series.items() if isinstance(v, np.ndarray)})
    json.dump(rep, open(os.path.join(a.out, "best.json"), "w"), indent=2, default=float)
    print("[sweep] " + json.dumps(rep["best_feasible"], default=float))
    print("[sweep] curobo-only " + json.dumps(rep["curobo_only_best"], default=float))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--_mj_stage", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/planner_sweep"))
    ap.add_argument("--n_knots", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--interp_dt", type=float, nargs="+", default=[0.01, 0.02])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--scale", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--finetune", type=int, nargs="+", default=[1, 0])
    ap.add_argument("--vlim", type=float, nargs="+", default=[25.0, 30.0, 36.0])
    ap.add_argument("--alim", type=float, nargs="+", default=[300.0, 450.0, 600.0])
    ap.add_argument("--vlim_hard", type=float, default=36.0)
    ap.add_argument("--alim_hard", type=float, default=600.0)
    ap.add_argument("--jerk_design", type=float, default=3000.0,
                    help="deg/s^3 design figure reported alongside the yml max_jerk")
    ap.add_argument("--case5_mount", default="slim", choices=["tight", "slim"])
    ap.add_argument("--_sim_stage", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--validate", action="store_true",
                    help="roll the best config in the measured-drive twin (2 cases)")
    ap.add_argument("--validate_cases", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--sim_steps", type=int, default=150)
    ap.add_argument("--sim_hold", type=int, default=20)
    ap.add_argument("--sim_dq_max", type=float, default=6.0,
                    help="env dq_max (deg/decision); raise so the executor does not clip")
    ap.add_argument("--analyze", action="store_true",
                    help="re-run the analysis/plot on an existing results.json")
    a = ap.parse_args()
    if a._mj_stage:
        mj_stage()
        return
    if a._sim_stage:
        sim_stage()
        return
    a.out = os.path.expanduser(a.out)
    a.finetune = [bool(x) for x in a.finetune]
    if a.analyze:
        rows = json.load(open(os.path.join(a.out, "results.json")))
        a.jerk_rad = 2000.0
        mj = MjLink()
        C = Curobo()
        p_c = C.fk(START_Q)[0]
        del C
        f = mj.rpc({"mode": "fk", "q": list(START_Q)})
        mj.close()
        a.corr = (quat_to_R(R_DOWN_YAW90) @ (np.asarray(f["R"]).T @
                                             (np.asarray(f["pos"]) - p_c))).tolist()
        write_outputs(a, rows)
        rep = analyze(a, rows)
        if a.validate and rep["best_feasible"]:
            validate(a, rep["best_feasible"], tuple(a.validate_cases))
        return
    run(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""sim_taskmpc :: closed-loop task-space MPC on the MuJoCo twin.

Layered exactly like deployment:
  50 Hz  task QP  : min ||J dq - dx_des||^2 + lam||dq||^2
                    s.t. |dq| <= dq_max,  n_wall . (x + J dq) >= wall_x + m
                    -> joint reference (this is the receding-horizon layer)
  250 Hz LQR      : u = -K [q - q_ref, qvel]  (the deployed lag-aware law)
  drives          : velocity servo emulation (first-order lag kv=40/s)

Emulated sensors:
  object tracker  : true pose sampled at 6 Hz with 1-sample lag + 2 mm noise
                    (the detect-once-track-fast D435 pattern)
  contact         : |joint torque| spike (qfrc_constraint) -> stop descent

Scenario: approach hover -> OBJECT MOVES 6 cm mid-approach (closed-loop
test) -> track -> descend to contact -> hold -> retreat -> home.

Outputs: per-joint command-vs-actual plot, task-error plot, rendered mp4.

    python sim_taskmpc.py --out ~/pnp_rl/taskmpc
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy import sparse
import osqp

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "rl", "scenes", "box_med_ped.xml")

DT = 0.002                  # physics
INNER_DT = 0.004            # 250 Hz LQR
OUTER_DT = 0.020            # 50 Hz task QP
TRACK_DT = 1.0 / 6.0        # 6 Hz object tracker
K_LQR = np.array([53.4798, 4.7131])          # deployed gains (deg units)
VEL_LIMIT = np.radians(40.0)
KV_DRIVE = 40.0
DQ_MAX = np.radians(1.2)    # per outer tick (60 deg/s task-layer ceiling)
WALL_X = -0.30
HOME = np.radians([0.0, -20.0, 80.0, 10.0, -90.0, 0.0])  # URDF-frame home


class TaskQP:
    """One-step task-space QP with wall half-space constraint."""
    def __init__(self, lam=1e-3):
        self.lam = lam

    def solve(self, J, dx, x_tcp):
        n = 6
        P = J.T @ J + self.lam * np.eye(n)
        q = -J.T @ dx
        # box on dq + wall: e_x . (x + J dq) >= WALL_X + 0.05
        A = np.vstack([np.eye(n), J[0:1, :]])
        lo = np.concatenate([np.full(n, -DQ_MAX),
                             [WALL_X + 0.05 - x_tcp[0]]])
        hi = np.concatenate([np.full(n, DQ_MAX), [np.inf]])
        s = osqp.OSQP()
        s.setup(sparse.csc_matrix((P + P.T) / 2), q, sparse.csc_matrix(A),
                lo, hi, verbose=False, max_iter=200)
        r = s.solve()
        if r.info.status.startswith("solved"):
            return r.x
        return np.clip(np.linalg.lstsq(J, dx, rcond=None)[0], -DQ_MAX, DQ_MAX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/taskmpc"))
    ap.add_argument("--T", type=float, default=14.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object0")
    jadr = m.jnt_qposadr[m.body_jntadr[bid]]
    half_z = float(m.geom_size[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM, "object0"), 2])
    d.qpos[:6] = HOME
    d.qpos[jadr:jadr + 3] = [0.36, 0.05, half_z + 0.001]
    d.qpos[jadr + 3:jadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    cup_gids = [g for g in range(m.ngeom)
                if "cup" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    obj_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_object0")

    def cup_contact_force():
        f = 0.0
        buf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = (c.geom1, c.geom2)
            if (pair[0] in cup_gids and pair[1] == obj_gid) or \
               (pair[1] in cup_gids and pair[0] == obj_gid):
                mujoco.mj_contactForce(m, d, i, buf)
                f += abs(buf[0])
        return f

    qp = TaskQP()
    ren = mujoco.Renderer(m, height=240, width=424)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0

    q_ref = d.qpos[:6].copy()
    tracked_obj = d.xpos[bid].copy()      # tracker state (lagged)
    last_track = -1.0
    pend_track = None
    phase = "approach"
    contact_t = None
    moved = False
    logs = dict(t=[], q=[], ref=[], err=[], contact=[], phase=[])
    frames = []
    n_outer = int(a.T / OUTER_DT)

    for ko in range(n_outer):
        t = ko * OUTER_DT
        # --- object moves mid-approach (closed-loop showcase) ---
        if phase == "approach" and t > 2.5 and not moved:
            d.qpos[jadr] += 0.06
            d.qpos[jadr + 1] -= 0.04
            mujoco.mj_forward(m, d)
            moved = True
        # --- 6 Hz tracker with one-sample lag + noise ---
        if t - last_track >= TRACK_DT:
            if pend_track is not None:
                tracked_obj = pend_track
            pend_track = d.xpos[bid].copy() + np.random.normal(0, 0.002, 3)
            last_track = t
        # --- task goal per phase ---
        obj_top = tracked_obj + np.array([0, 0, half_z])
        if phase == "approach":
            x_goal = obj_top + np.array([0, 0, 0.08])
            x_tcp = d.site_xpos[sid]
            if np.linalg.norm(x_tcp - x_goal) < 0.01 and t > 4.0:
                phase = "descend"
        elif phase == "descend":
            x_goal = obj_top + np.array([0, 0, -0.015])
            tau = cup_contact_force()
            if tau > 1.0:
                phase = "hold"
                contact_t = t
        elif phase == "hold":
            x_goal = d.site_xpos[sid].copy()
            if t - contact_t > 1.0:
                phase = "retreat"
        elif phase == "retreat":
            x_goal = obj_top + np.array([0, 0, 0.10])
            if d.site_xpos[sid][2] > obj_top[2] + 0.09:
                phase = "home"
        else:
            x_goal = None
        # --- 50 Hz task QP -> joint reference ---
        if phase == "home":
            dq = np.clip(HOME - q_ref, -DQ_MAX, DQ_MAX)
        else:
            x_tcp = d.site_xpos[sid].copy()
            dx = np.clip(x_goal - x_tcp, -0.03, 0.03)
            Jp = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, Jp, None, sid)
            dq = qp.solve(Jp[:, :6], dx, x_tcp)
        q_ref = q_ref + dq
        # --- inner 250 Hz LQR + drive emulation ---
        for ki in range(int(OUTER_DT / INNER_DT)):
            e_deg = np.degrees(d.qpos[:6] - q_ref)
            v_deg = np.degrees(d.qvel[:6])
            u = np.clip(-(K_LQR[0] * e_deg + K_LQR[1] * v_deg), -40.0, 40.0)
            v_cmd = np.clip(np.radians(u), -VEL_LIMIT, VEL_LIMIT)
            for _ in range(int(INNER_DT / DT)):
                qacc = KV_DRIVE * (v_cmd - d.qvel[:6])
                d.qfrc_applied[:6] = np.clip(
                    d.qfrc_bias[:6] + (m.dof_armature[:6] + 0.05) * qacc,
                    -100, 100)
                mujoco.mj_step(m, d)
        logs["t"].append(t)
        logs["q"].append(np.degrees(d.qpos[:6]).copy())
        logs["ref"].append(np.degrees(q_ref).copy())
        te = 0.0 if phase == "home" else float(np.linalg.norm(
            d.site_xpos[sid] - x_goal))
        logs["err"].append(te)
        logs["contact"].append(cup_contact_force())
        logs.setdefault("tcpz", []).append(float(d.site_xpos[sid][2]))
        logs["phase"].append(phase)
        if ko % 2 == 0:                    # 25 fps video
            ren.update_scene(d, camera="fixed_d435", scene_option=vopt)
            frames.append(ren.render().copy())

    # ---- outputs ----
    import cv2
    vw = cv2.VideoWriter(os.path.join(a.out, "_raw.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 25, (424, 240))
    ph_prev = None
    for f, ph in zip(frames, logs["phase"][::2]):
        img = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        cv2.putText(img, ph, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.array(logs["t"])
    Q = np.array(logs["q"])
    R = np.array(logs["ref"])
    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    for j in range(6):
        ax = axes[j // 2, j % 2]
        ax.plot(T, R[:, j], "--", color="#D65F5F", lw=1.2, label="command (q_ref)")
        ax.plot(T, Q[:, j], "-", color="#4878CF", lw=1.2, label="actual")
        ax.axvline(2.5, color="gray", ls=":", lw=1)
        ax.set_title(f"J{j+1}", fontsize=10)
        ax.set_ylabel("deg")
        if j == 0:
            ax.legend(fontsize=8)
            ax.text(2.55, ax.get_ylim()[0], "object\nmoves", fontsize=7,
                    color="gray")
    ax = axes[3, 0]
    ax.plot(T, np.array(logs["err"]) * 100, color="#6ACC65")
    ax.axvline(2.5, color="gray", ls=":", lw=1)
    ax.set_title("task-space error (cm)", fontsize=10)
    ax.set_xlabel("t (s)")
    ax = axes[3, 1]
    ax.plot(T, logs["contact"], color="#956CB4")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_title("cup-object contact force (N)", fontsize=10)
    ax.set_xlabel("t (s)")
    fig.suptitle("closed-loop task-space MPC on the MuJoCo twin "
                 "(50 Hz QP -> 250 Hz LQR -> drives; 6 Hz laggy tracker)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "taskmpc_cmd_vs_actual.png"), dpi=110)
    ph = np.array(logs["phase"])
    des = ph == "descend"
    if des.any():
        print(f"descend: min task err {np.array(logs['err'])[des].min()*100:.2f} cm, "
              f"max contact {np.array(logs['contact'])[des].max():.3f}, "
              f"tcp z min {min(z for z, p in zip([q for q in logs['tcpz']], ph) if p=='descend'):.4f}")
    mx = max(logs["err"][int(3.0/OUTER_DT):int(4.5/OUTER_DT)])
    print(f"phases reached: {sorted(set(logs['phase']), key=logs['phase'].index)}")
    print(f"post-move recovery peak task error: {mx*100:.1f} cm")
    print(f"contact detected at t={contact_t}s" if contact_t else "NO CONTACT")
    print(f"outputs -> {a.out}")


if __name__ == "__main__":
    np.random.seed(0)
    main()

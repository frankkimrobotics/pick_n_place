"""mjwarp_pick_demo :: validate robot_warp.xml under real dynamics + GPU batch.

Low-level control is the real robot's shape: torque motors driven at a 4 ms
tick (CTRL_DT) by PD + inverse-dynamics feedforward -- tau = tau_ff(qref,
qdref, qddref via mj_inverse) + KP*(qref-q) + KD*(qdref-qd), zero-order held
over the 2 ms physics substeps. The arm starts from config.HOME_Q.

Phase A (CPU mujoco, mj_step): gentle pick cycle on object0 -- hover, slow
descend until the tip rangefinder reads contact, activate the suction weld
(eq_data relpose = tcp->object pose at grasp, eq_active on), lift, swing over
the bin, descend, release, retreat. Reports worst-case joint tracking error.

Phase B (mujoco_warp, GPU): run the identical reference + weld schedule
closed-loop in a batch of NWORLD identical worlds; every world must land
object0 in the bin and world0 must track the CPU run.

Run in the mjwarp env:
    conda activate mjwarp && python mjwarp_pick_demo.py
"""
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C

XML = os.path.join(HERE, "outputs", "mujoco_sim", "robot_warp.xml")

NWORLD = 32
DT = 0.002                      # physics timestep (matches the MJCF)
CTRL_DT = 0.004                 # 4 ms control tick, as streamed to the robot
NSUB = int(round(CTRL_DT / DT))
Q_HOME = np.asarray(C.HOME_Q, float)
# PD gains are inertia-scaled in build_schedule(): KP_i = M_ii * W_BW^2,
# KD_i = 2 * ZETA * M_ii * W_BW  (computed-torque shape, diagonal approx).
# Wrist joints get higher bandwidth: their inertia is tiny, so the same rad/s
# costs little torque, and the weld latch dumps the payload weight on joint5.
W_BW = np.array([40.0, 40.0, 40.0, 60.0, 90.0, 90.0])
ZETA = 1.1
TAU_MAX = 100.0

BIN_XY = np.array([0.10, 0.40])
BIN_HALF = 0.15
OBJ = "object0"
OBJ_TOP = 0.060                 # cylinder pos z 0.030 + half height 0.030
HOVER = 0.12                    # tip clearance above object top for hover
CONTACT_MM = 8.0                # rangefinder threshold to latch suction

R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])  # tcp z-axis down

# elbow-safety joint box (matches the cuRobo planner constraint): |joint2|<=70deg
# and |joint3|<=145deg keep the link2/link3 chain above the table plane
# (FK sweep: elbow z >= +0.05 m inside this box, dips to -0.3 m outside)
ELBOW_LO = np.radians([-180, -70, -145, -170, -167, -173])
ELBOW_HI = np.radians([180, 70, 145, 170, 167, 173])


def _ik_once(m, d, sid, target_p, target_R, q0, iters=400):
    d.qpos[:6] = q0
    perr = rerr = None
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.site_xpos[sid].copy()
        R = d.site_xmat[sid].reshape(3, 3).copy()
        perr = target_p - p
        rq = np.zeros(4)
        mujoco.mju_mat2Quat(rq, (target_R @ R.T).ravel())
        rerr = np.zeros(3)
        mujoco.mju_quat2Vel(rerr, rq, 1.0)
        err = np.concatenate([perr, 0.5 * rerr])
        if np.linalg.norm(perr) < 1e-4 and np.linalg.norm(rerr) < 1e-3:
            break
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, :6], 0.5 * jr[:, :6]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        d.qpos[:6] = np.clip(d.qpos[:6] + dq,
                             np.maximum(m.jnt_range[:6, 0], ELBOW_LO),
                             np.minimum(m.jnt_range[:6, 1], ELBOW_HI))
    return d.qpos[:6].copy(), np.linalg.norm(perr), np.linalg.norm(rerr)


def ik(m, d, site, target_p, target_R, q0, restarts=20):
    """Multi-start damped least-squares 6D IK on the tcp site, joints 0..5."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    rng = np.random.default_rng(0)
    best = (None, np.inf, np.inf)
    seeds = [np.asarray(q0, float)]
    lo = np.maximum(m.jnt_range[:6, 0], ELBOW_LO)
    hi = np.minimum(m.jnt_range[:6, 1], ELBOW_HI)
    seeds += [rng.uniform(np.maximum(lo, -2.0), np.minimum(hi, 2.0))
              for _ in range(restarts)]
    for s in seeds:
        q, pe, re = _ik_once(m, d, sid, np.asarray(target_p, float), target_R, s)
        if pe + 0.3 * re < best[1] + 0.3 * best[2]:
            best = (q, pe, re)
        if pe < 1e-4 and re < 1e-3:
            break
    return best[0], best[1]


def _seg(out, q_from, q_to, seconds):
    n = max(2, int(seconds / CTRL_DT))
    for t in np.linspace(0.0, 1.0, n):
        a = 10 * t**3 - 15 * t**4 + 6 * t**5   # min-jerk: qd=qdd=0 at both ends
        out.append((1 - a) * q_from + a * q_to)


def build_schedule(m):
    """IK waypoints + 4ms reference (qref, qdref, qddref, tau_ff) and phase indices."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
    obj_p = d.xpos[obj_bid].copy()

    dik = mujoco.MjData(m)
    q_hover, e1 = ik(m, dik, "tcp", obj_p + [0, 0, OBJ_TOP - obj_p[2] + HOVER],
                     R_DOWN, Q_HOME)
    # descend endpoint 6mm above the top: suction latches at 8mm range, so the
    # welded object gets pressed at most ~2mm into the table before the dwell
    q_touch, e2 = ik(m, dik, "tcp", obj_p + [0, 0, OBJ_TOP - obj_p[2] + 0.006],
                     R_DOWN, q_hover)
    # carry above the 0.30m bin walls: object hangs ~0.09m below the tcp
    q_bin, e3 = ik(m, dik, "tcp", [BIN_XY[0], BIN_XY[1], 0.42], R_DOWN, q_hover)
    q_drop, e4 = ik(m, dik, "tcp", [BIN_XY[0], BIN_XY[1], 0.20], R_DOWN, q_bin)
    print(f"[ik] residuals hover {e1:.4f} touch {e2:.4f} bin {e3:.4f} drop {e4:.4f} m")

    qref = []
    _seg(qref, Q_HOME, q_hover, 2.5)
    i_descend_start = len(qref)
    _seg(qref, q_hover, q_touch, 2.5)         # gentle: 2.5 s descend
    i_descend_end = len(qref)
    _seg(qref, q_touch, q_touch, 0.3)         # dwell at contact
    _seg(qref, q_touch, q_hover, 1.5)         # lift
    _seg(qref, q_hover, q_bin, 2.5)           # carry
    _seg(qref, q_bin, q_drop, 1.0)            # move down over bin
    i_release = len(qref)                     # release here
    _seg(qref, q_drop, q_bin, 1.0)            # retreat
    _seg(qref, q_bin, q_bin, 1.5)             # settle
    qref = np.array(qref)
    qdref = np.gradient(qref, CTRL_DT, axis=0)
    qddref = np.gradient(qdref, CTRL_DT, axis=0)
    # box-smooth the accel so segment-boundary finite-diff spikes don't slam tau_ff
    kern = np.ones(9) / 9.0
    qddref = np.stack([np.convolve(qddref[:, j], kern, mode="same")
                       for j in range(6)], axis=1)

    # inverse-dynamics feedforward on the reference (same model, objects at rest)
    dinv = mujoco.MjData(m)
    tau_ff = np.zeros_like(qref)
    for k in range(len(qref)):
        dinv.qpos[:] = d.qpos
        dinv.qvel[:] = 0
        dinv.qacc[:] = 0
        dinv.qpos[:6] = qref[k]
        dinv.qvel[:6] = qdref[k]
        dinv.qacc[:6] = qddref[k]
        mujoco.mj_inverse(m, dinv)
        tau_ff[k] = dinv.qfrc_inverse[:6]
    # gain-scheduled computed-torque PD: per-tick diagonal of M(qref_k), so
    # joint1 is stiff when the arm is extended (I ~1.5) yet stable near home
    # where its inertia collapses to ~0.02
    dm = mujoco.MjData(m)
    Mii = np.zeros_like(qref)
    for k in range(len(qref)):
        dm.qpos[:6] = qref[k]
        mujoco.mj_forward(m, dm)
        for i in range(6):
            e = np.zeros(m.nv); r = np.zeros(m.nv)
            e[i] = 1.0
            mujoco.mj_mulM(m, dm, r, e)
            Mii[k, i] = r[i]
    Mii = np.maximum(Mii, 0.005)
    kp = Mii * W_BW ** 2
    kd = 2.0 * ZETA * Mii * W_BW
    print(f"[plan] {len(qref)} ticks ({len(qref)*CTRL_DT:.1f} s), "
          f"|tau_ff| max {np.abs(tau_ff).max():.1f} Nm, "
          f"KP range {np.round(kp.min(axis=0), 1)} .. {np.round(kp.max(axis=0), 1)}")
    return dict(qref=qref, qdref=qdref, tau_ff=tau_ff, kp=kp, kd=kd,
                i_descend_start=i_descend_start, i_descend_end=i_descend_end,
                i_release=i_release)


def pd_tau(sched, k, q, qd):
    e = sched["qref"][k] - q
    ed = sched["qdref"][k] - qd
    return np.clip(sched["tau_ff"][k] + sched["kp"][k] * e + sched["kd"][k] * ed,
                   -TAU_MAX, TAU_MAX)


def run_cpu(m, sched, frame_cb=None):
    """Closed-loop 4ms PD+ff pick cycle on CPU. Returns weld tick + logs."""
    d = mujoco.MjData(m)
    tcp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
    eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{OBJ}")
    rf_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "tip_range")]
    d.qpos[:6] = Q_HOME
    mujoco.mj_forward(m, d)

    qref = sched["qref"]
    weld_tick = -1
    eq_data_at_weld = None
    qlog = np.zeros_like(qref)
    for k in range(len(qref)):
        # suction may only latch during the descend (the rangefinder ray sweeps
        # other objects during the fast swing -- welding there flings them)
        if weld_tick < 0 and sched["i_descend_start"] <= k <= sched["i_descend_end"] + 100:
            rng = d.sensordata[rf_adr]
            if 0 <= rng * 1000.0 < CONTACT_MM:
                p1 = d.xpos[tcp_bid]; R1 = d.xmat[tcp_bid].reshape(3, 3)
                p2 = d.xpos[obj_bid]; R2 = d.xmat[obj_bid].reshape(3, 3)
                rq = np.zeros(4)
                mujoco.mju_mat2Quat(rq, (R1.T @ R2).ravel())
                m.eq_data[eq_id, :3] = 0
                m.eq_data[eq_id, 3:6] = R1.T @ (p2 - p1)
                m.eq_data[eq_id, 6:10] = rq
                d.eq_active[eq_id] = 1
                weld_tick = k
                eq_data_at_weld = m.eq_data[eq_id].copy()
                print(f"[cpu] contact at tick {k} (t={k*CTRL_DT:.2f}s, "
                      f"range {rng*1000:.1f} mm) -> suction ON")
        if k == sched["i_release"] and weld_tick >= 0:
            d.eq_active[eq_id] = 0
            print(f"[cpu] release at tick {k} (t={k*CTRL_DT:.2f}s) -> suction OFF")
        # reference held over the 4ms tick; feedback closes at the physics rate
        # (like the real drives' inner servo loop under the 4ms stream)
        for _ in range(NSUB):
            d.ctrl[:6] = pd_tau(sched, k, d.qpos[:6], d.qvel[:6])
            mujoco.mj_step(m, d)
        qlog[k] = d.qpos[:6]
        if frame_cb is not None:
            frame_cb(d, k, weld_tick)
    err = np.abs(qlog - qref)
    print(f"[cpu] tracking error: max {np.degrees(err.max()):.3f} deg "
          f"(per-joint max {np.round(np.degrees(err.max(axis=0)), 3)}), "
          f"rms {np.degrees(np.sqrt((err**2).mean())):.4f} deg")
    fp = d.xpos[obj_bid]
    in_bin = (abs(fp[0] - BIN_XY[0]) < BIN_HALF and abs(fp[1] - BIN_XY[1]) < BIN_HALF
              and fp[2] < 0.10)
    print(f"[cpu] {OBJ} final pos {np.round(fp, 3)}  in_bin={in_bin}")
    return dict(weld_tick=weld_tick, eq_data=eq_data_at_weld, qlog=qlog,
                in_bin=in_bin, track_max_deg=float(np.degrees(err.max())))


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    sched = build_schedule(m)
    res = run_cpu(m, sched)
    if res["weld_tick"] < 0:
        print("[cpu] FAIL: never reached contact"); sys.exit(1)
    if not res["in_bin"]:
        print("[cpu] FAIL: object not in bin"); sys.exit(1)

    # -------- Phase B: mujoco_warp, closed-loop batch with the same schedule
    import time
    import warp as wp
    import mujoco_warp as mjw
    wp.init()
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
    eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{OBJ}")
    d0 = mujoco.MjData(m)
    d0.qpos[:6] = Q_HOME
    m.eq_data[eq_id, :] = res["eq_data"]      # bake relpose; weld starts inactive
    mujoco.mj_forward(m, d0)
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, d0, nworld=NWORLD)

    base_eq = dw.eq_active.numpy().copy()
    ctrl_dtype = np.float32 if dw.ctrl.dtype == wp.float32 else np.float64
    qref = sched["qref"]

    t0 = time.time()
    for k in range(len(qref)):
        if k == res["weld_tick"]:
            e = base_eq.copy(); e[:, eq_id] = 1
            dw.eq_active.assign(e)
        if k == sched["i_release"]:
            e = base_eq.copy(); e[:, eq_id] = 0
            dw.eq_active.assign(e)
        for _ in range(NSUB):
            q = dw.qpos.numpy()[:, :6]
            qd = dw.qvel.numpy()[:, :6]
            tau = np.clip(sched["tau_ff"][k] + sched["kp"][k] * (qref[k] - q)
                          + sched["kd"][k] * (sched["qdref"][k] - qd), -TAU_MAX, TAU_MAX)
            cc = np.zeros(dw.ctrl.shape, dtype=ctrl_dtype)
            cc[:, :6] = tau
            dw.ctrl.assign(cc)
            mjw.step(mw, dw)
    wp.synchronize()
    dt_wall = time.time() - t0
    steps = len(qref) * NSUB * NWORLD
    print(f"[warp] {NWORLD} worlds x {len(qref)*NSUB} steps in {dt_wall:.1f}s "
          f"({steps/dt_wall/1000:.0f}k env-steps/s, closed-loop w/ host PD)")

    xpos = dw.xpos.numpy()
    qpos = dw.qpos.numpy()
    ok = 0
    for w in range(NWORLD):
        p = xpos[w, obj_bid]
        if (abs(p[0] - BIN_XY[0]) < BIN_HALF and abs(p[1] - BIN_XY[1]) < BIN_HALF
                and p[2] < 0.10):
            ok += 1
    qerr = np.abs(qpos[0, :6] - res["qlog"][-1]).max()
    print(f"[warp] in_bin {ok}/{NWORLD}   world0 final joint err vs CPU {qerr:.4f} rad")
    passed = ok == NWORLD and qerr < 0.05 and res["track_max_deg"] < 1.0
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

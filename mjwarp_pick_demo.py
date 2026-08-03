"""mjwarp_pick_demo :: validate robot_warp.xml under real dynamics + GPU batch.

Low-level control is the real robot's shape: torque motors driven at a 4 ms
tick (CTRL_DT) by PD + inverse-dynamics feedforward -- tau = tau_ff(qref,
qdref, qddref via mj_inverse) + KP*(qref-q) + KD*(qdref-qd), reference held
over the 2 ms physics substeps. The arm starts from config.START_Q (= BASE_Q,
the camera-down base pose the real pick-and-place starts from; HOME_Q is the
LinuxCNC mechanical home with the arm straight up).

Grasp strategy is PRESS-TO-SEAL: detecting "cup touched the object" from
range alone is unreliable, so once the grasp pose is computed the reference
pushes PRESS_M (15 mm) further along the suction approach axis and only then
applies suction. The cup tip is a real collision sphere (the only contact
geom on the arm), so the press is dynamics: the object is squeezed against
the table, and on a curved object an off-centre press slides the cup off and
shoves the object away. The rangefinder is a seal CHECK at press end (must
read < SEAL_MM), not the grasp trigger.

Phase A (CPU mujoco, mj_step): pick cycle on object0 (flat-top cylinder).
Phase A2: press test on object7 (sphere) with a lateral offset -- must show
the object dynamically escaping and the seal check failing.
Phase B (mujoco_warp, GPU): identical reference + weld schedule closed-loop
in NWORLD identical worlds; every world must land object0 in the bin and
world0 must track the CPU run.

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
Q_START = np.asarray(C.START_Q, float)
# PD gains are inertia-scaled in build_schedule(): KP_i = M_ii * W_BW^2,
# KD_i = 2 * ZETA * M_ii * W_BW  (computed-torque shape, diagonal approx).
W_BW = np.array([40.0, 40.0, 40.0, 60.0, 90.0, 90.0])
ZETA = 1.1
TAU_MAX = 100.0

BIN_XY = np.array([0.10, 0.40])
BIN_HALF = 0.15
OBJ = "object0"
OBJ_TOP = 0.060                 # cylinder pos z 0.030 + half height 0.030
HOVER = 0.12                    # tip clearance above object top for hover
CUP_R = 0.008                   # cup tip collision sphere radius
PRESS_M = 0.015                 # press-to-seal: push 15 mm past the grasp pose
SEAL_N = 2.0                    # seal check: cup contact force at press end must exceed this
SEAL_DEG = 25.0                 # ...and the contact normal must align with the cup axis
# (the rangefinder can't seal-check: pressed 15mm in, the ray origin sits inside
# the object and MuJoCo rays skip the containing geom -- it reads the table)

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


def _gains_and_ff(m, d0, qref):
    """mj_inverse feedforward + gain-scheduled diagonal PD along the reference."""
    qdref = np.gradient(qref, CTRL_DT, axis=0)
    qddref = np.gradient(qdref, CTRL_DT, axis=0)
    # box-smooth the accel so segment-boundary finite-diff spikes don't slam tau_ff
    kern = np.ones(9) / 9.0
    qddref = np.stack([np.convolve(qddref[:, j], kern, mode="same")
                       for j in range(6)], axis=1)
    dinv = mujoco.MjData(m)
    tau_ff = np.zeros_like(qref)
    Mii = np.zeros_like(qref)
    for k in range(len(qref)):
        dinv.qpos[:] = d0.qpos
        dinv.qvel[:] = 0
        dinv.qacc[:] = 0
        dinv.qpos[:6] = qref[k]
        dinv.qvel[:6] = qdref[k]
        dinv.qacc[:6] = qddref[k]
        mujoco.mj_inverse(m, dinv)
        tau_ff[k] = dinv.qfrc_inverse[:6]
        for i in range(6):
            e = np.zeros(m.nv); r = np.zeros(m.nv)
            e[i] = 1.0
            mujoco.mj_mulM(m, dinv, r, e)
            Mii[k, i] = r[i]
    # gain-scheduled computed-torque PD: per-tick diagonal of M(qref_k), so
    # joint1 is stiff when the arm is extended yet stable near home where its
    # inertia collapses
    Mii = np.maximum(Mii, 0.005)
    return qdref, tau_ff, Mii * W_BW ** 2, 2.0 * ZETA * Mii * W_BW


def build_schedule(m):
    """IK waypoints + 4ms reference (qref, qdref, tau_ff, gains) + phase indices."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
    obj_p = d.xpos[obj_bid].copy()
    grasp_z = OBJ_TOP + CUP_R              # cup surface touching the flat top

    dik = mujoco.MjData(m)
    q_hover, e1 = ik(m, dik, "tcp", [obj_p[0], obj_p[1], OBJ_TOP + HOVER],
                     R_DOWN, Q_START)
    q_grasp, e2 = ik(m, dik, "tcp", [obj_p[0], obj_p[1], grasp_z], R_DOWN, q_hover)
    # press-to-seal: 15 mm past the grasp pose along the approach axis (down)
    q_press, e3 = ik(m, dik, "tcp", [obj_p[0], obj_p[1], grasp_z - PRESS_M],
                     R_DOWN, q_grasp)
    # carry above the 0.30m bin walls: object hangs ~0.09m below the tcp
    q_bin, e4 = ik(m, dik, "tcp", [BIN_XY[0], BIN_XY[1], 0.42], R_DOWN, q_hover)
    q_drop, e5 = ik(m, dik, "tcp", [BIN_XY[0], BIN_XY[1], 0.20], R_DOWN, q_bin)
    print(f"[ik] residuals hover {e1:.4f} grasp {e2:.4f} press {e3:.4f} "
          f"bin {e4:.4f} drop {e5:.4f} m")

    qref = []
    _seg(qref, Q_START, q_hover, 2.5)
    i_descend_start = len(qref)
    _seg(qref, q_hover, q_grasp, 2.0)         # gentle descend to the grasp pose
    i_grasp_end = len(qref)
    _seg(qref, q_grasp, q_press, 0.8)         # press 15 mm, slow
    i_press_end = len(qref)                   # suction applied here (seal check)
    _seg(qref, q_press, q_press, 0.4)         # dwell under press
    _seg(qref, q_press, q_hover, 1.5)         # lift
    _seg(qref, q_hover, q_bin, 2.5)           # carry
    _seg(qref, q_bin, q_drop, 1.0)            # move down over bin
    i_release = len(qref)                     # release here
    _seg(qref, q_drop, q_bin, 1.0)            # retreat
    _seg(qref, q_bin, q_bin, 1.5)             # settle
    qref = np.array(qref)
    qdref, tau_ff, kp, kd = _gains_and_ff(m, d, qref)
    print(f"[plan] {len(qref)} ticks ({len(qref)*CTRL_DT:.1f} s), "
          f"|tau_ff| max {np.abs(tau_ff).max():.1f} Nm")
    return dict(qref=qref, qdref=qdref, tau_ff=tau_ff, kp=kp, kd=kd,
                i_descend_start=i_descend_start, i_grasp_end=i_grasp_end,
                i_press_end=i_press_end, i_release=i_release)


def pd_tau(sched, k, q, qd):
    e = sched["qref"][k] - q
    ed = sched["qdref"][k] - qd
    return np.clip(sched["tau_ff"][k] + sched["kp"][k] * e + sched["kd"][k] * ed,
                   -TAU_MAX, TAU_MAX)


def _cup_normal_force(m, d, cup_gid):
    return _cup_contact(m, d, cup_gid)[0]


def _cup_contact(m, d, cup_gid):
    """Total normal force on the cup tip + tilt of the strongest contact's
    normal vs the cup approach axis (deg). A suction cup only seals when it
    presses roughly face-on: force alone can't distinguish a seal from the
    cup skidding along the flank of a curved object."""
    tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip_rf")
    axis = d.site_xmat[tip_sid].reshape(3, 3)[:, 2]     # cup approach direction
    f_tot, f_best, ang = 0.0, 0.0, 180.0
    buf = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if cup_gid in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, i, buf)
            fn = abs(buf[0])
            f_tot += fn
            if fn > f_best:
                f_best = fn
                n = np.asarray(c.frame[:3])
                ang = np.degrees(np.arccos(min(1.0, abs(float(n @ axis)))))
    return f_tot, ang


def _latch_weld(m, d, eq_id, tcp_bid, obj_bid):
    p1 = d.xpos[tcp_bid]; R1 = d.xmat[tcp_bid].reshape(3, 3)
    p2 = d.xpos[obj_bid]; R2 = d.xmat[obj_bid].reshape(3, 3)
    rq = np.zeros(4)
    mujoco.mju_mat2Quat(rq, (R1.T @ R2).ravel())
    m.eq_data[eq_id, :3] = 0
    m.eq_data[eq_id, 3:6] = R1.T @ (p2 - p1)
    m.eq_data[eq_id, 6:10] = rq
    d.eq_active[eq_id] = 1
    return m.eq_data[eq_id].copy()


def run_cpu(m, sched, frame_cb=None):
    """Closed-loop 4ms PD+ff press-to-seal pick cycle on CPU."""
    d = mujoco.MjData(m)
    tcp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
    eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{OBJ}")
    rf_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "tip_range")]
    cup_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")
    d.qpos[:6] = Q_START
    mujoco.mj_forward(m, d)

    qref = sched["qref"]
    weld_tick = -1
    eq_data_at_weld = None
    press_force = 0.0
    qlog = np.zeros_like(qref)
    for k in range(len(qref)):
        if sched["i_grasp_end"] <= k <= sched["i_press_end"]:
            press_force = max(press_force, _cup_normal_force(m, d, cup_gid))
        if k == sched["i_press_end"]:
            f_now, tilt = _cup_contact(m, d, cup_gid)
            if f_now > SEAL_N and tilt < SEAL_DEG:
                eq_data_at_weld = _latch_weld(m, d, eq_id, tcp_bid, obj_bid)
                weld_tick = k
                print(f"[cpu] press done (t={k*CTRL_DT:.2f}s, force {f_now:.1f} N, "
                      f"tilt {tilt:.0f} deg) -> seal OK, suction ON")
            else:
                print(f"[cpu] press done but seal FAILED (force {f_now:.1f} N, "
                      f"tilt {tilt:.0f} deg) -- no suction")
        if k == sched["i_release"] and weld_tick >= 0:
            d.eq_active[eq_id] = 0
            print(f"[cpu] release at t={k*CTRL_DT:.2f}s -> suction OFF")
        # reference held over the 4ms tick; feedback closes at the physics rate
        # (like the real drives' inner servo loop under the 4ms stream)
        for _ in range(NSUB):
            d.ctrl[:6] = pd_tau(sched, k, d.qpos[:6], d.qvel[:6])
            mujoco.mj_step(m, d)
        qlog[k] = d.qpos[:6]
        if frame_cb is not None:
            frame_cb(d, k, weld_tick)
    err = np.abs(qlog - qref)
    # contact-critical tracking windows: approach-to-grasp, post-latch lift,
    # and drop/release. The press itself is intentional force application
    # (the reference is unreachable by design), so it is excluded.
    crit = np.zeros(len(qref), dtype=bool)
    crit[sched["i_descend_start"]:sched["i_grasp_end"]] = True
    crit[sched["i_press_end"] + 150:sched["i_press_end"] + 500] = True
    crit[sched["i_release"] - 250:] = True
    err_crit = float(np.degrees(err[crit].max()))
    print(f"[cpu] tracking error: max {np.degrees(err.max()):.3f} deg, "
          f"rms {np.degrees(np.sqrt((err**2).mean())):.4f} deg, "
          f"grasp/lift/place phases max {err_crit:.3f} deg")
    fp = d.xpos[obj_bid]
    in_bin = (abs(fp[0] - BIN_XY[0]) < BIN_HALF and abs(fp[1] - BIN_XY[1]) < BIN_HALF
              and fp[2] < 0.10)
    print(f"[cpu] {OBJ} final pos {np.round(fp, 3)}  in_bin={in_bin}")
    return dict(weld_tick=weld_tick, eq_data=eq_data_at_weld, qlog=qlog,
                in_bin=in_bin, track_max_deg=float(np.degrees(err.max())),
                track_crit_deg=err_crit, press_force=press_force)


def press_test(m, obj="object7", lateral=0.012, frame_cb=None):
    """Press-to-seal attempt on a curved object with a lateral aim offset.
    Under real contact the cup slides off the sphere and shoves it away;
    the seal check must then fail. Returns (displacement_m, sealed)."""
    d = mujoco.MjData(m)
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
    cup_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")
    mujoco.mj_forward(m, d)
    p0 = d.xpos[obj_bid].copy()
    top = p0[2] + 0.030                       # sphere radius
    aim = [p0[0] + lateral, p0[1]]

    dik = mujoco.MjData(m)
    q_hover, _ = ik(m, dik, "tcp", [aim[0], aim[1], top + HOVER], R_DOWN, Q_START)
    q_grasp, _ = ik(m, dik, "tcp", [aim[0], aim[1], top + CUP_R], R_DOWN, q_hover)
    q_press, _ = ik(m, dik, "tcp", [aim[0], aim[1], top + CUP_R - PRESS_M],
                    R_DOWN, q_grasp)
    qref = []
    _seg(qref, Q_START, q_hover, 2.0)
    _seg(qref, q_hover, q_grasp, 1.5)
    _seg(qref, q_grasp, q_press, 0.8)
    i_press_end = len(qref)
    _seg(qref, q_press, q_press, 0.4)
    qref = np.array(qref)
    qdref, tau_ff, kp, kd = _gains_and_ff(m, d, qref)
    sched = dict(qref=qref, qdref=qdref, tau_ff=tau_ff, kp=kp, kd=kd)

    d.qpos[:6] = Q_START
    mujoco.mj_forward(m, d)
    sealed = False
    for k in range(len(qref)):
        if k == i_press_end:
            f_now, tilt = _cup_contact(m, d, cup_gid)
            sealed = f_now > SEAL_N and tilt < SEAL_DEG
        for _ in range(NSUB):
            d.ctrl[:6] = pd_tau(sched, k, d.qpos[:6], d.qvel[:6])
            mujoco.mj_step(m, d)
        if frame_cb is not None:
            frame_cb(d, k, -1)
    disp = float(np.linalg.norm(d.xpos[obj_bid][:2] - p0[:2]))
    print(f"[press-test] curved {obj}, aim offset {lateral*1000:.0f} mm: "
          f"object displaced {disp*1000:.1f} mm, seal "
          f"{'OK (unexpected!)' if sealed else 'FAILED as expected'}")
    return disp, sealed


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    sched = build_schedule(m)
    res = run_cpu(m, sched)
    if res["weld_tick"] < 0:
        print("[cpu] FAIL: seal check failed on the flat object"); sys.exit(1)
    if not res["in_bin"]:
        print("[cpu] FAIL: object not in bin"); sys.exit(1)

    disp, sealed = press_test(m)

    # -------- Phase B: mujoco_warp, closed-loop batch with the same schedule
    import time
    import warp as wp
    import mujoco_warp as mjw
    wp.init()
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
    eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{OBJ}")
    d0 = mujoco.MjData(m)
    d0.qpos[:6] = Q_START
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
    passed = (ok == NWORLD and qerr < 0.05 and res["track_max_deg"] < 3.0
              and res["track_crit_deg"] < 1.0 and res["press_force"] > 1.0
              and disp > 0.010 and not sealed)
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

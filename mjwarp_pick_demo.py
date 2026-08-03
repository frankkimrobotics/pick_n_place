"""mjwarp_pick_demo :: validate robot_warp.xml under real dynamics + GPU batch.

Phase A (CPU mujoco, mj_step): drive the position actuators through a gentle
pick cycle on object0 -- hover, slow descend until the tip rangefinder reads
contact, activate the suction weld (eq_data relpose = tcp->object pose at
grasp, eq_active on), lift, swing over the bin, descend, release, retreat.
Records the per-step ctrl trajectory + the weld schedule.

Phase B (mujoco_warp, GPU): replay the identical ctrl/weld schedule in a
batch of NWORLD identical worlds and check every world lands object0 in the
bin, and that world0 joint angles track the CPU run.

Run in the mjwarp env:
    conda activate mjwarp && python mjwarp_pick_demo.py
"""
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "outputs", "mujoco_sim", "robot_warp.xml")

NWORLD = 32
DT = 0.002
BIN_XY = np.array([0.10, 0.40])
BIN_HALF = 0.15
OBJ = "object0"
OBJ_TOP = 0.060                 # cylinder pos z 0.030 + half height 0.030
HOVER = 0.12                    # tip clearance above object top for hover
CONTACT_MM = 8.0                # rangefinder threshold to latch suction

R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])  # tcp z-axis down


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
                             m.jnt_range[:6, 0], m.jnt_range[:6, 1])
    return d.qpos[:6].copy(), np.linalg.norm(perr), np.linalg.norm(rerr)


def ik(m, d, site, target_p, target_R, q0, restarts=20):
    """Multi-start damped least-squares 6D IK on the tcp site, joints 0..5."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    rng = np.random.default_rng(0)
    best = (None, np.inf, np.inf)
    seeds = [np.asarray(q0, float)]
    lo, hi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]
    seeds += [rng.uniform(np.maximum(lo, -2.0), np.minimum(hi, 2.0))
              for _ in range(restarts)]
    for s in seeds:
        q, pe, re = _ik_once(m, d, sid, np.asarray(target_p, float), target_R, s)
        if pe + 0.3 * re < best[1] + 0.3 * best[2]:
            best = (q, pe, re)
        if pe < 1e-4 and re < 1e-3:
            break
    return best[0], best[1]


def seg(ctrl_list, q_from, q_to, seconds):
    n = max(2, int(seconds / DT))
    for a in np.linspace(0.0, 1.0, n):
        ctrl_list.append((1 - a) * q_from + a * q_to)


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    tcp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
    eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{OBJ}")
    rf_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "tip_range")]
    obj_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj0_free")]

    obj_p = d.qpos[obj_qadr:obj_qadr + 3].copy()
    mujoco.mj_forward(m, d)
    obj_p = m.body_pos[obj_bid].copy() if False else d.xpos[obj_bid].copy()

    # -------- waypoints (IK on a scratch data so the sim state stays clean)
    dik = mujoco.MjData(m)
    q_home = np.array([0.0, -0.5, -1.0, 0.0, -1.1, 0.0])
    q_hover, e1 = ik(m, dik, "tcp", obj_p + [0, 0, OBJ_TOP - obj_p[2] + HOVER],
                     R_DOWN, q_home)
    q_touch, e2 = ik(m, dik, "tcp", obj_p + [0, 0, OBJ_TOP - obj_p[2] + 0.002],
                     R_DOWN, q_hover)
    q_bin, e3 = ik(m, dik, "tcp", [BIN_XY[0], BIN_XY[1], 0.30], R_DOWN, q_hover)
    q_drop, e4 = ik(m, dik, "tcp", [BIN_XY[0], BIN_XY[1], 0.20], R_DOWN, q_bin)
    print(f"[ik] residuals hover {e1:.4f} touch {e2:.4f} bin {e3:.4f} drop {e4:.4f} m")

    # -------- ctrl schedule: gentle approach (slow descend), lift, place
    ctrl = []
    seg(ctrl, q_home, q_hover, 2.0)
    seg(ctrl, q_hover, q_touch, 2.5)          # gentle: 2.5 s descend
    i_descend_end = len(ctrl)
    seg(ctrl, q_touch, q_touch, 0.3)          # dwell at contact
    seg(ctrl, q_touch, q_hover, 1.5)          # lift
    seg(ctrl, q_hover, q_bin, 2.5)            # carry
    seg(ctrl, q_bin, q_drop, 1.0)             # move down over bin
    i_release = len(ctrl)                     # release here
    seg(ctrl, q_drop, q_bin, 1.0)             # retreat
    seg(ctrl, q_bin, q_bin, 1.5)              # settle
    ctrl = np.array(ctrl)
    print(f"[plan] {len(ctrl)} steps ({len(ctrl)*DT:.1f} s)")

    # -------- Phase A: CPU dynamics with rangefinder-triggered weld
    d = mujoco.MjData(m)
    d.qpos[:6] = q_home
    d.ctrl[:] = q_home
    mujoco.mj_forward(m, d)
    weld_step = -1
    eq_data_at_weld = None
    qpos_log = np.zeros((len(ctrl), 6))
    for i, c in enumerate(ctrl):
        d.ctrl[:6] = c
        if weld_step < 0 and i < i_release:
            rng = d.sensordata[rf_adr]
            if 0 <= rng * 1000.0 < CONTACT_MM and i <= i_descend_end + 200:
                # latch suction: weld relpose = current tcp->object pose
                p1 = d.xpos[tcp_bid]; R1 = d.xmat[tcp_bid].reshape(3, 3)
                p2 = d.xpos[obj_bid]; R2 = d.xmat[obj_bid].reshape(3, 3)
                rp = R1.T @ (p2 - p1)
                rq = np.zeros(4)
                mujoco.mju_mat2Quat(rq, (R1.T @ R2).ravel())
                m.eq_data[eq_id, :3] = 0
                m.eq_data[eq_id, 3:6] = rp
                m.eq_data[eq_id, 6:10] = rq
                d.eq_active[eq_id] = 1
                weld_step = i
                eq_data_at_weld = m.eq_data[eq_id].copy()
                print(f"[cpu] contact at step {i} (t={i*DT:.2f}s, range {rng*1000:.1f} mm) -> suction ON")
        if i == i_release and weld_step >= 0:
            d.eq_active[eq_id] = 0
            print(f"[cpu] release at step {i} (t={i*DT:.2f}s) -> suction OFF")
        mujoco.mj_step(m, d)
        qpos_log[i] = d.qpos[:6]
    if weld_step < 0:
        print("[cpu] FAIL: never reached contact"); sys.exit(1)
    fp = d.xpos[obj_bid]
    in_bin = (abs(fp[0] - BIN_XY[0]) < BIN_HALF and abs(fp[1] - BIN_XY[1]) < BIN_HALF
              and fp[2] < 0.10)
    print(f"[cpu] {OBJ} final pos {np.round(fp, 3)}  in_bin={in_bin}")
    if not in_bin:
        print("[cpu] FAIL: object not in bin"); sys.exit(1)

    # -------- Phase B: mujoco_warp batched replay of the same schedule
    import warp as wp
    import mujoco_warp as mjw
    wp.init()
    d0 = mujoco.MjData(m)
    d0.qpos[:6] = q_home
    d0.ctrl[:] = q_home
    m.eq_data[eq_id, :] = eq_data_at_weld     # bake relpose; weld starts inactive
    mujoco.mj_forward(m, d0)
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, d0, nworld=NWORLD)

    ctrl_shape = dw.ctrl.shape
    eq_shape = dw.eq_active.shape
    on = np.ones(eq_shape, dtype=np.bool_ if dw.eq_active.dtype == wp.bool else np.int32)
    off = np.zeros_like(on)
    base_eq = dw.eq_active.numpy().copy()

    import time
    t0 = time.time()
    for i, c in enumerate(ctrl):
        cc = np.tile(c, (ctrl_shape[0], 1)).astype(np.float32 if dw.ctrl.dtype == wp.float32 else np.float64)
        dw.ctrl.assign(cc.reshape(ctrl_shape))
        if i == weld_step:
            e = base_eq.copy(); e[:, eq_id] = 1
            dw.eq_active.assign(e)
        if i == i_release:
            e = base_eq.copy(); e[:, eq_id] = 0
            dw.eq_active.assign(e)
        mjw.step(mw, dw)
    wp.synchronize()
    dt_wall = time.time() - t0
    steps = len(ctrl) * NWORLD
    print(f"[warp] {NWORLD} worlds x {len(ctrl)} steps in {dt_wall:.1f}s "
          f"({steps/dt_wall/1000:.0f}k env-steps/s)")

    xpos = dw.xpos.numpy()                      # (nworld, nbody, 3)
    qpos = dw.qpos.numpy()
    ok = 0
    for w in range(NWORLD):
        p = xpos[w, obj_bid]
        if (abs(p[0] - BIN_XY[0]) < BIN_HALF and abs(p[1] - BIN_XY[1]) < BIN_HALF
                and p[2] < 0.10):
            ok += 1
    qerr = np.abs(qpos[0, :6] - qpos_log[-1]).max()
    print(f"[warp] in_bin {ok}/{NWORLD}   world0 final joint err vs CPU {qerr:.4f} rad")
    print("PASS" if ok == NWORLD and qerr < 0.05 else "FAIL")
    sys.exit(0 if ok == NWORLD and qerr < 0.05 else 1)


if __name__ == "__main__":
    main()

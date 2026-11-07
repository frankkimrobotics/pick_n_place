#!/usr/bin/env python3
"""real_wrench_ctrl :: run the wrench_extract policy (policy_final_12of12,
33-D canonical obs, 7-D action) on the real Pro 630 through the proven
real_policy_ctrl plumbing (PiLink stream law, torque guard, scripted legs).

Sim -> real obs mapping (canonical obs make this direct):
  ac (array centre)      := measured object xy  (single object = 1x1 array)
  box_c (drop target)    := --goal xy
  bel_h                  := measured object height (touch-calib or --obj z*2)
  bel_nb                 := --nb_h (no neighbours: 0.03 -> clears right away)
  wrench dims            := nominal sim hanging stats when sealed, else idle
  sealed                 := executor attach emulation (press + pump + dwell)
  cleared                := sealed & believed bottom above believed nb top
Release gating mirrors the sim env: pump may switch off ONLY when cleared
and the object xy is within 0.12 m of the goal.

Safety: torque guard (hold at firm, abort at hard), Guard box + z floor,
--dq_scale 0.5 default for the first runs, --touch_only = never suction,
SIGINT -> stop stream + suction off.

    python3 real_wrench_ctrl.py --obj X Y Z_CENTRE --half hx hy hz \
        --goal GX GY --touch_only --exec
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from real_policy_ctrl import (  # noqa: E402
    PiLink, Guard, ObsBuilder, scripted_move,
    CTRL_DT, CUP_R, START_Q, STREAM_GAINS, LEAD_MAX_DEG,
    TAU_FIRM, TAU_HARD, TAU_PRESS, TAU_ABORT, TAU_ABORT_ATTACHED,
    ATTACH_AFTER, SCRIPT_VMAX_DEG,
)

OBS_DIM, EP = 33, 165
DQ_MAX_RAD = math.radians(3.2)        # wrench_extract training clamp
OBS_NOISE = 0.005                     # training DR noise; clones stall without it
F_V = 18.0
BOX_GATE = 0.12                       # release allowed within this xy of goal
SETTLE_S = 1.0


def load_wrench_policy(stem, h=512):
    import torch
    from wrench_extract import AC
    ac = AC(h=h)
    ac.load_state_dict(torch.load(stem + ".pt", map_location="cpu",
                                  weights_only=True))
    ac.eval()
    print(f"[policy] torch {stem}.pt (h={h})")

    def f(obs):
        with torch.no_grad():
            return torch.tanh(
                ac.pi(torch.as_tensor(obs, dtype=torch.float32)[None]))[0].numpy()
    return f


class WrenchObs:
    """33-D obs exactly as ExtractEnv.observe() builds it."""

    def __init__(self, ob, ac_xy, box_xy, bel_h, bel_nb, half_z):
        self.ob = ob                        # ObsBuilder (FK)
        self.ac = np.asarray(ac_xy, float)
        self.box = np.asarray(box_xy, float)
        self.bel_h = float(bel_h)
        self.bel_nb = float(bel_nb)
        self.half_z = float(half_z)
        self.rng = np.random.default_rng(0)

    def build(self, q, qd, p_obj, sealed, t_step):
        tcp, R = self.ob.fk(q)
        tcp_rel = tcp - np.array([self.ac[0], self.ac[1], 0.0])
        if sealed:
            # sim hanging stats (wr_last units): pull ~ mg, tiny shear
            wr = [0.7 / F_V, 0.02, 0.0, 0.0, 0.92]
        else:
            wr = [0.0, 0.0, 0.0, 0.0, 1.0]
        clear_bel = (p_obj[2] - self.bel_h / 2) - (self.bel_nb + 0.005)
        parts = [q, qd, tcp_rel, R[:, 2], p_obj - tcp,
                 self.box - tcp[:2],
                 [self.bel_h, self.bel_nb, clear_bel],
                 wr, [1.0 if sealed else 0.0, t_step / EP]]
        obs = np.concatenate([np.asarray(x, np.float32).reshape(-1)
                              for x in parts])
        assert obs.shape == (OBS_DIM,), obs.shape
        obs = obs + self.rng.normal(0, OBS_NOISE, size=obs.shape).astype(np.float32)
        return obs, tcp, R


def torque_metric(tau, tau0):
    """contact_detector style: weighted change vs rolling baseline."""
    d = np.abs(np.asarray(tau) - np.asarray(tau0))
    return float(d[1] + d[2] * 0.5 + d[3] * 0.5)


def run(a):
    exec_ = a.exec
    link = PiLink(a.pi, exec_, ref_mode="spline")
    time.sleep(1.5)
    q0, _, age = link.state()
    if q0 is None or age > 0.5:
        print("[abort] no feedback from the Pi (:9999)")
        return
    print(f"[fb] q = {np.degrees(q0).round(1)} deg")
    ob = ObsBuilder(os.path.join(HERE, "scenes", "box_med.xml"), a.half)
    p_obj = np.asarray(a.obj, float)
    guard = Guard(ob, p_obj[2] + a.half[2], force=a.force)

    # SIGINT: stop streaming (Pi holds) + suction off
    def _stop(sig, frm):
        print("\n[SIGINT] suction off, stopping")
        try:
            link.suction(False)
        finally:
            link.close()
            sys.exit(1)
    signal.signal(signal.SIGINT, _stop)

    # ---- probe: 1 deg J1 through the stream law (mandatory before moves)
    if exec_:
        qp = q0.copy()
        qp[0] += math.radians(1.0)
        scripted_move(link, guard, ob, qp, "probe +1deg J1", v_peak_deg=5.0)
        time.sleep(0.5)
        q1, _, _ = link.state()
        moved = math.degrees(abs(q1[0] - q0[0]))
        print(f"[probe] J1 moved {moved:.2f} deg")
        if moved < 0.5:
            print("[abort] probe did not move -- drives not following")
            link.close()
            return
        scripted_move(link, guard, ob, q0, "probe return", v_peak_deg=5.0)

    # ---- scripted leg to START_Q (policy's training start)
    if exec_ and np.degrees(np.abs(q0 - START_Q)).max() > 3.0:
        scripted_move(link, guard, ob, START_Q, "go home (START_Q)",
                      v_peak_deg=a.script_vmax)
        time.sleep(0.3)

    policy = load_wrench_policy(a.policy, h=a.h)
    wobs = WrenchObs(ob, p_obj[:2], a.goal[:2], bel_h=2 * a.half[2],
                     bel_nb=a.nb_h, half_z=a.half[2])

    sealed = False
    cleared = False
    released = False
    attach_t = None
    contact_hold = False
    tau_base = None
    q_target, _, _ = link.state()
    q_target = q_target.copy()
    log = []
    t0 = time.time()
    print(f"[run] obj ({p_obj[0]:.3f},{p_obj[1]:.3f},{p_obj[2]:.3f}) "
          f"goal ({a.goal[0]:.3f},{a.goal[1]:.3f}) dq_scale {a.dq_scale} "
          f"touch_only {a.touch_only}")
    for t in range(EP):
        tick = t0 + t * CTRL_DT
        while time.time() < tick:
            time.sleep(0.002)
        q, qd, age = link.state()
        tau = link.torque()
        if q is None or age > 0.5:
            print("[abort] feedback stale")
            break
        if qd is None:
            qd = np.zeros(6)
        if tau_base is None:
            tau_base = np.asarray(tau, float)
        # rolling baseline only while free (not pressing/holding)
        tm = torque_metric(tau, tau_base)
        if not contact_hold and tm < 0.03:
            tau_base = 0.98 * np.asarray(tau_base) + 0.02 * np.asarray(tau)

        # object pose estimate: static until sealed, then rides the cup
        tcp_now, R_now = ob.fk(q)
        if sealed:
            p_est = tcp_now - np.array([0.0, 0.0, CUP_R + a.half[2]])
        else:
            p_est = p_obj
        obs, tcp_now, R_now = wobs.build(q, qd, p_est, sealed,
                                         min(t, EP - 1))
        act = policy(obs)
        margin = 0.92 if sealed else 1.0
        dq = act[:6] * DQ_MAX_RAD * a.dq_scale * margin

        # ---- executor gates (mirror sim env.step) -------------------
        bel_bottom = p_est[2] - wobs.bel_h / 2
        cleared = cleared or (sealed and bel_bottom > wobs.bel_nb + 0.005)
        near_goal = np.linalg.norm(p_est[:2] - np.asarray(a.goal[:2])) < BOX_GATE
        allow_release = cleared and near_goal
        want_off = act[6] <= 0 and allow_release

        # ---- contact / attach emulation ----------------------------
        if not sealed and tm >= TAU_FIRM:
            contact_hold = True
            if a.touch_only:
                print(f"[touch] firm contact t={t} tm={tm:.3f} -- touch-only: retract")
                break
            if attach_t is None:
                attach_t = time.time()
                if exec_:
                    link.suction(True)
                print(f"[press] contact t={t} tm={tm:.3f}, pump on, dwell {ATTACH_AFTER}s")
            # hold position while vacuum builds: freeze dq z-descent
            dq = dq * 0.0
            if time.time() - attach_t >= ATTACH_AFTER:
                sealed = True
                contact_hold = False
                print(f"[attach] sealed at t={t}")
        abort_th = TAU_ABORT_ATTACHED if sealed else TAU_ABORT
        if tm >= abort_th:
            print(f"[abort] torque {tm:.3f} >= {abort_th}")
            break

        # ---- release ------------------------------------------------
        if sealed and want_off and not released:
            if exec_:
                link.suction(False)
            released = True
            print(f"[release] t={t} at ({p_est[0]:.3f},{p_est[1]:.3f},{p_est[2]:.3f})")
            break

        # ---- stream the decision ------------------------------------
        q_new = q_target + dq
        viol = guard.check_q(q_new)
        if viol:
            guard.n_clip += 1
            q_new = q_target        # hold rather than cross a limit
            if guard.n_clip % 20 == 1:
                print(f"[guard] {viol[0]}")
        # bounded reference lead over the measured joints
        q_send = q + np.clip(q_new - q, -math.radians(LEAD_MAX_DEG),
                             math.radians(LEAD_MAX_DEG))
        if exec_:
            link.send_segment(q_target, q_send, tick, dt=CTRL_DT, seq=t)
        q_target = q_new
        log.append(dict(t=t, q=np.degrees(q).round(2).tolist(), tm=round(tm, 3),
                        sealed=sealed, cleared=bool(cleared),
                        act6=round(float(act[6]), 2),
                        tcp=np.round(tcp_now, 3).tolist()))

    # ---- wrap up: settle, then scripted retract + home ---------------
    time.sleep(SETTLE_S)
    if exec_:
        if not a.keep_suction:
            link.suction(False)
        scripted_move(link, guard, ob, START_Q, "home", v_peak_deg=a.script_vmax)
    if a.log:
        json.dump(log, open(a.log, "w"))
        print(f"[log] {a.log} ({len(log)} steps)")
    link.close()
    print(f"[done] sealed={sealed} cleared={bool(cleared)} released={released}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.expanduser(
        "~/pnp_rl/wrench_extract/policy_final_12of12"))
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--pi", default="192.168.50.2")
    ap.add_argument("--obj", type=float, nargs=3, required=True,
                    help="object CENTRE in base frame (m)")
    ap.add_argument("--half", type=float, nargs=3, default=[0.025, 0.025, 0.025])
    ap.add_argument("--goal", type=float, nargs=2, required=True,
                    help="drop target xy (m)")
    ap.add_argument("--nb_h", type=float, default=0.03,
                    help="believed neighbour max height (single object: low)")
    ap.add_argument("--dq_scale", type=float, default=0.5,
                    help="action scale on top of the 3.2 deg clamp (first runs 0.5 = 16 deg/s)")
    ap.add_argument("--script_vmax", type=float, default=SCRIPT_VMAX_DEG)
    ap.add_argument("--touch_only", action="store_true")
    ap.add_argument("--keep_suction", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--log", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

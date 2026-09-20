#!/usr/bin/env python3
"""touch_verify :: slow straight-down touch on a detected object top with the torque contact guard.
Validates the camera-derived object height / xy against the robot (FK of the contact pose) before
running the policy. Hover above the point, descend at 2 cm/s, stop at firm contact, retract, go home.
  $PY rl/touch_verify.py --xy 0.374 0.082 --z_top 0.045 --exec
"""
import argparse, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from real_policy_ctrl import ObsBuilder, PiLink, Guard, START_Q, TAU_FIRM, TAU_HARD, W_J3, CTRL_DT  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xy", type=float, nargs=2, required=True)
    ap.add_argument("--z_top", type=float, required=True, help="expected top height (m), from the camera")
    ap.add_argument("--hover", type=float, default=0.06, help="start the descent this far above z_top")
    ap.add_argument("--speed", type=float, default=0.02, help="descent speed m/s")
    ap.add_argument("--pi", default="192.168.50.2")
    ap.add_argument("--exec", action="store_true")
    a = ap.parse_args()
    import mujoco
    demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    xml = os.path.join(HERE, "scenes", "box_med.xml")
    ob = ObsBuilder(xml, [0.04, 0.04, 0.02]); dik = mujoco.MjData(ob.m)
    guard = Guard(ob, a.z_top)
    link = PiLink(a.pi, a.exec)
    t0 = time.time()
    while not link.ok() and time.time() - t0 < 6: time.sleep(0.1)
    q, qd, age = link.state()
    if q is None: raise SystemExit("no feedback")
    tip0, _ = ob.fk(q); print(f"[touch] start q {np.round(np.degrees(q),1).tolist()} tip {np.round(tip0,3).tolist()}")
    def ik_to(p, seed):
        qq, err = demo["ik"](ob.m, dik, "tcp", [float(x) for x in p], demo["R_DOWN"], seed)
        if err > 0.005: raise SystemExit(f"IK failed for {p}: err {err:.4f}")
        return np.array(qq)
    def move_joint(q_from, q_to, deg_per_s=5.0, label=""):
        T = max(0.5, np.degrees(np.abs(q_to - q_from)).max() / deg_per_s); n = int(T / 0.1) + 1
        qs = [q_from + (q_to - q_from) * i / (n - 1) for i in range(n)]
        for qq in qs:
            v = guard.check_q(qq)
            if v: raise SystemExit(f"[guard] path {label} violates: {v}")
        print(f"[touch] {label}: {T:.1f} s{'' if a.exec else ' [dry]'}"); link.send_path(qs, 0.1, time.time() + 0.2); time.sleep(T + 0.8)
    p_hover = np.array([a.xy[0], a.xy[1], a.z_top + a.hover])
    q_hover = ik_to(p_hover, q)
    tip_h, R = ob.fk(q_hover); print(f"[touch] hover pose q {np.round(np.degrees(q_hover),1).tolist()} tip {np.round(tip_h,3).tolist()} axis {np.round(R[:,2],3).tolist()}")
    move_joint(q, q_hover, 6.0, "to hover")
    time.sleep(1.0)
    tau_base = link.torque_baseline(1.0); print(f"[touch] torque baseline at hover {np.round(tau_base,3).tolist()}")
    q_cur, _, _ = link.state(); tip, _ = ob.fk(q_cur); print(f"[touch] at hover: tip {np.round(tip,3).tolist()} (cmd {np.round(tip_h,3).tolist()})")
    # descend
    q_prev = q_cur.copy(); z = p_hover[2]; contact = None; k = 0; t_start = time.time()
    while z > a.z_top - 0.03:
        t_dec = t_start + k * CTRL_DT; d = t_dec - time.time()
        if d > 0: time.sleep(d)
        q_meas, _, _ = link.state(); tq = link.torque()
        tau = abs(tq[1] - tau_base[1]) + W_J3 * abs(tq[2] - tau_base[2])
        tip_m, _ = ob.fk(q_meas)
        if tau >= TAU_FIRM:
            contact = (tip_m.copy(), tau); print(f"[touch] CONTACT tau {tau:.3f} at tip {np.round(tip_m,4).tolist()} (cmd z {z:.4f})"); break
        z -= a.speed * CTRL_DT
        q_next = ik_to([a.xy[0], a.xy[1], z], q_prev)
        if guard.check_q(q_next): print("[guard]", guard.check_q(q_next)); break
        if a.exec: link.send_segment(q_prev, q_next, t_dec, seq=k)
        if k % 5 == 0: print(f"[touch] k={k} cmd z {z:.3f} meas tip z {tip_m[2]:.3f} tau {tau:.3f}")
        q_prev = q_next; k += 1
    if contact is None: print("[touch] no contact down to z", round(z, 3))
    # retract + home
    q_meas, _, _ = link.state(); tip_m, _ = ob.fk(q_meas)
    q_up = ik_to([a.xy[0], a.xy[1], tip_m[2] + 0.05], q_meas); move_joint(q_meas, q_up, 6.0, "retract")
    q_meas, _, _ = link.state(); move_joint(q_meas, START_Q, 6.0, "home")
    q_meas, _, _ = link.state(); print(f"[touch] final q {np.round(np.degrees(q_meas),1).tolist()}")
    if contact is not None:
        print(f"[touch] RESULT: contact z {contact[0][2]:.4f} vs camera top {a.z_top:.4f} (diff {1000*(contact[0][2]-a.z_top):+.1f} mm); contact xy ({contact[0][0]:.3f},{contact[0][1]:.3f})")
    link.close()
if __name__ == "__main__": main()

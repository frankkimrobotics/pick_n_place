#!/usr/bin/env python3
"""edge_calib :: measure a cylinder's TRUE centre with the arm (torque-guarded rim probing) and compare it with the
D435 tracker's estimate. Descends at the estimate to find the top, then bisects along +x/-x/+y/-y for the rim
(contact above top-12 mm = still on the object). Prints the centre, the radius and the tracker offset."""
import os, sys, time, json, socket, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.argv_backup = sys.argv; sys.argv = ['x']
import real_policy_ctrl as rc
import mujoco

def main():
    sys.argv = sys.argv_backup
    ap = argparse.ArgumentParser(); ap.add_argument("--obj", type=float, nargs=3, required=True, help="tracker cx cy top (raw, no bias)")
    ap.add_argument("--rmax", type=float, default=0.045); ap.add_argument("--tol", type=float, default=0.002); ap.add_argument("--exec", action="store_true")
    a = ap.parse_args()
    cx, cy, top = a.obj
    ob = rc.ObsBuilder(os.path.join(HERE, "scenes", "box_med.xml"), [0.04, 0.04, top / 2])
    guard = rc.Guard(ob, top, force=True)
    link = rc.PiLink("192.168.50.2", a.exec, ref_mode="spline")
    t = time.time()
    while not link.ok() and time.time() - t < 6: time.sleep(0.1)
    demo = {"__file__": os.path.join(os.path.dirname(HERE), "mjwarp_pick_demo.py")}; exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    dik = mujoco.MjData(ob.m); ik = lambda p, q0: demo["ik"](ob.m, dik, "tcp", [float(p[0]), float(p[1]), float(p[2])], demo["R_DOWN"], q0)

    def move(p, v=8.0):
        q0, _, _ = link.state(); q1, e = ik(p, q0); q1 = np.array(q1)
        if e > 0.005 or guard.check_q(q1): raise SystemExit(f"IK/guard failed at {p}: err {e:.4f} {guard.check_q(q1)}")
        T = max(0.6, np.degrees(np.abs(q1 - q0)).max() / v); n = int(T / 0.1) + 1
        link.send_path([q0 + (q1 - q0) * i / (n - 1) for i in range(n)], 0.1, time.time() + 0.2); time.sleep(T + 1.0)
        return np.array(link.state()[0])

    def probe(x, y, z_from, z_stop, base):
        """descend at 2 cm/s from z_from to z_stop; return contact tip z or None"""
        qprev = move([x, y, z_from]); z = z_from; t_s = time.time(); k = 0; hit = None
        while z > z_stop:
            t_dec = t_s + k * rc.CTRL_DT; d = t_dec - time.time()
            if d > 0: time.sleep(d)
            qm, _, _ = link.state(); tq = link.torque(); tau = abs(tq[1] - base[1]) + rc.W_J3 * abs(tq[2] - base[2]); tipm, _ = ob.fk(qm)
            if tau >= rc.TAU_FIRM: hit = float(tipm[2]); break
            z -= 0.02 * rc.CTRL_DT
            qn, e2 = ik([x, y, z], qprev); qn = np.array(qn)
            if e2 > 0.005 or guard.check_q(qn): break
            link.send_segment(qprev, qn, t_dec, seq=k); qprev = qn; k += 1
        time.sleep(0.15)
        return hit

    q, _, _ = link.state(); print("[edge] start q(deg)", np.round(np.degrees(q), 1).tolist())
    z_hover = top + 0.06
    move([cx, cy, z_hover]); base = link.torque_baseline(0.8)
    z_top = probe(cx, cy, z_hover, top - 0.03, base)
    if z_top is None: raise SystemExit("[edge] no contact at the estimate -> the estimate is off by more than the object radius")
    top_m = z_top - rc.CUP_R; print(f"[edge] top by touch {top_m:.3f} (camera {top:.3f})")
    z_clear, z_below = top_m + 0.030, top_m - 0.012
    edges = {}
    for name, ux, uy in (("+x", 1, 0), ("-x", -1, 0), ("+y", 0, 1), ("-y", 0, -1)):
        lo, hi = 0.0, a.rmax                       # lo = on the object, hi = off
        move([cx, cy, z_clear])
        while hi - lo > a.tol:
            r = 0.5 * (lo + hi); x, y = cx + ux * r, cy + uy * r
            base = link.torque_baseline(0.4)
            hit = probe(x, y, z_clear, z_below, base)
            on = hit is not None and hit > z_below + 0.004
            print(f"[edge] {name} r={1000*r:.0f} mm -> {'ON' if on else 'off'}{'' if hit is None else ' (contact z %.3f)' % hit}")
            if on: lo = r
            else: hi = r
            move([x, y, z_clear], v=10.0)
        edges[name] = 0.5 * (lo + hi)
    x_c = cx + (edges["+x"] - edges["-x"]) / 2; y_c = cy + (edges["+y"] - edges["-y"]) / 2
    rx = (edges["+x"] + edges["-x"]) / 2 - rc.CUP_R; ry = (edges["+y"] + edges["-y"]) / 2 - rc.CUP_R
    print(f"[edge] rim (from estimate): +x {1000*edges['+x']:.0f} -x {1000*edges['-x']:.0f} +y {1000*edges['+y']:.0f} -y {1000*edges['-y']:.0f} mm")
    print(f"[edge] TRUE centre ({x_c:.4f}, {y_c:.4f})  radius x {1000*rx:.0f} / y {1000*ry:.0f} mm   tracker raw ({cx:.4f}, {cy:.4f}) -> offset tracker-true dx {1000*(cx-x_c):+.1f} dy {1000*(cy-y_c):+.1f} mm")
    json.dump(dict(true=[x_c, y_c], est=[cx, cy], top_touch=top_m, top_cam=top, edges=edges), open(os.path.expanduser("~/pnp_rl/edge_calib_last.json"), "w"))
    move([x_c, y_c, z_hover])
    q0, _, _ = link.state(); T = max(1.0, np.degrees(np.abs(rc.START_Q - q0)).max() / 8.0); n = int(T / 0.1) + 1
    link.send_path([q0 + (rc.START_Q - q0) * i / (n - 1) for i in range(n)], 0.1, time.time() + 0.2); time.sleep(T + 1.2)
    print("[edge] home; dev %.1f deg" % np.degrees(np.abs(np.array(link.state()[0]) - rc.START_Q)).max()); link.close()

if __name__ == "__main__":
    main()

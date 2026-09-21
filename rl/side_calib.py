#!/usr/bin/env python3
"""side_calib :: true centre of a cylinder by SIDE contacts (torque-guarded lateral sweeps below the lid), compared with
the D435 tracker's RAW estimate. Four sweeps (+x -x +y -y) from 6 cm out toward the estimate at 1.5 cm/s at lid-15 mm;
stop when |dtau1| + |dtau2| + 0.5|dtau3| >= thr (j1 carries the y-direction contact, j2/j3 the x-direction)."""
import os, sys, time, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); _argv = sys.argv; sys.argv = ['x']
import real_policy_ctrl as rc
import mujoco

def main():
    sys.argv = _argv
    ap = argparse.ArgumentParser(); ap.add_argument("--obj", type=float, nargs=3, required=True, help="tracker RAW cx cy top")
    ap.add_argument("--r0", type=float, default=0.05); ap.add_argument("--thr", type=float, default=0.06); ap.add_argument("--exec", action="store_true")
    a = ap.parse_args(); cx, cy, top = a.obj
    ob = rc.ObsBuilder(os.path.join(HERE, "scenes", "box_med.xml"), [0.04, 0.04, top / 2]); guard = rc.Guard(ob, top, force=True); guard.z_floor = 0.015
    link = rc.PiLink("192.168.50.2", a.exec, ref_mode="spline"); t = time.time()
    while not link.ok() and time.time() - t < 6: time.sleep(0.1)
    demo = {"__file__": os.path.join(os.path.dirname(HERE), "mjwarp_pick_demo.py")}; exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    dik = mujoco.MjData(ob.m); ik = lambda p, q0: demo["ik"](ob.m, dik, "tcp", [float(p[0]), float(p[1]), float(p[2])], demo["R_DOWN"], q0)
    def move(p, v=8.0):
        q0, _, _ = link.state(); q1, e = ik(p, q0); q1 = np.array(q1)
        if e > 0.005 or guard.check_q(q1): raise SystemExit(f"IK/guard failed at {np.round(p,3)}: err {e:.4f} {guard.check_q(q1)}")
        T = max(0.6, np.degrees(np.abs(q1 - q0)).max() / v); n = int(T / 0.1) + 1
        link.send_path([q0 + (q1 - q0) * i / (n - 1) for i in range(n)], 0.1, time.time() + 0.2); time.sleep(T + 0.9)
        return np.array(link.state()[0])
    z_side, z_clear = top - 0.015, top + 0.05
    r = {}
    for name, ux, uy in (("+x", 1, 0), ("-x", -1, 0), ("+y", 0, 1), ("-y", 0, -1)):
        start = [cx + ux * a.r0, cy + uy * a.r0]
        move([start[0], start[1], z_clear]); qprev = move([start[0], start[1], z_side], v=6.0)
        base = link.torque_baseline(0.5); t_s = time.time(); k = 0; s = 0.0; hit = None
        while s < a.r0 - 0.005:
            t_dec = t_s + k * rc.CTRL_DT; d = t_dec - time.time()
            if d > 0: time.sleep(d)
            qm, _, _ = link.state(); tq = link.torque(); tau = abs(tq[0] - base[0]) + abs(tq[1] - base[1]) + 0.5 * abs(tq[2] - base[2]); tipm, _ = ob.fk(qm)
            if tau >= a.thr and s > 0.008: hit = (float(tipm[0]), float(tipm[1])); break     # armed after the start transient (8 mm)
            s += 0.015 * rc.CTRL_DT
            p = [start[0] - ux * s, start[1] - uy * s, z_side]; qn, e2 = ik(p, qprev); qn = np.array(qn)
            if e2 > 0.005 or guard.check_q(qn): break
            link.send_segment(qprev, qn, t_dec, seq=k); qprev = qn; k += 1
        time.sleep(0.1)
        if hit is None: print(f"[side] {name}: no contact within {1000*a.r0:.0f} mm"); r[name] = None
        else:
            r[name] = (hit[0] - cx) * ux + (hit[1] - cy) * uy
            print(f"[side] {name}: wall at {1000*r[name]:.1f} mm from the estimate (tip {np.round(hit,4).tolist()}, tau {tau:.3f})")
        qm, _, _ = link.state(); tipm, _ = ob.fk(qm)
        back = [tipm[0] + ux * 0.01, tipm[1] + uy * 0.01]                     # 1 cm back off the wall, then straight up
        try:
            move([back[0], back[1], z_side], v=6.0)
        except SystemExit:
            pass
        move([back[0], back[1], z_clear])
    out = dict(est=[cx, cy], top=top, r=r)
    if all(r.get(k) is not None for k in ("+x", "-x", "+y", "-y")):
        xc = cx + (r["+x"] - r["-x"]) / 2; yc = cy + (r["+y"] - r["-y"]) / 2
        rad = ((r["+x"] + r["-x"]) / 2 - rc.CUP_R, (r["+y"] + r["-y"]) / 2 - rc.CUP_R)
        print(f"[side] TRUE centre ({xc:.4f}, {yc:.4f}); radius x/y {1000*rad[0]:.0f}/{1000*rad[1]:.0f} mm; RAW estimate - true: dx {1000*(cx-xc):+.1f} dy {1000*(cy-yc):+.1f} mm")
        out.update(true=[xc, yc], radius=rad, err=[cx - xc, cy - yc])
    json.dump(out, open(os.path.expanduser(f"~/pnp_rl/side_calib_{int(time.time())}.json"), "w"))
    q0, _, _ = link.state(); move([cx, cy, z_clear + 0.05])
    q0, _, _ = link.state(); T = max(1.0, np.degrees(np.abs(rc.START_Q - q0)).max() / 8.0); n = int(T / 0.1) + 1
    link.send_path([q0 + (rc.START_Q - q0) * i / (n - 1) for i in range(n)], 0.1, time.time() + 0.2); time.sleep(T + 1.2); print("[side] home"); link.close()

if __name__ == "__main__":
    main()

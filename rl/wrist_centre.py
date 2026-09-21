#!/usr/bin/env python3
"""wrist_centre :: measure object centres with the WRIST D405 from a hover (ground truth for the D435 tracker).
For each object (tracker RAW cx cy top): hover the cup 12 cm above the lid, grab a 5-frame median of the aligned
depth, deproject with the colour intrinsics, transform to the base frame with fk(tcp) @ shift @ T_TCP_CAM (config.py
hand-eye, 2026-07-03), take the points inside a 7 cm box around the estimate whose height is within 12 mm of the lid,
and report their centroid as the true centre. Writes ~/pnp_rl/wrist_centre_<t>.json with raw/true pairs."""
import os, sys, time, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); _argv = sys.argv; sys.argv = ['x']
import real_policy_ctrl as rc
import config as C
import mujoco
import cv2

def main():
    sys.argv = _argv
    ap = argparse.ArgumentParser(); ap.add_argument("--objs", nargs="+", required=True, help="cx,cy,top per object (tracker RAW)")
    ap.add_argument("--hover", type=float, default=0.12); ap.add_argument("--exec", action="store_true"); ap.add_argument("--serial", default="218622271300")
    a = ap.parse_args()
    objs = [tuple(float(v) for v in o.split(",")) for o in a.objs]
    import subprocess, tempfile
    GRAB = os.path.join(HERE, "d405_grab.py"); npz = os.path.join(tempfile.gettempdir(), "wrist_grab.npz")
    def grab():
        subprocess.run(["python3", GRAB, a.serial, npz, "5"], check=True, env=dict(os.environ, PYTHONPATH=os.path.expanduser("~/librealsense/build/release")))
        d = np.load(npz); return d["col"], d["dep"], float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"])
    ob = rc.ObsBuilder(os.path.join(HERE, "scenes", "box_med.xml"), [0.04, 0.04, 0.02]); guard = rc.Guard(ob, 0.05, force=True)
    link = rc.PiLink("192.168.50.2", a.exec, ref_mode="spline"); t = time.time()
    while not link.ok() and time.time() - t < 6: time.sleep(0.1)
    demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}; exec(open(demo["__file__"]).read().split("if __name__")[0], demo); dik = mujoco.MjData(ob.m)
    def move(p, v=8.0):
        q0, _, _ = link.state(); q1, e = demo["ik"](ob.m, dik, "tcp", [float(x) for x in p], demo["R_DOWN"], q0); q1 = np.array(q1)
        if e > 0.005 or guard.check_q(q1): raise SystemExit(f"IK/guard failed at {np.round(p, 3)}: {e:.4f} {guard.check_q(q1)}")
        T = max(0.8, np.degrees(np.abs(q1 - q0)).max() / v); n = int(T / 0.1) + 1
        link.send_path([q0 + (q1 - q0) * i / (n - 1) for i in range(n)], 0.1, time.time() + 0.2); time.sleep(T + 1.3)
    shift = np.eye(4); shift[2, 3] = C.CAM_TCP_Z_SHIFT
    res = []
    for (cx, cy, top) in objs:
        move([cx, cy, top + rc.CUP_R + a.hover]); time.sleep(0.5)
        q, _, _ = link.state(); tcp, R = ob.fk(q); T_tcp = np.eye(4); T_tcp[:3, :3] = R; T_tcp[:3, 3] = tcp
        Tbc = T_tcp @ shift @ np.array(C.T_TCP_CAM)
        col, dep, fx, fy, cx0, cy0 = grab()
        Tcb = np.linalg.inv(Tbc)
        # lid = the circle (Hough on the grey image) nearest the projected estimate; the D405 depth on lids is holey
        pc = Tcb @ np.array([cx, cy, top, 1.0]); u_est, v_est = fx * pc[0] / pc[2] + cx0, fy * pc[1] / pc[2] + cy0
        dist = float(pc[2]); r_px = 0.033 * fx / dist
        g = cv2.GaussianBlur(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
        circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=90, param2=28, minRadius=int(0.5 * r_px), maxRadius=int(1.4 * r_px))
        if circ is None:
            print(f"[wrist] ({cx:.3f},{cy:.3f}): no circle found (expected r {r_px:.0f} px)"); res.append(dict(raw=[cx, cy], top=top, true=None)); continue
        circ = circ[0]; best = min(circ, key=lambda c: np.hypot(c[0] - u_est, c[1] - v_est))
        u_c, v_c, rr = float(best[0]), float(best[1]), float(best[2])
        ray = Tbc[:3, :3] @ np.array([(u_c - cx0) / fx, (v_c - cy0) / fy, 1.0]); o = Tbc[:3, 3]
        s_ = (top - o[2]) / ray[2]; pt = o + s_ * ray; xc, yc = float(pt[0]), float(pt[1]); top_w = top
        print(f"[wrist] raw ({cx:.4f},{cy:.4f}) top {top:.3f} -> circle r {rr:.0f}px (exp {r_px:.0f}) at ({u_c:.0f},{v_c:.0f}) vs est px ({u_est:.0f},{v_est:.0f}) -> TRUE ({xc:.4f},{yc:.4f})  raw-true dx {1000*(cx-xc):+.1f} dy {1000*(cy-yc):+.1f} mm")
        L = np.zeros((1, 3))
        res.append(dict(raw=[cx, cy], top=top, true=[xc, yc], top_w=top_w, tcp=tcp.tolist()))
        dbg = col.copy()
        cv2.circle(dbg, (int(u_est), int(v_est)), 8, (0, 0, 255), 2); cv2.circle(dbg, (int(u_c), int(v_c)), int(rr), (0, 255, 0), 2); cv2.circle(dbg, (int(u_c), int(v_c)), 4, (0, 255, 0), -1)
        cv2.imwrite(os.path.expanduser(f"~/pnp_rl/wrist_centre_{len(res)}.png"), dbg)
    json.dump(res, open(os.path.expanduser(f"~/pnp_rl/wrist_centre_{int(time.time())}.json"), "w"), indent=1)
    q0, _, _ = link.state(); T = max(1.0, np.degrees(np.abs(rc.START_Q - q0)).max() / 8.0); n = int(T / 0.1) + 1
    link.send_path([q0 + (rc.START_Q - q0) * i / (n - 1) for i in range(n)], 0.1, time.time() + 0.2); time.sleep(T + 1.2)
    print("[wrist] home"); link.close()

if __name__ == "__main__":
    main()

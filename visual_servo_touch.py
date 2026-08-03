#!/usr/bin/env python3
"""Visual-servo touch demo: fast approach to a pregrasp, then a SLOW, velocity-continuous
descent during which the wrist D405 (a) re-detects the object each tick and nulls the XY
error (pure horizontal correction) so the cup converges to the object CENTER, and (b)
uses a VISION-BASED contact signal -- the D405 depth GAP (median depth of the ring around
the cup minus the cup rim) -- which shrinks as the object nears the tip. Contact = gap
drops to ~0 (no torque; gravity-drift makes torque unreliable).

  source /opt/ros/humble/setup.bash
  python3 visual_servo_touch.py

Needs planner :9997 + :9994 (online_servo) + SAM3 up; cup_mask.npz; D405 wrist cam.
"""
import argparse, json, os, socket, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc", "session_tools")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, make_T, quat_wxyz_to_R
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg
from object_pointclouds import deproject_mask
from capture_and_plot import segment
from real_multi import detect_objects
from types import SimpleNamespace
import pyrealsense2 as rs

PI = "10.0.0.27"; SER = "218622271300"; W, H = 640, 480     # match cup_mask.npz
DOWN = list(map(float, R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0])))))

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def send_chunk(m):
    k = socket.create_connection((PI, 9994), timeout=3); k.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    k.sendall((json.dumps(m) + "\n").encode()); k.close()
def read_q():
    s = socket.create_connection((PI, 9999), timeout=3); b = b""
    while b"\n" not in b: b += s.recv(4096)
    s.close(); return list(json.loads(b.split(b"\n")[0])["joints_deg"])
def qrad(qd): return [float(v) for v in linuxcnc_deg_to_rad(qd)]
def fk(qd):
    r = rpc({"type": "fk", "q": qrad(qd)}); return np.array(r["pos"][0]), np.array(r["quat"][0])

def stream_to(goal_xyz, quat, Vmax, settle=True):
    qd = read_q()
    r = rpc({"type": "plan_pose", "start_q": qrad(qd), "goal_pose": list(goal_xyz) + list(quat), "max_attempts": 14})
    if not r.get("success"): return False
    traj = np.array(r["trajectory"]); dt = r["dt"]
    peak = np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt if len(traj) > 1 else Vmax
    sdt = dt * max(1.0, peak / Vmax)
    td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
    send_chunk({"trajectory": td, "traj_dt": sdt, "t_anchor": time.time() + 0.12})
    if settle: time.sleep(sdt * (len(traj) - 1) + 1.5)
    return True

def stream_joints(qd_target, dur=1.2, settle=True):
    """Stream a straight joint-space interpolation current->qd_target (linuxcnc deg)."""
    qd = np.array(read_q(), float); tgt = np.array(qd_target, float)
    n = 24; traj = [list(map(float, qd + (tgt - qd) * (i / (n - 1)))) for i in range(n)]
    send_chunk({"trajectory": traj, "traj_dt": dur / (n - 1), "t_anchor": time.time() + 0.12})
    if settle: time.sleep(dur + 1.2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-approach", type=float, default=28.0)
    ap.add_argument("--v-des", type=float, default=4.0, help="descent speed far from contact (deg/s) -- SLOW")
    ap.add_argument("--v-touch", type=float, default=2.0, help="descent speed within ~1.5cm of contact (deg/s) -- SLOWER")
    ap.add_argument("--standoff", type=float, default=0.05)
    ap.add_argument("--step", type=float, default=0.003, help="descent step per tick (m)")
    ap.add_argument("--abs-floor", type=float, default=-0.03,
                    help="SAFE absolute z stop (m); well above the table (-0.10) so a bad detection can't drive into it")
    ap.add_argument("--ring-margin", type=float, default=0.002,
                    help="ring surface reaches within this of the cup-tip plane = tip at the surface (m). SMALL = actually touch")
    ap.add_argument("--dome-drop", type=float, default=0.008,
                    help="cup dome (mushroom) depth drops this far below baseline = real compression/touch (m); above ~3mm jitter")
    ap.add_argument("--press", type=float, default=0.009,
                    help="once the ring reaches the surface, press this much further to actually contact+compress the cup (m, bounded)")
    ap.add_argument("--j6-span", type=float, default=0.0, help="J6 sweep half-span for the multi-view height (deg; 0=off)")
    ap.add_argument("--j6-dur", type=float, default=4.0, help="continuous J6 sweep duration (s)")
    args = ap.parse_args()

    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(SER)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]])
    for _ in range(12): pipe.wait_for_frames(2000)
    cm = np.load(os.path.join(C.OUT_DIR, "cup_mask.npz")); rim = cm["rim"]; ring = cm["ring"]; dome = cm["dome"]
    M_ct = make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
    tip_depth = float(np.linalg.inv(M_ct)[2, 3])       # cup-tip depth in the D405 optical frame (constant ~0.109m)
    dargs = SimpleNamespace(xmin=0.15, xmax=0.60, ymin=-0.30, ymax=0.30, max_h=0.16, max_foot=0.20)

    def fk_T(qd):
        p, q = fk(qd)
        return make_T(quat_wxyz_to_R(q), p) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM

    def sample_height_at(xy, z_guess, qd, depth):
        """FAST (no SAM3): project the known object xy into the current cam, sample depth there,
        deproject -> base-frame surface height at that xy. Cup-occluded azimuths -> outliers."""
        Tbc = fk_T(qd); Pc = np.linalg.inv(Tbc) @ np.array([xy[0], xy[1], z_guess, 1.0])
        if Pc[2] <= 0.02: return None
        u = int(round(K[0, 0] * Pc[0] / Pc[2] + K[0, 2])); v = int(round(K[1, 1] * Pc[1] / Pc[2] + K[1, 2]))
        if not (3 <= u < W - 3 and 3 <= v < H - 3): return None
        win = depth[v - 3:v + 4, u - 3:u + 4].ravel(); win = win[(win > 0.05) & (win < 0.5)]
        if len(win) < 6: return None
        d = float(np.median(win))
        Xc = (u - K[0, 2]) * d / K[0, 0]; Yc = (v - K[1, 2]) * d / K[1, 1]
        return float((Tbc @ np.array([Xc, Yc, d, 1.0]))[2])

    def ring_depth(depth):                             # cup-mask ring: surface depth just outside the tip
        d = depth[ring]; d = d[(d > 0.05) & (d < 0.5)]
        return float(np.median(d)) if len(d) >= 8 else None
    def dome_depth(depth):                             # cup-mask dome (mushroom): the cup's own depth (drops on compression)
        d = depth[dome]; d = d[(d > 0.03) & (d < 0.30)]
        return float(np.median(d)) if len(d) >= 50 else None

    def sense(target_xy):
        """One frame -> (object xy/top nearest target, depth-gap). gap = ring - rim median depth."""
        qd = read_q(); Tbc = fk_T(qd)
        f = align.process(pipe.wait_for_frames(1000))
        rgb = np.asanyarray(f.get_color_frame().get_data())
        depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
        rd = depth[rim]; rd = rd[(rd > 0.04) & (rd < 0.30)]
        ad = depth[ring]; ad = ad[(ad > 0.04) & (ad < 0.60)]
        gap = float(np.median(ad) - np.median(rd)) if (len(rd) >= 8 and len(ad) >= 8) else None
        objs = detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, dargs)
        det = None
        if objs:
            o = min(objs, key=lambda o: np.linalg.norm(o["centroid"][:2] - np.asarray(target_xy)))
            det = (np.asarray(o["centroid"][:2], float), float(o["hi"][2]))
        return det, gap

    print("[1] coarse detect...")
    det, _ = sense([0.4, 0.0])
    if det is None: print("no object"); pipe.stop(); return
    obj_xy, obj_top = det
    print(f"    object ~[{obj_xy[0]:.3f},{obj_xy[1]:.3f}] top~{obj_top:.3f}")

    pre = [float(obj_xy[0]), float(obj_xy[1]), float(obj_top + args.standoff)]
    print(f"[2] fast approach to pregrasp {np.round(pre,3).tolist()} @ {args.v_approach:.0f} deg/s")
    if not stream_to(pre, DOWN, args.v_approach): print("approach plan failed"); pipe.stop(); return
    time.sleep(1.0)

    # ---- 2b. J6 CONTINUOUS-SWEEP height: roll the wrist through a slow velocity sweep while
    #          streaming detections; the offset D405 orbits the tip so the cup occludes a
    #          different side each frame -> median over many views rejects the contaminated ones ----
    if args.j6_span >= 5.0:
        print(f"[2b] J6 continuous sweep height (+/-{args.j6_span:.0f}deg over {args.j6_dur:.1f}s, streaming depth)...")
        q_pre = read_q()
        q_start = list(q_pre); q_start[5] = q_pre[5] - args.j6_span
        q_end = list(q_pre);   q_end[5]   = q_pre[5] + args.j6_span
        stream_joints(q_start)                                              # settle at the sweep start
        qd = np.array(read_q(), float); tgt = np.array(q_end, float)        # stream ONE slow continuous sweep
        n = 60; sweep = [list(map(float, qd + (tgt - qd) * (i / (n - 1)))) for i in range(n)]
        send_chunk({"trajectory": sweep, "traj_dt": args.j6_dur / (n - 1), "t_anchor": time.time() + 0.15})
        t0 = time.time(); hs = []; ang = []                                # STREAM depth frames WHILE it rotates
        while time.time() - t0 < args.j6_dur + 0.2:
            f = align.process(pipe.wait_for_frames(1000))
            depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
            qd_now = read_q()                                              # actual joints @ grab time
            h = sample_height_at(obj_xy, obj_top, qd_now, depth)
            if h is not None: hs.append(h); ang.append(qd_now[5] - q_pre[5])
        stream_joints(list(q_pre))                                         # restore wrist roll
        if hs:
            hs = np.array(hs)
            obj_top = float(np.median(hs))                                 # median rejects the cup-occluded azimuths
            print(f"    -> {len(hs)} streamed depth samples over {min(ang):+.0f}..{max(ang):+.0f}deg, "
                  f"median height {obj_top:.3f}  (IQR {np.percentile(hs,75)-np.percentile(hs,25):.3f}, "
                  f"full-range {hs.max()-hs.min():.3f})")

    print(f"[3] SLOW descent -- cup-mask depth contact (ring->tip_depth {tip_depth:.3f}m, dome-compression)...")
    z = min(obj_top + args.standoff, 0.20); zmin = args.abs_floor   # start capped-high; SAFE absolute floor
    dome_base = []; dome0 = None; ring_buf = []; z_surface = None
    while z > zmin:
        f = align.process(pipe.wait_for_frames(1000))
        depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
        tip = fk(read_q())[0]                           # ACTUAL cup tip (FK of real joints)
        rd_raw = ring_depth(depth); dd = dome_depth(depth)
        if rd_raw is not None: ring_buf.append(rd_raw); ring_buf = ring_buf[-5:]
        rd = float(np.median(ring_buf)) if ring_buf else None   # temporal median -> rejects single-frame spikes
        if len(dome_base) < 6:                          # robust dome baseline from the first frames (rejects jitter)
            if dd is not None: dome_base.append(dd)
            dome0 = float(np.median(dome_base)) if dome_base else None
        # CONTACT from the pre-computed cup masks (no SAM3):
        gap = (rd - tip_depth) if rd is not None else None
        # 1) ring says we've reached the surface -> from here, keep PRESSING to truly contact
        if z_surface is None and gap is not None and len(ring_buf) >= 3 and gap <= args.ring_margin:
            z_surface = float(tip[2])
            print(f"    -- ring at surface (tip z={z_surface:.3f}); pressing {args.press*1000:.0f}mm to contact --")
        # 2) STOP on real cup compression (dome) OR when we've pressed the bounded amount past the surface
        dome_hit = (len(dome_base) >= 6 and dome0 is not None and dd is not None
                    and (dome0 - dd) > args.dome_drop)               # real cup compression
        pressed = z_surface is not None and float(tip[2]) <= z_surface - args.press
        if dome_hit or pressed:
            send_chunk({"hold": True})
            why = "dome-compress" if dome_hit else f"pressed {args.press*1000:.0f}mm past surface"
            rstr2 = "?" if rd is None else f"{rd:.3f}"; dstr2 = "?" if dd is None else f"{dd:.3f}"
            print(f"    >>> TOUCH ({why}): tip z={tip[2]:.3f}  ring_d={rstr2}  dome_d={dstr2} (base {dome0})  "
                  f"center=[{obj_xy[0]:.3f},{obj_xy[1]:.3f}]")
            break
        near = z_surface is not None or (gap is not None and gap < 0.015)   # slow near/after the surface
        z = z - args.step * (0.5 if near else 1.0)
        rstr = f"{rd:.3f}" if rd is not None else "?"; dstr = f"{dd:.3f}" if dd is not None else "?"
        print(f"    z={z:.3f}  tip={tip[2]:.3f}  ring_d={rstr}  dome_d={dstr}" + ("  [SLOW]" if near else ""))
        stream_to([float(obj_xy[0]), float(obj_xy[1]), float(z)], DOWN,
                  args.v_touch if near else args.v_des, settle=False)
        time.sleep(0.4)
    else:
        send_chunk({"hold": True}); print("    reached floor without a cup-mask contact")
    pipe.stop()
    print("done -- lifting cup")
    stream_to([float(obj_xy[0]), float(obj_xy[1]), float(obj_top + 0.10)], DOWN, 12.0)

if __name__ == "__main__":
    main()

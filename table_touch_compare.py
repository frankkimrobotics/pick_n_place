#!/usr/bin/env python3
"""Touch 5 random table points; on ONE slow descent per point, log where BOTH contact detectors
fire: DEPTH-based (cup-region above-band vs cup-hull) and TORQUE-based (J2 shoulder torque rise).
Reports per-point FK-z at each detection + consistency, so we can compare the two methods.
Safety: bounded abs-floor, capped press past depth-contact, hard torque abort.

  source /opt/ros/humble/setup.bash
  python3 table_touch_compare.py [--n 5]
"""
import argparse, json, os, socket, sys, time, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
import cup_region as CR
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg
import pyrealsense2 as rs
PI = "10.0.0.27"; SER = "218622271300"; W, H = 640, 480
DOWN = list(map(float, R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0])))))
CUP_REGION = os.path.join(C.OUT_DIR, "cup_region_fixed.npz")

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def send_chunk(m):
    k = socket.create_connection((PI, 9994), timeout=3); k.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    k.sendall((json.dumps(m) + "\n").encode()); k.close()
def read_state():
    s = socket.create_connection((PI, 9999), timeout=3); b = b""
    while b"\n" not in b: b += s.recv(4096)
    s.close(); d = json.loads(b.split(b"\n")[0]); return list(d["joints_deg"]), list(d.get("torque", [0] * 6))
def read_q(): return read_state()[0]
def qrad(qd): return [float(v) for v in linuxcnc_deg_to_rad(qd)]
def fk(qd): return np.array(rpc({"type": "fk", "q": qrad(qd)})["pos"][0])
def stream_to(xyz, V, settle=True):
    qd = read_q(); r = rpc({"type": "plan_pose", "start_q": qrad(qd), "goal_pose": list(xyz) + DOWN, "max_attempts": 16})
    if not r.get("success"): return False
    traj = np.array(r["trajectory"]); traj[:, 5] = float(C.BASE_Q[5]); dt = r["dt"]
    peak = np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt if len(traj) > 1 else V
    sdt = dt * max(1.0, peak / V)
    send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj], "traj_dt": sdt, "t_anchor": time.time() + 0.12})
    if settle: time.sleep(sdt * (len(traj) - 1) + 1.6)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start-z", type=float, default=0.06)
    ap.add_argument("--step", type=float, default=0.003)
    ap.add_argument("--contact-gap", type=float, default=0.0, help="depth touch: above_d-cup_d <= this")
    ap.add_argument("--torque-thresh", type=float, default=0.08, help="torque touch: |J2-baseline| > this")
    ap.add_argument("--torque-abort", type=float, default=0.25, help="hard abort")
    ap.add_argument("--max-press", type=float, default=0.014, help="max descent past depth-contact (m)")
    ap.add_argument("--abs-floor", type=float, default=-0.145)
    args = ap.parse_args()
    random.seed()

    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(SER)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30); cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale(); align = rs.align(rs.stream.color)
    for _ in range(12): pipe.wait_for_frames(2000)
    cup_m, above_m = CR.load(CUP_REGION)
    def depth_read():
        f = align.process(pipe.wait_for_frames(1000))
        _, sd = CR.read_depths(np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale, cup_m, above_m)
        return sd

    results = []
    for k in range(args.n):
        oxy = [round(random.uniform(0.30, 0.42), 3), round(random.uniform(-0.12, 0.12), 3)]
        print(f"\n=== point {k+1}/{args.n}: [{oxy[0]:.3f},{oxy[1]:.3f}] ===")
        if not stream_to([oxy[0], oxy[1], args.start_z], 20.0): print("   plan failed, skip"); continue
        time.sleep(0.8)
        cds = [d for d in (CR.read_depths(np.asanyarray(align.process(pipe.wait_for_frames(1000)).get_depth_frame().get_data()).astype(np.float32)*scale, cup_m, above_m)[0] for _ in range(8)) if d]
        cup_d = float(np.median(cds)) if cds else None
        _, tq0 = read_state(); base_t2 = tq0[2]
        depth_z = None; torque_z = None; z = args.start_z; aborted = False
        while z > args.abs_floor:
            sd = depth_read(); _, tq = read_state(); tip = fk(read_q())[2]
            if depth_z is None and sd is not None and cup_d is not None and (sd - cup_d) <= args.contact_gap:
                depth_z = tip; print(f"   [DEPTH]  contact at FK z={tip:.3f} (above_d {sd:.3f} = cup_d {cup_d:.3f})")
            if torque_z is None and abs(tq[2] - base_t2) > args.torque_thresh:
                torque_z = tip; print(f"   [TORQUE] contact at FK z={tip:.3f} (dtq {tq[2]-base_t2:+.3f})")
            if abs(tq[2] - base_t2) > args.torque_abort:
                print(f"   >>> TORQUE ABORT at FK z={tip:.3f}"); aborted = True; break
            if torque_z is not None: break                                   # both/torque done
            if depth_z is not None and tip <= depth_z - args.max_press:       # pressed enough, torque never fired
                print(f"   [TORQUE] not detected within {args.max_press*1000:.0f}mm press"); break
            z -= args.step
            stream_to([oxy[0], oxy[1], z], 2.0, settle=False); time.sleep(0.4)
        send_chunk({"hold": True}); time.sleep(0.4)
        results.append({"xy": oxy, "depth_z": depth_z, "torque_z": torque_z, "aborted": aborted})
        stream_to([oxy[0], oxy[1], 0.10], 12.0)                              # lift

    pipe.stop()
    print("\n================ COMPARISON (FK z at contact) ================")
    print("  pt   xy                 depth_z    torque_z   torque-depth")
    dz, tz = [], []
    for i, r in enumerate(results):
        d = r["depth_z"]; t = r["torque_z"]
        ds = f"{d:+.4f}" if d is not None else "  none "; ts = f"{t:+.4f}" if t is not None else "  none "
        diff = f"{(t-d)*1000:+.1f}mm" if (d is not None and t is not None) else "   -"
        print(f"  {i+1}   [{r['xy'][0]:.3f},{r['xy'][1]:+.3f}]   {ds}   {ts}   {diff}")
        if d is not None: dz.append(d)
        if t is not None: tz.append(t)
    if dz: print(f"  DEPTH  : {len(dz)}/{len(results)} fired, mean {np.mean(dz):+.4f}  std {np.std(dz)*1000:.1f}mm")
    if tz: print(f"  TORQUE : {len(tz)}/{len(results)} fired, mean {np.mean(tz):+.4f}  std {np.std(tz)*1000:.1f}mm")
    # return to base
    r = rpc({"type": "plan_joint", "start_q": qrad(read_q()), "goal_q": [float(v) for v in C.BASE_Q], "max_attempts": 12})
    if r.get("success"):
        traj = np.array(r["trajectory"]); dt = r["dt"]; sdt = dt * max(1.0, np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt / 20.0)
        send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj], "traj_dt": sdt, "t_anchor": time.time() + 0.12}); time.sleep(sdt * (len(traj) - 1) + 1.6)
    print("done; arm at base")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bare-table touch test: pick a random table point, descend the suction cup with the cup-region
depth contact (above-band vs cup-hull, no object needed), then PRESS ~1cm straight down while
watching J2 torque as a hard-surface safety-stop. Holds, then lifts.

  source /opt/ros/humble/setup.bash
  python3 table_touch.py                 # random point
  python3 table_touch.py --xy 0.36,-0.05 # specific point
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
def fk(qd):
    r = rpc({"type": "fk", "q": qrad(qd)}); return np.array(r["pos"][0])
def stream_to(xyz, quat, V, settle=True):
    qd = read_q(); r = rpc({"type": "plan_pose", "start_q": qrad(qd), "goal_pose": list(xyz) + list(quat), "max_attempts": 16})
    if not r.get("success"): return False
    traj = np.array(r["trajectory"]); traj[:, 5] = float(C.BASE_Q[5]); dt = r["dt"]
    peak = np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt if len(traj) > 1 else V
    sdt = dt * max(1.0, peak / V)
    send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj], "traj_dt": sdt, "t_anchor": time.time() + 0.12})
    if settle: time.sleep(sdt * (len(traj) - 1) + 1.6)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xy", default="", help="table x,y (else random)")
    ap.add_argument("--start-z", type=float, default=0.06, help="FK z to start the descent (well above table)")
    ap.add_argument("--step", type=float, default=0.004)
    ap.add_argument("--v-touch", type=float, default=2.5)
    ap.add_argument("--contact-gap", type=float, default=0.0, help="touch when above_d-cup_d <= this (m)")
    ap.add_argument("--press", type=float, default=0.010, help="press this far past table contact (m)")
    ap.add_argument("--torque-stop", type=float, default=0.12, help="abort press if |J2 torque-baseline| exceeds this")
    ap.add_argument("--abs-floor", type=float, default=-0.145, help="SAFE hard z floor (table ~-0.10 + offset + press)")
    args = ap.parse_args()

    if args.xy:
        oxy = np.array([float(v) for v in args.xy.split(",")])
    else:
        oxy = np.array([round(random.uniform(0.30, 0.42), 3), round(random.uniform(-0.12, 0.12), 3)])
    print(f"[table-touch] random point [{oxy[0]:.3f},{oxy[1]:.3f}]")

    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(SER)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30); cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    for _ in range(12): pipe.wait_for_frames(2000)
    cup_m, above_m = CR.load(CUP_REGION)
    def grab_depth():
        f = align.process(pipe.wait_for_frames(1000))
        return np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale

    # pregrasp well above the table
    if not stream_to([float(oxy[0]), float(oxy[1]), args.start_z], DOWN, 20.0): print("pregrasp plan failed"); pipe.stop(); return
    time.sleep(0.8)
    # cup reference depth (constant)
    cds = []
    for _ in range(8):
        cd, _ = CR.read_depths(grab_depth(), cup_m, above_m)
        if cd is not None: cds.append(cd)
    if not cds: print("no cup depth"); pipe.stop(); return
    cup_d = float(np.median(cds)); print(f"   cup_d={cup_d:.3f}")

    # slow descent to TABLE contact (above-band depth reaches cup rim)
    z = args.start_z; touched = False
    while z > args.abs_floor:
        _, surf_d = CR.read_depths(grab_depth(), cup_m, above_m)
        gap = (surf_d - cup_d) if surf_d is not None else None
        if gap is not None and gap <= args.contact_gap:
            tip = fk(read_q()); touched = True
            print(f"   >>> TABLE CONTACT: above_d {surf_d:.3f} = cup_d {cup_d:.3f} at FK z={tip[2]:.3f}")
            break
        z -= args.step
        print(f"   z={z:.3f}  above_d={surf_d:.3f}" if surf_d is not None else f"   z={z:.3f}")
        stream_to([float(oxy[0]), float(oxy[1]), z], DOWN, args.v_touch, settle=False); time.sleep(0.45)
    if not touched:
        send_chunk({"hold": True}); print("   reached floor, no depth contact -- aborting press"); pipe.stop(); return

    # PRESS ~1cm past contact, small steps, torque safety-stop
    _, tq0 = read_state(); base_t2 = tq0[2]; z_c = fk(read_q())[2]
    target = max(z_c - args.press, args.abs_floor)
    print(f"   pressing {args.press*1000:.0f}mm: FK {z_c:.3f} -> {target:.3f} (torque-stop {args.torque_stop})")
    zp = z_c
    while zp > target:
        zp = max(zp - 0.002, target)
        stream_to([float(oxy[0]), float(oxy[1]), zp], DOWN, 1.5, settle=False); time.sleep(0.4)
        _, tq = read_state()
        if abs(tq[2] - base_t2) > args.torque_stop:
            send_chunk({"hold": True}); print(f"   >>> TORQUE STOP at FK z={fk(read_q())[2]:.3f} (dtq {tq[2]-base_t2:+.2f}) -- cup bottomed / firm press"); break
    else:
        send_chunk({"hold": True}); print(f"   pressed to FK z={fk(read_q())[2]:.3f}")
    time.sleep(1.2)                                     # hold the press
    pipe.stop()
    print("   lifting + returning to base")
    stream_to([float(oxy[0]), float(oxy[1]), 0.12], DOWN, 12.0)
    r = rpc({"type": "plan_joint", "start_q": qrad(read_q()), "goal_q": [float(v) for v in C.BASE_Q], "max_attempts": 12})
    if r.get("success"):
        traj = np.array(r["trajectory"]); dt = r["dt"]; sdt = dt * max(1.0, np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt / 20.0)
        send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj], "traj_dt": sdt, "t_anchor": time.time() + 0.12}); time.sleep(sdt * (len(traj) - 1) + 1.6)
    print("done")

if __name__ == "__main__":
    main()

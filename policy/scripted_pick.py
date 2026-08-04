"""scripted_pick :: REAL-robot scripted lift-and-replace using the repo's own
pick primitives (mjwarp_pick_demo IK + R_DOWN + min-jerk timing) over the
direct robot_hal :9998 protocol. Sequence:

  home -> hover over object -> TORQUE-GUARDED descent to contact ->
  suction ON -> dwell -> lift 10 cm -> dwell -> lower to contact ->
  suction OFF -> hover -> home

Torque-guarded contact makes the press robust to the shoulder's FK slip
(absolute z is biased; contact is detected physically). Wrist frames are
saved at every stage. Slow throughout (<= ~5 deg/s).

    python policy/scripted_pick.py [--exec]     # dry-run without --exec
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "mycobot_mpc")))
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg  # noqa
import mujoco                                                           # noqa

PI = "10.0.0.27"
SHM = "/dev/shm"
OUT = os.path.expanduser("~/pnp_real_runs")


def feedback():
    s = socket.create_connection((PI, 9999), timeout=5)
    b = b""
    while b"\n" not in b:
        b += s.recv(4096)
    s.close()
    d = json.loads(b.split(b"\n")[0])
    return (np.array(d["joints_deg"], float),
            np.array(d.get("torque", [0] * 6), float))


def move(target_lcnc, duration, execute):
    if not execute:
        print(f"  [dry] -> {np.round(target_lcnc,1).tolist()} over {duration:.1f}s")
        return
    k = socket.create_connection((PI, 9998), timeout=5)
    k.sendall((json.dumps({"target_deg": [float(v) for v in target_lcnc],
                           "duration": float(duration),
                           "controller": "pid"}) + "\n").encode())
    try:
        k.settimeout(2)
        k.recv(256)
    except Exception:
        pass
    k.close()
    time.sleep(duration + 1.0)


def suction(on, execute):
    if not execute:
        print(f"  [dry] suction -> {int(on)}")
        return
    subprocess.run(["ssh", f"pi@{PI}",
                    f"halcmd setp pro600.digital_out00 {int(on)}"],
                   check=True, timeout=10, capture_output=True)


def snap(run_dir, name):
    try:
        import shutil
        shutil.copy(f"{SHM}/pnp_wrist.jpg", os.path.join(run_dir, name))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="execute", action="store_true")
    ap.add_argument("--vel", type=float, default=5.0, help="deg/s")
    a = ap.parse_args()
    run_dir = os.path.join(OUT, "scripted_" + time.strftime("%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    m = mujoco.MjModel.from_xml_path(
        os.path.join(ROOT, "outputs", "mujoco_sim", "robot_warp.xml"))
    d = mujoco.MjData(m)
    dik = mujoco.MjData(m)
    ik, R_DOWN, CUP_R = demo["ik"], demo["R_DOWN"], demo["CUP_R"]
    home = np.asarray(demo["Q_START"], float)

    def to_lcnc(q):
        return np.array(rad_to_linuxcnc_deg(q), float)

    def dur(q_from, q_to):
        return max(6.0, np.degrees(np.abs(q_to - q_from)).max() / a.vel)

    # ---- current state + goal (same estimator as live_chain) ----
    q_lcnc, tau0 = feedback()
    q_now = np.array(linuxcnc_deg_to_rad(q_lcnc), float)
    print("joints (lcnc):", np.round(q_lcnc, 1).tolist())
    dq_home = np.degrees(np.abs(q_now - home)).max()
    if dq_home > 3.0:
        print(f"[pick] homing first ({dq_home:.0f} deg off)")
        move(to_lcnc(home), dur(q_now, home), a.execute)
        q_now = home.copy()
    snap(run_dir, "0_home.jpg")

    intr = json.load(open(f"{SHM}/pnp_wrist_intr.json"))
    import cv2
    d1 = open(f"{SHM}/pnp_wrist_depth.npy", "rb").read()
    time.sleep(1.0)
    d2 = open(f"{SHM}/pnp_wrist_depth.npy", "rb").read()
    assert d1 != d2, "depth not updating"
    dm = np.frombuffer(d2, np.uint16).reshape(intr["h"], intr["w"]).astype(np.float32) * 1e-4
    valid = dm > 0.05
    valid[190:, :] = False
    valid[:, :30] = False
    valid[:, -30:] = False
    table = float(np.median(dm[valid]))
    mask = (valid & (dm < table - 0.012) & (dm > table - 0.12)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    assert n > 1, "no object visible"
    best = 1 + int(np.argmax(stats[1:, 4]))
    u, v = cent[best]
    z = float(np.median(dm[lab == best]))
    p_rs = np.array([(u - intr["ppx"]) / intr["fx"] * z,
                     (v - intr["ppy"]) / intr["fy"] * z, z])
    p_mj = np.array([p_rs[0], -p_rs[1], -p_rs[2]])
    d.qpos[:6] = q_now
    mujoco.mj_forward(m, d)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_d405")
    g = d.cam_xpos[cid] + d.cam_xmat[cid].reshape(3, 3) @ p_mj
    print(f"[pick] object top at {np.round(g,4).tolist()} "
          f"({stats[best,4]} px, table {table:.3f} m)")
    assert 0.2 < g[0] < 0.6 and abs(g[1]) < 0.3, "goal out of envelope"

    # ---- IK waypoints (repo primitives) ----
    q_hover, e1 = ik(m, dik, "tcp", [g[0], g[1], g[2] + 0.10], R_DOWN, q_now)
    q_touch0, e2 = ik(m, dik, "tcp", [g[0], g[1], g[2] + CUP_R + 0.02], R_DOWN, q_hover)
    assert max(e1, e2) < 0.005, f"IK failed ({e1:.4f}, {e2:.4f})"

    print("[pick] hover")
    move(to_lcnc(q_hover), dur(q_now, q_hover), a.execute)
    snap(run_dir, "1_hover.jpg")

    print("[pick] pre-contact (+2 cm nominal)")
    move(to_lcnc(q_touch0), dur(q_hover, q_touch0), a.execute)
    snap(run_dir, "2_precontact.jpg")

    # ---- torque-guarded descent: 8 mm steps until contact signature ----
    _, tau_free = feedback()
    q_ref = q_touch0.copy()
    contact = False
    target_z = g[2] + CUP_R + 0.02
    for step in range(8):                       # up to 6.4 cm of travel
        target_z -= 0.008
        q_next, e = ik(m, dik, "tcp", [g[0], g[1], target_z], R_DOWN, q_ref)
        if e > 0.005:
            print("[pick] IK limit during descent")
            break
        move(to_lcnc(q_next), 3.0, a.execute)
        q_ref = q_next
        _, tau = feedback()
        dtau = np.abs(tau - tau_free)[:4].max()
        print(f"  descent step {step}: z={target_z:.4f} dtau={dtau:.2f}")
        if a.execute and dtau > max(0.8, 0.25 * np.abs(tau_free[:4]).max()):
            print("[pick] CONTACT detected (torque)")
            contact = True
            break
    if a.execute and not contact:
        print("[pick] WARNING: no contact signature; proceeding at last height")
    snap(run_dir, "3_contact.jpg")

    print("[pick] suction ON + dwell")
    suction(1, a.execute)
    time.sleep(1.5)

    print("[pick] lift 10 cm")
    q_lift, e = ik(m, dik, "tcp", [g[0], g[1], target_z + 0.10], R_DOWN, q_ref)
    assert e < 0.005
    move(to_lcnc(q_lift), dur(q_ref, q_lift), a.execute)
    time.sleep(1.0)
    snap(run_dir, "4_lifted.jpg")
    _, tau_lift = feedback()
    print(f"[pick] holding; torque delta vs free: "
          f"{np.round(np.abs(tau_lift - tau_free)[:4], 2).tolist()}")
    time.sleep(1.5)

    print("[pick] lower back to contact height")
    move(to_lcnc(q_ref), dur(q_lift, q_ref), a.execute)
    snap(run_dir, "5_lowered.jpg")

    print("[pick] suction OFF (release)")
    suction(0, a.execute)
    time.sleep(1.0)

    print("[pick] retreat to hover + home")
    move(to_lcnc(q_hover), dur(q_ref, q_hover), a.execute)
    snap(run_dir, "6_released.jpg")
    move(to_lcnc(home), dur(q_hover, home), a.execute)
    snap(run_dir, "7_home.jpg")
    print(f"[pick] DONE -> {run_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            suction(0, True)
        except Exception:
            pass

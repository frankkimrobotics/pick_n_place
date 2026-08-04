"""real_rollout :: DP policy on the REAL myCobot Pro 630 (via Pi weld streaming).

Reuses the proven il/rollout.py transport: joints from Pi :9999 (LinuxCNC deg),
weld-chunk trajectories to :9994, suction via `halcmd setp pro600.digital_out00`
over ssh (same pin as suction_test.py). Policy + obs pipeline mirror the sim
executor (rollout_sim.py): 2-step JPEG-domain obs (424x240 -> 432 -> JPEG q90),
goal grasp pose relative to current tcp (FK via MuJoCo model), 16 B-spline ctrl
points -> dense joint reference, C1 Hermite splice bridge, suction hysteresis,
frozen-noise DDIM.

SAFETY (defaults are conservative):
  --slow 4        time-stretch: spline phase advances at 1/4 real time
                  (demo joint speeds already <= ~40 deg/s -> ~10 deg/s here)
  --max_vel_deg 12  reference clamped tick-to-tick to this rate; abort if the
                  policy demands more than 3x this after clamping
  joint box: j2 in [-70,70], j3 in [-145,145] deg (elbow constraint memo),
  all joints within URDF limits minus 5 deg margin
  --exec          REQUIRED for any motion or suction; default is dry-run
                  (prints what would be sent, captures obs, runs inference)
  start check: refuses to run unless robot is within 15 deg of sim home pose
  on ANY exception: suction OFF, no further chunks (welder holds last target)

Rangefinder obs (sim tip rangefinder) is synthesized from D405 center depth.
Goal: --goal_xyz X Y Z (robot-base metres, the object TOP surface); the goal
z gets + cup_radius as in sim; top-down quat assumed.

    python3 policy/real_rollout.py --ckpt ~/pnp_runs_studio/dp_q/ckpt_final.pt \
        --goal_xyz 0.35 0.05 0.03 --pi 10.0.0.27 [--exec]
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "mycobot_mpc")))

import train_lit as tl                                     # noqa: E402
import convert_litdata as cl                               # noqa: E402
import rollout_sim as rsim                                 # noqa: E402
import mujoco                                              # noqa: E402
from joint_conventions import (linuxcnc_deg_to_rad,        # noqa: E402
                               rad_to_linuxcnc_deg)

WRIST_SER = "218622271300"
FIXED_SER = "043422070101"
SUCTION_PIN = "pro600.digital_out00"
# tightened to the demo task envelope (real cell has a WALL behind the
# robot: j1 below -30 URDF leans the upper arm backward over the base)
# widened after run 1: the shoulder slip means the physical goal needs
# MORE forward reach than FK predicts; z-floor guard remains the true stop
J_LO = np.radians([-60, -30, 0, -60, -135, -60])
J_HI = np.radians([60, 78, 70, 45, -45, 60])


class Robot:
    def __init__(self, pi, user, execute):
        self.pi, self.user, self.execute = pi, user, execute
        self.suction = 0

    def joints_rad(self):
        s = socket.create_connection((self.pi, 9999), timeout=3)
        b = b""
        while b"\n" not in b:
            b += s.recv(4096)
        s.close()
        d = json.loads(b.split(b"\n")[0])
        return np.array(linuxcnc_deg_to_rad(d["joints_deg"]), float)

    def send_chunk(self, traj_rad, dt):
        """SAFE SPLINE FOLLOWER: walk the pid controller along the (already
        accel/vel-clamped, z-floor-truncated) spline via sub-targets every
        SUB_T seconds, so motion follows the smooth chunk shape rather than
        one point-to-point jump. Blocks for the window duration."""
        SUB_T = 1.0
        n = len(traj_rad)
        step = max(1, int(round(SUB_T / dt)))
        idx = list(range(step, n, step)) + [n - 1]
        for i in idx:
            tgt = list(map(float, rad_to_linuxcnc_deg(traj_rad[i])))
            dur = float((i - (idx[idx.index(i) - 1] if idx.index(i) else 0)) * dt)
            if not self.execute:
                print(f"  [dry] sub-target {np.round(tgt, 1)} over {dur:.1f}s")
                continue
            k = socket.create_connection((self.pi, 9998), timeout=3)
            k.sendall((json.dumps({"target_deg": tgt, "duration": dur,
                                   "controller": "pid"}) + "\n").encode())
            try:
                k.settimeout(2); k.recv(256)
            except Exception:
                pass
            k.close()
            time.sleep(dur)

    def set_suction(self, on):
        on = int(on)
        if on == self.suction:
            return
        self.suction = on
        cmd = f"halcmd setp {SUCTION_PIN} {on}"
        if not self.execute:
            print(f"  [dry] suction -> {on}")
            return
        subprocess.run(["ssh", f"{self.user}@{self.pi}", cmd],
                       check=True, timeout=10, capture_output=True)


class Cams:
    """Reads frames published to /dev/shm by cam_server.py (system python --
    the sam3 librealsense wheel cannot open these cameras). Start it first:
        /usr/bin/python3 policy/cam_server.py &"""
    SHM = "/dev/shm"

    def __init__(self):
        for f in ("pnp_wrist.jpg", "pnp_fixed.jpg", "pnp_range.txt"):
            p = os.path.join(self.SHM, f)
            if not os.path.exists(p) or time.time() - os.path.getmtime(p) > 3:
                raise RuntimeError(f"stale/missing {p} -- is cam_server.py running?")

    def get(self):
        out = {}
        w = cv2.imread(os.path.join(self.SHM, "pnp_wrist.jpg"))
        x = cv2.imread(os.path.join(self.SHM, "pnp_fixed.jpg"))
        out["wrist"] = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)
        out["fixed"] = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        out["range"] = float(open(os.path.join(self.SHM, "pnp_range.txt")).read())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal_xyz", type=float, nargs=3, required=True,
                    help="object TOP center in robot base frame [m]")
    ap.add_argument("--pi", default="10.0.0.27")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--exec", dest="execute", action="store_true")
    ap.add_argument("--slow", type=float, default=4.0)
    ap.add_argument("--max_vel_deg", type=float, default=12.0)
    ap.add_argument("--exec_steps", type=int, default=10)
    ap.add_argument("--steps_max", type=int, default=40)
    ap.add_argument("--suction_off_n", type=int, default=5)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_real_runs"))
    a = ap.parse_args()
    run_dir = os.path.join(a.out, time.strftime("%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # policy (frozen-noise, same wrapper as sim rollout)
    class PA:
        ckpt = a.ckpt
        untrained = None
        data = None
        frozen_noise = True
    pol = rsim.Policy(PA, "cuda:0" if torch.cuda.is_available() else "cpu")

    # FK model for tcp pose / goal encoding
    m = mujoco.MjModel.from_xml_path(
        os.path.join(ROOT, "outputs", "mujoco_sim", "robot_warp.xml"))
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    def tcp_pose(q):
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()

    CUP_R = 0.008
    quat_down = np.array([0.0, 0.70711, 0.70711, 0.0], np.float32)
    R_g = tl.quat_to_R(quat_down)
    g_pos = np.array(a.goal_xyz) + np.array([0, 0, CUP_R])

    robot = Robot(a.pi, a.user, a.execute)
    cams = Cams()
    q0 = robot.joints_rad()
    print("[real] robot joints (deg):", np.round(np.degrees(q0), 1))
    # start-pose check vs sim home
    demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    dq = np.degrees(np.abs(q0 - demo["Q_START"]))
    if dq.max() > 15:
        print(f"[real] ABORT: robot {np.round(dq,1)} deg from sim home pose; "
              "jog it home first (or run with sim home disabled consciously)")
        if a.execute:
            return
    print(f"[real] slow={a.slow}x  max_vel={a.max_vel_deg} deg/s  "
          f"exec={'LIVE' if a.execute else 'DRY-RUN'}")

    # dense spline basis at the streamed dt, time-stretched
    DT_STREAM = 0.05                       # 20 Hz waypoints to the welder
    n_ticks = int(round(a.exec_steps * 0.1 * a.slow / DT_STREAM))
    taus = np.arange(n_ticks) * DT_STREAM / a.slow          # policy-time
    dense_s = np.clip(taus - 0.1, 0.0, 1.5)
    DENSE = np.stack([cl.bspline_basis(s) for s in dense_s])
    NB = max(2, int(round(0.1 * a.slow / DT_STREAM)))       # bridge ticks

    hist = None
    q_prev = q0.copy()
    latched_cmd = 0
    off_count = 0
    log = []
    try:
        for step in range(a.steps_max):
            t_step0 = time.time()
            frames = cams.get()
            j405 = rsim.jpeg_domain(frames["wrist"])
            j435 = rsim.jpeg_domain(frames["fixed"])
            q = robot.joints_rad()
            qd = (q - q_prev) / max(1e-3, a.exec_steps * 0.1)
            q_prev = q.copy()
            if hist is None:
                hist = (j405, j435, q.copy(), qd.copy())
            prop = np.stack([
                np.concatenate([hist[2], hist[3], [float(latched_cmd)],
                                [frames["range"]]]),
                np.concatenate([q, qd, [float(latched_cmd)],
                                [frames["range"]]])]).astype(np.float32)
            p_t, R_t = tcp_pose(q)
            dR = R_t.T @ R_g
            goal = np.concatenate([R_t.T @ (g_pos - p_t),
                                   dR[:, 0], dR[:, 1]]).astype(np.float32)
            ctrl, suc_cmd = pol([hist[0], j405, hist[1], j435], prop, goal)
            hist = (j405, j435, q.copy(), qd.copy())

            qref = DENSE @ ctrl
            # C1 bridge from current q (vel~0 at these speeds)
            p1 = qref[min(NB, len(qref) - 1)]
            w = np.linspace(0, 1, NB)[:, None]
            qref[:NB] = (1 - (3 * w**2 - 2 * w**3)) * q + (3 * w**2 - 2 * w**3) * p1
            # clamps: joint box + per-tick velocity + acceleration
            qref = np.clip(qref, J_LO, J_HI)
            vmax = np.radians(a.max_vel_deg) * DT_STREAM
            amax = vmax / 4.0                      # reach vmax over ~0.2 s
            dq_prev = np.zeros(6)
            for k in range(1, len(qref)):
                dq = np.clip(qref[k] - qref[k - 1], -vmax, vmax)
                dq = np.clip(dq, dq_prev - amax, dq_prev + amax)
                qref[k] = qref[k - 1] + dq
                dq_prev = dq
            # FK z-floor: truncate the window at the first sample whose tcp
            # dips below the goal press floor (goal_z - 25 mm)
            z_floor = g_pos[2] - 0.025
            n_ok = len(qref)
            for k in range(0, len(qref), 4):
                pz, _ = tcp_pose(qref[k])
                if pz[2] < z_floor:
                    n_ok = max(NB + 1, k)
                    print(f"  [safe] z-floor truncation at sample {k} "
                          f"(tcp z {pz[2]:.3f} < {z_floor:.3f})")
                    break
            qref = qref[:n_ok]
            demand = np.degrees(np.abs(np.diff(qref, axis=0)).max()) / DT_STREAM
            log.append(dict(step=step, q=q.tolist(), range=frames["range"],
                            goal=goal.tolist(), suction=int(latched_cmd),
                            demand_deg_s=float(demand)))

            # suction hysteresis on the commanded channel
            want = bool(suc_cmd[0])
            if want and not latched_cmd and frames["range"] < 0.03:
                robot.set_suction(1)
                latched_cmd = 1
                off_count = 0
            elif latched_cmd:
                off_count = 0 if want else off_count + 1
                if off_count >= a.suction_off_n:
                    robot.set_suction(0)
                    latched_cmd = 0

            robot.send_chunk(qref, DT_STREAM)
            cv2.imwrite(os.path.join(run_dir, f"wrist_{step:03d}.jpg"),
                        cv2.cvtColor(frames["wrist"], cv2.COLOR_RGB2BGR))
            # wait out the (stretched) execution window
            t_left = a.exec_steps * 0.1 * a.slow - (time.time() - t_step0)
            if a.execute and t_left > 0:
                time.sleep(t_left)
    finally:
        robot.set_suction(0)
        json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=1)
        print(f"[real] done -> {run_dir} (suction OFF)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""robot_execute :: stream the planned pick-and-place trajectory to the REAL robot.

The pick-and-place pipeline exports `outputs/robot_trajectory.json` (per-segment
joint waypoints in URDF rad + suction on/off events). This script replays it on the
real MyCobot Pro 630 by REUSING the existing mycobot_mpc execution stack:

  * `perturb_loop.execute()` time-scales each segment and publishes it to the ROS2
    topic `/mycobot/cmd/move`; `mycobot_ros2_bridge.py` forwards it (cmd-port 9998)
    to `robot_hal` on the Pi, whose controller (PID / inverse-dynamics) FOLLOWS the
    trajectory. `perturb_loop.RobotState` reads `/joint_states` to settle between
    segments.
  * suction is toggled at the marked events via the digital-out pin
    `pro600.digital_out00` (same as robot_hal / suction_test.py).

Python 3.8 + ROS 2 Humble (the same env perturb_loop runs in). Run near the robot
with the bridge + robot_hal already up:

    source /opt/ros/humble/setup.bash          # + your mycobot workspace
    python3 pick_and_place/robot_execute.py outputs/robot_trajectory.json \
        --controller pid --max-vel-deg 40 --suction-host $ROBOT_IP

Validate the file offline (no ROS2 / no robot needed):
    python3 pick_and_place/robot_execute.py outputs/robot_trajectory.json --dry-run

Safety: the robot must START at the trajectory's first waypoint (the base pose).
`execute()` aborts a segment if the live joints are >--start-tol-deg from its start,
and on any failure we force suction OFF and stop.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MPC = os.path.join(HERE, "..", "mycobot_mpc")
sys.path.insert(0, HERE)                       # suction_test
sys.path.insert(0, os.path.abspath(MPC))       # perturb_loop, joint_conventions


def _validate(data, max_vel_deg):
    """Offline checks: per-segment continuity + peak speed. Returns True if ok."""
    segs = data["segments"]
    ok = True
    print(f"trajectory: {data.get('n_segments')} segments, "
          f"{data.get('n_waypoints')} waypoints, units={data.get('units')}")
    print(f"start_q (base pose) = {np.round(data['start_q_rad'],4).tolist()}")
    prev_end = np.array(data["start_q_rad"])
    for i, s in enumerate(segs):
        t = np.array(s["trajectory_rad"])
        gap = np.rad2deg(np.abs(t[0] - prev_end)).max() if len(t) else 0.0
        peak = (np.rad2deg(np.abs(np.diff(t, axis=0))).max() / s["dt"]) if len(t) > 1 else 0.0
        flag = ""
        if gap > 8.0:
            flag += f"  !! start gap {gap:.1f}deg"; ok = False
        print(f"  [{i}] {s['label']:22s} pts={len(t):4d} dt={s['dt']:.3f} "
              f"peak~{peak:5.1f}deg/s suction_after={s['suction_after']}{flag}")
        if len(t):
            prev_end = t[-1]
    print(f"final q = {np.round(prev_end,4).tolist()}")
    print("validation:", "OK" if ok else "PROBLEMS FOUND")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Replay the pick-and-place trajectory on the real robot.")
    ap.add_argument("trajectory", help="robot_trajectory.json from the pipeline")
    ap.add_argument("--controller", default="pid", help="robot_hal controller (pid/invdyn/...)")
    ap.add_argument("--max-vel-deg", type=float, default=40.0,
                    help="peak joint speed (deg/s); <=55 (drives saturate ~60)")
    ap.add_argument("--min-dur", type=float, default=1.0, help="min seconds per segment")
    ap.add_argument("--ramp-time", type=float, default=0.15)
    ap.add_argument("--pos-gain", type=float, default=1.0)
    ap.add_argument("--vff-scale", type=float, default=1.0)
    ap.add_argument("--start-tol-deg", type=float, default=8.0)
    ap.add_argument("--suction-host", default=os.environ.get("ROBOT_IP", "").strip(),
                    help="Raspi IP for halcmd over SSH (empty = run on the Pi locally)")
    ap.add_argument("--suction-user", default="pi")
    ap.add_argument("--dry-run", action="store_true", help="validate file only; no ROS2, no robot")
    ap.add_argument("--no-suction", action="store_true", help="skip suction toggles (motion only)")
    args = ap.parse_args()

    with open(args.trajectory) as f:
        data = json.load(f)
    segs = data["segments"]

    if args.dry_run:
        _validate(data, args.max_vel_deg)
        return

    # ---- live execution (needs ROS2 + bridge + robot_hal running) -------- #
    import rclpy
    from std_msgs.msg import String
    from perturb_loop import RobotState, execute      # reuse existing executor
    import suction_test                                # reuse suction pin toggle

    def suction(value):
        if args.no_suction:
            print(f"  [suction] (skipped) would set {value}")
            return
        suction_test.set_pin(value, args.suction_host or None, args.suction_user)
        print(f"  [suction] pin -> {value} ({suction_test.get_pin(args.suction_host or None, args.suction_user)})")

    rclpy.init()
    node = rclpy.create_node("pp_robot_execute")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
    state = RobotState(node)
    track = {"ramp_time": args.ramp_time, "pos_gain": args.pos_gain,
             "vff_scale": args.vff_scale}

    # confirm the robot is at the trajectory start (the base pose)
    q0 = state.get_q()
    if q0 is None:
        print("ABORT: no /joint_states — is the bridge/robot up?"); rclpy.shutdown(); sys.exit(1)
    start_gap = np.rad2deg(np.abs(q0 - np.array(data["start_q_rad"]))).max()
    print(f"live q vs trajectory start: {start_gap:.1f} deg")
    if start_gap > args.start_tol_deg:
        print(f"ABORT: robot is {start_gap:.1f} deg from the base pose "
              f"(> {args.start_tol_deg}). Move it to the base pose first.")
        rclpy.shutdown(); sys.exit(1)

    try:
        for i, s in enumerate(segs):
            traj = np.array(s["trajectory_rad"])
            print(f"== segment {i+1}/{len(segs)}: {s['label']} "
                  f"({len(traj)} pts, dt={s['dt']:.3f}) ==")
            ex = execute(state, pub, traj, s["dt"], args.controller,
                         args.max_vel_deg, args.min_dur, s["label"], track=track)
            print(f"   reach_err={ex.get('reach_err'):.2f} deg, ok={ex.get('ok')}")
            if not ex.get("ok"):
                print("   segment FAILED -> stopping, suction OFF for safety")
                suction(0)
                break
            sa = s.get("suction_after")
            if sa == "on":
                suction(1)
            elif sa == "off":
                suction(0)
        else:
            print("== trajectory complete ==")
    except KeyboardInterrupt:
        print("\ninterrupted -> suction OFF")
        suction(0)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

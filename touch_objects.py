#!/usr/bin/env python3
"""Touch the CENTRE of each tabletop object (no suction) with a small push -- a
calibration/perception check that reuses the pick pipeline's object detection.

From base: detect objects (SAM3 'everything', reuses real_multi.detect_objects), and for
each, move the tip to the centre of its TOP surface, descend to touch, then push --push
deeper. No suction is ever engaged. Returns to base.

Run in the ROS env with the D405 on PYTHONPATH; planner+bridge+SAM3 up.
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, quat_wxyz_to_R, make_T, transform_points
from real_multi import detect_objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--max-h", type=float, default=0.13); ap.add_argument("--max-foot", type=float, default=0.16)
    ap.add_argument("--xmin", type=float, default=0.15); ap.add_argument("--xmax", type=float, default=0.55)
    ap.add_argument("--ymin", type=float, default=-0.28); ap.add_argument("--ymax", type=float, default=0.30)
    ap.add_argument("--push", type=float, default=0.01, help="push this far below the touched surface (m)")
    ap.add_argument("--pre", type=float, default=0.06, help="pre-approach height above the surface (m)")
    ap.add_argument("--dwell", type=float, default=1.5)
    ap.add_argument("--vmax", type=float, default=18.0)
    ap.add_argument("--approach-vmax", type=float, default=12.0,
                    help="peak speed for the welded approach+descent (contact stays gentle "
                         "because the trajectory decelerates to rest at the goal)")
    ap.add_argument("--detect-tries", type=int, default=3, help="union this many SAM3 passes (robust to flaky masks)")
    ap.add_argument("--no-touch", action="store_true", help="detect + report only, no motion")
    args = ap.parse_args()

    import rclpy
    from std_msgs.msg import String
    from object_pointclouds import capture_aligned, deproject_mask
    from capture_and_plot import segment
    from multiview_fuse import pick_res
    from perturb_loop import PlannerClient, RobotState, execute

    pc = PlannerClient()
    rclpy.init(); node = rclpy.create_node("touch_objects")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))
    CW, CH = pick_res(args.serial)

    def fk_T(q):
        r = pc.rpc({"type": "fk", "q": list(map(float, q))})
        return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))

    def cam_pose(q):
        return fk_T(q) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM

    def fix_j6(traj):
        t = np.array(traj, float); t[:, 5] = t[0, 5]; return t

    def goto(xyz, label, vmax=None):
        q = state.get_q()
        r = pc.plan_pose(list(map(float, q)), list(map(float, xyz)) + down, max_attempts=14)
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED"); return False
        ex = execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid", vmax or args.vmax, 2.0, label, track=track)
        return ex.get("ok")

    def goto_welded(via_xyz, goal_xyz, label, vmax):
        """Plan current->via and via->goal, WELD into ONE trajectory and send as a single
        command. robot_hal's B-spline tracker streams through `via` as an interior point
        (no stop / re-plan), then decelerates to rest at `goal` (gentle contact)."""
        q = state.get_q()
        r1 = pc.plan_pose(list(map(float, q)), list(map(float, via_xyz)) + down, max_attempts=14)
        if not r1.get("success"):
            print(f"  [{label}] approach plan FAILED"); return False
        r2 = pc.plan_pose(list(map(float, r1["trajectory"][-1])), list(map(float, goal_xyz)) + down,
                          max_attempts=14)
        if not r2.get("success"):
            print(f"  [{label}] descent plan FAILED"); return False
        t1, t2 = np.array(r1["trajectory"], float), np.array(r2["trajectory"], float)
        traj = fix_j6(np.vstack([t1, t2[1:]]))            # weld (drop the duplicate seam waypoint)
        ex = execute(state, pub, traj, min(r1["dt"], r2["dt"]), "pid", vmax, 2.0, label, track=track)
        return ex.get("ok")

    def to_base():
        r = pc.plan_joint(list(map(float, state.get_q())), list(map(float, C.BASE_Q)))
        if r.get("success"):
            execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid", 22.0, 3.0, "to-base", track=track)

    to_base(); time.sleep(0.3)
    # union several SAM3 passes (per-call masks are flaky -> sometimes miss an object)
    objs = []
    for t in range(args.detect_tries):
        Tbc = cam_pose(state.get_q())
        rgb, depth, K, _ = capture_aligned(args.serial, CW, CH, 30, 30)
        for o in detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, args):
            if all(np.linalg.norm(o["centroid"][:2] - a["centroid"][:2]) > 0.04 for a in objs):
                objs.append(o)
    print(f"detected {len(objs)} object(s) over {args.detect_tries} passes:")
    targets = []
    for o in sorted(objs, key=lambda d: -d["n"]):
        P = o["pts"]
        cxy = o["centroid"][:2].copy()
        # surface z at the centre: median of the INTERIOR column (within 2 cm of the centre).
        # The camera only sees the real top face there -> no oblique edge/mixed-pixel floaters
        # (which sit ~2 cm above the object and inflate the cloud max).
        col = P[np.linalg.norm(P[:, :2] - cxy, axis=1) < 0.02]
        cz = float(np.median(col[:, 2])) if len(col) > 20 else float(np.percentile(P[:, 2], 80))
        face = P[np.abs(P[:, 2] - cz) < 0.010]            # refine centre on the top-face plane
        if len(face) > 20:
            cxy = face[:, :2].mean(0)
        print(f"  centre=[{cxy[0]:.3f}, {cxy[1]:.3f}] surf_z={cz:.3f} "
              f"(cloud-max {o['hi'][2]:.3f}) foot={o['foot']:.3f} n={o['n']}")
        targets.append((cxy, cz))

    if args.no_touch or not targets:
        rclpy.shutdown(); return

    for i, (cxy, cz) in enumerate(targets):
        print(f"\n== object {i+1}: touch centre [{cxy[0]:.3f}, {cxy[1]:.3f}] (surf_z={cz:.3f}) ==")
        via = [cxy[0], cxy[1], cz + args.pre]                 # pre-grasp standoff (vertical column)
        goal = [cxy[0], cxy[1], cz - args.push]               # through touch (cz) down to +push press
        # ONE smooth welded motion: gross approach -> straight descent -> gentle press
        if not goto_welded(via, goal, f"approach-touch{i}", args.approach_vmax):
            continue
        time.sleep(args.dwell)
        goto([cxy[0], cxy[1], cz + args.pre], f"lift{i}", vmax=10.0)            # lift off

    to_base()
    print("== done; touched each object centre (no suction) ==")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

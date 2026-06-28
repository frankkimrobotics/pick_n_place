#!/usr/bin/env python3
"""real_place :: place the HELD object into the box, considering the carried object volume.

Plans a top-down place over a 25 cm cube box (bottom-centre --box), and along every
segment VERIFIES that the carried object's OBB (measured from the grasp cloud, riding
the tcp) clears the box walls + ground -- raising the transit height until it does.
Then moves there and releases so the object drops into the box.

This is the "consider object volume during planning" requirement: the object's swept
volume is checked against the obstacles and the plan is rejected/raised if it collides.

Reuses perturb_loop (PlannerClient/RobotState/execute), sim_planner.box_wall_obstacles,
collision.obb_vs_walls, suction_test. Run in the ROS2 env (planner :9997, bridge up).
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc"))):
    sys.path.insert(0, p)
import config as C
from geometry import (R_from_two_axes, R_to_quat_wxyz, make_T, pose_to_T, T_to_pose,
                      inv_T, quat_wxyz_to_R, transform_points)
from collision import obb_vs_walls
from sim_planner import box_wall_obstacles


def read_ply_xyz(path):
    data = open(path, "rb").read()
    he = data.index(b"end_header\n") + len(b"end_header\n")
    n = [int(l.split()[-1]) for l in data[:he].decode().splitlines()
         if l.startswith("element vertex")][0]
    a = np.frombuffer(data[he:], dtype=np.dtype(
        [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]), count=n)
    return np.stack([a['x'], a['y'], a['z']], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="grasp output dir (has object.ply + grasp.json)")
    ap.add_argument("--box", default="0.1,0.4,0.0", help="box bottom-centre x,y,z")
    ap.add_argument("--box-size", type=float, default=0.25, help="cube outer size (m)")
    ap.add_argument("--wall", type=float, default=0.01)
    ap.add_argument("--release-z", type=float, default=0.30, help="tcp z to release over the box")
    ap.add_argument("--high-z", type=float, default=0.40, help="transit height for the tcp")
    ap.add_argument("--clear", type=float, default=0.01, help="required OBB clearance margin (m)")
    ap.add_argument("--vmax", type=float, default=22.0)
    ap.add_argument("--suction-host", default=os.environ.get("ROBOT_IP", "").strip())
    ap.add_argument("--suction-user", default="pi")
    ap.add_argument("--dry-run", action="store_true", help="plan + verify only, NO motion / no release")
    args = ap.parse_args()
    BOX = np.array([float(v) for v in args.box.split(",")])

    # ---- carried object OBB in the TCP frame (from the grasp cloud) ----
    cloud = read_ply_xyz(os.path.join(args.run_dir, "object.ply"))
    g = json.load(open(os.path.join(args.run_dir, "grasp.json")))
    T_bt_grasp = pose_to_T(np.array(g["grasp_pose_tcp"]))
    pts_tcp = transform_points(inv_T(T_bt_grasp), cloud)
    lo, hi = pts_tcp.min(0), pts_tcp.max(0)
    obb_dims = (hi - lo) + 2 * args.clear            # inflate by clearance margin
    T_tcp_obb = make_T(np.eye(3), (lo + hi) / 2.0)   # OBB centre in tcp frame
    print(f"carried OBB dims={np.round(obb_dims,3).tolist()} (tcp-frame centre "
          f"{np.round((lo+hi)/2,3).tolist()})")

    # ---- box walls + ground as (centre, dims) obstacles ----
    inner = args.box_size - 2 * args.wall
    walls = []
    for dims, pose in box_wall_obstacles(BOX, (inner, inner, args.box_size), args.wall).values():
        walls.append((np.array(pose[:3]), np.array(dims)))
    walls.append((np.array([0.0, 0.0, -0.12]), np.array([2.0, 2.0, 0.04])))  # ground

    import rclpy
    from std_msgs.msg import String
    from perturb_loop import PlannerClient, RobotState, execute
    import suction_test
    pc = PlannerClient()
    rclpy.init(); node = rclpy.create_node("real_place")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))

    def fk_many(qs):
        r = pc.rpc({"type": "fk", "q": [list(map(float, q)) for q in qs]})
        return [make_T(quat_wxyz_to_R(np.array(r["quat"][i])), np.array(r["pos"][i]))
                for i in range(len(qs))]

    def sweep_ok(traj):
        """Min clearance of the carried OBB to box+ground along traj; collided?"""
        Ts = fk_many(traj)
        collided, mc = False, np.inf
        for T in Ts:
            c, clr = obb_vs_walls(T @ T_tcp_obb, obb_dims, walls)
            collided = collided or c; mc = min(mc, clr)
        return collided, mc

    def plan_verify(q0, goal_pose, label):
        r = pc.plan_pose(list(map(float, q0)), goal_pose, max_attempts=12)
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED: {r.get('status')}"); return None
        collided, mc = sweep_ok(r["trajectory"])
        print(f"  [{label}] planned {len(r['trajectory'])} wpts; carried-OBB "
              f"{'COLLIDES' if collided else f'clear (min {mc*1e3:.0f}mm)'}")
        return None if collided else r

    q = state.get_q()
    if q is None:
        print("ABORT: no /joint_states"); return
    cur = pc.rpc({"type": "fk", "q": list(map(float, q))})
    cur_xyz = np.array(cur["pos"][0]); print("current tcp:", np.round(cur_xyz, 3).tolist())

    # segments: lift straight up -> over box (high) -> descend to release
    segs = [
        (np.r_[cur_xyz[:2], args.high_z], "lift-up"),
        (np.r_[BOX[:2], args.high_z], "over-box"),
        (np.r_[BOX[:2], args.release_z], "descend-to-release"),
    ]
    plans = []
    qref = q
    for xyz, label in segs:
        r = plan_verify(qref, list(map(float, xyz)) + down, label)
        if r is None:
            print("  -> aborting (plan failed or carried OBB would hit the box)."); rclpy.shutdown(); return
        plans.append((r, label)); qref = r["trajectory"][-1]

    if args.dry_run:
        print("DRY RUN ok: all segments planned and carried OBB clears the box. No motion.")
        rclpy.shutdown(); return

    for r, label in plans:
        ex = execute(state, pub, np.array(r["trajectory"]), r["dt"], "pid",
                     args.vmax, 2.0, label, track=track)
        print(f"  [{label}] {'OK' if ex.get('ok') else 'FAILED'} reach_err={ex.get('reach_err'):.1f}")
        if not ex.get("ok"):
            print("  segment failed; stopping (object still held)."); rclpy.shutdown(); return

    suction_test.set_pin(0, args.suction_host or None, args.suction_user)
    print(f"  RELEASE over box -> object drops in. pin="
          f"{suction_test.get_pin(args.suction_host or None, args.suction_user)}")
    print("== place complete ==")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

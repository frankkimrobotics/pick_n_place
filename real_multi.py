#!/usr/bin/env python3
"""real_multi :: pick-and-place ONE tabletop object per run (call repeatedly).

Geometric tabletop detection (no SAM3 text prompt): from the base pose it deprojects
the whole scene, finds the table height, clusters the points sitting ABOVE the table
within --max-h (default 0.13 m) into objects, and excludes the place box + oversized
clusters (cardboard). It then picks the best flat-top object (vertical suction, tcp
sealed with a small press), places it into the box (carried-OBB verified clear of the
box), verifies, and returns to base.

Reuses real_grasp (estimate_normals/detect_suction_point), collision.obb_vs_walls,
sim_planner.box_wall_obstacles, perturb_loop (PlannerClient/RobotState/execute),
suction_test. Run in the ROS2 env with the D405 + planner + bridge + SAM-not-needed.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from geometry import (R_from_two_axes, R_to_quat_wxyz, quat_wxyz_to_R, make_T,
                      transform_points)
from collision import obb_vs_walls
from sim_planner import box_wall_obstacles
from real_grasp import estimate_normals, detect_suction_point


def detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, args):
    """SAM3 'everything' object masks -> deproject -> filter to reachable tabletop
    objects <= max-h, excluding the place box / oversized cardboard. Returns list."""
    try:
        label, inst = segment("tcp://127.0.0.1:5599", rgb[:, :, ::-1].copy(), "",
                               50, 30000, extra={"mode": "everything"})
    except Exception as e:
        print("SAM3 everything failed:", e); return []
    objs = []
    for ins in inst:
        m = label == ins["id"]
        pts, _ = deproject_mask(m, depth, rgb, K, 0.05, 1.5)
        if len(pts) < 300:
            continue
        Pb = transform_points(Tbc, pts)
        cen = Pb.mean(0); lo = Pb.min(0); hi = Pb.max(0)
        foot = max(hi[0] - lo[0], hi[1] - lo[1]); h = hi[2] - lo[2]
        if not (args.xmin < cen[0] < args.xmax and args.ymin < cen[1] < args.ymax):
            continue
        if lo[2] > 0.12 or h > args.max_h or foot > args.max_foot:   # off-table / too tall / too big
            continue
        if np.linalg.norm(cen[:2] - np.array([0.1, 0.4])) < 0.16:    # the place box
            continue
        flat = float(np.mean(Pb[:, 2] > hi[2] - 0.008))   # fraction near top (flat-top -> high)
        objs.append({"id": int(ins["id"]), "centroid": cen, "lo": lo, "hi": hi,
                     "foot": float(foot), "height": float(h), "n": int(len(Pb)),
                     "pts": Pb.astype(np.float32), "flatness": flat})
    # dedup: drop objects whose centroid is within 4 cm of a larger kept one
    objs.sort(key=lambda d: -d["n"])
    keep = []
    for o in objs:
        if all(np.linalg.norm(o["centroid"][:2] - k["centroid"][:2]) > 0.04 for k in keep):
            keep.append(o)
    return keep


def detect_target(rgb, depth, K, Tbc, prompt, segment, deproject_mask, args):
    """Segment ONE specific named object by SAM3 text prompt (no size filtering -- these
    are the large cartons/box the user named explicitly). Returns the best reachable,
    well-populated instance as a single-element obj list, or [] if not found."""
    try:
        label, inst = segment("tcp://127.0.0.1:5599", rgb[:, :, ::-1].copy(), prompt,
                               10, 30000)
    except Exception as e:
        print("SAM3 text segment failed:", e); return []
    cand = []
    for ins in inst:
        m = label == ins["id"]
        pts, _ = deproject_mask(m, depth, rgb, K, 0.05, 1.5)
        if len(pts) < 300:
            continue
        Pb = transform_points(Tbc, pts)
        cen = Pb.mean(0); lo = Pb.min(0); hi = Pb.max(0)
        foot = max(hi[0] - lo[0], hi[1] - lo[1]); h = hi[2] - lo[2]
        if not (args.xmin < cen[0] < args.xmax and args.ymin < cen[1] < args.ymax):
            continue
        if np.linalg.norm(cen[:2] - np.array([0.1, 0.4])) < 0.16:    # the place box itself
            continue
        flat = float(np.mean(Pb[:, 2] > hi[2] - 0.008))
        cand.append({"id": int(ins["id"]), "centroid": cen, "lo": lo, "hi": hi,
                     "foot": float(foot), "height": float(h), "n": int(len(Pb)),
                     "pts": Pb.astype(np.float32), "flatness": flat,
                     "score": float(ins.get("score", 0.0))})
    if not cand:
        return []
    if args.near is not None:
        nx, ny = args.near
        cand = [c for c in cand if np.linalg.norm(c["centroid"][:2] - np.array([nx, ny])) < 0.10]
        if not cand:
            return []
        cand.sort(key=lambda d: np.linalg.norm(d["centroid"][:2] - np.array([nx, ny])))  # closest to hint
        return [cand[0]]
    cand.sort(key=lambda d: -(d["score"] * 1e6 + d["n"]))   # best-scoring, well-populated
    return [cand[0]]


def grasp_held(depth, K, T_base_cam, T_base_tcp, win=22):
    """Is an object held? Project the suction tip into the image and read the depth
    there: a held object sits right at the tip (small depth) while an empty cup sees
    the far table. Returns (held_bool_or_None, tip_depth, tip_optical_z)."""
    tip = T_base_tcp[:3, 3]
    pc = transform_points(np.linalg.inv(T_base_cam), tip[None])[0]   # tip in cam frame
    if pc[2] <= 0:
        return None, None, None
    u = int(K[0, 0] * pc[0] / pc[2] + K[0, 2]); v = int(K[1, 1] * pc[1] / pc[2] + K[1, 2])
    Hh, Ww = depth.shape
    if not (0 <= u < Ww and 0 <= v < Hh):
        return None, None, float(pc[2])
    w = depth[max(0, v - win):v + win, max(0, u - win):u + win]
    valid = w[w > 0.02]
    if len(valid) < 20:
        return None, None, float(pc[2])
    med = float(np.median(valid))
    held = med < pc[2] + 0.10                 # object within ~10 cm beyond the tip
    return bool(held), med, float(pc[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--target", default=None,
                    help="SAM3 text prompt for ONE specific object (e.g. 'orange juice carton'). "
                         "When set, size filters are skipped and this named object is picked.")
    ap.add_argument("--near", default=None,
                    help="X,Y hint (base frame): when --target matches several instances, "
                         "pick the one whose centroid is closest to this point (within 10 cm).")
    ap.add_argument("--max-h", type=float, default=0.13, help="max object height above table (m)")
    ap.add_argument("--max-foot", type=float, default=0.16, help="max object footprint (m)")
    ap.add_argument("--xmin", type=float, default=0.15); ap.add_argument("--xmax", type=float, default=0.50)
    ap.add_argument("--ymin", type=float, default=-0.28); ap.add_argument("--ymax", type=float, default=0.30)
    ap.add_argument("--box", default="0.1,0.4,0.0"); ap.add_argument("--box-size", type=float, default=0.25)
    ap.add_argument("--grasp-dz", type=float, default=0.005, help="fixed press fallback (if --no-contact)")
    ap.add_argument("--no-contact", action="store_true", help="use fixed grasp-dz press instead of vision contact")
    ap.add_argument("--contact-steps", type=int, default=8, help="max descend iterations")
    ap.add_argument("--contact-thresh", type=float, default=0.007,
                    help="when the measured gap is <= this, do the final seal press")
    ap.add_argument("--margin", type=float, default=0.005,
                    help="descend to this margin above the measured surface each step (then re-measure)")
    ap.add_argument("--seal-press", type=float, default=0.003, help="over-press past the surface to seal (m)")
    ap.add_argument("--pre-dist", type=float, default=0.06, help="pre-grasp standoff above the object (m)")
    ap.add_argument("--max-descend", type=float, default=0.03,
                    help="hard limit: cup may not descend more than this below the detected surface "
                         "(prevents diving to the floor on a bad/edge target)")
    ap.add_argument("--min-flat", type=float, default=0.05,
                    help="skip targets whose flat-top fraction is below this (likely floor/edge, not an object)")
    ap.add_argument("--release-z", type=float, default=0.30); ap.add_argument("--high-z", type=float, default=0.40)
    ap.add_argument("--vmax", type=float, default=45.0, help="joint speed for fast moves (approach/lift/place); <=55")
    ap.add_argument("--contact-vmax", type=float, default=8.0, help="slow joint speed for the contact descent")
    ap.add_argument("--suction-host", default=os.environ.get("ROBOT_IP", "").strip())
    ap.add_argument("--suction-user", default="pi")
    ap.add_argument("--dry-run", action="store_true", help="detect + plan/verify only, NO motion")
    args = ap.parse_args()
    BOX = np.array([float(v) for v in args.box.split(",")])
    if args.near is not None:
        args.near = tuple(float(v) for v in args.near.split(","))

    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from object_pointclouds import capture_aligned, deproject_mask
    from capture_and_plot import segment
    from multiview_fuse import pick_res
    from perturb_loop import PlannerClient, RobotState, execute
    import suction_test

    pc = PlannerClient()
    rclpy.init(); node = rclpy.create_node("real_multi")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))
    CW, CH = pick_res(args.serial)

    def fix_j6(traj):
        """Hold joint6 (wrist roll) constant at its start value across the trajectory.
        J6 rotates about the tool axis, so the tcp position + approach are unchanged --
        only the (irrelevant for a symmetric suction cup) yaw differs. Protects the J6
        cable from being wound up by the planner's free wrist roll."""
        t = np.array(traj, float)
        t[:, 5] = t[0, 5]
        return t

    def fk_T(q):
        r = pc.rpc({"type": "fk", "q": list(map(float, q))})
        return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))

    def fk_many(qs):
        r = pc.rpc({"type": "fk", "q": [list(map(float, q)) for q in qs]})
        return [make_T(quat_wxyz_to_R(np.array(r["quat"][i])), np.array(r["pos"][i])) for i in range(len(qs))]

    def cam_pose(q):
        return fk_T(q) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM

    def goto(goal, label, vmax=None):
        q = state.get_q()
        r = pc.plan_pose(list(map(float, q)), goal, max_attempts=12)
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED {r.get('status')}"); return None
        ex = execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid", vmax or args.vmax, 2.0, label, track=track)
        print(f"  [{label}] {'OK' if ex.get('ok') else 'FAIL'} err={ex.get('reach_err'):.1f}")
        return r["trajectory"][-1] if ex.get("ok") else None

    def _descend_to(newz, label):
        q = state.get_q(); Tbt = fk_T(q)
        goal = [float(Tbt[0, 3]), float(Tbt[1, 3]), float(newz)] + down
        r = pc.plan_pose(list(map(float, q)), goal, max_attempts=8)
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED"); return False
        execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid", args.contact_vmax, 1.0, label, track=track)
        return True

    def measure_gap():
        """Live gap (m) = surface depth in the annulus around the cup tip minus the cup
        tip depth; 0 if the cup occludes the surface. Returns (gap, tcp_z)."""
        q = state.get_q(); Tbt = fk_T(q); Tbc = cam_pose(q)
        rgb, depth, K, _ = capture_aligned(args.serial, CW, CH, 30, 30)
        tip = Tbt[:3, 3]
        pcam = transform_points(np.linalg.inv(Tbc), tip[None])[0]; tip_z = float(pcam[2])
        u = int(K[0, 0] * pcam[0] / pcam[2] + K[0, 2]); v = int(K[1, 1] * pcam[1] / pcam[2] + K[1, 2])
        Hh, Ww = depth.shape; w = 60
        sub = depth[max(0, v - w):min(Hh, v + w), max(0, u - w):min(Ww, u + w)]
        valid = sub[sub > 0.02]; surf = valid[valid > tip_z + 0.012]
        gap = 0.0 if len(surf) < 25 else float(np.percentile(surf, 25)) - tip_z
        return gap, float(Tbt[2, 3])

    def descend_to_contact(P):
        """Descend using the MEASURED gap (not fixed steps): each step, move to --margin
        above the measured surface, then re-measure (the cup gets closer -> far less depth
        noise than the base view) until gap <= --contact-thresh, then a final seal press
        onto the surface. Floor-limited so a bad/edge target can't reach the floor."""
        floor_z = float(P[2]) - args.max_descend
        for it in range(args.contact_steps):
            gap, tcpz = measure_gap()
            print(f"  [descend {it}] gap={gap*1000:.0f}mm  (tcp z={tcpz:.3f})")
            if tcpz <= floor_z + 1e-4:
                print("  hit floor limit -> NO OBJECT"); return False
            if gap <= args.contact_thresh:                  # close enough -> seal onto surface
                _descend_to(max(floor_z, tcpz - (gap + args.seal_press)), "seal-press")
                return True
            target = max(floor_z, tcpz - (gap - args.margin))   # leave --margin above the surface
            if not _descend_to(target, f"descend{it}"):
                return False
        gap, tcpz = measure_gap()                           # out of steps -> seal at current
        _descend_to(max(floor_z, tcpz - (gap + args.seal_press)), "seal-press")
        return True

    def refine_grasp_at(P_coarse):
        """Re-capture at the (close) current pose and recompute the central flat-top
        suction point near P_coarse. The close view has far less depth noise than the
        base view, so this recalculates the desired grasp point/depth. Returns refined P."""
        q = state.get_q(); Tbc = cam_pose(q)
        rgb, depth, K, _ = capture_aligned(args.serial, CW, CH, 30, 30)
        pts, _ = deproject_mask(np.ones(depth.shape, bool), depth, rgb, K, 0.05, 0.6)
        Pb = transform_points(Tbc, pts)
        sel = ((np.abs(Pb[:, 0] - P_coarse[0]) < 0.05) & (np.abs(Pb[:, 1] - P_coarse[1]) < 0.05) &
               (Pb[:, 2] > -0.02) & (Pb[:, 2] < float(P_coarse[2]) + 0.04))
        objp = Pb[sel]
        if len(objp) < 100:
            print("  refine: too few points; keeping coarse P"); return np.asarray(P_coarse, float)
        nrm = estimate_normals(objp, Tbc[:3, 3])
        found = detect_suction_point(objp, nrm, normal_cone_deg=45.0, select="central")
        if found is None:
            print("  refine: no patch; keeping coarse P"); return np.asarray(P_coarse, float)
        print(f"  refined grasp point {np.round(found[0],4).tolist()} (was {np.round(P_coarse,4).tolist()})")
        return np.asarray(found[0], float)

    # ---- 1. capture scene + detect objects ----
    q0 = state.get_q()
    if q0 is None:
        print("ABORT: no /joint_states"); rclpy.shutdown(); return
    Tbc = cam_pose(q0)
    rgb, depth, K, _ = capture_aligned(args.serial, CW, CH, 30, 30)
    if args.target:
        objs = detect_target(rgb, depth, K, Tbc, args.target, segment, deproject_mask, args)
        print(f"target {args.target!r}: {len(objs)} instance(s)")
        for o in objs:
            print(f"  centroid={np.round(o['centroid'],3).tolist()} h={o['height']:.3f} "
                  f"foot={o['foot']:.3f} flat={o['flatness']:.2f} n={o['n']} score={o['score']:.2f}")
        if not objs:
            print(f"target {args.target!r} NOT FOUND."); rclpy.shutdown(); return
        obj = objs[0]
    else:
        objs = detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, args)
        print(f"detected {len(objs)} pickable objects (<= {args.max_h} m):")
        for o in sorted(objs, key=lambda d: -d["flatness"]):
            print(f"  centroid={np.round(o['centroid'],3).tolist()} h={o['height']:.3f} "
                  f"foot={o['foot']:.3f} flat={o['flatness']:.2f} n={o['n']}")
        objs = [o for o in objs if o["flatness"] >= args.min_flat]   # drop floor/edge phantoms
        if not objs:
            print("NO OBJECTS LEFT (none above min-flat)."); rclpy.shutdown(); return
        # pick the flattest, well-populated object (best suction target)
        obj = max(objs, key=lambda d: (d["flatness"], d["n"]))
    P0 = obj["centroid"]
    print(f"\n== TARGET object @ {np.round(P0,3).tolist()} (flat={obj['flatness']:.2f}) ==")

    # ---- 2. detect the suction point on this object (vertical) ----
    nrm = estimate_normals(obj["pts"], Tbc[:3, 3])
    found = detect_suction_point(obj["pts"], nrm, normal_cone_deg=45.0,
                                 select="central", centroid=obj["centroid"])
    if found is None:
        print("  no vertical suction patch on this object."); rclpy.shutdown(); return
    P, n = found
    Pc = P - np.array([0, 0, args.grasp_dz])
    Rg = R_from_two_axes(np.array([0, 0, -1.0]))
    print(f"  suction P={np.round(P,4).tolist()} -> grasp contact {np.round(Pc,4).tolist()}")

    # carried OBB (this object) in tcp frame, assuming grasp at Pc with vertical tcp
    Tbt_g = make_T(Rg, Pc)
    pts_tcp = transform_points(np.linalg.inv(Tbt_g), obj["pts"])
    lo, hi = pts_tcp.min(0), pts_tcp.max(0)
    obb_dims = (hi - lo) + 0.02; T_tcp_obb = make_T(np.eye(3), (lo + hi) / 2)

    # A carried object hangs below the tcp by down_extent. To drop it into the box its
    # bottom must clear the rim (= box bottom z + box height), so raise the transit /
    # release height for tall objects. (Small flat objects keep the default low release.)
    rim_z = float(BOX[2]) + args.box_size
    down_extent = float(Pc[2] - obj["lo"][2])
    need_z = rim_z + down_extent + 0.03
    if need_z > args.high_z:
        print(f"  tall object (hangs ~{down_extent*100:.0f}cm below tip) -> "
              f"raising transit/release height to {need_z:.2f} m (clears {rim_z:.2f} m rim)")
        args.high_z = need_z
    args.release_z = max(args.release_z, need_z)

    inner = args.box_size - 0.02
    walls = [(np.array(p[:3]), np.array(d)) for d, p in
             box_wall_obstacles(BOX, (inner, inner, args.box_size), 0.01).values()]
    walls.append((np.array([0, 0, -0.12]), np.array([2, 2, 0.04])))

    def sweep_clear(traj):
        for T in fk_many(traj):
            c, _ = obb_vs_walls(T @ T_tcp_obb, obb_dims, walls)
            if c:
                return False
        return True

    if args.dry_run:
        print("DRY RUN: object + suction point detected; skipping motion.")
        rclpy.shutdown(); return

    # ---- 3. PICK: pre-grasp(6cm) -> REFINE grasp from a close low-noise capture ->
    #         suction ON -> margin-based descend-to-contact -> lift ----
    print("== PICK ==")
    pre = list(map(float, P + [0, 0, args.pre_dist])) + down
    if goto(pre, "pre-grasp") is None: rclpy.shutdown(); return
    if not args.no_contact:
        time.sleep(0.3)
        P = refine_grasp_at(P)                            # recompute desired grasp point/depth, close-up
        Pc = P - np.array([0, 0, args.grasp_dz])
        q = state.get_q(); Tbt = fk_T(q)
        goto([float(P[0]), float(P[1]), float(Tbt[2, 3])] + down, "re-centre")   # over the refined xy
    suction_test.set_pin(1, args.suction_host or None, args.suction_user)
    print("  suction ON")
    if args.no_contact:                                   # fixed-distance press fallback
        if goto(list(map(float, Pc)) + down, "press", vmax=12) is None: rclpy.shutdown(); return
    else:                                                 # vision contact descent (slow, floor-limited)
        if not descend_to_contact(P):
            suction_test.set_pin(0, args.suction_host or None, args.suction_user)
            rb = pc.plan_joint(list(map(float, state.get_q())), list(map(float, C.BASE_Q)))
            if rb.get("success"):
                execute(state, pub, fix_j6(rb["trajectory"]), rb["dt"], "pid", 25.0, 3.0, "to-base", track=track)
            print("== no object at target (floor limit); to base =="); rclpy.shutdown(); return
    time.sleep(1.0)
    q = state.get_q(); Tbt = fk_T(q)
    goto([float(Tbt[0, 3]), float(Tbt[1, 3]), float(Tbt[2, 3] + 0.12)] + down, "lift")  # fast

    # ---- 3b. VERIFY grasp via depth at the suction tip ----
    time.sleep(0.5)
    qv = state.get_q()
    rgb2, depth2, K2, _ = capture_aligned(args.serial, CW, CH, 30, 30)
    held, tipd, tipz = grasp_held(depth2, K2, cam_pose(qv), fk_T(qv))
    print(f"  GRASP CHECK: held={held} (tip depth={tipd}, tip optical z={tipz})")
    if held is not True:
        print("  -> object NOT grasped; suction OFF, skipping place, returning to base.")
        suction_test.set_pin(0, args.suction_host or None, args.suction_user)
        rb = pc.plan_joint(list(map(float, state.get_q())), list(map(float, C.BASE_Q)))
        if rb.get("success"):
            execute(state, pub, np.array(rb["trajectory"]), rb["dt"], "pid", 25.0, 3.0, "to-base", track=track)
        print("== pick FAILED for this object; re-run (more press) ==")
        rclpy.shutdown(); return

    # ---- 4. PLACE into box (carried-OBB verified each segment) ----
    print("== PLACE (object confirmed held) ==")
    # Aim so the OBJECT's centroid (not the tcp) lands at the bin centre: a large object
    # grasped off-centre would overhang a wall if we centred the tcp. The grasp is rigid
    # and vertical (no rotation), so the world xy offset (centroid - grasp point) is
    # preserved through transit; subtract it from the bin target.
    place_off = obj["centroid"][:2] - P[:2]
    tgt = BOX[:2] - place_off
    print(f"  place offset (centroid-grasp)={np.round(place_off,3).tolist()} -> tcp target {np.round(tgt,3).tolist()}")
    # Go directly up-and-over to the box (the planner arcs up as it moves inboard, where
    # the higher z is reachable). A straight vertical lift at a FAR pick x (e.g. milk at
    # x=0.52) is out of reach, so we don't pre-lift -- the over-box plan rises on its own.
    segs = [(np.r_[tgt, args.high_z], "over-box"),
            (np.r_[tgt, args.release_z], "descend")]
    for xyz, label in segs:
        q = state.get_q()
        r = pc.plan_pose(list(map(float, q)), list(map(float, xyz)) + down, max_attempts=12)
        traj = fix_j6(r["trajectory"]) if r.get("success") else None
        if traj is None or not sweep_clear(traj):
            print(f"  [{label}] plan/clear FAILED -> stopping (object held)."); rclpy.shutdown(); return
        execute(state, pub, traj, r["dt"], "pid", args.vmax, 2.0, label, track=track)
        print(f"  [{label}] ok")
    suction_test.set_pin(0, args.suction_host or None, args.suction_user)
    print("  RELEASE -> object drops in box")

    # ---- 5. back to base ----
    rb = pc.plan_joint(list(map(float, state.get_q())), list(map(float, C.BASE_Q)))
    if rb.get("success"):
        execute(state, pub, fix_j6(rb["trajectory"]), rb["dt"], "pid", 25.0, 3.0, "to-base", track=track)
    print("== object done; re-run for the next one ==")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""real_grasp :: REAL-robot suction pick (no place), reusing the online-viewpoint
capture pattern from mycobot_mpc/multiview_fuse.py.

Sequence:
  1. MOVE the eye-in-hand D405 to a VIEWPOINT looking at the object (plan_pose +
     execute -- the same path multiview_fuse uses), settle, read the ACTUAL q,
     FK -> T_base_cam = FK(tcp) @ T_tcp_cam (hand-eye).
  2. CAPTURE aligned RGBD, segment (SAM 3), deproject -> partial cloud in base frame.
  3. DETECT the 1 cm circular suction point.
  4. MOVE the suction tip onto it (plan_pose to the grasp pose), execute.
  5. suction ON (digital-out pin), LIFT. No box.

Reuses (un-edited): perturb_loop.PlannerClient / RobotState / execute,
object_pointclouds.capture_aligned / deproject_mask / write_ply,
capture_and_plot.segment, the hand-eye T_tcp_cam (config.T_TCP_CAM).

Tip convention -- REAL robot: the cuRobo planner's tool frame `tcp` (0.145 m, touch-
tested and tuned on hardware) IS the suction tip, so grasp goals are published
directly (NO offset). (Sim uses eef=0.105 m; that's separate.)

Infra up: ROS bridge (-> /joint_states), planner :9997 (fk+plan), SAM 3 :5599, D405.
Run:
  source ~/Desktop/2026/ros2node/config/ros2node.env
  PYTHONPATH=~/librealsense/build/release python3 pick_and_place/real_grasp.py \
      --obj 0.35,0.08,0.0 --prompt object
  # safe checkpoint (move to viewpoint + capture + detect, NO grasp descent / suction):
  ... --no-grasp
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
MPC = os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc"))
PERC = os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))
for p in (HERE, MPC, PERC):
    sys.path.insert(0, p)

import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, quat_wxyz_to_R, make_T, transform_points

BASE_Q = [0.0, -0.349066, 1.396263, 0.174533, -1.570796, 0.0]    # robot rest pose (LinuxCNC [0,-110,80,-80,-90,0])


# ---- small helpers (mirror multiview_fuse conventions) -------------------- #
def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def lookat_R(p_cam, target, up_ref=(0, 0, 1.)):
    """Camera optical frame at p_cam looking at target (+Z fwd, +Y down, +X right)."""
    z = np.array(target, float) - np.array(p_cam, float); z /= np.linalg.norm(z)
    up = np.array(up_ref, float)
    if abs(np.dot(up, z)) > 0.95:
        up = np.array([0, 1., 0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def pose7(T):
    return list(T[:3, 3]) + list(R_to_quat_wxyz(T[:3, :3]))


# ---- 1 cm circular suction-point detection (numpy + scipy) ---------------- #
def estimate_normals(pts, cam, k=30):
    tree = cKDTree(pts); nrm = np.zeros_like(pts); kk = min(k, len(pts))
    for i, p in enumerate(pts):
        _, idx = tree.query(p, k=kk)
        c = pts[idx].mean(0)
        _, _, Vt = np.linalg.svd(pts[idx] - c, full_matrices=False)
        n = Vt[-1]
        if n @ (cam - p) < 0:
            n = -n
        nrm[i] = n
    return nrm


def detect_suction_point(pts, nrm, up=np.array([0.0, 0.0, 1.0]),
                         flat_tol=None, normal_cone_deg=70.0, top_band=0.010,
                         select="highest", centroid=None, support_r=None):
    """Point whose 3 cm-DIAMETER patch is low-curvature + fully supported + graspable.

    The cup is 1 cm, but it may land up to ~1 cm off the commanded point, so we require
    the SUPPORT region (radius `support_r`, default 1.5 cm -> 3 cm dia) around each
    candidate to be smooth and flat: local curvature = RMS deviation of that neighbourhood
    from its best-fit plane (low RMS = flat over the whole 3 cm circle). Among all patches
    that are flat enough over 3 cm, well supported, and whose normal faces up enough to
    suck from above:
      select="highest" -> take the flattest near the top (within `top_band`).
      select="central" -> take the one NEAREST the object centroid (xy) -- the
                          airtight middle of a flat top, best for sealing.
    flat_tol auto-relaxes if nothing qualifies. Returns (P, n) or None.
    """
    r = C.GRASP_SUPPORT_RADIUS if support_r is None else support_r
    up = up / np.linalg.norm(up)
    tree = cKDTree(pts)
    samp = pts[np.linspace(0, len(pts) - 1, min(400, len(pts))).astype(int)]
    counts = np.array([len(tree.query_ball_point(p, r)) for p in samp])
    min_support = max(6, int(0.55 * np.percentile(counts, 80)))
    cone = np.cos(np.deg2rad(normal_cone_deg))

    cand = []                                          # (z, rms, P, plane_normal)
    for i in np.where((nrm @ up) > cone)[0]:
        p = pts[i]
        nb = tree.query_ball_point(p, r)
        if len(nb) < min_support:
            continue
        loc = pts[nb]; cc = loc.mean(0)
        _, _, Vt = np.linalg.svd(loc - cc, full_matrices=False)
        pn = Vt[-1]
        if pn @ nrm[i] < 0:
            pn = -pn
        rms = float(np.sqrt(np.mean(((loc - cc) @ pn) ** 2)))   # local curvature proxy
        cand.append((float(p[2]), rms, p, pn))
    if not cand:
        return None

    base = C.FLAT_RMS_TOL if flat_tol is None else flat_tol
    for tol in (base, 1.5 * base, 2.5 * base, 4.0 * base):       # relax if needed
        flat = [c for c in cand if c[1] <= tol]
        if flat:
            if select == "central":
                # centre of the FLAT TOP region (upward-facing flat points) -- robust to
                # the oblique-view bias that pulls the whole-cloud centroid toward a side face.
                fxy = np.mean([c[2][:2] for c in flat], axis=0)
                best = min(flat, key=lambda c: np.linalg.norm(c[2][:2] - fxy))
                how = f"flat-top centre {np.round(fxy,3).tolist()}"
            else:
                zmax = max(c[0] for c in flat)
                top = [c for c in flat if c[0] >= zmax - top_band]
                best = min(top, key=lambda c: c[1])
                how = f"highest (top-band z>={zmax-top_band:.3f})"
            print(f"  [detect] flat_tol={tol*1e3:.1f}mm: {len(flat)} flat patches; "
                  f"{how}; picked xyz={np.round(best[2],3).tolist()} "
                  f"curv_rms={best[1]*1e3:.2f}mm")
            return best[2], best[3]
    return None


def grasp_pose(P, n):
    n = n / (np.linalg.norm(n) + 1e-12)
    return np.concatenate([P, R_to_quat_wxyz(R_from_two_axes(-n))])   # tcp +Z = -normal


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Real-robot suction pick (no place) with online viewpoint capture.")
    ap.add_argument("--obj", default="0.35,0.08,0.0", help="approx object location x,y,z (m) to aim the camera")
    ap.add_argument("--prompt", default="object", help="SAM3 text prompt")
    ap.add_argument("--serial", default="218622271300", help="D405 serial")
    ap.add_argument("--slant", type=float, default=0.25, help="camera slant range to OBJ (m)")
    ap.add_argument("--elev-deg", type=float, default=55.0, help="viewpoint elevation")
    ap.add_argument("--az-deg", type=float, default=0.0, help="viewpoint azimuth")
    ap.add_argument("--vmax", type=float, default=25.0, help="peak joint speed deg/s (<=55)")
    ap.add_argument("--crop", type=float, default=0.15, help="half-box around OBJ for the cloud (m)")
    ap.add_argument("--zmax", type=float, default=0.35)
    ap.add_argument("--grasp-dz", type=float, default=0.0,
                    help="lower the grasp contact this many metres along world -Z (e.g. 0.01 = 1 cm down)")
    ap.add_argument("--standoff", type=float, default=C.PREGRASP_STANDOFF)
    ap.add_argument("--lift", type=float, default=C.LIFT_HEIGHT)
    ap.add_argument("--suction-dwell", type=float, default=1.5,
                    help="seconds held at the pressed pose (let the cup conform/seal) before lifting")
    ap.add_argument("--suction-host", default=os.environ.get("ROBOT_IP", "").strip())
    ap.add_argument("--suction-user", default="pi")
    ap.add_argument("--here", action="store_true", help="capture from the CURRENT pose (no viewpoint move)")
    ap.add_argument("--vertical", action="store_true",
                    help="grasp straight DOWN (top-down) on the highest top-facing point, "
                         "instead of along the local surface normal")
    ap.add_argument("--no-grasp", action="store_true", help="viewpoint + capture + detect only; no grasp/suction")
    ap.add_argument("--verify", action="store_true",
                    help="after the lift, re-capture RGBD and judge pick success (bottle lifted vs still on table)")
    ap.add_argument("--no-suction", action="store_true", help="do the motion but never toggle suction")
    args = ap.parse_args()
    OBJ = np.array([float(v) for v in args.obj.split(",")])

    # reused infra (un-edited): planner client, robot state, executor, perception
    import rclpy
    from std_msgs.msg import String
    from perturb_loop import PlannerClient, RobotState, execute
    from object_pointclouds import capture_aligned, deproject_mask, write_ply
    from capture_and_plot import segment
    import suction_test

    T_tcp_cam = C.T_TCP_CAM
    T_cam_tcp = np.linalg.inv(T_tcp_cam)
    pc = PlannerClient()
    print("planner:", pc.rpc({"type": "ping"}).get("backend"), flush=True)

    def fk_T(q):
        r = pc.rpc({"type": "fk", "q": list(map(float, q))})
        return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))

    rclpy.init()
    node = rclpy.create_node("real_grasp")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
    state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}

    def move_to_tcp(goalT, label, vmax=None):
        """Plan current_q -> tcp pose goalT and execute. Returns ok."""
        q = state.get_q()
        if q is None:
            print("ABORT: no /joint_states (bridge up?)"); return False
        r = pc.plan_pose(list(map(float, q)), pose7(goalT))
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED: {r.get('status')}"); return False
        ex = execute(state, pub, np.array(r["trajectory"]), r["dt"], "pid",
                     vmax or args.vmax, 2.0, label, track=track)
        ok = ex.get("ok") and ex.get("reach_err", 999) <= 12
        print(f"  [{label}] {'OK' if ok else 'FAILED'} reach_err={ex.get('reach_err'):.1f} deg")
        return ok

    try:
        if state.get_q() is None:
            print("ABORT: no /joint_states — start mycobot_ros2_bridge.py."); return

        # ---- 1. move to a VIEWPOINT looking at OBJ (roll sweep for reachability) ----
        if args.here:
            print("== 1. VIEWPOINT skipped (--here: capturing from current pose) ==")
        else:
            el, az = math.radians(args.elev_deg), math.radians(args.az_deg)
            d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            p_cam = OBJ + args.slant * d
            Rbc = lookat_R(p_cam, OBJ)
            q_cur = state.get_q()
            best = None
            for th in range(0, 360, 30):
                Tbc = make_T(Rbc @ Rz(math.radians(th)), p_cam)
                Tbt = Tbc @ T_cam_tcp
                r = pc.plan_pose(list(map(float, q_cur)), pose7(Tbt))
                if not r.get("success"):
                    continue
                travel = float(np.rad2deg(np.abs(np.array(r["trajectory"][-1]) - q_cur)).max())
                if best is None or travel < best[0]:
                    best = (travel, Tbt, th)
            if best is None:
                print("ABORT: viewpoint UNREACHABLE at all rolls; adjust --obj/--slant/--elev-deg")
                return
            print(f"== 1. VIEWPOINT az={args.az_deg:.0f} elev={args.elev_deg:.0f} "
                  f"roll={best[2]} (travel {best[0]:.0f} deg) ==")
            if not move_to_tcp(best[1], "viewpoint"):
                return
            time.sleep(0.4)

        # ---- 2. CAPTURE partial cloud in base frame ----
        print("== 2. CAPTURE ==")
        q_act = state.get_q()
        # hand-eye correction: T_tcp_cam was calibrated at the old (longer) tcp; the
        # planner tcp is now PLANNER_TCP_LEN, so shift the extrinsic by CAM_TCP_Z_SHIFT
        # along tcp +Z to keep the camera pose (hence the grasp point) accurate.
        T_base_cam = fk_T(q_act) @ make_T(np.eye(3), [0.0, 0.0, C.CAM_TCP_Z_SHIFT]) @ T_tcp_cam
        from multiview_fuse import pick_res
        CW, CH = pick_res(args.serial)
        rgb, depth, K, _ = capture_aligned(args.serial, CW, CH, 30, 30)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        label, inst = segment("tcp://127.0.0.1:5599", bgr, args.prompt, 20, 20000)
        best_pts = None; bestd = 1e9
        for ins in inst:
            m = label == ins["id"]
            pts, cols = deproject_mask(m, depth, rgb, K, 0.05, 2.0)
            if len(pts) == 0:
                continue
            Pb = transform_points(T_base_cam, pts)
            sel = (np.abs(Pb[:, 0] - OBJ[0]) < args.crop) & (np.abs(Pb[:, 1] - OBJ[1]) < args.crop) & \
                  (Pb[:, 2] > -0.03) & (Pb[:, 2] < args.zmax)
            if sel.sum() < 30:
                continue
            cen = Pb[sel].mean(0); dd = np.linalg.norm(cen[:2] - OBJ[:2])
            if dd < bestd:
                bestd = dd; best_pts = (Pb[sel].astype(np.float32), cols[sel])
        if best_pts is None:
            print("ABORT: no object points in the crop (check --obj / prompt / view)"); return
        pts_base, cols = best_pts
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(C.OUT_DIR, f"real_grasp_{stamp}")
        os.makedirs(out, exist_ok=True)
        write_ply(os.path.join(out, "object.ply"), pts_base, cols)
        print(f"  object cloud: {len(pts_base)} pts, centroid={np.round(pts_base.mean(0),4).tolist()} -> {out}/object.ply")

        # ---- 3. DETECT 1 cm suction point ----
        print("== 3. DETECT suction point ==")
        nrm = estimate_normals(pts_base, T_base_cam[:3, 3])
        # --vertical: only consider TOP-facing patches (normal near +Z) so a straight
        # top-down cup seals flat on the highest point of the (lying) bottle.
        cone = 30.0 if args.vertical else 70.0
        found = detect_suction_point(pts_base, nrm, normal_cone_deg=cone)
        if found is None:
            print("ABORT: no valid suction patch found "
                  f"({'top-facing ' if args.vertical else ''}1 cm, low curvature)."); return
        P, n = found
        # lower the contact point along world -Z by --grasp-dz (corrects the cup stopping
        # short of the surface; e.g. 0.01 = descend 1 cm past the detected top).
        Pc = P - np.array([0.0, 0.0, args.grasp_dz])
        if args.vertical:
            Rg = R_from_two_axes(np.array([0.0, 0.0, -1.0]))      # tcp +Z = straight down
            pre_xyz = Pc + np.array([0.0, 0.0, args.standoff])    # pre-grasp straight above
            approach = "VERTICAL top-down"
        else:
            Rg = R_from_two_axes(-n / (np.linalg.norm(n) + 1e-12))
            pre_xyz = Pc + args.standoff * n
            approach = "surface-normal"
        lift_xyz = Pc + np.array([0.0, 0.0, args.lift])
        gp = np.concatenate([Pc, R_to_quat_wxyz(Rg)])
        print(f"  detected top P={np.round(P,4).tolist()} surf-normal={np.round(n,3).tolist()} "
              f"approach={approach}")
        print(f"  grasp contact (P - {args.grasp_dz*100:.1f}cm z)={np.round(Pc,4).tolist()}")
        print(f"  grasp pose (tcp)={np.round(gp,4).tolist()}")
        json.dump({"obj": OBJ.tolist(), "P": P.tolist(), "grasp_contact": Pc.tolist(),
                   "grasp_dz": args.grasp_dz, "normal": n.tolist(), "approach": approach,
                   "grasp_pose_tcp": gp.tolist(), "q_view": list(map(float, q_act))},
                  open(os.path.join(out, "grasp.json"), "w"), indent=2)

        if args.no_grasp:
            print("== --no-grasp: stopping after capture+detect (no descent, no suction) ==")
            return

        # ---- 4. GRASP: pre-grasp -> suction ON (before contact) -> press -> lift ----
        print("== 4. GRASP ==")
        pre = make_T(Rg, pre_xyz)
        gT = make_T(Rg, Pc)
        if not move_to_tcp(pre, "pre-grasp"):
            return
        if not args.no_suction:                      # engage vacuum BEFORE contact
            suction_test.set_pin(1, args.suction_host or None, args.suction_user)
            print(f"  [suction] ON (before contact) "
                  f"pin={suction_test.get_pin(args.suction_host or None, args.suction_user)}")
        if not move_to_tcp(gT, "press onto object", vmax=min(args.vmax, 12)):
            return
        time.sleep(args.suction_dwell)               # let the cup conform + seal
        move_to_tcp(make_T(Rg, lift_xyz), "lift", vmax=min(args.vmax, 15))

        # ---- 6. VERIFY: post-lift RGBD -> is the bottle up at the tip or still on the table? ----
        if args.verify:
            print("== 6. VERIFY (post-lift RGBD) ==")
            time.sleep(0.6)
            q_v = state.get_q()
            T_cam_v = fk_T(q_v) @ make_T(np.eye(3), [0.0, 0.0, C.CAM_TCP_Z_SHIFT]) @ T_tcp_cam
            rgb2, depth2, K2, _ = capture_aligned(args.serial, CW, CH, 30, 30)
            lab2, inst2 = segment("tcp://127.0.0.1:5599", np.ascontiguousarray(rgb2[:, :, ::-1]),
                                  args.prompt, 20, 20000)
            best_b = None
            for ins in inst2:
                m = lab2 == ins["id"]
                pts, _ = deproject_mask(m, depth2, rgb2, K2, 0.05, 2.0)
                if len(pts) < 50:
                    continue
                Pb = transform_points(T_cam_v, pts)
                if best_b is None or len(Pb) > len(best_b):
                    best_b = Pb
            thresh = float(P[2]) + 0.05                  # lifted well above the grasp/table height
            if best_b is None:
                verdict, ok = "UNKNOWN (no bottle segmented after lift)", None
            else:
                cz = float(np.median(best_b[:, 2]))
                ok = cz > thresh
                verdict = (f"{'PICKED' if ok else 'NOT PICKED'} — bottle median z={cz:.3f} m "
                           f"vs threshold {thresh:.3f} (grasp z={P[2]:.3f}), {len(best_b)} pts")
            print(f"  RESULT: {verdict}")
            import cv2
            vis = np.ascontiguousarray(rgb2[:, :, ::-1])
            col = (0, 200, 0) if ok else ((0, 0, 230) if ok is False else (0, 180, 230))
            cv2.putText(vis, ("PICKED" if ok else ("NOT PICKED" if ok is False else "UNKNOWN")),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.imwrite(os.path.join(out, "verify.png"), vis)
            json.dump({"picked": ok, "verdict": verdict}, open(os.path.join(out, "verify.json"), "w"))
        print("== done: pick attempt complete (not placed). ==")
    except KeyboardInterrupt:
        if not args.no_suction:
            try:
                suction_test.set_pin(0, args.suction_host or None, args.suction_user)
            except Exception:
                pass
        print("\ninterrupted -> suction OFF")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

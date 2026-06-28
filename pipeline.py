"""pipeline :: the full pick-and-place sequence (simulation).

Stages:
  1. OBSERVE   move so the eye-in-hand camera looks down at the object; render RGBD
  2. PERCEIVE  segment object mask -> deproject -> base-frame cloud (+normals)
  3. DETECT    find the 1 cm circular suction grasp point; compute object OBB
  4. PICK      plan->pre-grasp, descend, ACTIVATE SUCTION, lift
  5. PLACE     transport to the box (segmented OR planned w/ OBB collision volume),
               lower, RELEASE SUCTION
  6. (evaluation is done by evaluate.py on the returned record)

Returns a `record` dict consumed by evaluate.py / run_demo.py.
"""
import numpy as np

import config as C
import scene as SCN
import perception as PCP
import grasp_detection as GD
import obb as OBB
from geometry import (make_T, look_at_R, pose_to_T, T_to_pose, inv_T,
                      quat_wxyz_to_R)
from sim_planner import SimPlanner, grasp_pose_from_point_normal
from simulator import SuctionSim
import viz


def _record_exec(rec, label, traj, dt, suction_after=None):
    """Append a robot-executable joint segment (URDF rad waypoints + dt)."""
    rec.setdefault("exec_segments", []).append({
        "label": label,
        "trajectory_rad": np.asarray(traj, float).tolist(),
        "dt": float(dt) if dt else 0.02,
        "suction_after": suction_after})


def _plan_and_exec(planner, sim, start_q, goal_pose, label, rec, max_attempts=10,
                   capture_every=6, use_tip=False):
    """Plan start_q->goal_pose, execute on the sim. Returns (ok, end_q).

    use_tip=True -> goal_pose is a SUCTION-TIP (eef, 0.105 m) pose; otherwise it is a
    tcp pose (used for the camera-relative observe move)."""
    planfn = planner.plan_pose_tip if use_tip else planner.plan_pose
    res = planfn(start_q, goal_pose, max_attempts=max_attempts)
    seg = {"label": label, "planned": bool(res["success"]),
           "status": res.get("status"), "n_waypts": len(res.get("trajectory", [])),
           "motion_time": res.get("motion_time")}
    if not res["success"]:
        rec["segments"].append(seg)
        print(f"  [{label}] PLAN FAILED ({res.get('status')})")
        return False, np.asarray(start_q)
    traj = np.asarray(res["trajectory"])
    ex = sim.execute(traj, capture_every=capture_every, label=label)
    seg.update(ex)
    rec["segments"].append(seg)
    _record_exec(rec, label, traj, res.get("dt"))
    cl = "" if ex["min_clearance"] is None else f", OBB clearance={ex['min_clearance']*1e3:.0f}mm"
    print(f"  [{label}] ok: {len(traj)} steps{cl}"
          f"{', COLLISION!' if ex['collided'] else ''}")
    return True, traj[-1]


def _planned_place(planner, sim, start_q, place_pose, hover_pose, rec):
    """Single collision-free planned motion to the in-box pose, OBB-gated.

    Plans start->place_pose directly; if the carried OBB would clip the box
    (our sweep), reroutes via a safe hover and lowers. Returns (ok4, ok5, q).
    """
    res = planner.plan_pose_tip(start_q, place_pose, max_attempts=16)
    if res["success"]:
        collided, clr = sim.dry_sweep(res["trajectory"])
        if not collided:
            ex = sim.execute(res["trajectory"], capture_every=3,
                             label="planned-direct-to-box")
            rec["segments"].append({"label": "planned-direct-to-box", "planned": True,
                                    "status": res.get("status"),
                                    "n_waypts": len(res["trajectory"]), **ex})
            _record_exec(rec, "planned-direct-to-box", res["trajectory"], res.get("dt"))
            print(f"  [planned-direct-to-box] single collision-free motion, "
                  f"OBB clearance={clr*1e3:.0f}mm")
            return True, True, np.asarray(res["trajectory"])[-1]
        print(f"  [planned] direct path would clip box (OBB) -> rerouting via hover")
    else:
        print(f"  [planned] direct plan failed ({res.get('status')}) -> via clear pose")
    # fallback: explicit safe route via the high clear pose
    ok4, q = _plan_and_exec(planner, sim, start_q, hover_pose, "reroute-high",
                            rec, max_attempts=14, use_tip=True)
    ok5, q = _plan_and_exec(planner, sim, q, place_pose, "to-release", rec,
                            capture_every=3, use_tip=True)
    return ok4, ok5, q


def run_pipeline(place_mode="planned", box_xyz=None, object_xyz=None,
                 save_artifacts=True, verbose=True):
    box_xyz = C.BOX_POSE_XYZ if box_xyz is None else np.asarray(box_xyz, float)
    scene = SCN.build_scene(object_pose_xyz=object_xyz, box_xyz=box_xyz)
    oi, bi = scene["object_info"], scene["box_info"]

    rec = {"place_mode": place_mode, "box_xyz": list(map(float, box_xyz)),
           "object_xyz": list(map(float, oi["pose_xyz"])), "segments": [],
           "stages": {}}

    print(f"== Building planner (box at {np.round(box_xyz,3)}) ==")
    planner = SimPlanner(box_xyz=box_xyz, include_box=True)
    renderer = viz.SceneRenderer(scene, bi) if save_artifacts else None
    sim = SuctionSim(planner, scene, frame_renderer=renderer)
    sim.goto(C.START_Q, capture=save_artifacts)

    # ---------- 1. OBSERVE ------------------------------------------------- #
    print("== 1. OBSERVE ==")
    top = oi["top_centre"]
    eye = top + np.array([0.0, 0.0, 0.30])                  # 30 cm above object
    T_cam_des = make_T(look_at_R(eye, top), eye)
    T_tcp_des = T_cam_des @ inv_T(C.T_TCP_CAM)
    ok, q_obs = _plan_and_exec(planner, sim, C.START_Q, T_to_pose(T_tcp_des),
                               "observe", rec, max_attempts=12)
    if ok:
        T_base_cam = sim.P.fk_T(q_obs) @ C.T_TCP_CAM         # actual camera pose
        cam_mode = "eye-in-hand (reached)"
    else:
        T_base_cam = T_cam_des                               # virtual fallback
        cam_mode = "virtual (pose unreached)"
        print("  observe pose unreached; rendering from desired camera pose.")
    frame = PCP.render_rgbd(T_base_cam, scene)
    rec["stages"]["observe"] = {"camera": cam_mode,
                                "cam_pos": list(map(float, T_base_cam[:3, 3]))}

    # ---------- 2. PERCEIVE ------------------------------------------------ #
    print("== 2. PERCEIVE ==")
    mask = PCP.segment_object(frame)
    dp = PCP.deproject(mask, frame)
    print(f"  segmented object: {dp['count']} pts; "
          f"centroid={np.round(dp['pts_base'].mean(0),3)}")
    rec["stages"]["perceive"] = {"n_points": dp["count"],
                                 "centroid": list(map(float, dp["pts_base"].mean(0)))}

    # ---------- 3. DETECT grasp + OBB ------------------------------------- #
    print("== 3. DETECT (1 cm suction circle) ==")
    grasp = GD.detect_suction_grasp(dp["pts_base"], dp["normals_base"])
    if grasp is None:
        rec["stages"]["detect"] = {"found": False}
        rec["failed"] = "no_grasp_detected"
        return _finish(rec, sim, frame, mask, None, None, scene, save_artifacts)
    objc = OBB.compute_obb_grounded(dp["pts_base"], support_z=oi["pose_xyz"][2])
    rec["stages"]["detect"] = {
        "found": True, "point": list(map(float, grasp["point"])),
        "normal": list(map(float, grasp["normal"])),
        "plane_rms_mm": grasp["plane_rms"] * 1e3,
        "support_pts": grasp["n_support"], "candidates": grasp["candidates"],
        "obb_dims": list(map(float, objc["dims"]))}
    print(f"  grasp @ {np.round(grasp['point'],4)} normal {np.round(grasp['normal'],2)} "
          f"rms={grasp['plane_rms']*1e3:.2f}mm; OBB dims={np.round(objc['dims'],3)}")

    # ---------- 4. PICK ---------------------------------------------------- #
    print("== 4. PICK ==")
    q = q_obs if ok else C.START_Q
    ok1, q = _plan_and_exec(planner, sim, q, grasp["pregrasp_pose"], "pre-grasp", rec,
                            use_tip=True)
    ok2, q = _plan_and_exec(planner, sim, q, grasp["grasp_pose"], "approach", rec,
                            capture_every=3, use_tip=True)
    seal_ok, pe, ae = (False, None, None)
    if ok2:
        seal_ok, pe, ae = sim.activate_suction(grasp["point"], grasp["normal"], obb=objc)
        print(f"  ACTIVATE SUCTION: {'SEALED' if seal_ok else 'FAILED'} "
              f"(pos_err={pe*1e3:.1f}mm, ang_err={ae:.1f}deg)")
        if seal_ok and rec.get("exec_segments"):
            rec["exec_segments"][-1]["suction_after"] = "on"   # ON after the approach
        if seal_ok and place_mode == "planned":
            attached = planner.attach_obb(q, objc["pose7"], objc["dims"])
            rec["stages"]["curobo_attach"] = attached
    rec["stages"]["pick"] = {"pregrasp_ok": ok1, "approach_ok": ok2,
                             "seal_ok": seal_ok,
                             "seal_pos_err_mm": None if pe is None else pe * 1e3,
                             "seal_ang_err_deg": ae}
    # place geometry (TALL box): grasped on top, so object bottom = tcp.z - h.
    # Release with the object bottom a clear margin ABOVE the box rim (z=0.25),
    # per the task ("move to height greater than 25 cm, then release").
    grasp_quat = grasp["grasp_pose"][3:7]
    h = oi["height"]
    gp = grasp["point"]
    bx, by = bi["centre_xy"]
    release_tcp_z = bi["rim_z"] + C.RELEASE_CLEARANCE + h
    clear_pose = np.concatenate([[gp[0], gp[1], release_tcp_z], grasp_quat])  # straight up
    release_pose = np.concatenate([[bx, by, release_tcp_z], grasp_quat])      # over the box

    # lift the object straight up, clear above the box rim, before traversing
    ok3, q = _plan_and_exec(planner, sim, q, clear_pose, "lift-above-rim", rec,
                            capture_every=3, max_attempts=12, use_tip=True)

    # ---------- 5. PLACE: move over the box, release ABOVE the rim --------- #
    print(f"== 5. PLACE (mode={place_mode}) ==")
    if place_mode == "segmented":
        ok4, q = _plan_and_exec(planner, sim, q, release_pose, "move-over-box",
                                rec, max_attempts=16, use_tip=True)
        ok5 = ok4
    else:
        # planned: single collision-free motion to the release pose with the object
        # OBB as the collision volume; reroute via the lift-clear pose if the carried
        # OBB would clip the box.
        ok4, ok5, q = _planned_place(planner, sim, q, release_pose, clear_pose, rec)

    obj_z = float(sim.object_T[2, 3])
    obj_bottom_z = obj_z - h / 2.0
    released_above_rim = bool(obj_bottom_z > bi["rim_z"])
    if sim.attached:
        if rec.get("exec_segments"):
            rec["exec_segments"][-1]["suction_after"] = "off"  # OFF after reaching release
        sim.release_suction()
        print(f"  RELEASE SUCTION over box: object bottom z={obj_bottom_z:.3f} "
              f"(rim {bi['rim_z']:.2f}) -> {'ABOVE rim ✓' if released_above_rim else 'NOT above rim ✗'}")
        sim.settle_into_box()                            # object drops into the bin
        if place_mode == "planned":
            planner.detach()
    ok6, q = _plan_and_exec(planner, sim, q, clear_pose, "retract", rec, use_tip=True)

    rec["stages"]["place"] = {
        "transport_ok": ok4, "move_ok": ok5, "retract_ok": ok6,
        "rim_z": float(bi["rim_z"]),
        "release_obj_bottom_z": obj_bottom_z,
        "released_above_rim": released_above_rim,
        "object_final_xyz": list(map(float, sim.object_T[:3, 3]))}

    return _finish(rec, sim, frame, mask, grasp, objc, scene, save_artifacts)


def _export_robot_trajectory(rec):
    """Write the robot-executable joint trajectory (URDF rad + suction events).

    Consumed by robot_execute.py, which streams it to /mycobot/cmd/move so the
    Pi's robot_hal controller follows it.
    """
    import os, json
    segs = rec.get("exec_segments", [])
    out = {
        "joint_names": list(C.JOINT_NAMES),
        "units": "urdf_rad",
        "start_q_rad": list(map(float, C.START_Q)),
        "suction_pin": "pro600.digital_out00",
        "place_mode": rec.get("place_mode"),
        "box_xyz": rec.get("box_xyz"),
        "n_segments": len(segs),
        "n_waypoints": int(sum(len(s["trajectory_rad"]) for s in segs)),
        "segments": segs,
    }
    os.makedirs(C.OUT_DIR, exist_ok=True)
    path = os.path.join(C.OUT_DIR, "robot_trajectory.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"  robot trajectory -> {path} "
          f"({out['n_segments']} segments, {out['n_waypoints']} waypoints)")
    return path


def _finish(rec, sim, frame, mask, grasp, objc, scene, save_artifacts):
    rec["_sim"] = sim
    rec["_scene"] = scene
    _export_robot_trajectory(rec)
    if save_artifacts:
        import os
        os.makedirs(C.OUT_DIR, exist_ok=True)
        viz.save_eye_in_hand(frame, mask, os.path.join(C.OUT_DIR, "eye_in_hand.png"))
        if grasp is not None:
            viz.save_grasp_still(PCP.deproject(mask, frame), grasp, objc,
                                 os.path.join(C.OUT_DIR, "grasp_detection.png"))
        vid = viz.save_video(sim.frames, os.path.join(C.OUT_DIR, "demo.mp4"))
        rec["video"] = vid
        print(f"  artifacts -> {C.OUT_DIR} (frames={len(sim.frames)}, video={vid})")
    return rec

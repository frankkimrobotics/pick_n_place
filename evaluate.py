"""evaluate :: success criteria + metrics for a pick-and-place run.

A run PASSES iff all of:
  1. detection      — a valid 1 cm circular suction patch was found
  2. reachability    — every planned motion segment succeeded
  3. seal            — suction seal check passed (pos & angle tolerances)
  4. collision_free  — no carried-OBB collision with the box during transport
  5. placed          — object final pose lies inside the box footprint and rests
                       on/within the bin, roughly upright
"""
import numpy as np

import config as C


def evaluate(rec):
    st = rec.get("stages", {})
    crit = {}
    details = {}

    # 1. detection
    det = st.get("detect", {})
    crit["detection"] = bool(det.get("found", False))
    details["detection"] = {"candidates": det.get("candidates"),
                            "plane_rms_mm": det.get("plane_rms_mm")}

    # 2. reachability — all segments planned & executed
    segs = rec.get("segments", [])
    all_planned = len(segs) > 0 and all(s.get("planned") for s in segs)
    crit["reachability"] = bool(all_planned)
    details["reachability"] = {"n_segments": len(segs),
                               "failed": [s["label"] for s in segs
                                          if not s.get("planned")]}

    # 3. seal
    pick = st.get("pick", {})
    crit["seal"] = bool(pick.get("seal_ok", False))
    details["seal"] = {"pos_err_mm": pick.get("seal_pos_err_mm"),
                       "ang_err_deg": pick.get("seal_ang_err_deg"),
                       "tol_pos_mm": C.SEAL_POS_TOL * 1e3,
                       "tol_ang_deg": C.SEAL_ANG_TOL_DEG}

    # 4. collision-free transport (carried OBB vs box)
    collided = any(s.get("collided") for s in segs if s.get("collided") is not None)
    clears = [s["min_clearance"] for s in segs
              if s.get("min_clearance") is not None]
    crit["collision_free"] = (not collided)
    details["collision_free"] = {
        "any_collision": bool(collided),
        "min_clearance_mm": (min(clears) * 1e3) if clears else None}

    # 5. released above the box rim (task: lift above 25 cm before release)
    pl_stage = st.get("place", {})
    crit["released_above_rim"] = bool(pl_stage.get("released_above_rim", False))
    details["released_above_rim"] = {
        "object_bottom_z": pl_stage.get("release_obj_bottom_z"),
        "rim_z": pl_stage.get("rim_z")}

    # 6. placed inside box
    placed_ok, place_det = _check_placed(rec)
    crit["placed"] = placed_ok
    details["placed"] = place_det

    overall = all(crit.values())
    return {"PASS": bool(overall), "criteria": crit, "details": details}


def _check_placed(rec):
    sim = rec.get("_sim")
    scene = rec.get("_scene")
    if sim is None or scene is None:
        return False, {"reason": "no sim state"}
    bi = scene["box_info"]
    info = scene["object_info"]
    T = sim.object_T
    c = T[:3, 3]
    # XY inside inner footprint?
    dxy = np.abs(c[:2] - bi["centre_xy"])
    margin = bi["half_xy"] - dxy - info["radius"]
    inside_xy = bool(np.all(margin > -1e-3))
    # within bin height (object centre between floor and rim)
    inside_z = bool(bi["floor_z"] - 1e-3 <= c[2] <= bi["rim_z"] + info["height"])
    # upright: object +Z still ~ world +Z
    upz = T[:3, 2]
    upright = bool(np.degrees(np.arccos(np.clip(upz @ [0, 0, 1], -1, 1))) < 20.0)
    ok = inside_xy and inside_z and upright
    return ok, {"final_xyz": list(map(float, c)),
                "inside_xy": inside_xy, "inside_z": inside_z, "upright": upright,
                "xy_margin_mm": list(map(lambda v: float(v * 1e3), margin))}


def format_report(rec, ev):
    lines = []
    lines.append("=" * 56)
    lines.append(f" PICK-AND-PLACE EVALUATION  —  {'PASS ✅' if ev['PASS'] else 'FAIL ❌'}")
    lines.append("=" * 56)
    lines.append(f" place mode : {rec.get('place_mode')}")
    lines.append(f" box xyz    : {np.round(rec.get('box_xyz'),3).tolist()}")
    lines.append(f" object xyz : {np.round(rec.get('object_xyz'),3).tolist()}")
    lines.append("-" * 56)
    names = {"detection": "1cm suction-circle detection",
             "reachability": "all motions reachable/planned",
             "seal": "suction seal valid",
             "collision_free": "collision-free transport (OBB)",
             "released_above_rim": "released above box rim (>25cm)",
             "placed": "object placed inside box"}
    for k, v in ev["criteria"].items():
        lines.append(f"  [{'PASS' if v else 'FAIL'}] {names.get(k,k)}")
    lines.append("-" * 56)
    d = ev["details"]
    if d["seal"]["pos_err_mm"] is not None:
        lines.append(f"  seal: pos_err={d['seal']['pos_err_mm']:.1f}mm "
                     f"(tol {d['seal']['tol_pos_mm']:.0f}), "
                     f"ang_err={d['seal']['ang_err_deg']:.1f}deg "
                     f"(tol {d['seal']['tol_ang_deg']:.0f})")
    mc = d["collision_free"]["min_clearance_mm"]
    if mc is not None:
        lines.append(f"  carried-OBB min clearance to box: {mc:.0f} mm")
    rr = d.get("released_above_rim", {})
    if rr.get("object_bottom_z") is not None:
        lines.append(f"  release height: object bottom z={rr['object_bottom_z']:.3f} m "
                     f"vs rim z={rr['rim_z']:.2f} m")
    pl = d["placed"]
    if "final_xyz" in pl:
        lines.append(f"  object final xyz: {np.round(pl['final_xyz'],3).tolist()} "
                     f"(inside_xy={pl['inside_xy']}, upright={pl['upright']})")
    lines.append("=" * 56)
    return "\n".join(lines)

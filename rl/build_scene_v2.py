#!/usr/bin/env python3
"""build_scene_v2 :: MJCF for the DIVERSIFIED paper pick task (rl/scenes/diverse_v2.xml).

Starts from the repo's generated warp scene (rl/scenes/box_med.xml, produced by
sim_robot_mjcf.build(warp=True, ...) -- see env_warp.build_scene_xml) so the robot model,
cameras, materials and the place bin are exactly the ones the rest of the pipeline uses, then:

  * TABLE grown to x in [0.12, 0.60], y in [-0.40, 0.40], top at z = 0 (2 cm slab).
    The real table top is at z = 0; keep it there.
  * WALLS as collision geoms: back slab at x = -0.30 and side slabs at y = +-0.50
    (2 cm thick, z 0..1.0).  They use contype=0 / conaffinity=5 so they collide with the
    default-mask world (object, cup tip) AND with the arm collision proxies below.
  * ARM COLLISION PROXIES: the generated robot's link meshes are contype=0/conaffinity=0
    (the cup tip sphere is the only collision geom on the arm), so "robot touches wall"
    could never fire.  Massless capsule proxies on link1..link5 + the suction cup with
    contype=4 / conaffinity=0 pair ONLY with the walls (4 & 5 != 0), never with each
    other (4 & 0 == 0), the table or the object -- so no existing dynamics change.
  * ONE object body with THREE geoms (box / cylinder / hex-prism mesh).  mujoco_warp 3.12
    batches geom_size, geom_pos, geom_dataid, geom_rbound, geom_aabb, geom_friction,
    geom_rgba, body_mass, body_inertia, body_ipos per world (put_model(batch_sizes=...)),
    but geom_TYPE is NOT batched -- so the shape is selected by keeping exactly one geom
    at its sampled size and shrinking the other two to a 1 mm stub parked at the object's
    centroid, where the active geom fully encloses them (they can never touch anything).
    The mesh geom's "stub" is a 1 mm hex mesh selected through the batched geom_dataid.
    Meshes cannot be rescaled at runtime, so the hex prism comes from a baked
    len(HEX_R) x len(HEX_H) mesh library and its dimensions are quantised to that grid;
    box and cylinder dimensions are continuous.
  * The object body's frame origin sits at the BOTTOM-CENTRE of the object (the geoms are
    offset up by their half height, body_ipos likewise).  That makes xpos[object0].z the
    object's lift above the table for ANY size, which is what lets DiverseEnv reuse the
    parent's `float(self.half[2])` arithmetic unchanged with half = (0, 0, 0).

    $PY rl/build_scene_v2.py            # -> rl/scenes/diverse_v2.xml
"""
import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCENES = os.path.join(HERE, "scenes")
CUROBO_PY = "/home/lisc-frank/miniconda3/envs/curobo2/bin/python"

OUT_XML = os.path.join(SCENES, "diverse_v2.xml")
OUT_XML_V3 = os.path.join(SCENES, "obstacle_v3.xml")
TEMPLATE = os.path.join(SCENES, "box_med.xml")

# ---- scene geometry (robot base frame, table top at z = 0) -------------------
TABLE_X2 = (0.12, 0.60)
TABLE_Y2 = (-0.40, 0.40)
TABLE_TH = 0.02
WALL_TH = 0.02
WALL_Z = 1.0
WALL_BACK_X = -0.30
WALL_SIDE_Y = 0.50
WALL_BACK_Y = 0.60          # back slab spans y +- 0.60
WALL_SIDE_X = (-0.30, 0.80)  # side slabs span x -0.30 .. 0.80
WALL_NAMES = ("wall_back", "wall_ym", "wall_yp")

# ---- object variant ranges ---------------------------------------------------
BOX_HALF = (0.015, 0.045)        # per axis
CYL_R = (0.018, 0.045)
CYL_H = (0.012, 0.050)           # half height
MASS_RANGE = (0.03, 0.15)
FRIC_RANGE = (0.6, 1.0)
# baked hexagonal-prism mesh library (circumradius x half height)
HEX_R = np.linspace(CYL_R[0], CYL_R[1], 6)
HEX_H = np.linspace(CYL_H[0], CYL_H[1], 5)
HEX_STUB = 0.001                 # 1 mm stub mesh used to "switch the mesh geom off"
STUB = 0.001                     # stub half-size for the box / cylinder geoms

# collision masks
MASK_PROXY_TYPE, MASK_PROXY_AFF = 4, 0
MASK_WALL_TYPE, MASK_WALL_AFF = 0, 5
# DISTRACTORS (v3).  contype 1 / conaffinity 5 makes them collide with
#   the table plane, the target object and the cup tip   (1 & 1)
#   the arm collision proxies (contype 4, conaffinity 0)  (4 & 5)
#   the walls (contype 0, conaffinity 5)                  (1 & 5)
# and with each other -- i.e. with everything that matters and nothing that does not.
MASK_DIST_TYPE, MASK_DIST_AFF = 1, 5
N_DIST = 3                       # bodies present in the v3 scene (1-3 active per world)
DIST_PARK = (2.0, 2.0)           # xy where an inactive distractor is parked (outside the walls)
DIST_BOX_HW = (0.020, 0.050)     # half width of a normal distractor
DIST_BOX_HH = (0.015, 0.060)     # half height of a normal distractor
POST_HW = (0.018, 0.035)         # half width / radius of a TALL POST
POST_HH = (0.050, 0.125)         # half height of a tall post (0.10 - 0.25 m tall)
DIST_MASS = (0.15, 0.60)


def dist_geom_names(k):
    return (f"d{k}_box", f"d{k}_cyl")


def dist_body_name(k):
    return f"dist{k}"

# massless capsule proxies: body name -> (fromto, radius)
ARM_PROXIES = {
    "link1": ((0, 0, -0.06, 0, 0, 0.03), 0.070),
    "link2": ((0, 0, 0, 0.27, 0, 0), 0.060),
    "link3": ((0, 0, 0, 0.267, 0, -0.0745), 0.055),
    "link4": ((0, 0, 0, 0, -0.095, 0), 0.045),
    "link5": ((0, 0, 0, -0.054, 0, 0), 0.042),
    "suction_cup": ((0, 0, 0, 0, 0, -0.10), 0.030),
}


def hex_vertices(r, h, n=6):
    """Vertices of a regular n-gon prism, circumradius r, half height h, axis = +z."""
    a = np.arange(n) * 2 * np.pi / n
    top = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, h)], 1)
    bot = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, -h)], 1)
    return np.concatenate([top, bot], 0)


def hex_mesh_name(i, j):
    return f"hex_r{i}_h{j}"


def hex_library():
    """[(name, r, h)] of the baked hex meshes, index 0 = the 1 mm stub."""
    lib = [("hex_stub", HEX_STUB, HEX_STUB)]
    for i, r in enumerate(HEX_R):
        for j, h in enumerate(HEX_H):
            lib.append((hex_mesh_name(i, j), float(r), float(h)))
    return lib


def _f(v):
    return " ".join(f"{float(x):.6f}" for x in np.atleast_1d(v))


def _ensure_template(path):
    if os.path.exists(path):
        return path
    tup = [("object0", "box", "0.0250 0.0250 0.0200", "0.38 0.0 0.0200", "0.8 0.3 0.3 1", "1 0 0 0")]
    code = (f"import sys; sys.path.insert(0, {ROOT!r});\n"
            f"import sim_robot_mjcf as g;\n"
            f"g.build(warp=True, objects={tup!r}, light_pos=(0.3, 0.1, 1.5), out_path={path!r})")
    subprocess.run([CUROBO_PY, "-c", code], check=True, capture_output=True)
    return path


def build(out_path=OUT_XML, template=TEMPLATE, n_dist=0):
    template = _ensure_template(template)
    tree = ET.parse(template)
    root = tree.getroot()
    root.set("model", "mycobot_diverse_v2")
    comp = root.find("compiler")
    comp.set("meshdir", "meshes")            # bundle like the other rl/scenes/*.xml
    wb = root.find("worldbody")
    asset = root.find("asset")

    # ---- baked hex-prism mesh library ---------------------------------------
    for name, r, h in hex_library():
        ET.SubElement(asset, "mesh", {"name": name, "vertex": _f(hex_vertices(r, h).ravel())})
    ET.SubElement(asset, "material", {"name": "wall", "rgba": "0.62 0.62 0.66 0.35"})

    # ---- table -> big table, top at z = 0 -----------------------------------
    # The table SLAB is visual-only and a co-planar collision PLANE carries the contacts:
    # mujoco_warp 3.12 has no multicontact for the CYLINDER-BOX CCD pair ("MULTICCD is
    # enabled, but the scene contains CCD pairs without multicontact support"), so a wide
    # upright cylinder resting on a box table gets ONE contact point and sinks up to 1 cm
    # into it.  PLANE-<primitive> pairs are dedicated multi-contact colliders, so every
    # object shape rests correctly on the plane.  The plane is infinite, so "the object
    # slid off the table" is a code-level termination (DiverseEnv.reward), not a fall.
    for g in wb.findall("geom"):
        if g.get("name") == "table":
            g.set("pos", _f([0.5 * (TABLE_X2[0] + TABLE_X2[1]), 0.5 * (TABLE_Y2[0] + TABLE_Y2[1]), -0.5 * TABLE_TH]))
            g.set("size", _f([0.5 * (TABLE_X2[1] - TABLE_X2[0]), 0.5 * (TABLE_Y2[1] - TABLE_Y2[0]), 0.5 * TABLE_TH]))
            g.set("contype", "0")
            g.set("conaffinity", "0")
            idx = list(wb).index(g)
            wb.insert(idx + 1, ET.Element("geom", {
                "name": "table_top", "type": "plane", "size": "1.5 1.5 0.05", "pos": "0 0 0",
                "rgba": "0 0 0 0", "group": "3", "contype": "1", "conaffinity": "1"}))

    # ---- walls ---------------------------------------------------------------
    walls = [
        ("wall_back", (WALL_BACK_X, 0.0, 0.5 * WALL_Z), (0.5 * WALL_TH, WALL_BACK_Y, 0.5 * WALL_Z)),
        ("wall_ym", (0.5 * (WALL_SIDE_X[0] + WALL_SIDE_X[1]), -WALL_SIDE_Y, 0.5 * WALL_Z),
         (0.5 * (WALL_SIDE_X[1] - WALL_SIDE_X[0]), 0.5 * WALL_TH, 0.5 * WALL_Z)),
        ("wall_yp", (0.5 * (WALL_SIDE_X[0] + WALL_SIDE_X[1]), WALL_SIDE_Y, 0.5 * WALL_Z),
         (0.5 * (WALL_SIDE_X[1] - WALL_SIDE_X[0]), 0.5 * WALL_TH, 0.5 * WALL_Z)),
    ]
    anchor = list(wb).index([g for g in wb.findall("geom") if g.get("name") == "table"][0]) + 1
    for k, (nm, pos, size) in enumerate(walls):
        e = ET.Element("geom", {"name": nm, "type": "box", "pos": _f(pos), "size": _f(size),
                                "material": "wall", "contype": str(MASK_WALL_TYPE),
                                "conaffinity": str(MASK_WALL_AFF)})
        wb.insert(anchor + k, e)

    # ---- arm collision proxies (wall-only mask) ------------------------------
    for body in root.iter("body"):
        spec = ARM_PROXIES.get(body.get("name"))
        if spec is None:
            continue
        ft, r = spec
        ET.SubElement(body, "geom", {
            "name": f"prox_{body.get('name')}", "type": "capsule", "fromto": _f(ft),
            "size": f"{r:.4f}", "mass": "0", "group": "3", "rgba": "0.9 0.2 0.2 0.12",
            "contype": str(MASK_PROXY_TYPE), "conaffinity": str(MASK_PROXY_AFF)})

    # ---- object body: 3 geoms, origin at the object's BOTTOM centre ----------
    old = [b for b in wb.findall("body") if b.get("name") == "object0"]
    for b in old:
        wb.remove(b)
    hz0 = 0.02
    obj = ET.SubElement(wb, "body", {"name": "object0", "pos": "0.38 0 0.0", "quat": "1 0 0 0"})
    ET.SubElement(obj, "freejoint", {"name": "obj0_free"})
    # explicit inertial => the three geoms contribute NO mass; body_mass / body_inertia /
    # body_ipos are written per world at reset.
    ET.SubElement(obj, "inertial", {"pos": _f([0, 0, hz0]), "mass": "0.05",
                                    "diaginertia": _f([1e-5, 1e-5, 1e-5])})
    ET.SubElement(obj, "geom", {"name": "g_box", "type": "box", "size": _f([0.025, 0.025, hz0]),
                                "pos": _f([0, 0, hz0]), "material": "o0", "mass": "0"})
    ET.SubElement(obj, "geom", {"name": "g_cyl", "type": "cylinder", "size": _f([STUB, STUB]),
                                "pos": _f([0, 0, hz0]), "material": "o0", "mass": "0"})
    ET.SubElement(obj, "geom", {"name": "g_hex", "type": "mesh", "mesh": "hex_stub",
                                "pos": _f([0, 0, hz0]), "material": "o0", "mass": "0"})

    # ---- distractor bodies (v3): free, 2 geoms (box | upright cylinder) -------
    # Same trick as the target object: ONE body per distractor carrying both primitive
    # geoms; the inactive one is shrunk to a 1 mm stub parked at the active geom's
    # centroid (geom_TYPE is not batchable in mujoco_warp).  The body origin is the
    # BOTTOM CENTRE, so body_pos.z is the distractor's lift for any size, which is what
    # the "displaced > 1 cm" test and the obstacle cuboids sent to cuRobo both want.
    for k in range(n_dist):
        nb, (gb, gc) = dist_body_name(k), dist_geom_names(k)
        hz0 = 0.03
        b = ET.SubElement(wb, "body", {"name": nb,
                                       "pos": _f([DIST_PARK[0] + 0.15 * k, DIST_PARK[1], 0.0]),
                                       "quat": "1 0 0 0"})
        ET.SubElement(b, "freejoint", {"name": f"{nb}_free"})
        ET.SubElement(b, "inertial", {"pos": _f([0, 0, hz0]), "mass": "0.3",
                                      "diaginertia": _f([1e-4, 1e-4, 1e-4])})
        for nm, typ, size in ((gb, "box", [0.03, 0.03, hz0]), (gc, "cylinder", [STUB, STUB])):
            ET.SubElement(b, "geom", {
                "name": nm, "type": typ, "size": _f(size), "pos": _f([0, 0, hz0]),
                "rgba": "0.35 0.35 0.40 1", "mass": "0",
                "contype": str(MASK_DIST_TYPE), "conaffinity": str(MASK_DIST_AFF)})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--n_dist", type=int, default=0, help="distractor bodies (v3 scene: 3)")
    a = ap.parse_args()
    out = a.out or (OUT_XML_V3 if a.n_dist else OUT_XML)
    p = build(out, a.template, n_dist=a.n_dist)
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(p)
        print(f"[build_scene_v2] {p}: nbody {m.nbody} ngeom {m.ngeom} nmesh {m.nmesh} nq {m.nq}")
    except Exception as e:                                       # pragma: no cover
        print(f"[build_scene_v2] wrote {p} (mujoco check failed: {e})", file=sys.stderr)

"""mujoco_export :: run the pick-and-place and export a MuJoCo replay package.

Runs in the `curobo2` env (it needs the cuRobo planner + yourdfpy + trimesh).
Produces, under outputs/mujoco/:
  * meshes/<link>.stl   — each robot link's visual mesh, baked into the LINK frame
                          (DAE->STL, mm->m, with the URDF visual origin applied)
  * scene.xml           — an MJCF: robot links as mocap bodies (kinematic replay),
                          table, place-box walls, the pick object, a camera + light
  * traj.npz            — per-frame mocap poses for every link + the object pose

The companion `mujoco_play.py` (run in a MuJoCo env, headless OSMesa) renders it.

We use mocap bodies so the replay is purely kinematic: MuJoCo just displays the
exact arm/object poses our planner produced — no contact/grasp physics needed.
"""
import os
import sys
import numpy as np
import trimesh
import yourdfpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from geometry import R_to_quat_wxyz

MJ_DIR = os.path.join(C.OUT_DIR, "mujoco")
MESH_DIR = os.path.join(MJ_DIR, "meshes")
# robot links that carry a visual mesh (base_link is the static mount at origin)
VIS_LINKS = ["base", "link1", "link2", "link3", "link4", "link5", "link6",
             "camera_mount", "suction_cup"]


def bake_link_meshes():
    """DAE->STL per link, baked into the link frame (mm->m + visual origin)."""
    os.makedirs(MESH_DIR, exist_ok=True)
    urdf = yourdfpy.URDF.load(C.URDF_PATH, build_scene_graph=True, load_meshes=False)
    made = {}
    for lk in VIS_LINKS:
        link = urdf.link_map[lk]
        parts = []
        for vis in link.visuals:
            geom = vis.geometry
            if geom.mesh is None:
                continue
            fn = geom.mesh.filename.replace("package://mycobot_description/", "")
            path = os.path.join(C.DESC_DIR, fn)
            m = trimesh.load(path, force="mesh")
            # scale mm -> m (URDF carries no scale; DAE verts are in mm)
            scale = geom.mesh.scale
            s = np.asarray(scale, float) if scale is not None else np.array([1, 1, 1.])
            if np.allclose(s, 1.0) and m.extents.max() > 5.0:
                s = np.array([1e-3, 1e-3, 1e-3])
            m.apply_scale(s)
            origin = vis.origin if vis.origin is not None else np.eye(4)
            m.apply_transform(np.asarray(origin))      # visual-origin -> link frame
            parts.append(m)
        if not parts:
            continue
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        out = os.path.join(MESH_DIR, f"{lk}.stl")
        mesh.export(out)
        made[lk] = out
    print(f"[mj] baked {len(made)} link meshes -> {MESH_DIR}")
    return made


def _q(M):
    return R_to_quat_wxyz(M[:3, :3])


def write_mjcf(scene, made_links):
    """Write scene.xml. Order of mocap bodies = VIS_LINKS then 'object'."""
    bi = scene["box_info"]; oi = scene["object_info"]
    W, D, H = C.BOX_INNER; t = C.BOX_WALL
    bx, by, bz = bi["pose_xyz"]
    zc = bz + H / 2.0
    walls = [  # name, size(half-extents), pos
        ("bottom", (W/2+t, D/2+t, t/2), (bx, by, bz - t/2)),
        ("xm", (t/2, D/2+t, H/2), (bx-(W/2+t/2), by, zc)),
        ("xp", (t/2, D/2+t, H/2), (bx+(W/2+t/2), by, zc)),
        ("ym", (W/2, t/2, H/2), (bx, by-(D/2+t/2), zc)),
        ("yp", (W/2, t/2, H/2), (bx, by+(D/2+t/2), zc)),
    ]
    tw, td, th = C.TABLE_DIMS
    obj_r, obj_h = oi["radius"], oi["height"]

    L = []
    L.append('<mujoco model="mycobot_pick_place">')
    L.append('  <compiler angle="radian" meshdir="meshes" autolimits="true"/>')
    L.append('  <option gravity="0 0 0"/>')
    L.append('  <visual><global offwidth="960" offheight="720"/>'
             '<headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/>'
             '<map znear="0.01"/></visual>')
    L.append('  <asset>')
    L.append('    <texture name="grid" type="2d" builtin="checker" '
             'rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="300" height="300"/>')
    L.append('    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>')
    L.append('    <material name="metal" rgba="0.7 0.72 0.75 1"/>')
    L.append('    <material name="obj" rgba="0.8 0.35 0.24 1"/>')
    L.append('    <material name="box" rgba="0.27 0.43 0.78 1"/>')
    L.append('    <material name="table" rgba="0.6 0.6 0.62 1"/>')
    for lk in made_links:
        L.append(f'    <mesh name="{lk}" file="{lk}.stl"/>')
    L.append('  </asset>')
    L.append('  <worldbody>')
    L.append('    <light pos="0.3 0.1 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>')
    L.append('    <geom name="floor" type="plane" size="2 2 0.1" '
             f'pos="0 0 {C.TABLE_Z-th:.4f}" material="grid"/>')
    # table top
    L.append(f'    <geom name="table" type="box" pos="0.30 0.10 {C.TABLE_Z-th/2:.4f}" '
             f'size="{tw/2:.4f} {td/2:.4f} {th/2:.4f}" material="table"/>')
    # box walls (static)
    for nm, sz, ps in walls:
        L.append(f'    <geom name="box_{nm}" type="box" pos="{ps[0]:.4f} {ps[1]:.4f} '
                 f'{ps[2]:.4f}" size="{sz[0]:.4f} {sz[1]:.4f} {sz[2]:.4f}" material="box"/>')
    # cameras
    L.append('    <camera name="iso" pos="1.05 -0.55 0.78" '
             'xyaxes="0.45 0.89 0 -0.5 0.25 0.83"/>')
    L.append('    <camera name="front" pos="0.30 -0.95 0.35" xyaxes="1 0 0 0 0.35 0.94"/>')
    # robot link mocap bodies
    for lk in made_links:
        L.append(f'    <body name="L_{lk}" mocap="true">')
        L.append(f'      <geom type="mesh" mesh="{lk}" material="metal" '
                 f'contype="0" conaffinity="0"/>')
        L.append('    </body>')
    # object mocap body (cylinder; freejoint not needed for kinematic replay)
    L.append('    <body name="object" mocap="true">')
    L.append(f'      <geom type="cylinder" size="{obj_r:.4f} {obj_h/2:.4f}" '
             f'material="obj" contype="0" conaffinity="0"/>')
    L.append('    </body>')
    L.append('  </worldbody>')
    L.append('</mujoco>')
    os.makedirs(MJ_DIR, exist_ok=True)
    path = os.path.join(MJ_DIR, "scene.xml")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[mj] wrote MJCF -> {path}")
    return path, [f"L_{lk}" for lk in made_links] + ["object"]


def export_trajectory(sim, made_links, mocap_order):
    """Per-frame mocap pos/quat for every body, from the recorded sim path."""
    from viz import ArmFK
    arm = ArmFK()
    nF = len(sim.path)
    bodies = mocap_order
    pos = np.zeros((nF, len(bodies), 3))
    quat = np.zeros((nF, len(bodies), 4))
    obj_T0 = sim.obj_T0
    for i, (q, objT, attached) in enumerate(sim.path):
        Ts = arm.link_transforms(q, made_links)
        for j, lk in enumerate(made_links):
            T = Ts[lk]
            pos[i, j] = T[:3, 3]
            quat[i, j] = _q(T)
        # object body (its geom is centred at object centre frame)
        pos[i, len(made_links)] = objT[:3, 3]
        quat[i, len(made_links)] = _q(objT)
    path = os.path.join(MJ_DIR, "traj.npz")
    np.savez(path, pos=pos, quat=quat, bodies=np.array(bodies),
             fps=C.VIDEO_FPS)
    print(f"[mj] wrote trajectory ({nF} frames, {len(bodies)} bodies) -> {path}")
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--place-mode", choices=["planned", "segmented"], default="planned")
    ap.add_argument("--box-pose", type=float, nargs=3, default=None)
    args = ap.parse_args()

    import pipeline
    box = np.array(args.box_pose) if args.box_pose else None
    # save_artifacts=False -> skip the matplotlib video (we want the recorded path only)
    rec = pipeline.run_pipeline(place_mode=args.place_mode, box_xyz=box,
                                save_artifacts=False)
    sim = rec["_sim"]; scene = rec["_scene"]
    made = bake_link_meshes()
    made_links = [lk for lk in VIS_LINKS if lk in made]
    _, mocap_order = write_mjcf(scene, made_links)
    export_trajectory(sim, made_links, mocap_order)
    print(f"\n[mj] export complete. Now render with a MuJoCo env:\n"
          f"  MUJOCO_GL=osmesa /home/lisc-frank/miniconda3/bin/python "
          f"pick_and_place/mujoco_play.py")


if __name__ == "__main__":
    main()

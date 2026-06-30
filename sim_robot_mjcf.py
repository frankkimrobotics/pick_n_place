"""sim_robot_mjcf :: generate a JOINT-DRIVEN MuJoCo model of the MyCobot Pro 630
from the URDF, with baked STL meshes, for the ROS2 MuJoCo controller node.

The exported scene.xml from mujoco_export.py is mocap-only (kinematic replay).
For a *controller* we want a model we can drive by setting 6 joint angles, so
MuJoCo computes FK itself. MuJoCo can't read the URDF's DAE meshes, so we:
  1. reuse mujoco_export.bake_link_meshes() (DAE->STL, baked into each link frame),
  2. walk the URDF tree (yourdfpy) emitting nested <body> with <joint type="hinge">
     for the revolute joints and the baked mesh as each link's <geom>,
  3. add a ground, light, camera, a free-joint pick object, and the place-box walls.

Joint order in qpos is joint1..joint6 (URDF radians). The tcp tool frame is a
<site> so the node can read the tip pose. Set data.qpos[:6] + mj_forward for FK.

Run in the curobo2 env (needs yourdfpy + trimesh):
    /home/lisc-frank/miniconda3/envs/curobo2/bin/python sim_robot_mjcf.py
Writes outputs/mujoco_sim/robot_sim.xml + meshes/.
"""
import os
import sys

import numpy as np
import yourdfpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
from geometry import R_to_quat_wxyz
import mujoco_export as mex

SIM_DIR = os.path.join(C.OUT_DIR, "mujoco_sim")
MESH_DIR = os.path.join(SIM_DIR, "meshes")
REVOLUTE = {"revolute", "continuous"}


def _pose_attrs(T):
    p = T[:3, 3]
    q = R_to_quat_wxyz(T[:3, :3])
    return (f'pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
            f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"')


def build(box_xyz=(0.20, 0.30, 0.0), box_fp=0.20, box_h=0.12, box_wall=0.01,
          obj_xyz=(0.30, 0.0, 0.03), obj_r=0.03, obj_h=0.06):
    os.makedirs(MESH_DIR, exist_ok=True)
    # bake the STL meshes into our sim mesh dir (reuse mujoco_export's baker)
    mex.MESH_DIR = MESH_DIR
    made = mex.bake_link_meshes()                      # {link: path}
    urdf = yourdfpy.URDF.load(C.URDF_PATH, build_scene_graph=True, load_meshes=False)

    children = {}
    for j in urdf.robot.joints:
        children.setdefault(j.parent, []).append(j)
    root = urdf.base_link

    def emit(link, depth):
        ind = "    " * (depth + 2)
        L = []
        if link in made:
            L.append(f'{ind}<geom type="mesh" mesh="{link}" material="metal" '
                     f'contype="0" conaffinity="0"/>')
        if link == "tcp" or link.endswith("tcp"):
            L.append(f'{ind}<site name="tcp" size="0.005" rgba="1 0 0 1"/>')
        for j in children.get(link, []):
            T = np.asarray(j.origin if j.origin is not None else np.eye(4))
            L.append(f'{ind}<body name="{j.child}" {_pose_attrs(T)}>')
            if j.type in REVOLUTE:
                ax = np.asarray(j.axis, float)
                lo = hi = None
                if j.limit is not None and j.limit.lower is not None:
                    lo, hi = float(j.limit.lower), float(j.limit.upper)
                rng = f' range="{lo:.4f} {hi:.4f}"' if lo is not None else ' limited="false"'
                L.append(f'{ind}    <joint name="{j.name}" type="hinge" '
                         f'axis="{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f}"{rng} '
                         f'damping="1.0"/>')
            L += emit(j.child, depth + 1)
            L.append(f'{ind}</body>')
        return L

    body_tree = emit(root, 0)

    # actuators only for the 6 revolute joints (position servos; the node also
    # supports direct qpos playback, but actuators let it be stepped if wanted)
    act = "".join(
        f'    <position name="act_{n}" joint="{n}" kp="80" '
        f'ctrlrange="-6.5 6.5"/>\n' for n in [f"joint{i}" for i in range(1, 7)])

    bx, by, bz = box_xyz
    h = box_fp / 2.0
    t = box_wall
    walls = [
        ("floor", (box_fp/2+t, box_fp/2+t, t/2), (bx, by, bz - t/2)),
        ("xm", (t/2, box_fp/2+t, box_h/2), (bx-(h+t/2), by, bz+box_h/2)),
        ("xp", (t/2, box_fp/2+t, box_h/2), (bx+(h+t/2), by, bz+box_h/2)),
        ("ym", (box_fp/2, t/2, box_h/2), (bx, by-(h+t/2), bz+box_h/2)),
        ("yp", (box_fp/2, t/2, box_h/2), (bx, by+(h+t/2), bz+box_h/2)),
    ]
    wall_geoms = "".join(
        f'    <geom name="box_{nm}" type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
        f'size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" material="box"/>\n'
        for nm, s, p in walls)

    meshes = "".join(f'    <mesh name="{lk}" file="{lk}.stl"/>\n' for lk in made)
    tz = C.TABLE_Z if hasattr(C, "TABLE_Z") else 0.0

    xml = f"""<mujoco model="mycobot_sim">
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/><map znear="0.01"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>
    <material name="metal" rgba="0.7 0.72 0.75 1"/>
    <material name="obj" rgba="0.85 0.35 0.24 1"/>
    <material name="box" rgba="0.27 0.43 0.78 1"/>
    <material name="table" rgba="0.6 0.6 0.62 1"/>
{meshes}  </asset>
  <worldbody>
    <light pos="0.3 0.1 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 {tz-0.02:.4f}" material="grid"/>
    <geom name="table" type="box" pos="0.30 0.10 {tz-0.01:.4f}" size="0.35 0.35 0.01" material="table"/>
{wall_geoms}    <camera name="iso" pos="1.05 -0.55 0.78" xyaxes="0.45 0.89 0 -0.5 0.25 0.83"/>
    <camera name="front" pos="0.30 -0.95 0.55" xyaxes="1 0 0 0 0.5 0.87"/>
    <body name="{root}" pos="0 0 0">
{chr(10).join(body_tree)}
    </body>
    <body name="object" pos="{obj_xyz[0]:.4f} {obj_xyz[1]:.4f} {obj_xyz[2]:.4f}">
      <freejoint name="obj_free"/>
      <geom name="object" type="cylinder" size="{obj_r:.4f} {obj_h/2:.4f}" material="obj" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
{act}  </actuator>
</mujoco>
"""
    os.makedirs(SIM_DIR, exist_ok=True)
    out = os.path.join(SIM_DIR, "robot_sim.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"[sim-mjcf] wrote {out}  ({len(made)} link meshes, root='{root}')")
    return out


if __name__ == "__main__":
    build()

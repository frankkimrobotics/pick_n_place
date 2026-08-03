"""sim_robot_mjcf :: generate a JOINT-DRIVEN MuJoCo model of the MyCobot Pro 630
from the URDF, with baked STL meshes, for the ROS2 MuJoCo controller node.

The exported scene.xml from mujoco_export.py is mocap-only (kinematic replay).
For a *controller* we want a model we can drive by setting 6 joint angles, so
MuJoCo computes FK itself. MuJoCo can't read the URDF's DAE meshes, so we:
  1. reuse mujoco_export.bake_link_meshes() (DAE->STL, baked into each link frame),
  2. walk the URDF tree (yourdfpy) emitting nested <body> with <joint type="hinge">
     for the revolute joints and the baked mesh as each link's <geom>,
  3. add a ground, light, cameras, the cluttered pick objects, the place-box
     walls, the tip rangefinder (tip_rf site + sensor) and position actuators.

Joint order in qpos is joint1..joint6 (URDF radians). The tcp tool frame is a
<site> so the node can read the tip pose. Set data.qpos[:6] + mj_forward for FK.

The sim tcp/eef bodies OVERRIDE the URDF origins (-0.135/-0.105): in the twin
the tip sits at the VISIBLE suction surface, cup-frame z=-0.115 (eef -0.105),
so the red dot + rangefinder originate at the mushroom tip and contact depth is
right (see commit 7bf3b0b). The planner keeps the URDF's calibrated -0.135.

build() writes outputs/mujoco_sim/robot_sim.xml (the ROS2 twin, kinematic
attach for suction). build_warp() writes robot_warp.xml for mujoco_warp: same
model plus per-object <weld> equality constraints (inactive; the runtime sets
eq_data relpose at grasp and flips eq_active) so suction is real physics that
survives GPU-batched stepping.

Run in the curobo2 env (needs yourdfpy + trimesh):
    /home/lisc-frank/miniconda3/envs/curobo2/bin/python sim_robot_mjcf.py [--warp]
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

# sim-only tip override (visible cup surface, not the planner's calibrated tcp)
SIM_TIP_Z = {"tcp": -0.115, "eef": -0.105}

# cluttered objects on the table: name, shape, size, pos, rgba
OBJECTS = [
    ("object0", "cylinder", "0.026 0.030", "0.340 0.050 0.030", "0.85 0.35 0.24 1"),
    ("object1", "box",      "0.025 0.025 0.040", "0.405 -0.055 0.040", "0.30 0.69 0.31 1"),
    ("object2", "cylinder", "0.030 0.025", "0.375 0.105 0.025", "0.20 0.55 0.86 1"),
    ("object3", "cylinder", "0.022 0.035", "0.445 0.030 0.035", "0.95 0.61 0.07 1"),
    ("object4", "box",      "0.030 0.030 0.022", "0.330 -0.110 0.022", "0.61 0.35 0.71 1"),
    ("object5", "cylinder", "0.028 0.030", "0.460 -0.085 0.030", "0.90 0.80 0.16 1"),
    ("object6", "cylinder", "0.024 0.028", "0.300 0.095 0.028", "0.40 0.76 0.75 1"),
]
# warp-only extras: a curved (sphere) object for press-to-seal tests -- a cup
# pressing off-centre on it slides and shoves it away under real contact
WARP_OBJECTS = [
    ("object7", "sphere", "0.030", "0.300 -0.020 0.030", "0.95 0.30 0.55 1"),
]


def _pose_attrs(T):
    p = T[:3, 3]
    q = R_to_quat_wxyz(T[:3, :3])
    return (f'pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
            f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"')


def build(box_xyz=(0.10, 0.40, 0.0), box_fp=0.30, box_h=0.30, box_wall=0.01,
          warp=False):
    os.makedirs(MESH_DIR, exist_ok=True)
    # bake the STL meshes into our sim mesh dir (reuse mujoco_export's baker)
    mex.MESH_DIR = MESH_DIR
    made = mex.bake_link_meshes()                      # {link: path}
    urdf = yourdfpy.URDF.load(C.URDF_PATH, build_scene_graph=True, load_meshes=False)

    children = {}
    for j in urdf.robot.joints:
        children.setdefault(j.parent, []).append(j)
    root = urdf.base_link

    link_by_name = {l.name: l for l in urdf.robot.links}

    def _inertial_xml(link, ind):
        """Bake the URDF <inertial> (rotated into the body frame) so mj_inverse
        RNEA uses the same masses/inertias the cuRobo planner does, instead of
        mesh-density guesses."""
        li = link_by_name.get(link)
        li = getattr(li, "inertial", None)
        if li is None or li.mass is None or li.mass < 1e-6:
            return None
        T = np.asarray(li.origin if li.origin is not None else np.eye(4), float)
        R, p = T[:3, :3], T[:3, 3]
        I = R @ np.asarray(li.inertia, float) @ R.T
        return (f'{ind}<inertial pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
                f'mass="{li.mass:.6f}" fullinertia="{I[0,0]:.3e} {I[1,1]:.3e} '
                f'{I[2,2]:.3e} {I[0,1]:.3e} {I[0,2]:.3e} {I[1,2]:.3e}"/>')

    def emit(link, depth):
        ind = "    " * (depth + 2)
        L = []
        if warp:
            ine = _inertial_xml(link, ind)
            if ine:
                L.append(ine)
        if link in made:
            L.append(f'{ind}<geom type="mesh" mesh="{link}" material="metal" '
                     f'contype="0" conaffinity="0"/>')
        if link == "tcp" or link.endswith("tcp"):
            L.append(f'{ind}<site name="tcp" size="0.005" rgba="1 0 0 1"/>')
            L.append(f'{ind}<site name="tip_rf" size="0.004" rgba="0 1 0 0.6"/>')
            if warp:
                # the only contact geom on the arm: the cup tip, so press-to-seal
                # is a real dynamic interaction (robot meshes stay contactless)
                L.append(f'{ind}<geom name="cup_tip" type="sphere" size="0.008" '
                         f'rgba="0.25 0.25 0.28 1"/>')
                # wrist D405 from the calibrated hand-eye extrinsic (OpenCV
                # optical frame -> MuJoCo camera: flip y,z)
                Tc = np.asarray(C.T_TCP_CAM, float)
                Rmj = Tc[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                qc = R_to_quat_wxyz(Rmj)
                p = Tc[:3, 3]
                L.append(f'{ind}<camera name="wrist_d405" '
                         f'pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
                         f'quat="{qc[0]:.6f} {qc[1]:.6f} {qc[2]:.6f} {qc[3]:.6f}" '
                         f'fovy="58.0"/>')
        for j in children.get(link, []):
            T = np.asarray(j.origin if j.origin is not None else np.eye(4), float).copy()
            if j.child in SIM_TIP_Z:
                T[2, 3] = SIM_TIP_Z[j.child]
            L.append(f'{ind}<body name="{j.child}" {_pose_attrs(T)}>')
            if j.type in REVOLUTE:
                ax = np.asarray(j.axis, float)
                lo = hi = None
                if j.limit is not None and j.limit.lower is not None:
                    lo, hi = float(j.limit.lower), float(j.limit.upper)
                rng = f' range="{lo:.4f} {hi:.4f}"' if lo is not None else ' limited="false"'
                # warp variant: reflected rotor inertia of the harmonic drives
                # (~1e-4 kg m^2 * gear^2). Without it the coupled link modes have
                # near-zero modal inertia and discrete-time PD limit-cycles.
                arm = ' armature="0.15"' if warp else ''
                L.append(f'{ind}    <joint name="{j.name}" type="hinge" '
                         f'axis="{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f}"{rng} '
                         f'damping="1.0"{arm}/>')
            L += emit(j.child, depth + 1)
            L.append(f'{ind}</body>')
        return L

    body_tree = emit(root, 0)

    # robot_sim.xml keeps position servos (the ROS2 node uses direct qpos
    # playback); the warp variant exposes raw torque motors so the controller
    # can run PD + inverse-dynamics feedforward at the 4ms tick
    if warp:
        act = "".join(
            f'    <motor name="act_{n}" joint="{n}" '
            f'ctrlrange="-100 100"/>\n' for n in [f"joint{i}" for i in range(1, 7)])
    else:
        act = "".join(
            f'    <position name="act_{n}" joint="{n}" kp="80" '
            f'ctrlrange="-6.5 6.5"/>\n' for n in [f"joint{i}" for i in range(1, 7)])

    bx, by, bz = box_xyz
    h = box_fp / 2.0
    t = box_wall
    walls = [
        ("floor", (h, h, t/2), (bx, by, bz - t/2)),
        ("xm", (t/2, h, box_h/2), (bx-(h+t/2), by, bz+box_h/2)),
        ("xp", (t/2, h, box_h/2), (bx+(h+t/2), by, bz+box_h/2)),
        ("ym", (h, t/2, box_h/2), (bx, by-(h+t/2), bz+box_h/2)),
        ("yp", (h, t/2, box_h/2), (bx, by+(h+t/2), bz+box_h/2)),
    ]
    wall_geoms = "".join(
        f'    <geom name="box_{nm}" type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
        f'size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" material="box"/>\n'
        for nm, s, p in walls)

    objs = OBJECTS + (WARP_OBJECTS if warp else [])
    obj_mats = "".join(
        f'    <material name="o{i}" rgba="{rgba}"/>\n'
        for i, (_, _, _, _, rgba) in enumerate(objs))
    obj_bodies = "".join(
        f'    <body name="{nm}" pos="{pos}"><freejoint name="obj{i}_free"/>\n'
        f'      <geom name="g_{nm}" type="{shape}" size="{size}" material="o{i}" mass="0.05"/></body>\n'
        for i, (nm, shape, size, pos, _) in enumerate(objs))

    # suction welds for the warp variant: inactive until the controller sets
    # eq_data[3:10] to the tcp->object relpose at grasp and flips eq_active
    welds = "".join(
        f'    <weld name="suction_{nm}" body1="tcp" body2="{nm}" active="false" '
        f'solref="0.005 1"/>\n' for nm, *_ in objs)
    equality = f"  <equality>\n{welds}  </equality>\n" if warp else ""

    # fixed D435 on the far side of the table (opposite the robot), centred on
    # that edge at 0.6 m, aimed at the table centre. MuJoCo cameras look along
    # -z with +y up: build the frame from the aim direction.
    d435 = ""
    if warp:
        cam_p = np.array([0.66, 0.0, 0.60])
        aim = np.array([0.38, 0.0, 0.03])
        f = aim - cam_p; f /= np.linalg.norm(f)
        zc = -f
        xc = np.cross([0.0, 0.0, 1.0], zc); xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        d435 = (f'    <camera name="fixed_d435" '
                f'pos="{cam_p[0]:.3f} {cam_p[1]:.3f} {cam_p[2]:.3f}" '
                f'xyaxes="{xc[0]:.4f} {xc[1]:.4f} {xc[2]:.4f} '
                f'{yc[0]:.4f} {yc[1]:.4f} {yc[2]:.4f}" fovy="42.0"/>\n')

    meshes = "".join(f'    <mesh name="{lk}" file="{lk}.stl"/>\n' for lk in made)

    xml = f"""<mujoco model="mycobot_sim">
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/><map znear="0.01"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>
    <material name="metal" rgba="0.7 0.72 0.75 1"/>
    <material name="box" rgba="0.27 0.43 0.78 1"/>
    <material name="table" rgba="0.55 0.55 0.58 1"/>
{obj_mats}{meshes}  </asset>
  <worldbody>
    <light pos="0.3 0.1 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.1200" material="grid"/>
    <geom name="table" type="box" pos="0.38 0.00 -0.0100" size="0.20 0.22 0.01" material="table"/>
    <!-- place bin at [0.10,0.40], floor top at 0.0, ~12 cm walls -->
{wall_geoms}    <camera name="iso" pos="1.05 -0.55 0.78" xyaxes="0.45 0.89 0 -0.5 0.25 0.83"/>
    <camera name="front" pos="0.38 -0.85 0.45" xyaxes="1 0 0 0 0.45 0.89"/>
    <camera name="top" pos="0.38 0.0 0.9" xyaxes="1 0 0 0 1 0"/>
{d435}    <body name="{root}" pos="0 0 0">
{chr(10).join(body_tree)}
    </body>
    <!-- cluttered objects on the table (kinematic; freejoints) -->
{obj_bodies}  </worldbody>
{equality}  <actuator>
{act}  </actuator>
  <sensor>
    <rangefinder name="tip_range" site="tip_rf"/>
  </sensor>
</mujoco>
"""
    os.makedirs(SIM_DIR, exist_ok=True)
    out = os.path.join(SIM_DIR, "robot_warp.xml" if warp else "robot_sim.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"[sim-mjcf] wrote {out}  ({len(made)} link meshes, root='{root}')")
    return out


if __name__ == "__main__":
    build(warp="--warp" in sys.argv)

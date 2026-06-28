"""config :: all constants, paths and calibration for the pick-and-place sim.

Everything is grounded in the existing (un-edited) repo assets:
  * URDF / cuRobo robot cfg / the V2 Planner class
  * hand-eye extrinsic  T_tcp_cam   (camera pose in the suction-tip / tcp frame)
  * D405 colour intrinsics
  * joint conventions (home pose, joint order)

Frames & conventions (verified against the repo):
  * cuRobo ee_link = "tcp" = the real suction-cup tip (0.145 m below the flange,
    tcp +Z is the approach axis and points DOWN at home-ish poses).
  * Quaternions are wxyz:  pose = [x, y, z, qw, qx, qy, qz], metres.
  * Joints are URDF radians, order [joint1 .. joint6].
  * T_base_cam = FK_tcp(q) @ T_tcp_cam.  Camera optical frame: +X right, +Y down, +Z fwd.
"""
import os
import numpy as np

# --------------------------------------------------------------------------- #
# Paths into the existing repo (read-only; never edited)
# --------------------------------------------------------------------------- #
REPO_ROOT = "/home/lisc-frank/Desktop/2026"
DESC_DIR  = os.path.join(REPO_ROOT,
                         "frankkimrobotics/ros2_mycobot/src/mycobot_description")
URDF_PATH = os.path.join(DESC_DIR, "urdf/mycobot_pro_630.urdf")
MESH_DIR  = os.path.join(DESC_DIR, "meshes")
CUROBO_CFG_DIR = os.path.join(DESC_DIR, "curobo")          # holds curobo_planner_server_v2.py
ROBOT_YML = os.path.join(CUROBO_CFG_DIR, "mycobot_pro_630.yml")

CALIB_DIR = os.path.join(REPO_ROOT,
                         "mycobot_mpc/captures/calib_20260622_215855")
HANDEYE_JSON    = os.path.join(CALIB_DIR, "handeye_result.json")
INTRINSICS_JSON = os.path.join(CALIB_DIR, "d405_intrinsics.json")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --------------------------------------------------------------------------- #
# Hand-eye extrinsic  T_tcp_cam  (camera pose in tcp frame), from handeye_result.json
# --------------------------------------------------------------------------- #
T_TCP_CAM = np.array([
    [-0.9983998722356762, -0.055265655956255856, 0.011975073758070786, 0.01111958419898037],
    [0.051808921950663604, -0.9788432303796386, -0.1979438454366913, 0.06128205328288515],
    [0.02266121634194691, -0.19700669433214563, 0.9801402102057766, -0.09794176471947463],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# T_TCP_CAM was calibrated with the tcp at HANDEYE_TCP_LEN below the flange. The real
# robot's planner tcp was later shortened to PLANNER_TCP_LEN (URDF tcp_joint) so the
# planned tcp matches the physical cup tip. Because camera pose is computed as
# fk(tcp) @ T_tcp_cam, a consistent correction of (HANDEYE_TCP_LEN - PLANNER_TCP_LEN)
# along the tcp +Z axis must be applied so perception stays accurate after the change.
HANDEYE_TCP_LEN = 0.145          # tcp length when the hand-eye was calibrated
PLANNER_TCP_LEN = 0.145          # current URDF/planner tcp length (must match the URDF).
                                 # Restored to 0.145 after the 2026-06-27 floor touch-test
                                 # (the 0.13 setting made grasps dive ~1.5 cm too deep).
CAM_TCP_Z_SHIFT = HANDEYE_TCP_LEN - PLANNER_TCP_LEN   # now 0 -> tcp == hand-eye calib frame

# --------------------------------------------------------------------------- #
# D405 colour intrinsics (from d405_intrinsics.json)
# --------------------------------------------------------------------------- #
IMG_W, IMG_H = 848, 480
FX, FY = 433.3826599121094, 432.8515625
CX, CY = 429.44384765625, 243.3206787109375
K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float64)
# We render at a downscaled resolution for speed; intrinsics scale with it.
RENDER_SCALE = 0.5                       # 424 x 240 render

DEPTH_MIN, DEPTH_MAX = 0.05, 1.0         # metres (matches perception_node defaults)

# --------------------------------------------------------------------------- #
# Joint conventions (mirrors mycobot_mpc/joint_conventions.py)
# --------------------------------------------------------------------------- #
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# Home in URDF radians == [-90, 0, 0, 0, 0, 0] deg  (HOME_LINUXCNC_DEG mapped through cal)
HOME_Q = np.array([-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
# BASE / rest pose (URDF rad). LinuxCNC deg [0, -110, 80, -80, -90, 0]
# (mapped through joint_conventions: offsets [0,90,0,90,0,0]). FK(tcp@0.13) ~ [0.291, 0.073, 0.476].
BASE_Q = np.array([0.0, -0.349066, 1.396263, 0.174533, -1.570796, 0.0], dtype=np.float64)
START_Q = BASE_Q                          # pick-and-place starts from the base pose
# URDF joint limits (lower, upper) rad, from mycobot_pro_630.urdf
JOINT_LIMITS = np.array([
    (-3.14159, 3.14159),
    (-3.14159, 3.14159),
    (-2.61,    2.618),
    (-2.9670,  2.9670),
    (-2.93,    2.9321),
    (-3.03,    3.0368),
], dtype=np.float64)

# --------------------------------------------------------------------------- #
# Suction cup model
# --------------------------------------------------------------------------- #
SUCTION_DIAMETER   = 0.010               # 1 cm circular sealing footprint
SUCTION_RADIUS     = SUCTION_DIAMETER / 2.0
# Grasp-point selection evaluates flatness/smoothness over a LARGER region than the cup
# itself, so the chosen point stays on a smooth flat patch even if the cup lands up to
# ~1 cm off the commanded point (placement error). 3 cm diameter -> 1.5 cm radius.
GRASP_SUPPORT_DIAMETER = 0.030
GRASP_SUPPORT_RADIUS   = GRASP_SUPPORT_DIAMETER / 2.0
SEAL_POS_TOL       = 0.005               # tip must be within 5 mm of the surface
SEAL_ANG_TOL_DEG   = 15.0                # approach axis within 15 deg of surface normal
FLAT_RMS_TOL       = 0.0012              # patch planarity RMS must be < 1.2 mm
NORMAL_CONE_DEG    = 12.0                # neighbour-normal agreement cone

# --------------------------------------------------------------------------- #
# Motion standoffs (metres)
# --------------------------------------------------------------------------- #
PREGRASP_STANDOFF  = 0.06                # pre-grasp sits this far along +normal from grasp pt
LIFT_HEIGHT        = 0.10                # lift this far straight up after suction
PLACE_HOVER        = 0.08                # hover this high above the box before lowering
RELEASE_CLEARANCE  = 0.04                # at release, object bottom this far ABOVE the box rim

# --------------------------------------------------------------------------- #
# World / scene defaults (base_link frame, metres)
# --------------------------------------------------------------------------- #
TABLE_Z            = -0.10               # table top height (== Planner ground_z default)
TABLE_DIMS         = (1.2, 1.2, 0.04)

# Pick object: a low cylinder "puck" with a clean flat top (good suction target).
OBJECT_POSE_XYZ    = np.array([0.30, 0.06, TABLE_Z])   # base of object sits on table
OBJECT_RADIUS      = 0.030
OBJECT_HEIGHT      = 0.040

# Place box: a 25 cm CUBE, open top, resting ON the table at xy = [0.1, 0.4]
# (so it does not float). Its rim sits 25 cm above the table surface. The object is
# lifted clear of the 25 cm rim and released over the box. BOX_POSE_XYZ is the box's
# bottom-centre (on the table). Frame note: the table top is at z = TABLE_Z = -0.10
# in the robot base frame, so the rim is at z = TABLE_Z + 0.25 = 0.15.
BOX_POSE_XYZ       = np.array([0.10, 0.40, TABLE_Z])   # bottom sits on the table
BOX_INNER          = (0.23, 0.23, 0.25)  # inner W, D, H -> ~25 cm cube outer (1 cm walls)
BOX_WALL           = 0.010

# Rendering / video
VIDEO_FPS          = 20
SHADE_NORMAL_DIR   = np.array([0.2, 0.3, 1.0])   # light dir for simple Lambert shading


def scaled_intrinsics():
    """Return (K_scaled, w, h) at RENDER_SCALE for the synthetic camera."""
    s = RENDER_SCALE
    w, h = int(round(IMG_W * s)), int(round(IMG_H * s))
    Ks = K.copy()
    Ks[0, 0] *= s; Ks[1, 1] *= s
    Ks[0, 2] *= s; Ks[1, 2] *= s
    return Ks, w, h

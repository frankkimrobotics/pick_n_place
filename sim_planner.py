"""sim_planner :: thin wrapper around the REUSED cuRobo V2 Planner.

We import the existing, un-edited ``curobo_planner_server_v2.Planner`` and use it
in-process (no socket, no ROS) for:
  * forward kinematics of the tcp (suction tip)            -> fk()
  * collision-free Cartesian planning                      -> plan_pose()
  * collision-free joint planning                          -> plan_joint()

Obstacles (table + place-box walls) are injected through the Planner's own
supported ``world_path`` argument: we generate a cuRobo world yaml and hand it to
the constructor. For the carried object we attempt cuRobo's real attachment
(``mp.attachment_manager.attach``) as a bonus; collision-safety of the carried
OBB is *always* additionally guaranteed by our own deterministic sweep in
``simulator.py``, so the pipeline is robust even if the low-level attach API
differs.

Runs in the ``curobo2`` conda env.
"""
import os
import sys
import tempfile

import numpy as np
import yaml

import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, make_T, inv_T, pose_to_T, T_to_pose


# Fixed transform: the real suction tip (URDF `eef` link, 0.105 m from the flange)
# expressed in the cuRobo tool frame (`tcp`, 0.145 m). cuRobo plans the tcp, so to
# place the real tip at a target we offset by this constant. Identity rotation,
# translation [0,0,-0.04] (tip is 4 cm back toward the flange from tcp).
def _tcp_to_tip_transform():
    import yourdfpy
    u = yourdfpy.URDF.load(C.URDF_PATH, build_scene_graph=True, load_meshes=False)
    u.update_cfg({f"joint{i+1}": 0.0 for i in range(6)})
    return np.asarray(u.get_transform("eef", "tcp"))


def _ensure_planner_on_path():
    if C.CUROBO_CFG_DIR not in sys.path:
        sys.path.insert(0, C.CUROBO_CFG_DIR)


def write_world_yaml(path, obstacles):
    """Write a cuRobo world yaml from a dict of {name: (dims_xyz, pose7)}.

    pose7 = [x,y,z, qw,qx,qy,qz] in base_link.
    """
    cub = {name: {"dims": [float(d) for d in dims],
                  "pose": [float(p) for p in pose]}
           for name, (dims, pose) in obstacles.items()}
    with open(path, "w") as f:
        yaml.safe_dump({"cuboid": cub}, f, sort_keys=False)
    return path


def box_wall_obstacles(box_xyz, inner, wall, name="box"):
    """Return obstacle dict for an open-top bin: 4 walls + bottom (cuboids).

    box_xyz = bottom-centre of the bin on the table; inner = (W,D,H) inner size.
    """
    bx, by, bz = box_xyz
    W, D, H = inner
    t = wall
    q = [1.0, 0.0, 0.0, 0.0]
    zc = bz + H / 2.0
    obs = {
        f"{name}_bottom": ((W + 2 * t, D + 2 * t, t), [bx, by, bz - t / 2.0, *q]),
        f"{name}_xm": ((t, D + 2 * t, H), [bx - (W / 2 + t / 2), by, zc, *q]),
        f"{name}_xp": ((t, D + 2 * t, H), [bx + (W / 2 + t / 2), by, zc, *q]),
        f"{name}_ym": ((W, t, H), [bx, by - (D / 2 + t / 2), zc, *q]),
        f"{name}_yp": ((W, t, H), [bx, by + (D / 2 + t / 2), zc, *q]),
    }
    return obs


class SimPlanner:
    def __init__(self, box_xyz=None, include_box=True, ground_z=None, verbose=True):
        _ensure_planner_on_path()
        from curobo_planner_server_v2 import Planner   # reused, un-edited

        ground_z = C.TABLE_Z if ground_z is None else ground_z
        obstacles = {}
        # NB: the Planner already adds a 2x2 ground slab at ground_z, so the table
        # surface is modelled. We only add the box walls here.
        if include_box and box_xyz is not None:
            obstacles.update(box_wall_obstacles(box_xyz, C.BOX_INNER, C.BOX_WALL))

        world_path = None
        if obstacles:
            fd, world_path = tempfile.mkstemp(prefix="pp_world_", suffix=".yml")
            os.close(fd)
            # include the ground slab too so the single yaml is the whole world
            obstacles_full = dict(obstacles)
            obstacles_full["ground"] = ((2.0, 2.0, 0.04),
                                        [0.0, 0.0, ground_z - 0.02, 1, 0, 0, 0])
            write_world_yaml(world_path, obstacles_full)

        if verbose:
            print(f"[sim_planner] building cuRobo Planner "
                  f"(ground_z={ground_z}, box={'yes' if obstacles else 'no'}) ...")
        self.P = Planner(ground_z=ground_z, world_path=world_path)
        self.joint_names = self.P.joint_names
        self.tool_frame = self.P.tool_frame
        self.T_tcp_tip = _tcp_to_tip_transform()     # real suction tip (eef) in tcp frame
        self.T_tip_tcp = inv_T(self.T_tcp_tip)
        self._attached = False
        if verbose:
            print(f"[sim_planner] ready. (suction tip = eef, "
                  f"{abs(self.T_tcp_tip[2,3])*1000:.0f} mm back from tcp)")

    # ---- kinematics ------------------------------------------------------- #
    def fk(self, q):
        """tcp pose for q [6] (rad). Returns (pos[3], quat_wxyz[4]) numpy."""
        r = self.P.fk(list(np.asarray(q, float)))
        return np.array(r["pos"][0]), np.array(r["quat"][0])

    def fk_T(self, q):
        from geometry import make_T, quat_wxyz_to_R
        pos, quat = self.fk(q)
        return make_T(quat_wxyz_to_R(quat), pos)

    # ---- suction TIP (eef, 0.105 m) kinematics --------------------------- #
    def fk_tip_T(self, q):
        """Pose (4x4) of the real suction tip (eef) in base for config q."""
        return self.fk_T(q) @ self.T_tcp_tip

    def fk_tip(self, q):
        """Real suction-tip pose: (pos[3], quat_wxyz[4])."""
        return T_to_pose(self.fk_tip_T(q))[:3], T_to_pose(self.fk_tip_T(q))[3:7]

    # ---- planning --------------------------------------------------------- #
    def plan_pose(self, start_q, goal_pose, max_attempts=8):
        """goal_pose = [x,y,z, qw,qx,qy,qz]. Returns the Planner result dict."""
        return self.P.plan_pose(list(np.asarray(start_q, float)),
                                list(np.asarray(goal_pose, float)),
                                max_attempts=max_attempts)

    def plan_pose_tip(self, start_q, tip_pose, max_attempts=8):
        """Plan so the real suction TIP (eef) reaches `tip_pose` [x,y,z,qw..qz].

        cuRobo plans the tcp, so we convert: tcp_goal = tip_goal @ inv(T_tcp_tip).
        """
        tcp_goal_T = pose_to_T(tip_pose) @ self.T_tip_tcp
        return self.plan_pose(start_q, T_to_pose(tcp_goal_T), max_attempts=max_attempts)

    def plan_joint(self, start_q, goal_q, max_attempts=8):
        return self.P.plan_joint(list(np.asarray(start_q, float)),
                                 list(np.asarray(goal_q, float)),
                                 max_attempts=max_attempts)

    # ---- carried-object attach (best effort; collision safety is also
    #      independently guaranteed by simulator.py's OBB sweep) ------------ #
    def attach_obb(self, q_grasp, obb_pose7, obb_dims):
        """Try to attach the object OBB to the tool link for collision-aware planning.

        Returns True on success, False if the low-level API is unavailable (the
        pipeline then relies on the independent trimesh OBB sweep)."""
        try:
            import torch
            from curobo._src.geom.types import Cuboid
            from curobo._src.types.pose import Pose
            from curobo._src.state.state_joint import JointState
            mp = self.P.mp
            am = mp.attachment_manager
            dev = self.P.device
            obs = [Cuboid(name="carried_obj",
                          pose=[float(x) for x in obb_pose7],
                          dims=[float(d) for d in obb_dims])]
            qt = torch.tensor([list(np.asarray(q_grasp, float))],
                              dtype=torch.float32, device=dev)
            js = JointState.from_position(qt, joint_names=self.joint_names)
            am.attach(js, obs, link_name=self.tool_frame)
            self._attached = True
            return True
        except Exception as e:                       # link missing / API mismatch
            print(f"[sim_planner] cuRobo attach unavailable ({type(e).__name__}: {e}); "
                  f"using independent OBB sweep for carried-object safety.")
            return False

    def detach(self):
        if not self._attached:
            return
        try:
            self.P.mp.attachment_manager.detach()
        except Exception:
            pass
        self._attached = False


# ---- convenience: a top-down grasp pose from point + normal --------------- #
def grasp_pose_from_point_normal(point, normal, x_hint=np.array([1.0, 0.0, 0.0])):
    """tcp pose [x,y,z,qw,qx,qy,qz] that places the suction tip at `point` with the
    approach axis (tcp +Z) pointing INTO the surface (= -normal)."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    Rm = R_from_two_axes(-n, x_hint=x_hint)          # tcp +Z = -normal
    return np.concatenate([np.asarray(point, float), R_to_quat_wxyz(Rm)])

"""simulator :: kinematic world-state simulation of the pick-and-place.

The real robot is OFF, so this is a *kinematic* simulator: it advances the arm
along joint trajectories using cuRobo FK (via SimPlanner), tracks the suction
state, rigidly carries the grasped object (and its OBB collision volume) with the
tcp, checks the carried OBB against the place-box with trimesh, and captures
rendered frames for a video.

State:
  q          current joints (rad)        attached    suction on/off
  T_tcp_obj  object pose in tcp frame     T_tcp_obb   OBB pose in tcp frame
  object_T   object pose (4x4) in base    obb_dims    carried OBB dimensions
"""
import numpy as np
import trimesh

import config as C
from geometry import inv_T, quat_wxyz_to_R, make_T


class SuctionSim:
    def __init__(self, planner, scene, frame_renderer=None):
        self.P = planner
        self.scene = scene
        self.render = frame_renderer
        self.q = C.START_Q.copy()
        self.attached = False
        self.T_tcp_obj = None
        self.T_tcp_obb = None
        self.obb_dims = None
        self.object_T = scene["object_info"]["T_base_obj"].copy()
        self.obj_mesh0 = scene["object"].copy()
        self.obj_T0 = scene["object_info"]["T_base_obj"].copy()
        # box walls (the obstacle to avoid while carrying), as (center, dims) AABBs
        from sim_planner import box_wall_obstacles
        bi = scene["box_info"]
        wall_obs = box_wall_obstacles(bi["pose_xyz"], C.BOX_INNER, C.BOX_WALL)
        self._walls = [(np.array(pose[:3]), np.array(dims))
                       for dims, pose in wall_obs.values()]
        self.frames = []
        self.path = []          # full per-step (q, object_T, attached) for MuJoCo replay
        self.log = []

    def _record(self):
        self.path.append((self.q.copy(), self.object_T.copy(), bool(self.attached)))

    # ---- geometry helpers ------------------------------------------------ #
    def object_mesh_at(self, T):
        m = self.obj_mesh0.copy()
        m.apply_transform(T @ inv_T(self.obj_T0))
        return m

    def obb_T_now(self):
        """Carried OBB pose (4x4) at the current q (rides the tcp)."""
        if self.T_tcp_obb is None:
            return None
        return self.P.fk_T(self.q) @ self.T_tcp_obb

    # ---- suction --------------------------------------------------------- #
    def seal_check(self, grasp_point, grasp_normal):
        pos, quat = self.P.fk_tip(self.q)                 # real suction tip (eef, 0.105 m)
        approach = quat_wxyz_to_R(quat)[:, 2]             # tip +Z in base
        pos_err = float(np.linalg.norm(pos - grasp_point))
        n = grasp_normal / (np.linalg.norm(grasp_normal) + 1e-12)
        ang = float(np.degrees(np.arccos(np.clip(-approach @ n, -1, 1))))
        ok = (pos_err <= C.SEAL_POS_TOL) and (ang <= C.SEAL_ANG_TOL_DEG)
        return ok, pos_err, ang

    def activate_suction(self, grasp_point, grasp_normal, obb=None):
        ok, pe, ae = self.seal_check(grasp_point, grasp_normal)
        if ok:
            T_base_tcp = self.P.fk_T(self.q)
            inv_tcp = inv_T(T_base_tcp)
            self.T_tcp_obj = inv_tcp @ self.object_T
            if obb is not None:
                self.T_tcp_obb = inv_tcp @ obb["T_base_obb"]
                self.obb_dims = np.asarray(obb["dims"], float)
            self.attached = True
        self.log.append(("activate_suction", dict(ok=ok, pos_err=pe, ang_err=ae)))
        return ok, pe, ae

    def release_suction(self):
        self.attached = False
        self.log.append(("release_suction", {}))

    # ---- carried OBB vs box ---------------------------------------------- #
    def carried_collision(self):
        """(collided, clearance_m) of the carried OBB against the place-box walls."""
        from collision import obb_vs_walls
        T = self.obb_T_now()
        if T is None:
            return False, np.inf
        return obb_vs_walls(T, self.obb_dims, self._walls)

    # ---- stepping -------------------------------------------------------- #
    def _update_object(self):
        if self.attached and self.T_tcp_obj is not None:
            self.object_T = self.P.fk_T(self.q) @ self.T_tcp_obj

    def goto(self, q, capture=True):
        self.q = np.asarray(q, float)
        self._update_object()
        self._record()
        if capture and self.render is not None:
            self.frames.append(self.render(self.q, self))

    def execute(self, trajectory, capture_every=6, label=""):
        traj = np.asarray(trajectory, float)
        collided = False
        min_clear = np.inf
        for k, q in enumerate(traj):
            self.q = q
            self._update_object()
            self._record()
            if self.attached and self.T_tcp_obb is not None:
                c, clr = self.carried_collision()
                min_clear = min(min_clear, clr)
                collided = collided or c
            if self.render is not None and (k % capture_every == 0 or k == len(traj) - 1):
                self.frames.append(self.render(self.q, self))
        mc = None if min_clear is np.inf else float(min_clear)
        self.log.append((f"execute:{label}", dict(n=len(traj), collided=collided,
                                                   min_clearance=mc)))
        return dict(collided=collided, min_clearance=mc, n_steps=len(traj))

    def dry_sweep(self, trajectory):
        """Check carried-OBB clearance along a trajectory WITHOUT side effects.

        Used by 'planned' place mode to gate/reroute a candidate plan using the
        object OBB as the collision volume. Returns (collided, min_clearance_m).
        """
        if self.T_tcp_obb is None:
            return False, np.inf
        q_save = self.q.copy()
        collided, min_clear = False, np.inf
        for q in np.asarray(trajectory, float):
            self.q = q
            c, clr = self.carried_collision()
            min_clear = min(min_clear, clr)
            collided = collided or c
        self.q = q_save
        return collided, (None if min_clear is np.inf else float(min_clear))

    # ---- settle on release ----------------------------------------------- #
    def settle_into_box(self):
        """After release, drop the object so it rests on the box floor (kinematic)."""
        info = self.scene["box_info"]
        m = self.object_mesh_at(self.object_T)
        obj_bottom_z = float(m.bounds[0, 2])
        dz = info["floor_z"] - obj_bottom_z
        T = self.object_T.copy()
        T[2, 3] += dz
        self.object_T = T
        self._update_object_static(T)
        for _ in range(8):                      # hold a moment on the settled object
            self._record()
        if self.render is not None:
            self.frames.append(self.render(self.q, self))

    def _update_object_static(self, T):
        self.object_T = T

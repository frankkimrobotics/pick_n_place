#!/usr/bin/env python3
"""env_v2 :: DiverseEnv -- the paper pick task with per-world object diversity, a big table
and walls, for the measured-drive RL/DAgger pipeline.

Everything is the paper task (env_paper.PaperPickEnv: same reward terms, weights, success
criterion) except:

  * OBJECT DIVERSITY, per world, resampled at every reset:
      shape    box | upright cylinder | upright hexagonal prism        (one-hot, 3)
      dims     box half sizes 0.015-0.045 m per axis (continuous)
               cylinder radius 0.018-0.045, half height 0.012-0.050 (continuous)
               hex      circumradius / half height quantised to the baked mesh grid
      mass     0.03-0.15 kg  (body_inertia kept consistent with mass and dims)
      friction 0.6-1.0       colour  random rgba (render-only)
  * SPAWN REGION x in [0, 0.5], y in [-0.3, 0.3] (robot base frame), rejection-sampled
    against: the table top (x 0.12-0.60, |y| <= 0.40), >= 0.05 m clear of the base
    footprint (radius 0.12) and of the bin footprint (bin (0.10, 0.40), 0.30 m square),
    and a top-down-cup IK REACHABILITY grid solved once per model and cached.
  * WALLS (back slab at x = -0.30, sides at y = +-0.50).  Any robot geom touching a wall
    geom ends the episode with info["wall_hit"] and the same -1 "fail" penalty the paper
    reward already gives for the object leaving the table.
  * OBSERVATION = the paper observation, with a 7-D object descriptor
    [hx, hy, hz, onehot(box, cyl, hex), mass] appended at the END.

BATCHING PATH (checked at runtime, mujoco_warp 3.12): the warp Model DOES support per-world
model fields through `put_model(mjm, batch_sizes={field: nworld})` for every field whose
array spec starts with "*", which includes geom_size, geom_pos, geom_dataid, geom_rbound,
geom_aabb, geom_friction, geom_rgba, body_mass, body_inertia, body_ipos.  geom_TYPE is NOT
batchable, so the three shapes live as three geoms on ONE object body and the two inactive
ones are shrunk to a 1 mm stub parked at the object's centroid, fully enclosed by the active
geom (see rl/build_scene_v2.py).  So there is ONE object body and ONE free joint per world --
no K-variant parked bodies, and no per-world qpos/dof address bookkeeping.

GEOMETRY TRICK: the object body origin is the BOTTOM CENTRE of the object (geom_pos and
body_ipos are offset up by the half height).  xpos[object0].z is therefore the object's lift
above the table for ANY size, so the parent's `float(self.half[2])` arithmetic stays correct
with self.half = (0, 0, 0) and the reward / teacher code needs no per-world patching.

    $PY rl/env_v2.py --nworld 64 --steps 20
"""
import os
import sys

import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import build_scene_v2 as B  # noqa: E402
import env_paper as EP  # noqa: E402
import env_warp as E  # noqa: E402
from env_paper import PaperPickEnv  # noqa: E402
from env_warp import CUP_R  # noqa: E402

SCENE_V2 = B.OUT_XML
# spawn region requested in the robot base frame
SPAWN_DEFAULT = (0.0, 0.5, -0.3, 0.3)
GOAL_X = (0.15, 0.50)
GOAL_Y = (-0.30, 0.30)
GOAL_Z = (0.05, 0.25)
# goals keep >= 0.10 m from every wall: GOAL_X/GOAL_Y above already satisfy that
# (back wall x = -0.30, side walls y = +-0.50).
WALL_CLEAR = 0.10
BASE_R = 0.12               # robot base footprint radius
BASE_CLEAR = 0.05
BIN_XY = (0.10, 0.40)
BIN_HALF = 0.15
BIN_CLEAR = 0.05
TABLE_MARGIN = 0.01
# off-table termination: env_paper's test is `obj < TABLE_X[0] - 0.08` etc., so feed it
# constants whose padded box is exactly the v2 table.
OFF_TABLE_X = (B.TABLE_X2[0] + 0.08, B.TABLE_X2[1] - 0.08)
OFF_TABLE_Y = (B.TABLE_Y2[0] + 0.10, B.TABLE_Y2[1] - 0.10)

SHAPES = ("box", "cyl", "hex")
# IK reachability grid (top-down cup pose)
GRID_DX = 0.025
GRID_Z = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18])

BATCH_FIELDS = ("geom_size", "geom_pos", "geom_quat", "geom_rbound", "geom_aabb", "geom_dataid",
                "geom_friction", "geom_rgba", "body_mass", "body_inertia", "body_ipos",
                "body_subtreemass", "body_invweight0", "dof_invweight0")


class DiverseEnv(PaperPickEnv):
    """PaperPickEnv + per-world object diversity + walls.  Drop-in for PaperPickEnv."""

    def __init__(self, nworld=1024, device="cuda:0", seed=0, xml=None,
                 variants="box,cyl,hex", spawn=SPAWN_DEFAULT, reach_cache=True, **kw):
        self._v2_ready = False
        self.nworld = nworld
        self.device = device
        if isinstance(variants, str):
            variants = [v.strip() for v in variants.split(",") if v.strip()]
        self.variants = [SHAPES.index(v) for v in variants]
        assert self.variants, "need at least one object variant"
        if isinstance(spawn, str):
            spawn = [float(v) for v in spawn.split(",")]
        self.spawn = tuple(float(v) for v in spawn)
        assert len(self.spawn) == 4
        if xml is None:
            xml = SCENE_V2
            if not os.path.exists(xml):
                B.build(xml)
        # tensors the overridden helpers need BEFORE the parent's first reset() call
        self.half_w = torch.zeros(nworld, 3, device=device)
        self.top_h = torch.zeros(nworld, device=device)
        self.shape_w = torch.zeros(nworld, dtype=torch.long, device=device)
        self.mass_w = torch.full((nworld,), 0.05, device=device)
        self.fric_w = torch.full((nworld,), 1.0, device=device)
        self.rgba_w = torch.zeros(nworld, 4, device=device)
        self.wall_hit_ep = torch.zeros(nworld, dtype=torch.bool, device=device)
        # the v2 scene (walls + 6 arm collision proxies + the object) overflows
        # mujoco_warp's default njmax 64 / nconmax 48 ("nefc overflow ... increase njmax
        # to 112" in dagger_v2); dropped constraints let a carried object tunnel into a
        # wall and blow the world up.  ~2 MB extra at nworld 256.
        self.data_kw = dict(njmax=int(kw.pop("njmax", 256)), nconmax=int(kw.pop("nconmax", 160)))
        # half_extents (0,0,0) => the parent's `float(self.half[2])` terms measure the lift
        # of the body origin, which we pin to the object's bottom face.
        kw.pop("half_extents", None)
        super().__init__(nworld=nworld, device=device, seed=seed, xml=xml,
                         half_extents=(0.0, 0.0, 0.0), **kw)
        # ---- swap in a per-world-batched warp Model -------------------------
        M = self.mjm
        import mujoco
        nid = mujoco.mj_name2id
        self.gid_box = nid(M, mujoco.mjtObj.mjOBJ_GEOM, "g_box")
        self.gid_cyl = nid(M, mujoco.mjtObj.mjOBJ_GEOM, "g_cyl")
        self.gid_hex = nid(M, mujoco.mjtObj.mjOBJ_GEOM, "g_hex")
        assert min(self.gid_box, self.gid_cyl, self.gid_hex) >= 0, "scene is not a v2 scene"
        self.gid_shape = [self.gid_box, self.gid_cyl, self.gid_hex]
        self.mesh_of = {}
        for name, r, h in B.hex_library():
            self.mesh_of[name] = nid(M, mujoco.mjtObj.mjOBJ_MESH, name)
        self.hex_dataid = np.array([[self.mesh_of[B.hex_mesh_name(i, j)] for j in range(len(B.HEX_H))]
                                    for i in range(len(B.HEX_R))], dtype=np.int64)
        self.stub_dataid = self.mesh_of["hex_stub"]
        # MuJoCo re-expresses every mesh in its own principal-inertia frame and folds the
        # transform into the REFERENCING geom's pos/quat at compile time.  For a squat hex
        # prism the principal axes get permuted, so swapping geom_dataid alone would leave
        # the prism lying on its side.  Keep the per-mesh frame + exact aabb / rbound and
        # write them together with the dataid (geom_quat is batchable too).
        self.mesh_quat = np.asarray(M.mesh_quat, float)                 # (nmesh, 4) wxyz
        self.mesh_off = np.asarray(M.mesh_pos, float)                   # (nmesh, 3)
        nm = M.nmesh
        self.mesh_aabb = np.zeros((nm, 2, 3))
        self.mesh_rbound = np.zeros(nm)
        for i in range(nm):
            a, k = int(M.mesh_vertadr[i]), int(M.mesh_vertnum[i])
            v = np.asarray(M.mesh_vert[a:a + k], float)
            lo, hi = v.min(0), v.max(0)
            self.mesh_aabb[i, 0] = 0.5 * (lo + hi)
            self.mesh_aabb[i, 1] = 0.5 * (hi - lo)
            self.mesh_rbound[i] = np.linalg.norm(self.mesh_aabb[i, 0]) + np.linalg.norm(self.mesh_aabb[i, 1])
        self.m = mjw.put_model(self.mjm, batch_sizes={f: nworld for f in BATCH_FIELDS})
        self.mw = {f: wp.to_torch(getattr(self.m, f)) for f in BATCH_FIELDS}
        for f in BATCH_FIELDS:            # fail loudly if a field silently stayed shared
            assert self.mw[f].shape[0] == nworld, f"{f} not batched: {self.mw[f].shape}"
        # reference inertial scales for the constraint-solver invweights
        self._iw_ref = self.mw["body_invweight0"][0, self.bid_obj].clone()
        self._dw_ref = self.mw["dof_invweight0"][0, self.vadr_obj:self.vadr_obj + 6].clone()
        self._m_ref = float(self.mjm.body_mass[self.bid_obj])
        self._I_ref = float(np.mean(self.mjm.body_inertia[self.bid_obj]))
        # wall / robot geom masks for the contact test
        self.wall_gids = [nid(M, mujoco.mjtObj.mjOBJ_GEOM, n) for n in B.WALL_NAMES]
        rob = [nid(M, mujoco.mjtObj.mjOBJ_GEOM, f"prox_{b}") for b in B.ARM_PROXIES]
        rob.append(nid(M, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip"))
        self.robot_gids = [g for g in rob if g >= 0]
        self.wall_mask = torch.zeros(M.ngeom + 1, dtype=torch.bool, device=device)
        self.wall_mask[self.wall_gids] = True
        self.robot_mask = torch.zeros(M.ngeom + 1, dtype=torch.bool, device=device)
        self.robot_mask[self.robot_gids] = True
        self._con_ar = None
        # ---- reachability ---------------------------------------------------
        self._build_reach_grid(cache=reach_cache)
        # ---- divergence guard -------------------------------------------------
        # A world whose physics blows up (object dragged into a wall while the episode is
        # already `off`/done and auto_reset is False) leaves NaN in the SOLVER state, not just
        # in qpos/qvel.  Clearing qpos/qvel on reset is therefore not enough: the warm start
        # re-NaNs the world on its very next step, for every later episode (dagger_v2 world
        # 210).  Keep torch views on everything that has to be zeroed.
        self._solver_state = []
        for f in ("qacc_warmstart", "cqacc_warmstart", "qacc", "cqacc", "qacc_smooth",
                  "cqacc_smooth", "qfrc_applied", "qfrc_constraint", "cqfrc_constraint",
                  "act", "actuator_velocity", "actuator_force", "cvel", "cacc"):
            arr = getattr(self.d, f, None)
            if arr is None:
                continue
            try:
                t = wp.to_torch(arr)
            except Exception:
                continue
            if t.ndim >= 1 and t.shape[0] == nworld and t.is_floating_point():
                self._solver_state.append(t)
        self.diverged_ep = torch.zeros(nworld, dtype=torch.bool, device=device)
        self.n_diverged = 0
        self._v2_ready = True
        self.reset(torch.ones(nworld, dtype=torch.bool, device=device))

    # ================= reachability =================
    def _build_reach_grid(self, cache=True):
        gx = np.arange(self.spawn[0], self.spawn[1] + 1e-9, GRID_DX)
        gy = np.arange(self.spawn[2], self.spawn[3] + 1e-9, GRID_DX)
        self.gx, self.gy, self.gz = gx, gy, GRID_Z
        tag = f"{len(gx)}x{len(gy)}x{len(GRID_Z)}_{self.spawn[0]:.3f}_{self.spawn[1]:.3f}_{self.spawn[2]:.3f}_{self.spawn[3]:.3f}"
        path = os.path.join(os.path.dirname(SCENE_V2), f"_reach_v2_{tag}.npz")
        reach = None
        # The grid depends only on the ROBOT kinematics + joint/elbow limits (IK ignores
        # collisions), which the scene template fixes -- so the cache is keyed on the grid
        # geometry alone and survives edits to the table / walls / object.
        if cache and os.path.exists(path):
            try:
                reach = np.load(path)["reach"]
            except Exception:
                reach = None
        if reach is None:
            import mujoco
            demo = {"__file__": os.path.join(os.path.dirname(HERE), "mjwarp_pick_demo.py")}
            exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
            dik = mujoco.MjData(self.mjm)
            reach = np.zeros((len(gx), len(gy), len(GRID_Z)), bool)
            static = self._static_ok(gx[:, None], gy[None, :], 0.03)
            home = np.asarray(self.q_home, float)
            qsol = np.zeros((len(gx), len(gy), len(GRID_Z), 6))
            # NOTE: seed every solve from the HOME pose.  Warm-seeding from the previous
            # cell's solution walks the damped-LS solver into local minima and carved
            # spurious 5-cm-wide "unreachable" stripes out of the middle of the table.
            for i, x in enumerate(gx):
                for j, y in enumerate(gy):
                    if not static[i, j]:
                        continue
                    for k, z in enumerate(GRID_Z):
                        q, err = demo["ik"](self.mjm, dik, "tcp", [float(x), float(y), float(z)],
                                            demo["R_DOWN"], home, restarts=4)
                        if err < 0.005:
                            reach[i, j, k] = True
                            qsol[i, j, k] = q
            # second pass: retry the failures from their reachable neighbours' solutions
            for _ in range(2):
                todo = np.argwhere(static[:, :, None] & ~reach)
                if not len(todo):
                    break
                got = 0
                for i, j, k in todo:
                    seeds = [qsol[a, b, c] for a, b, c in
                             ((i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j + 1, k),
                              (i, j, k - 1), (i, j, k + 1))
                             if 0 <= a < len(gx) and 0 <= b < len(gy) and 0 <= c < len(GRID_Z)
                             and reach[a, b, c]]
                    for s in seeds:
                        q, err = demo["ik"](self.mjm, dik, "tcp",
                                            [float(gx[i]), float(gy[j]), float(GRID_Z[k])],
                                            demo["R_DOWN"], s, restarts=0)
                        if err < 0.005:
                            reach[i, j, k] = True
                            qsol[i, j, k] = q
                            got += 1
                            break
                if got == 0:
                    break
            if cache:
                try:
                    np.savez(path, reach=reach, mtime=os.path.getmtime(SCENE_V2))
                except Exception:
                    pass
        self.reach = reach
        self.reach_cum = np.concatenate([np.zeros((len(gx), len(gy), 1), np.int32),
                                         np.cumsum(reach.astype(np.int32), axis=2)], axis=2)
        # "usable" = valid for a median object (top 0.06 m -> grasp/hover band 0.04 .. 0.12 m)
        k0 = int(np.searchsorted(GRID_Z, 0.04))
        k1 = int(np.searchsorted(GRID_Z, 0.12))
        band = reach[:, :, k0:k1 + 1].all(-1)
        stat = self._static_ok(gx[:, None], gy[None, :], 0.03)
        usable = stat & band
        self.static_frac = float(stat.mean())
        self.ik_frac = float(band[stat].mean()) if stat.any() else 0.0
        # fallback cells: valid for the LARGEST object over the whole grasp band
        big = self._static_ok(gx[:, None], gy[None, :], np.hypot(0.045, 0.045)) & \
            reach[:, :, k0:int(np.searchsorted(GRID_Z, 0.16)) + 1].all(-1)
        ii, jj = np.nonzero(big if big.any() else usable)
        self.fallback_xy = np.stack([gx[ii], gy[jj]], -1) if len(ii) else np.array([[0.38, 0.0]])
        self.reach_frac = float(band.mean())
        self.spawn_frac = float(usable.mean())
        print(f"[env_v2] reach grid {reach.shape}: of the requested spawn box "
              f"x{self.spawn[:2]} y{self.spawn[2:]}, {100 * self.static_frac:.0f} % is on the table and "
              f"clear of the base/bin, and {100 * self.ik_frac:.0f} % of that is IK-reachable with a "
              f"top-down cup over the 4-12 cm grasp band -> {100 * self.spawn_frac:.0f} % usable "
              f"(fallback cells {len(self.fallback_xy)})", flush=True)

    def _static_ok(self, x, y, rad, in_box=True):
        """Table / base / bin clearance for an object of circumradius `rad` at (x, y).
        `in_box=False` drops the spawn-box test (env_v3 places distractors anywhere on
        the table, not only inside the TARGET spawn box)."""
        on_table = ((x - rad >= B.TABLE_X2[0] + TABLE_MARGIN) & (x + rad <= B.TABLE_X2[1] - TABLE_MARGIN)
                    & (np.abs(y) + rad <= B.TABLE_Y2[1] - TABLE_MARGIN))
        base_ok = np.hypot(x, y) >= BASE_R + BASE_CLEAR + rad
        b = BIN_HALF + BIN_CLEAR
        bin_ok = ~((np.abs(x - BIN_XY[0]) < b + rad) & (np.abs(y - BIN_XY[1]) < b + rad))
        if not in_box:
            return on_table & base_ok & bin_ok
        box = (x >= self.spawn[0]) & (x <= self.spawn[1]) & (y >= self.spawn[2]) & (y <= self.spawn[3])
        return on_table & base_ok & bin_ok & box

    def _sample_spawn(self, rad, z_lo, z_hi, n, tries=48):
        """Rejection-sample (n, 2) object centres that are on the table, clear of the base and
        the bin, and IK-reachable with a top-down cup pose over [z_lo, z_hi]."""
        klo = np.clip(np.searchsorted(self.gz, z_lo, "right") - 1, 0, len(self.gz) - 1)
        khi = np.clip(np.searchsorted(self.gz, z_hi, "left"), 0, len(self.gz) - 1)
        khi = np.maximum(khi, klo)
        xs = self.rng.uniform(self.spawn[0], self.spawn[1], size=(n, tries))
        ys = self.rng.uniform(self.spawn[2], self.spawn[3], size=(n, tries))
        ix = np.clip(np.rint((xs - self.gx[0]) / GRID_DX).astype(int), 0, len(self.gx) - 1)
        iy = np.clip(np.rint((ys - self.gy[0]) / GRID_DX).astype(int), 0, len(self.gy) - 1)
        need = (khi - klo + 1)[:, None]
        got = self.reach_cum[ix, iy, (khi + 1)[:, None]] - self.reach_cum[ix, iy, klo[:, None]]
        ok = self._static_ok(xs, ys, rad[:, None]) & (got == need)
        first = np.argmax(ok, axis=1)
        out = np.stack([xs[np.arange(n), first], ys[np.arange(n), first]], -1)
        bad = ~ok[np.arange(n), first]
        if bad.any():
            pick = self.rng.integers(0, len(self.fallback_xy), size=int(bad.sum()))
            out[bad] = self.fallback_xy[pick] + self.rng.uniform(-0.004, 0.004, size=(int(bad.sum()), 2))
        self._last_reject = float(bad.mean())
        return out

    # ================= object helpers =================
    def _grasp_point(self):
        g = self._obj_pos().clone()
        g[:, 2] = g[:, 2] + self.top_h + CUP_R
        return g

    def _sample_variants(self, n):
        rng = self.rng
        shape = np.asarray(self.variants)[rng.integers(0, len(self.variants), size=n)]
        hx = np.empty(n); hy = np.empty(n); hz = np.empty(n)
        hi = np.zeros(n, int); hj = np.zeros(n, int)
        m_box = shape == 0
        if m_box.any():
            k = int(m_box.sum())
            hx[m_box] = rng.uniform(*B.BOX_HALF, size=k)
            hy[m_box] = rng.uniform(*B.BOX_HALF, size=k)
            hz[m_box] = rng.uniform(*B.BOX_HALF, size=k)
        m_cyl = shape == 1
        if m_cyl.any():
            k = int(m_cyl.sum())
            r = rng.uniform(*B.CYL_R, size=k)
            hx[m_cyl] = r; hy[m_cyl] = r
            hz[m_cyl] = rng.uniform(*B.CYL_H, size=k)
        m_hex = shape == 2
        if m_hex.any():
            k = int(m_hex.sum())
            r = rng.uniform(*B.CYL_R, size=k)
            h = rng.uniform(*B.CYL_H, size=k)
            # meshes cannot be rescaled at runtime -> quantise to the baked mesh grid
            i_idx = np.argmin(np.abs(B.HEX_R[None, :] - r[:, None]), axis=1)
            j_idx = np.argmin(np.abs(B.HEX_H[None, :] - h[:, None]), axis=1)
            hx[m_hex] = B.HEX_R[i_idx]; hy[m_hex] = B.HEX_R[i_idx]; hz[m_hex] = B.HEX_H[j_idx]
            hi[m_hex] = i_idx; hj[m_hex] = j_idx
        mass = rng.uniform(*B.MASS_RANGE, size=n)
        fric = rng.uniform(*B.FRIC_RANGE, size=n)
        rgba = np.concatenate([rng.uniform(0.15, 0.95, size=(n, 3)), np.ones((n, 1))], 1)
        return shape, hx, hy, hz, hi, hj, mass, fric, rgba

    def _inertia(self, shape, hx, hy, hz, mass):
        """Diagonal inertia about the centroid (principal axes = body axes, upright)."""
        I = np.zeros((len(shape), 3))
        a, b, c = 2 * hx, 2 * hy, 2 * hz                     # full sizes
        m_box = shape == 0
        I[m_box, 0] = mass[m_box] / 12 * (b[m_box] ** 2 + c[m_box] ** 2)
        I[m_box, 1] = mass[m_box] / 12 * (a[m_box] ** 2 + c[m_box] ** 2)
        I[m_box, 2] = mass[m_box] / 12 * (a[m_box] ** 2 + b[m_box] ** 2)
        rnd = ~m_box
        # hex prism about its axis: I_z/m = 5 r^2 / 12 = (5/6) * (r^2 / 2)  -> r_eff^2 = 5/6 r^2
        k = np.where(shape[rnd] == 2, 5.0 / 6.0, 1.0)
        r2 = k * hx[rnd] ** 2
        I[rnd, 0] = mass[rnd] * (3 * r2 + (2 * hz[rnd]) ** 2) / 12
        I[rnd, 1] = I[rnd, 0]
        I[rnd, 2] = mass[rnd] * r2 / 2
        return I

    def _write_object(self, idx, shape, hx, hy, hz, hi, hj, mass, fric, rgba):
        """Per-world model fields for the sampled objects (torch views on the warp Model)."""
        dev, f32 = self.device, torch.float32
        t = lambda v, d=f32: torch.as_tensor(np.ascontiguousarray(v), device=dev, dtype=d)  # noqa: E731
        hzt = t(hz)
        stub = B.STUB
        # --- all three geoms: stub at the centroid; the active one is overwritten below ---
        for g in self.gid_shape:
            self.mw["geom_size"][idx, g] = stub
            self.mw["geom_rbound"][idx, g] = float(np.sqrt(3) * stub)
            self.mw["geom_aabb"][idx, g, 0] = 0.0
            self.mw["geom_aabb"][idx, g, 1] = stub
            self.mw["geom_pos"][idx, g, 0] = 0.0
            self.mw["geom_pos"][idx, g, 1] = 0.0
            self.mw["geom_pos"][idx, g, 2] = hzt
            self.mw["geom_friction"][idx, g, 0] = t(fric)
            self.mw["geom_rgba"][idx, g] = t(rgba)
        sd = int(self.stub_dataid)
        self.mw["geom_dataid"][idx, self.gid_hex] = sd
        self.mw["geom_quat"][idx, self.gid_hex] = t(self.mesh_quat[sd])
        self.mw["geom_rbound"][idx, self.gid_hex] = float(self.mesh_rbound[sd])
        self.mw["geom_aabb"][idx, self.gid_hex] = t(self.mesh_aabb[sd])
        # --- active geom ---
        for s, gid in enumerate(self.gid_shape):
            sel = np.nonzero(shape == s)[0]
            if not len(sel):
                continue
            ii = idx[sel]
            if s == 0:
                size = np.stack([hx[sel], hy[sel], hz[sel]], -1)
            elif s == 1:
                size = np.stack([hx[sel], hz[sel], np.zeros(len(sel))], -1)
            else:
                # mesh: geom_size is ignored; the mesh's own frame / aabb / rbound apply
                did = self.hex_dataid[hi[sel], hj[sel]]
                size = np.stack([hx[sel], hy[sel], hz[sel]], -1)
                self.mw["geom_dataid"][ii, gid] = t(did, torch.int32)
                self.mw["geom_quat"][ii, gid] = t(self.mesh_quat[did])
                self.mw["geom_pos"][ii, gid] = t(self.mesh_off[did] + np.stack(
                    [np.zeros(len(sel)), np.zeros(len(sel)), hz[sel]], -1))
                self.mw["geom_size"][ii, gid] = t(size)
                self.mw["geom_rbound"][ii, gid] = t(self.mesh_rbound[did])
                self.mw["geom_aabb"][ii, gid] = t(self.mesh_aabb[did])
                continue
            self.mw["geom_size"][ii, gid] = t(size)
            ext = np.stack([hx[sel], hy[sel], hz[sel]], -1)
            self.mw["geom_rbound"][ii, gid] = t(np.linalg.norm(ext, axis=-1))
            self.mw["geom_aabb"][ii, gid, 0] = 0.0
            self.mw["geom_aabb"][ii, gid, 1] = t(ext)
        # --- body inertial (origin at the bottom face, centroid at +hz) ---
        Idiag = self._inertia(shape, hx, hy, hz, mass)
        self.mw["body_mass"][idx, self.bid_obj] = t(mass)
        self.mw["body_subtreemass"][idx, self.bid_obj] = t(mass)
        self.mw["body_ipos"][idx, self.bid_obj, 0] = 0.0
        self.mw["body_ipos"][idx, self.bid_obj, 1] = 0.0
        self.mw["body_ipos"][idx, self.bid_obj, 2] = hzt
        self.mw["body_inertia"][idx, self.bid_obj] = t(Idiag)
        # solver reference weights: invweight ~ 1/m (translation), 1/I (rotation)
        sm = t(self._m_ref / mass)
        si = t(self._I_ref / np.mean(Idiag, axis=-1).clip(1e-9))
        self.mw["body_invweight0"][idx, self.bid_obj, 0] = self._iw_ref[0] * sm
        self.mw["body_invweight0"][idx, self.bid_obj, 1] = self._iw_ref[1] * si
        va = self.vadr_obj
        for k in range(3):
            self.mw["dof_invweight0"][idx, va + k] = self._dw_ref[k] * sm
            self.mw["dof_invweight0"][idx, va + 3 + k] = self._dw_ref[3 + k] * si
        # --- torch mirrors used by the env / obs / teacher ---
        self.half_w[idx] = t(np.stack([hx, hy, hz], -1))
        self.top_h[idx] = 2 * hzt
        self.shape_w[idx] = t(shape, torch.long)
        self.mass_w[idx] = t(mass)
        self.fric_w[idx] = t(fric)
        self.rgba_w[idx] = t(rgba)

    # ================= reset =================
    def reset(self, mask):
        if not self._v2_ready:
            return super().reset(mask)
        super().reset(mask)                     # drive state, DR, home start, legacy obj pose
        idx_t = torch.nonzero(mask).squeeze(-1)
        if idx_t.numel() == 0:
            return
        idx = idx_t.cpu().numpy()
        n = len(idx)
        shape, hx, hy, hz, hi, hj, mass, fric, rgba = self._sample_variants(n)
        self._write_object(idx_t, shape, hx, hy, hz, hi, hj, mass, fric, rgba)
        # ---- spawn pose -------------------------------------------------------
        rad = np.hypot(hx, hy)
        top = 2 * hz
        z_lo = np.maximum(GRID_Z[0], top + CUP_R - 0.025)
        z_hi = top + CUP_R + 0.05
        xy = self._sample_spawn(rad, z_lo, z_hi, n)
        yaw = self.rng.uniform(0, np.pi, size=n)
        qa = self.jadr_obj
        dev = self.device
        self.qpos[idx_t, qa:qa + 2] = torch.as_tensor(xy, device=dev, dtype=torch.float32)
        self.qpos[idx_t, qa + 2] = 0.001                      # bottom face 1 mm above the table
        self.qpos[idx_t, qa + 3] = torch.as_tensor(np.cos(yaw / 2), device=dev, dtype=torch.float32)
        self.qpos[idx_t, qa + 4:qa + 6] = 0.0
        self.qpos[idx_t, qa + 6] = torch.as_tensor(np.sin(yaw / 2), device=dev, dtype=torch.float32)
        self.qvel[idx_t, self.vadr_obj:self.vadr_obj + 6] = 0.0
        self.xfrc[idx_t, self.bid_obj] = 0.0
        # ---- 3-D goal ---------------------------------------------------------
        ang = self.rng.uniform(0, 2 * np.pi, size=n)
        rr = self.rng.uniform(0.08, self.target_max, size=n)
        gx = np.clip(xy[:, 0] + rr * np.cos(ang), *GOAL_X)
        gy = np.clip(xy[:, 1] + rr * np.sin(ang), *GOAL_Y)
        gz = np.clip(self.rng.uniform(*self.paper["goal_z"], size=n), *GOAL_Z)
        g = torch.as_tensor(np.stack([gx, gy], -1), device=dev, dtype=torch.float32)
        self._assign_target(idx_t, g)
        self.goal[idx_t, :2] = g
        self.goal[idx_t, 2] = torch.as_tensor(gz, device=dev, dtype=torch.float32)
        self.wall_hit_ep[idx_t] = False
        self.diverged_ep[idx_t] = False
        # the free-joint quaternion and the whole solver warm start must be sane before the
        # next forward(), otherwise a world that diverged once stays NaN for ever
        for t in self._solver_state:
            t[idx_t] = 0.0
        self.qpos[idx_t] = torch.nan_to_num(self.qpos[idx_t])
        self.qvel[idx_t] = 0.0
        E.mjw.forward(self.m, self.d)
        tcp, _ = self._tcp()
        self.phi_approach[idx_t] = -torch.norm(tcp[idx_t] - self._grasp_point()[idx_t], dim=-1)

    # ================= observation =================
    def observe(self):
        obs = super().observe()
        oh = torch.zeros(self.nworld, 3, device=self.device)
        oh.scatter_(1, self.shape_w[:, None], 1.0)
        desc = torch.cat([self.half_w, oh, self.mass_w[:, None]], -1)
        obs = torch.cat([obs, desc], -1)
        # last line of defence: a single NaN row here NaNs a whole BC fit (dagger_v2).
        return torch.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)

    # ================= walls =================
    def _wall_contacts(self):
        """(N,) bool: a robot geom is penetrating a wall geom this step."""
        con = self.d.contact
        g = wp.to_torch(con.geom).long()
        w = wp.to_torch(con.worldid).long()
        dist = wp.to_torch(con.dist)
        if self._con_ar is None or self._con_ar.numel() != g.shape[0]:
            self._con_ar = torch.arange(g.shape[0], device=self.device)
        nac = wp.to_torch(self.d.nacon).reshape(-1)[:1].long()
        valid = self._con_ar < nac                     # no host sync
        ng = self.mjm.ngeom
        g0, g1 = g[:, 0].clamp(0, ng), g[:, 1].clamp(0, ng)
        pair = (self.wall_mask[g0] & self.robot_mask[g1]) | (self.wall_mask[g1] & self.robot_mask[g0])
        hit = valid & pair & (dist < 0.0)
        out = torch.zeros(self.nworld, device=self.device)
        out.index_add_(0, w.clamp(0, self.nworld - 1), hit.float())
        return out > 0

    # ================= reward / termination =================
    def reward(self, want, latched_now, released, broke, tcp_before, obj_before, a):
        # env_paper's off-table test reads module-level TABLE_X / TABLE_Y (the legacy small
        # table).  Patch them for the duration of the parent call so `off` means "off the v2
        # table" -- everything else in the paper reward is untouched.
        ox, oy = EP.TABLE_X, EP.TABLE_Y
        EP.TABLE_X, EP.TABLE_Y = OFF_TABLE_X, OFF_TABLE_Y
        try:
            r, done, info = super().reward(want, latched_now, released, broke, tcp_before, obj_before, a)
        finally:
            EP.TABLE_X, EP.TABLE_Y = ox, oy
        hit_now = self._wall_contacts()
        # charge the fail term ONCE per episode (auto_reset=False keeps the contact alive)
        new = hit_now & ~self.wall_hit_ep & ~info["off"]
        self.wall_hit_ep |= hit_now
        if new.any():
            r = r - new.float()                          # same -1 as the paper's off-table fail
            self.ep_comp_p[:, self.RKEYS_PAPER.index("fail")] -= new.float()
            info["ep_comp"] = self.ep_comp_p.clone()
        done = done | self.wall_hit_ep
        info["placed"] = info["placed"] & ~self.wall_hit_ep
        info["wall_hit"] = self.wall_hit_ep.clone()
        info["wall_hit_now"] = hit_now
        info["half"] = self.half_w.clone()
        info["shape"] = self.shape_w.clone()
        info["mass"] = self.mass_w.clone()
        return r, done, info


    # ================= step (divergence guard) =================
    DIVERGE_POS = 5.0            # m: no legal object/site position is anywhere near this
    DIVERGE_VEL = 500.0          # rad/s or m/s

    def _diverged(self):
        """(N,) bool: this world's physics state is NaN/Inf or absurd."""
        q, v = self.qpos, self.qvel
        bad = ~torch.isfinite(q).all(-1) | ~torch.isfinite(v).all(-1)
        bad |= ~torch.isfinite(self.xpos).flatten(1).all(-1)
        bad |= self.xpos[:, self.bid_obj].abs().amax(-1) > self.DIVERGE_POS
        bad |= v.abs().amax(-1) > self.DIVERGE_VEL
        return bad

    def step(self, action):
        obs, r, done, info = super().step(torch.nan_to_num(action).clamp(-1, 1))
        bad = self._diverged()
        if bool(bad.any()):
            # such a world is always already terminated (off-table / wall hit) -- bc_curobo
            # runs with auto_reset=False, so it would otherwise keep integrating garbage and
            # feed NaN rows into the DAgger set.  Recycle it now and keep it marked failed.
            self.n_diverged += int(bad.sum())
            r = torch.nan_to_num(r)
            self.reset(bad)
            self.diverged_ep |= bad
            obs = self.observe()
        done = done | self.diverged_ep                  # tensor ops: no host sync
        info["placed"] = info["placed"] & ~self.diverged_ep
        info["diverged"] = self.diverged_ep.clone()
        return obs, torch.nan_to_num(r), done, info


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    args = ap.parse_args()
    wp.init()
    env = DiverseEnv(nworld=args.nworld, drive=args.drive, dr=True, ep_len=150)
    obs = env.observe()
    print(f"obs dim {obs.shape[-1]} | nworld {args.nworld} | substeps {env.substeps}")
    t0 = time.time()
    for _ in range(args.steps):
        a = torch.rand(args.nworld, 7, device=env.device) * 2 - 1
        obs, r, done, info = env.step(a)
    dt = time.time() - t0
    print(f"{args.steps} x {args.nworld} in {dt:.2f}s = {args.nworld * args.steps / dt:,.0f} env-steps/s "
          f"| reward {r.mean():.3f} | wall_hit {int(info['wall_hit'].sum())}")

"""batchsim.world :: N copies of one MuJoCo scene on the GPU (mujoco_warp), with zero-copy torch views.

All worlds share one MjModel (a mujoco_warp constraint). What may differ per world: the state (object layout,
arm pose), the joint physics (armature / damping / friction, see `set_joint_physics`) and everything the layers
above keep per world (drive parameters, latencies). Different object SETS need one BatchWorld per scene.
"""
import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mw

NJ = 6


class BatchWorld:
    def __init__(self, xml, nworld, device="cuda:0", joint_physics=None, nconmax=None, njmax=None):
        wp.config.quiet = True
        self.device, self.N = device, int(nworld)
        self.mjm = mujoco.MjModel.from_xml_path(xml) if isinstance(xml, str) else xml
        if joint_physics is not None:                      # nominal values go in BEFORE put_model: friction adds
            for k, f in (("armature", "dof_armature"), ("damping", "dof_damping"), ("frictionloss", "dof_frictionloss")):
                getattr(self.mjm, f)[:NJ] = np.asarray(joint_physics[k], float)      # constraint rows that are sized there
        self.mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, self.mjd)
        self.dt = float(self.mjm.opt.timestep)
        # contact / constraint buffers PER WORLD. mujoco_warp silently drops what does not fit (NaNs follow), so the
        # defaults are generous: every object can touch the table, the box and the cup at once.
        nobj = sum(1 for i in range(self.mjm.nbody) if (mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("object"))
        self.nconmax = int(nconmax or max(64, 12 * nobj + 32))
        self.njmax = int(njmax or max(256, 4 * self.nconmax + 64))
        with wp.ScopedDevice(device):
            self.m = mw.put_model(self.mjm)
            self.d = mw.put_data(self.mjm, self.mjd, nworld=self.N, nconmax=self.nconmax, njmax=self.njmax)
        T = wp.to_torch
        self.qpos, self.qvel, self.ctrl = T(self.d.qpos), T(self.d.qvel), T(self.d.ctrl)
        self.qfrc_bias, self.xfrc = T(self.d.qfrc_bias), T(self.d.xfrc_applied)
        self.xpos, self.xquat = T(self.d.xpos), T(self.d.xquat)
        self.site_xpos, self.site_xmat = T(self.d.site_xpos), T(self.d.site_xmat)
        self.nefc = T(self.d.nefc)
        nid = lambda t, n: mujoco.mj_name2id(self.mjm, t, n)
        self.sid_tcp = nid(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.obj_bid = [i for i in range(self.mjm.nbody) if (mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("object")]
        self.obj_qadr = [int(self.mjm.jnt_qposadr[self.mjm.body_jntadr[b]]) for b in self.obj_bid]
        self.obj_vadr = [int(self.mjm.jnt_dofadr[self.mjm.body_jntadr[b]]) for b in self.obj_bid]
        self.qpos0 = torch.tensor(self.mjd.qpos, dtype=torch.float32, device=device)
        self.cams = [mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.mjm.ncam)]

    # ---------------------------------------------------------------- per-world joint physics
    def set_joint_physics(self, armature=None, damping=None, frictionloss=None):
        """Per-world arm joint physics, each [N, 6] (torch or numpy). Replaces the shared (1, nv) model rows."""
        for val, f in ((armature, "dof_armature"), (damping, "dof_damping"), (frictionloss, "dof_frictionloss")):
            if val is None:
                continue
            full = torch.tensor(getattr(self.mjm, f), dtype=torch.float32, device=self.device).repeat(self.N, 1)
            full[:, :NJ] = torch.as_tensor(val, dtype=torch.float32, device=self.device)
            setattr(self.m, f, wp.from_torch(full.contiguous()))
            setattr(self, "_" + f, full)                    # keep the torch owner alive

    # ---------------------------------------------------------------- stepping
    def step(self):
        mw.step(self.m, self.d)

    def forward(self):
        mw.forward(self.m, self.d)

    def reset(self, mask=None, qpos=None):
        """Write qpos (default: the scene's own) into the masked worlds, zero their velocities, run forward."""
        mask = torch.ones(self.N, dtype=torch.bool, device=self.device) if mask is None else mask
        q = self.qpos0.expand(self.N, -1) if qpos is None else qpos
        with wp.ScopedDevice(self.device):                 # clears warm starts / contacts too: a world that went NaN
            mw.reset_data(self.m, self.d, reset=wp.from_torch(mask.to(torch.int32).contiguous(), dtype=wp.int32))   # must not stay NaN
        self.qpos[mask] = q[mask] if q.shape[0] == self.N else q
        self.qvel[mask] = 0.0
        self.ctrl[mask] = 0.0
        self.xfrc[mask] = 0.0
        self.forward()

    def healthy(self):
        """False for worlds whose state went non-finite or whose constraint buffer is (nearly) full."""
        return torch.isfinite(self.qpos).all(1) & (self.nefc < int(0.95 * self.njmax))

    def tcp(self):
        return self.site_xpos[:, self.sid_tcp], self.site_xmat[:, self.sid_tcp]

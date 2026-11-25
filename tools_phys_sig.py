"""Deterministic physics signature for one scene, so MuJoCo versions can be compared
numerically instead of by assertion.

This project's behaviour (suction seal, crush guard, contact-force gating) depends on
contact dynamics, so a major-version bump has to be checked for drift before anything is
migrated. Identical initial state -> step -> dump qpos/contact/force summary.
"""
import json, os, sys
import numpy as np
import mujoco

XML = sys.argv[1]
OUT = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 600

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)

# a fixed, non-trivial arm pose so the scene is in contact, not floating at the default
q0 = np.zeros(m.nq)
q0[:min(6, m.nq)] = [0.0, -0.6, 1.2, -0.6, -1.57, 0.0][:min(6, m.nq)]
# keep any free-joint objects at their model defaults
q0[min(6, m.nq):] = m.qpos0[min(6, m.nq):]
d.qpos[:] = q0
d.qvel[:] = 0.0
mujoco.mj_forward(m, d)

ncon_trace, force_trace = [], []
fbuf = np.zeros(6)
for i in range(N):
    mujoco.mj_step(m, d)
    if i % 50 == 0:
        ncon_trace.append(int(d.ncon))
        tot = 0.0
        for ci in range(d.ncon):
            mujoco.mj_contactForce(m, d, ci, fbuf)
            tot += abs(float(fbuf[0]))
        force_trace.append(round(tot, 6))

sig = dict(
    mujoco=mujoco.__version__,
    xml=os.path.basename(XML),
    nq=int(m.nq), nv=int(m.nv), nbody=int(m.nbody), ngeom=int(m.ngeom),
    steps=N, timestep=float(m.opt.timestep),
    solver=int(m.opt.solver), integrator=int(m.opt.integrator),
    iterations=int(m.opt.iterations), cone=int(m.opt.cone),
    qpos_final=[round(float(x), 8) for x in d.qpos],
    qvel_norm=round(float(np.linalg.norm(d.qvel)), 8),
    ncon_trace=ncon_trace,
    force_trace=force_trace,
    energy=round(float(d.energy[0] + d.energy[1]), 8) if hasattr(d, "energy") else None,
)
json.dump(sig, open(OUT, "w"), indent=1)
print(f"{mujoco.__version__}: nq {m.nq} solver {m.opt.solver} integrator {m.opt.integrator} "
      f"cone {m.opt.cone} dt {m.opt.timestep} | final |qvel| {sig['qvel_norm']:.6f} "
      f"| ncon {ncon_trace}")

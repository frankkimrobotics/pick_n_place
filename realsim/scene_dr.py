"""realsim.scene_dr :: per-episode scene randomisation for the closed-loop evaluator.

Appearance (lights, colours, camera extrinsics) and physics (object pose / mass / friction).
Everything is drawn from the caller's rng, so a (scene, seed) pair is reproducible.
"""
import mujoco
import numpy as np


def _small_rot(rng, deg):
    v = rng.normal(0, np.radians(deg), 3)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, v / (np.linalg.norm(v) + 1e-12), np.linalg.norm(v))
    return q


def randomize_scene(m, d, objs, rng):
    # ---- lights + headlight
    for i in range(m.nlight):
        m.light_pos[i] += rng.normal(0, 0.15, 3)
        m.light_diffuse[i] = np.clip(m.light_diffuse[i] * rng.uniform(0.7, 1.3), 0, 1)
    m.vis.headlight.ambient[:] = np.clip(np.array(m.vis.headlight.ambient) * rng.uniform(0.7, 1.4), 0, 1)
    m.vis.headlight.diffuse[:] = np.clip(np.array(m.vis.headlight.diffuse) * rng.uniform(0.8, 1.2), 0, 1)
    # ---- colours: every geom / material, +-12 % per channel
    m.geom_rgba[:, :3] = np.clip(m.geom_rgba[:, :3] * rng.uniform(0.88, 1.12, (m.ngeom, 3)), 0, 1)
    if m.nmat:
        m.mat_rgba[:, :3] = np.clip(m.mat_rgba[:, :3] * rng.uniform(0.88, 1.12, (m.nmat, 3)), 0, 1)
    # ---- camera extrinsics (calibration error): ZED 5 mm / 0.7 deg, wrist 2 mm / 0.5 deg
    for name, dp, dr in (("zed", 0.005, 0.7), ("wrist_d405", 0.002, 0.5)):
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if c < 0:
            continue
        m.cam_pos[c] += rng.normal(0, dp, 3)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, m.cam_quat[c], _small_rot(rng, dr))
        m.cam_quat[c] = q
    # ---- objects: pose jitter (1 cm, free yaw 20 deg), mass x0.6-1.6, friction x0.7-1.3
    for o in objs:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
        j = m.body_jntadr[b]
        if j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = m.jnt_qposadr[j]
            d.qpos[a:a + 2] += rng.uniform(-0.01, 0.01, 2)
            d.qpos[a + 2] += 0.002
            yaw = np.radians(rng.uniform(-20, 20))
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), d.qpos[a + 3:a + 7])
            d.qpos[a + 3:a + 7] = q
        s = rng.uniform(0.6, 1.6)
        m.body_mass[b] *= s
        m.body_inertia[b] *= s
        for g in np.where(m.geom_bodyid == b)[0]:
            m.geom_friction[g, 0] *= rng.uniform(0.7, 1.3)

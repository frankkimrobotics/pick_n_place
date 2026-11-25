# MuJoCo 4 environment (`mj4`)

MuJoCo 4.0.0 was released 2026-11-06. This is a **separate** conda env on purpose: the
existing `mjwarp` and `ilpolicy` envs have pinned torch/CUDA combinations that have broken
before when upgraded in place, so nothing here touches them.

## Create

```bash
conda create -n mj4 python=3.12 -y          # mujoco 4 requires >=3.11
conda activate mj4
pip install mujoco==4.0.0 numpy imageio imageio-ffmpeg opencv-python-headless
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mujoco-warp==4.0.0
pip install "warp-lang==1.15.0"             # see the pin note below -- NOT the latest
```

## Verified versions

| package | version |
|---|---|
| mujoco | 4.0.0 (+ `mujoco-core` 4.0.0, `mujoco-studio` 4.0.0) |
| mujoco-warp | 4.0.0 |
| warp-lang | **1.15.0** (pinned, see below) |
| torch | 2.5.1+cu121 |
| numpy | 2.5.4 |
| python | 3.12.16 |

MuJoCo 4 is **re-packaged**: the `mujoco` wheel is now pure Python (`py3-none-any`) and the
native code lives in `mujoco-core`. A `py3-none-any` wheel for a physics engine looks wrong
at first glance but is correct here — same publisher as 3.x (`Google DeepMind
<mujoco@deepmind.com>`).

## The warp-lang pin (do not "upgrade" this)

`warp-lang` 1.19.0 ships **CUDA Toolkit 13.4 and requires driver >= 13.0**. This machine has
driver 570.124.06 = **CUDA 12.8**, so 1.19.0 silently falls back to **CPU only** — it still
imports and runs, it just lists no `cuda:*` devices, which looks like a slow run rather than
a misconfiguration. 1.15.0 ships CUDA 12.9 and sees both GPUs. `mujoco-warp==4.0.0` only
requires `warp-lang>=1.15`, so the pin costs nothing.

## Compatibility with this project — measured, not assumed

- **Physics is bit-identical to 3.x.** Same scene, same initial state, 600 steps:
  `max |dqpos| = 0.0` exactly against mujoco 3.15.0 (`ilpolicy`) and 3.11.0 (`mjwarp`),
  identical contact counts and solver/integrator/cone settings. Script:
  `scratchpad/mj4_check/phys_sig.py`. This matters because the suction seal, crush guard and
  contact-force gating all depend on contact dynamics.
- **MJCF loads unchanged** — `rl/scenes/*.xml` and the generated `~/pnp_dataset_{mega,zed}/
  scene_*/scene.xml`, with all cameras preserved (`iso, front, top, zed, fixed_d435,
  wrist_d405`).
- **Full 3.x Python API surface is present**: `MjModel/MjData/MjvOption/Renderer`,
  `mj_name2id`, `mj_kinematics`, `mj_comPos`, `mj_forward`, `mj_jacSite`, `mju_mat2Quat`,
  `mju_quat2Vel`, `mj_contactForce`, `mj_step`, `mjtObj`, `mjtVisFlag.mjVIS_RANGEFINDER`,
  `mj_rnePostConstraint`. Nothing the project calls has been removed.
- **Offscreen rendering works** under `MUJOCO_GL=osmesa` for `wrist_d405` and `fixed_d435`.
- **GPU warp works** and matches CPU MuJoCo to `9.8e-05` over 20 steps (ordinary
  float32-vs-float64 difference, same as under 3.x).

### One cosmetic wart

`Renderer.__del__` raises a `ctypes.ArgumentError` from `mujoco/osmesa/__init__.py:113` at
interpreter shutdown, *after* rendering has succeeded. Harmless, but it prints a traceback
that looks like a crash in logs. Call `ren.close()` explicitly before exit to avoid it.

## Status

Nothing in the project has been migrated. The existing pipelines still run on `mjwarp`
(mujoco 3.11) and `ilpolicy` (mujoco 3.15) — see `MEMORY.md`. Given physics is bit-identical,
migration is low-risk, but the eval harness and RL code have not been run end-to-end here yet.

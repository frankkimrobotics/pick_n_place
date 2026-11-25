# AGENTS.md — how to work in pick_and_place

Read this before the README. The README (869 lines) is the project history; this is the map.

## What this is
Suction pick-and-place on a myCobot Pro 630 with a MuJoCo/mujoco_warp twin, cuRobo planning, visual
policies (DP/CFM/ACT over frozen vision encoders) and an RL contact controller. Two hosts: this desktop
and a Raspberry Pi running LinuxCNC + `robot_hal.py` over a direct cable (`cfg.pi.host`).

## Start here
    bin/pnp tasks -v          # every supported thing to run, with its interpreter, inputs, outputs
    bin/pnp env check         # are the environments present and importable
    bin/pnp config show --diff
    bin/pnp config validate   # runs automatically before anything that moves the robot

## Rules
1. **Never hard-code** a path, serial, host, port or gain. Read it from `pnp.config` (`configs/defaults.toml`
   for tunables, `configs/machine/<hostname>.toml` for this machine). Add new tunables there with a unit and
   a source comment; add them to `pnp/config/schema.py` if they have a sane range.
2. **Interpreter per task.** Five conda envs plus system Python; `tasks.toml` says which. `mj4` has no
   matplotlib; `curobo2`/`curobo3` are Python 3.10; `system` is 3.8 and owns the cameras and ROS 2.
3. **The Pi controller lives in `cfg.paths.mycobot_mpc`** (`~/Desktop/2026/mycobot_mpc`, September 2026).
   `~/Desktop/2026/frankkimrobotics/mycobot_mpc` is a stale June copy: do not edit it. The Pi's own copy may be
   newer than git; `pnp pi sync` is the only sanctioned way to change it. The repo copy still lacks the
   `torque` field and `seq` the host already parses.
4. **Do not edit `outputs/`** by hand: calibrations (`extrinsics_*.json`, `handeye_*.json`) are measured files.
5. **Live vs dead code.** Scripts at the repo root untouched since before August 2026 are archive candidates
   (72 of 116; `docs/restructure_plan.md` phase 1). Prefer `policy/`, `rl/`, `realsim/`, `dataset_gen_curobo.py`,
   `dataset_gen_zed.py`, `rim_seal.py`, `sim_robot_mjcf.py`, `config.py`, `geometry.py`.
6. **Eval hygiene.** `realsim/eval_policy.py`: at most 3 workers on GPU 0 (6.4 GB each), fresh `--out` per run
   (reusing one deletes the previous traces), run in `mj4`. Offline loss has repeatedly anti-correlated with
   closed-loop success: only closed-loop numbers count.
7. **Before any robot motion**: `bin/pnp config validate`; cameras up (`cam_server`); planner up; START pressed
   on the base after every `robot_hal` restart; stop = SIGINT `robot_hal` (never `online_servo`).
8. **Memory of past decisions** is in `docs/` and the README's dated sections; check `docs/restructure_plan.md`
   and `docs/runlog_design.md` before proposing structure or logging changes.

## Layout (phase 0; see docs/restructure_plan.md for the target)
    pnp/            config system, task registry, CLI            (new, importable, no heavy deps)
    configs/        defaults.toml, machine/<host>.toml, experiments/
    policy/         visual policies: features, training, sim rollout, real executor, camera server
    rl/             twin (env_warp), seal RL, DAgger, planner sweeps, PiLink (real_policy_ctrl.py)
    realsim/        measured-drive twin, plunger, validation, eval harness
    il/             episode datasets from bags
    docs/           designs (runlog, restructure), diagrams, reports
    outputs/        calibrations and measured artifacts (do not edit)
    *.py at root    116 scripts; the live ones are listed in tasks.toml, the rest is history

## Conventions
- Angles in radians in the URDF frame inside Python; the Pi speaks LinuxCNC degrees
  (`mycobot_mpc/joint_conventions.py` converts). `cfg.robot.base_q` is the start pose.
- Front of the robot is +x; table top at `cfg.task.table_z`.
- Checkpoints: `ckpt_final_ema.pt` is what evals load; intermediate `ckpt_NNk.pt` are 7 GB each.

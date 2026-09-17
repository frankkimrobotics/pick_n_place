# RL pick-and-place training (myCobot Pro 630, suction)

Batched GPU training on `env_warp.py` (mujoco_warp, N parallel worlds of one
compiled model). Suction = force/torque-capped compliant attachment with
peel + torsion limits; decisions at 10 Hz (Δq ≤ 2°/tick + suction logit).
Hard-won design rules live in [`FINDINGS.md`](FINDINGS.md); curated
checkpoints with results in [`weights/`](weights/README.md).

Environment: conda env with `mujoco`, `mujoco_warp`, `warp-lang`, `torch`
(referred to below as `$PY`, e.g. `~/miniconda3/envs/mjwarp/bin/python`).

## Drive-model probe (hardware replay)

```bash
$PY rl/drive_probe.py --drive real     # 10 deg step + 12 deg 0.5 Hz sine vs the 2026-09-17 hardware numbers
```

## Measured-drive attach bootstrap (behaviour cloning)

```bash
$PY rl/bc_bootstrap.py --drive real --out ~/pnp_rl/bc_attach_real/bc_init.pt   # scripted teacher -> actor (63.5 % seal+lift)
$PY rl/ppo.py --mode attach --dr --drive real --init ~/pnp_rl/bc_attach_real/bc_init.pt --out ~/pnp_rl/attach_real
```

Random exploration cannot discover the sustained press the real drive needs (FINDINGS);
clone the scripted descend-press-lift first, then let PPO refine.

## Paper setup: Arafat et al., "Efficient Parallel PPO-Based RL for Generalized Pick and Place
with Dense Reward Shaping" (IEEE QPAIN 2026, DOI 10.1109/QPAIN69676.2026.11545619)

`rl/env_paper.py` re-hosts the paper's task on this twin (same physics, suction and the
calibrated drive model):

| paper | here |
|---|---|
| SO-101 + gripper, one cube, Isaac Lab / PhysX 5 | Pro 630 + suction cup, one box, mujoco_warp (`--drive real` or `ideal`) |
| randomised object + goal poses | object xy/yaw on the table; goal = random 3-D point (xy within `--target_max`, z 5–25 cm) |
| obs `[q, q̇, p_obj, p_goal, a_prev]` | same (25-D; +6 drive-lag block under the real drive) |
| action `[a_arm, a_grip]` | joint-delta targets + suction logit |
| reward reach / lift / track (coarse + fine) + λ(t)·reg | same forms; weights are documented assumptions in the file (reach σ 0.25 m: our home pose is 45 cm from the table) |
| 5 s episodes, failure = root height | `--ep_len 100` decisions (10 s at our 10 Hz), failure = object off table |
| success = final object–goal distance | object lifted and within `--succ_tol` (3.5 cm) at the time limit |
| PPO [256,128,64] ELU, adaptive LR (KL 0.01), γ 0.98, 5 epochs, 4 minibatches, entropy 0.006, value clip | `--arch paper --kl_target 0.01 --gamma 0.98 --epochs 5 --minibatch 12288 --ent 0.006 --value_clip --vf_coef 1.0 --init_std 0.0` |

Note the paper has **no release phase**: "placement" is holding the lifted object at the goal.

Embodiment adaptation (`--grasp_shaping 1`, default): the paper's parallel gripper grasps a cube
by closing on it; our suction cup must be pressed 3 mm into the top at low speed with suction on.
Under the pure paper reward neither drive discovered a single seal in 2–4M steps (`paper_real`,
`paper_ideal`), so the env adds env_warp's dense press term and a one-time seal bonus.
`--grasp_shaping 0` gives the pure paper reward.

Observation extension (`--obs_ee 1`, default): tcp position, cup axis and the grasp-point-relative
vector. With the paper's `[q, q̇, p_obj, p_goal, a_prev]` alone (5-DoF SO-101 in the paper, 6-DoF
here) both drives plateaued with the cup ~10 cm from the object and never pressed (`paper2_*`).
`--obs_ee 0` gives the paper's observation.

```bash
$PY rl/env_paper.py --nworld 512 --steps 30 --drive real         # smoke test
$PY rl/ppo.py --env paper --arch paper --nworld 2048 --steps 8000000 --rollout 24 --epochs 5 \
    --minibatch 12288 --gamma 0.98 --lam 0.95 --clip 0.2 --ent 0.006 --lr 1e-4 --kl_target 0.01 \
    --vf_coef 1.0 --value_clip --init_std 0.0 --dr --drive real --scene rl/scenes/box_med.xml \
    --out ~/pnp_rl/paper_real
```

## Environment smoke test

```bash
$PY rl/env_warp.py --nworld 64 --steps 20
```

## Training

PPO is the workhorse (`rl/ppo.py`); SAC (`rl/sac.py`) is the historical
comparator (collapses when warm-started into shifted dynamics — see
FINDINGS). All runs write `log.jsonl` + periodic `ac.pt` + final `final.pt`
into `--out`.

```bash
# 1. ATTACH curriculum (from scratch; teaches seal + lift; ~6M steps)
$PY rl/ppo.py --nworld 4096 --steps 6000000 --mode attach --dr \
    --scene rl/scenes/box_med_ped.xml --out ~/pnp_rl/attach1

# 2. Full pick-and-place, warm-started from attach
$PY rl/ppo.py --nworld 4096 --steps 20000000 --mode pnp --dr \
    --init ~/pnp_rl/attach1/final.pt --target_max 0.3 --lift_req 0.35 \
    --speed_bonus 0.3 --scene rl/scenes/box_med_ped.xml --out ~/pnp_rl/pnp1

# 3. Mixed-mode curriculum (pnp/carry/place worlds 40/30/30 in ONE network;
#    prevents the catastrophic forgetting that killed sequential stages)
$PY rl/ppo.py --nworld 4096 --steps 30000000 --mode mix --dr \
    --release_mask --mask_h 0.008 --tilt_pen -0.15 --target_max 0.3 \
    --lift_req 0.35 --speed_bonus 0.3 --scene rl/scenes/box_med_ped.xml \
    --init ~/pnp_rl/attach1/final.pt --out ~/pnp_rl/mix1
```

Key flags: `--drive {real,ideal}` (measured Pro 630 velocity-drive model — the
default since 2026-09-17 — vs the legacy stiff PD; see FINDINGS "Dynamics update"),
`--dq_max DEG` (per-decision joint delta, 2 = 20 °/s), `--obs_lag {0,1}` (append
`q_target − q`; auto-on for the real drive), `--mode {attach,pnp,place,carry,mix}` (curriculum stage),
`--dr` (gain/seal/delay/obs randomization), `--release_mask --mask_h H`
(suction release held while sealed > H above the target surface — a
skill-forcing trainer aid; per-world in mix mode), `--tilt_pen W` (dense
airborne tilt cost), `--init CKPT` (warm start; obs growth is zero-padded).

## Monitoring & evaluation

```bash
# 6-panel training dashboard (success/seal, return, per-component rewards)
$PY rl/plot_training.py ~/pnp_rl/pnp1

# standardized benchmark: criteria ladder V1/V2/V3 on the CURRENT physics,
# deterministic, across the era checkpoints listed in the script
$PY rl/eval_bench.py --episodes 256

# rollout video (state policy)
$PY rl/demo_video.py --actor ~/pnp_rl/pnp1/final.pt --algo ppo --mode pnp \
    --scene rl/scenes/box_med_ped.xml --out /tmp/demo

# sequential multi-object clearing demo (scripted reach + learned pnp)
$PY rl/multi_demo.py --actor rl/weights/ppo5_workspace.pt
```

Read episode results from the `info` dict fields (`final_d`, `max_lift`,
`release_h`, `max_tilt`, `final_spd`) — never from post-step env state
(auto-reset zeroes it).

## Vision distillation (DAgger)

State-policy teacher → dual-RGBD student (wrist D405 + fixed D435, 96×96,
no privileged object state). Renders on GPU via EGL (~630 fps; multiprocess
CPU pool fallback). Visual DR (object color / lighting / camera jitter)
re-rolled per episode.

```bash
$PY rl/distill.py --teacher rl/weights/ppo7_ped_teacher.pt \
    --nworld 128 --iters 4 --steps_per_iter 60000 --epochs 6 \
    --out ~/pnp_rl/distill1
```

## Criteria ladder (what "success" means)

| Tier | Requirement |
|---|---|
| V1 | object at rest ≤ 3.5 cm from target |
| V2 | V1 + max lift ≥ 0.30 m + landing speed < 8 cm/s |
| V3 | V2 + release height < 1 cm (contact release) + object tilt < 25° |

Status: V1/V2 solved by the ppo4→ppo7 line (see `weights/README.md`);
V3 remains open — all component skills train (see place-curriculum results
in FINDINGS) but no checkpoint assembles the full V3 task yet.

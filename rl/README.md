# RL pick-and-place training (myCobot Pro 630, suction)

Batched GPU training on `env_warp.py` (mujoco_warp, N parallel worlds of one
compiled model). Suction = force/torque-capped compliant attachment with
peel + torsion limits; decisions at 10 Hz (Δq ≤ 2°/tick + suction logit).
Hard-won design rules live in [`FINDINGS.md`](FINDINGS.md); curated
checkpoints with results in [`weights/`](weights/README.md).

Environment: conda env with `mujoco`, `mujoco_warp`, `warp-lang`, `torch`
(referred to below as `$PY`, e.g. `~/miniconda3/envs/mjwarp/bin/python`).

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

Key flags: `--mode {attach,pnp,place,carry,mix}` (curriculum stage),
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

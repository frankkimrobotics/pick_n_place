# SAC pick-and-place: batched-sim RL design

Why RL here: the v1 BC program proved the ceiling is data coverage
(perfect-only demos, no recovery) and plan churn — not fit quality
(12×-MSE-range, flat behavior). RL optimizes success directly and
manufactures its own corrective experience. The catch: pixel-SAC from
scratch is sample-hungry and brittle; the robust two-stage recipe is
**privileged-state SAC (batched) → vision distillation (DAgger)**, which
also produces exactly the recovery-rich visual data BC lacked.

## Stage A — state-based SAC on mujoco_warp (batched)

**Why mujoco_warp over Isaac**: physics, suction weld logic, MJCF scene,
and controllers already exist and are validated here; mjwarp batches
identical models on GPU. Isaac Lab's tiled cameras only matter for
pixel-RL, which Stage A avoids (Stage B renders unbatched).

**Batching constraint & DR**: mjwarp replicates ONE compiled model across
N worlds — geom sizes are model-level. Solution: fixed layout (1 object +
table + bin per world), sizes/colors resampled by **recompiling the model
every rollout phase** (~seconds, amortized over ~100k env-steps), object
pose randomized per-world per-reset via qpos. N_env = 4096.

**Episode** (matches the BC single-pick structure): one object, one goal;
reset with random object pose/dims + random arm start near home;
10 Hz decisions, 150-step cap (15 s), terminate on place / off-table /
timeout.

**Observation (privileged, ~40-D)**: q(6) qd(6) tcp pos+rot6d(9)
obj pos+rot6d(9) obj half-extents(3) rel goal grasp(9)
suction(1) cup contact force(1) — all computed on-GPU, no rendering.

**Action (7-D @ 10 Hz)**: Δtcp position (3, ±2 cm clamp) + Δyaw (1) +
z-approach scale (1, lets policy modulate press depth) + suction logit
(1) + gripper-frame lateral micro-correction (1); mapped to joints via
damped-least-squares IK on GPU, tracked by the existing PD. (Joint-space
Δq(6)+suction is the fallback if GPU-IK is a bottleneck.)

**Suction physics fixes (required before RL — policies exploit sims)**:
compliant latch (ramp weld solref over 50 ms; kills the ±250°/s jolt),
force-limited release (break weld when constraint tension > ~20 N),
seal check unchanged (force + tilt gates).

**Reward** (dense shaping, weights to tune):
  r = 1.0·Δ(−‖tcp − grasp_point‖)          # approach progress
    + 0.3·axis_alignment(cup, obj_top)
    + 5.0·[seal event]                      # one-time bonus
    + 2.0·clip(lift_height/0.1, 0, 1)       # while sealed
    + 1.0·Δ(−‖obj − bin‖) while sealed
    + 10.0·[placed in bin]                  # terminal
    − 0.01·‖a‖² − 0.1·[dropped early]

**SAC config**: twin Q + target nets, auto-entropy (target −dim(A)),
LayerNorm critics, UTD 4-8 (RLPD-style), batch 4096 from a 2M-transition
GPU replay buffer, γ=0.99, lr 3e-4, MLP 3×512.
**Demo seeding (RLPD)**: v1 demos lack per-frame object poses, so either
(a) regenerate demos with full qpos logging (already a v2-data item) and
mix 50/50 demo/online in replay, or (b) run pure online — with this
shaping and 4096 envs, single-object pick typically converges in 3–10M
env-steps ≈ minutes-to-an-hour of wall time on one GPU.

**Throughput estimate**: mjwarp ~1M env-steps/min at 4096 envs on H100
(small scene, 10 ms physics per 10 Hz decision = 10 substeps); SAC
updates dominate at high UTD → expect 30–90 min to a competent
state-based picker. Runs on the local A5000 at ~1/4 speed.

## Stage B — vision distillation (teacher → cameras)

DAgger: roll the STUDENT (two RGBD + proprio + goal — the exact BC
architecture, CLIP-ViT context + DiT or plain regression head) in the
standard non-batched renderer env; supervise every visited state with
the teacher's action; aggregate and iterate. 50–200k labeled frames.

Why this beats both BC-v1 and pixel-SAC:
- coverage: student visits its OWN mistakes; teacher labels the fix —
  the recovery data BC never had, generated automatically;
- cameras only render at distillation scale (unbatched is fine);
- output is drop-in compatible with rollout_sim / real_rollout /
  the TRT export path — directly comparable to DP/CFM v1;
- depth obs comes free (RGBD requested): concat D405/D435 depth as
  extra channels in the student encoder.

## Deliverables / build order

1. `rl/env_warp.py` — batched single-pick env (model builder + reset/
   step/reward on GPU) + suction fixes ported into the MJCF.
2. `rl/sac.py` — SAC + replay (torch, single file, cleanrl-style).
3. Stage-A training run + success-rate curve (target ≥90% sim pick-place).
4. `rl/distill.py` — DAgger loop reusing train_lit's encoder/context.
5. Evaluate student via rollout_sim (same scene-609 protocol + rate
   sweep) vs v1 scorecard; then real-robot via real_rollout unchanged.

## Honest risk register

- Reward hacking around the seal (e.g., pinning object against bin wall)
  → keep the physical seal gates + audit rollouts visually.
- IK-action singularities near workspace edges → clamp + fallback to Δq.
- mjwarp recompile-per-phase DR is coarser than per-episode DR — accept,
  or hold K precompiled variants round-robin.
- Distillation gap (student plateau below teacher) → standard; DAgger
  iterations + depth channel usually close most of it.
- Sim-exploited contact quirks transferring badly → the compliant-latch
  + force-limit fixes are load-bearing, do them first.

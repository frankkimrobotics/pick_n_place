# Imitation learning v2: cuRobo expert in mujoco_warp → BC / Diffusion Policy / CFM / ACT

Date: 2026-09-20. Status: PLAN (nothing below is built yet unless marked EXISTS).

Goal: a vision policy for **pick → fast, obstacle-aware return to the bin → release**, trained
entirely in the mujoco_warp twin from a cuRobo expert, with diverse objects (colour, size,
primitive shape, position), a cell with walls (back wall x = −0.30 m, side walls y = ±0.50 m),
and distractor objects the return path must avoid. Deployable through the same chunk welder
that ran the RL demos (`rl/real_policy_ctrl.py`, Pi `robot_hal` stream mode).

Why v2 (lessons already paid for, see `rl/FINDINGS.md`, memory notes):
- v1 (Aug 2026) trained DP/CFM/ACT on 600 kinematic scenes: offline chunk MSE improved 12×
  while closed-loop success stayed 1/5 → the data lacked recovery/contact coverage and used
  an idealised drive. v2 collects **on the measured drive model**, adds **perturbation +
  relabel (DAgger) rounds**, and evaluates closed-loop in batch from day one.
- The RL/DAgger work showed the measured drive (dead-time 45 ms, 50 °/s cap, 800 °/s²) is
  faithful (real touch at 5.1 s = sim) and that the Pi welder tracks a 10 Hz reference to ≤2°.
- The 10 Hz per-step reference is what looks jerky on the robot; the policy here outputs
  **1.5 s B-spline chunks** (16 control points, EXISTS in `policy/convert_litdata.py`) that the
  welder can follow C² (TODO_BSPLINE in mycobot_mpc).

---------------------------------------------------------------------------------------------
## 0. Task and cell specification

| item | spec |
|---|---|
| start | arm at `config.START_Q` (± 2° noise), suction off |
| target object | one per episode, on the table, random primitive (below) |
| distractors | 0–3 objects that must NOT be touched; ≥1 of them tall (0.10–0.25 m) placed on the straight-line corridor between the pick point and the bin, so the return has to go over/around |
| bin | existing 0.30 × 0.30 × 0.30 m bin at (0.10, 0.40) (walls 0.30 m) |
| walls | back wall: plane at **x = −0.30** (full height); side walls: planes at **y = ±0.50**; both as MuJoCo geoms (collision on) AND cuRobo cuboids |
| object spawn region | x ∈ [0.22, 0.55], y ∈ [−0.35, 0.35] (table enlarged to 0.7 × 0.8 m, top at z = 0), rejection-sampled: ≥ 0.05 m from other objects, outside the bin footprint, inside the walls with 0.10 m margin; reachability check by cuRobo IK of the grasp pose (top-down, cup radius 8 mm above the top, +90° yaw) |
| success | object released inside the bin (xy within the floor, resting), no distractor displaced > 1 cm, no wall/camera-mount contact, episode ≤ 8 s |
| speed target | expert return retimed to the measured limits: 50 °/s, 800 °/s² per joint; pick + return ≤ 6 s (RL touch = 5 s today) |

Object primitives (all with a flat, suction-able top; target only from the "flat-top" set):
- box: half-sizes U(0.015, 0.045)³, top area ≥ 4× cup area
- cylinder (upright): r U(0.018, 0.045), half-height U(0.012, 0.05)
- hexagonal / octagonal prism (mesh baked once per size class, reuse STLs — parallel workers must not re-bake)
- distractor-only shapes: sphere r U(0.02, 0.05), capsule lying flat, tall post 0.03 × 0.03 × (0.10–0.25)
- colour: HSV random (S ≥ 0.4, V ≥ 0.3) per object; table/bin/wall albedo jittered ±20 %; light position from 3 presets + jitter (EXISTS in `rl/distill.py` DR params)
- mass 0.03–0.15 kg, friction 0.6–1.0

---------------------------------------------------------------------------------------------
## 1. Scene generation (mujoco_warp constraint: geometry is model-level)

mujoco_warp batches worlds over one model: positions/orientations/colours can differ per
world (qpos, mocap, rgba are per-world or render-only), but **geom sizes and body counts
cannot**. Use a **scene bank**:

- `dataset_gen.sample_scene` → extend to `sample_scene_v2(seed)` emitting: target spec,
  0–3 distractor specs, wall/table geoms, camera DR params. Build with
  `sim_robot_mjcf.build(...)` (EXISTS; add `walls=True`, `table_dims`, prism meshes).
- Bank of **B = 512 models** (distinct shape/size combos). Each model runs as one
  `PickEnv`-style warp env with **W = 128 worlds** that randomise per world: target xy/yaw,
  distractor xy/yaw (rejection-sampled so a tall one blocks the corridor in ≥ 50 % of
  worlds), colours (render-side), light, camera jitter, drive DR (EXISTS: `DRIVE_DR`).
- Bodies per model: `object0` (target) + `dist0..2` (unused distractors parked under the
  table at z = −1 so body count is constant).
- Cameras: `wrist_d405` (fovy 58, hand-eye), `fixed_d435` (0.66, 0, 0.60) — EXISTS in
  `sim_robot_mjcf.py`. The fixed camera mount is a cuRobo obstacle already.

---------------------------------------------------------------------------------------------
## 2. Expert: cuRobo v3 planner (batched) + scripted press/seal + fast retime

Today's server (`curobo_planner_server_v2.py`, :9997) plans one world at a time under a
global lock — it was the DAgger bottleneck. v3:

1. **Batched planning**: `MotionGen.plan_batch_env` with per-env `WorldConfig` (cuboids:
   table, bin walls, back/side walls, camera mount, every distractor as a cuboid/its OBB,
   the target while approaching). Batch = 32–64 worlds per call; `collision_cache
   {"primitive": 32}`. New RPC `plan_batch` {starts[N,6], goals[N] (pose or joint), worlds[N]}
   → trajectories + status. Keep v2 RPCs for single calls.
2. **Phases per episode** (all in joint space, +90° tool yaw, elbow box
   `position_limit_clip` EXISTS):
   a. hover: plan_pose to grasp point + 0.06 m
   b. press: straight-line IK descent to grasp point − 0.02 m at 0.6 °/decision equivalent
      (EXISTS as `bc_curobo.Teacher` press; 20 mm depth was the fix that made the measured
      drive seal 91 %), suction ON at hover exit
   c. seal check (contact force > 2 N, normal within 25°, EXISTS) → `attach` the object as
      spheres (EXISTS RPC) and lift 0.06 m
   d. **return**: plan_pose to bin drop pose (bin centre, z = 0.30 + object height + 0.05)
      with the attached object and all distractors in the world → **retime with TOPP-RA
      to vlim 50 °/s, alim 800 °/s²** (EXISTS in `curobo_pick_traj.py` for vlim 0.6; raise to
      the measured ceiling) — this is the "fast return"
   e. release over the bin (suction OFF), short retreat to a park pose above the bin
3. **Execution in warp**: the expert reference is streamed through the measured drive
   model (`env_warp._step_real_drive`, EXISTS) at 10 Hz decisions exactly as the robot will
   see it, so recorded proprio/images carry realistic lag. Failed episodes (no seal, distractor
   moved, wall contact, timeout) are dropped from the clean set but kept, labelled, for §4.
4. Throughput estimate: cuRobo batch of 64 ≈ 0.5–1.5 s per phase on the A5000; 5 phases →
   ~6 s per 64 episodes ⇒ ~40 k episodes/hour planning-bound. Rendering (below) is the real
   limit.

---------------------------------------------------------------------------------------------
## 3. Observations, actions, dataset

- **Obs** (10 Hz, To = 2 steps): RGB **and depth** from both cameras at 96×96 (train_lit uses
  RGB 432×240 with CLIP tokens; v2 default = raw 4-channel 96×96 conv towers as in
  `rl/distill.py` Student, ablation = CLIP tokens), proprio [q, qd, suction, last-chunk phase],
  **no object state, no goal token** (the bin is fixed; the policy must find the target
  visually — that is what the colour/shape diversity is for). Optional privileged variant
  (object pose in obs) as a fast ablation to separate perception from control failures.
- **Action**: 16 B-spline control points over 1.5 s in joint space + 16 suction bits
  (EXISTS: `convert_litdata.py`, `dataset_spec.json`); executor overlap 1.0 s (re-plan every
  0.5–1.0 s — the v1 rate sweep found 1 Hz best, 5 Hz+ dithers).
- **Rendering** (the bottleneck; mujoco_warp does not render): EGL render farm from
  `rl/distill.py` (~630 render-units/s ≈ 300 image-pairs/s at 96×96). 100 k episodes × 6 s ×
  10 Hz × 2 cams = 12 M images ≈ 11 h on one box; run 2 GPUs (A5000 + 2080 Ti) or render at
  5 Hz for the first pass. Store JPEG + uint16 depth in litdata shards (EXISTS format).
- **Dataset size targets**: clean expert 60 k episodes; perturbation/recovery 40 k (below);
  val = held-out 5 % of the scene bank (unseen shape/size combos), plus a held-out colour
  range.

---------------------------------------------------------------------------------------------
## 4. Recovery data (the v1 gap) — perturbation + DAgger relabel

1. **Perturbed expert rollouts** (40 % of data): at random times inject a joint offset
   (≤ 4°), a 0.1–0.3 s hold, or a suction dropout; re-plan from the perturbed state with the
   expert and continue. Records how to recover, not just how to succeed.
2. **DAgger rounds** (after the first policy): roll the student in warp (batched, with
   rendering), relabel every visited state with the cuRobo expert (batched `plan_batch`
   from the student's state), β = 0.5 → 0.3 → 0.1 → 0 mixing, 3–4 rounds. This is the recipe
   that took the measured-drive RL policy from 0 → 88 % (`dagger5/6_real`), and relabelling
   from student states is what fixed v1's "perfect approaches only" defect.
3. Keep failed-seal episodes with the expert's retry (lift 2 cm, re-press) as demonstrations
   of retry behaviour.

---------------------------------------------------------------------------------------------
## 5. Training (four heads, one trainer: `policy/train_lit.py`, EXISTS)

| head | changes vs v1 | schedule |
|---|---|---|
| BC (MSE on control points) | new `--model bc` = DiT context + MLP head, deterministic | 100 k steps, baseline |
| Diffusion Policy | DDPM 50 → DDIM 16 at inference (EXISTS) | 200 k |
| CFM | as v1 (`x1 − x0` MSE), add V2_TRAINING_PLAN's boundary + smoothness losses | 200 k |
| ACT | CVAE, L1 + 10·KL (EXISTS), chunk 16 | 100 k |

Common: batch 256, lr 3e-4, EMA, To = 2, DR at the image level again during training
(colour jitter, crop), **early stopping on closed-loop success in warp, not on MSE**
(v1 lesson: MSE < 0.002 is decoupled from success). Add `--obs raw96` path (4-ch conv towers)
next to the CLIP path. Compute: A5000 locally ~6 it/s (UNet) — 200 k steps ≈ 10 h per head;
or the Lightning H100 studio (EXISTS, needs balance).

---------------------------------------------------------------------------------------------
## 6. Evaluation

- **Batched closed-loop in warp with rendering**: 512 episodes per checkpoint on the held-out
  bank, measured drive + DR, executor = 1.5 s chunks with 1 s overlap and the C² splice
  bridge (EXISTS in `policy/rollout_sim.py`; port to the warp env as `rl/eval_il.py`).
  Report: seal %, binned %, distractor-touch %, wall-contact %, mean pick + return time,
  chunk-splice velocity jump.
- Ablations: privileged-state vs vision; RGB vs RGB-D; with/without recovery data; with/without
  DAgger rounds; return retime 30 vs 50 °/s.
- **Real robot** (touch-only rule until you lift it): the same executor streams chunks to
  `robot_hal`; first tests with `--touch_only` (no suction) on the D435-scanned scene, then
  suction picks into the bin.

---------------------------------------------------------------------------------------------
## 7. Work breakdown (order matters)

| # | deliverable | est. |
|---|---|---|
| 1 | `sim_robot_mjcf`: walls, big table, prism meshes, distractor bodies; `sample_scene_v2`; scene bank builder (512 models) | 1 day |
| 2 | `env_warp`: multi-body worlds (target + 3 distractors, parked when unused), distractor-displacement + wall-contact detectors in `info`, bin-placement success | 1 day |
| 3 | cuRobo v3 server: `plan_batch` with per-env worlds, TOPP-RA retime to 50 °/s / 800 °/s², attach on seal | 1–2 days |
| 4 | expert episode generator in warp (phases a–e, measured drive, perturbations) + litdata writer with RGB-D | 1–2 days |
| 5 | render farm run: 100 k episodes | ~1 day of GPU |
| 6 | train BC / DP / CFM / ACT (`--obs raw96`), closed-loop eval `rl/eval_il.py` | 2 days + GPU |
| 7 | DAgger rounds ×3 with batched relabel | 1 day + GPU |
| 8 | real-robot: chunk executor over robot_hal (spline welder = TODO_BSPLINE steps 1–4), touch-only demos, then picks | 1–2 days |

Risks: (i) render throughput — mitigate with 5 Hz images or two GPUs; (ii) cuRobo batch
planning with attached objects may reject corridor-blocked returns — fall back to a two-
segment plan via a fixed 0.35 m "carry height" waypoint; (iii) per-world size diversity is
bank-level only — 512 models × 128 worlds is enough variety for the shape classes above;
(iv) mujoco_warp contact with thin wall planes — use 0.02 m-thick boxes, not planes.

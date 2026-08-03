# Pick-and-Place Imitation Learning — Plan (2026-07)

Goal: collect ~100 pick-and-place episodes (3–5 random objects) as **rosbags**, build a MuJoCo sim that
runs the **same control stack** to also produce synthetic data, and train a policy with **four swappable
model heads**: UMI, Diffusion Policy, Conditional Flow Matching, and a "drift" generative model.

---

## 0. What already exists (reuse — don't rebuild)

| Need | Already there | File |
|---|---|---|
| Real control stack | cuRobo planner `:9997` (pose→joint traj), welder/servo `:9994`, feedback `:9999` @50 Hz | `curobo_planner_server_v2.py`, `online_servo.py` |
| ROS2 bridge | `/joint_states`, `/mycobot/drive_feedback` (pos+vel+torque), `/mycobot/cmd/move`, weld path `/planner/weld_chunks`→`chunk_to_pi`→`:9994` | `mycobot_ros2_bridge.py`, `chunk_to_pi.py` |
| Cameras (ROS) | `/d405/color`+`/d405/aligned_depth_to_color` (wrist, ser 218622271300), `/d435/*` (fixed) @30 Hz | `realsense_rgbd_node.py` |
| **Rosbag recorder** | per-episode `.mcap`: color(jpeg)+depth(16UC1)+camera_info+joint_states+joint_vel+joint_cmd+phase | `episode_bag.py` (+ `servo_touch.py --bag`) |
| Autonomous real p&p | cuRobo-planned pick→carry→place + suction | `real_pipeline.py`, `real_multi.py`, `real_grasp.py`, `real_place.py` |
| **MuJoCo sim p&p** | full pick→place via the SAME `:9997` + 10 Hz weld chunks; articulated `robot_sim.xml` (6 joints, actuators, freejoint objects, rangefinder, iso/front/top cams) | `sim_pick_rangefinder.py`, `sim_mujoco_node.py` |
| Success scorer | rule-based (detect/reach/seal/collision/placed) | `evaluate.py` |
| Conventions | pose `[x,y,z,qw,qx,qy,qz]` (wxyz, m) at planner; joint weld-chunk actions (URDF rad); suction = binary; `ee_link=tcp` 0.135 m; 50 Hz | `config.py`, `mycobot_pro_630_v2.yml` |

**Absent (we add):** any learned policy / training code; a zarr/LeRobot dataset layer; a checkpoint→action
rollout harness; contact-dynamics + camera obs in sim.

**Gaps to close before collecting (from the surveys):**
1. **Suction not recorded on hardware** (it's an SSH GPIO toggle, not a topic) → publish `/mycobot/suction` + add to `episode_bag._TOPICS`.
2. **No TCP/EE-pose topic** → add an FK publisher (`:9997 type:"fk"` or URDF FK) so Cartesian obs/action can be logged.
3. **D405 device contention** — ROS `realsense_rgbd_node` vs the pipeline's direct grabber can't both own the D405; `episode_bag` uses the direct grabber, so collect with the D405 ROS node OFF.
4. **D435 not in the bag** — add `/d435/*` if we want the third-person view.
5. Bag velocity = `velfb` @50 Hz (set Pi `STREAM_RATE_HZ=100` if we want 100 Hz).

---

## 1. Observation / Action spec (Phase 0) — DECISIONS LOCKED (2026-07)

> **Action space = JOINT-SPACE (direct)** · **Obs vision = wrist D405 RGB + fixed D435 RGB** ·
> **Drift = stochastic-interpolant drift field** · **Dataset = LeRobot**

- **Observation** (history `To=2`):
  - `wrist_rgb`: D405 color 224×224 (RGB).
  - `fixed_rgb`: D435 color 224×224 (RGB).
  - `proprio`: **joint positions (6, URDF rad) + suction (1)** = **7-d** (also log TCP pose for eval/analysis).
- **Action** (predict `Tp=16`, execute `Ta=8`): **joint targets** `q(6)` + `suction(1)` = **7-d**.
  - `diffusion`/`cfm`/`drift` presets → **absolute joints**; `umi` preset → **Δjoints relative to current q** (UMI-style).
  - At inference the joint chunk is streamed straight to the welder `/planner/weld_chunks → :9994` — **no IK**.
- **Rates:** log obs+action at **10 Hz**; the executed joint chunk is welded/resampled to the 50 Hz servo.
- No depth, no EE-pose action. TCP pose still logged as auxiliary proprio for eval.

**Consequence:** simpler inference (policy joints → weld chunk), and sim↔real transfer holds because the sim
uses the same MyCobot MJCF + the same joint weld-chunk control. "UMI" vs "Diffusion Policy" now differ only
in action representation (relative vs absolute joints), so they share the diffusion head.

---

## 2. Phase 1 — Real data collection (~100 episodes)

- [ ] **Suction publisher**: small node/tap publishing `/mycobot/suction` (Bool) whenever the pin toggles; add to `episode_bag._TOPICS` and to the `real_*` executors.
- [ ] **EE-pose publisher**: FK from `/joint_states` (URDF FK or planner `fk`) → `/mycobot/tcp_pose` (PoseStamped); add to the bag.
- [ ] **Episode driver** = extend `real_multi.py`: randomize **3–5 objects** on the table (or hand-place per episode with a randomized target bin slot), run pick→carry→place per object, write **one bag per episode** with `/phase` markers (`start / reach_obj_i / grasp / lift / place / release / end / success|fail`). Wrap `episode_bag` (D405 direct grabber; D405 ROS node off).
- [ ] **Auto-reset / randomization**: object identity (3–5 of a fixed set), XY pose + yaw, and target slot; log the seed. Human resets clutter between episodes (prompted).
- [ ] **Collect ~100 episodes**, tagged with success from `evaluate.py`; keep failures (labeled) for diagnostics.
- [ ] **Converter** `bags_to_dataset.py`: rosbag(mcap) → chosen dataset format (§8 Q4). Sync topics to 10 Hz, compute TCP-relative actions from the future TCP-pose trajectory + suction, resize images, write normalization stats.

**Data schema (per timestep):** `{ wrist_rgb, [depth], [d435_rgb], tcp_pose(10 proprio), action(10 rel), phase, success, episode_id, t }`.

---

## 3. Phase 2 — Sim-aligned p&p + synthetic capture

Build on `sim_pick_rangefinder.py` + `sim_mujoco_node.py` (same `:9997` + weld-chunk protocol as real → the
control stack is *identical*; only the executor endpoint differs). Upgrades:
- [ ] **Physics**: switch `sim_mujoco_node` from kinematic `mj_forward` playback to **`mj_step`** with the position actuators tracking the welded reference (contacts on), so grasps can slip/fail — realistic data.
- [ ] **Physical suction**: replace the rigid qpos-slave attach with a **MuJoCo `equality/weld` (or connect) constraint** activated when the cup is sealed (tip within tol + normal aligned + `mj_step` contact), released on suction-off. Gives grasp-failure modes the policy must learn.
- [ ] **Wrist camera obs**: add a MuJoCo camera at the `suction_cup`/`tcp` site matching D405 intrinsics + `T_TCP_CAM`; render RGB(+depth) each obs tick → **same schema as real**.
- [ ] **Domain randomization**: object set/pose/mass/friction/color, lighting, table texture, camera jitter, small proprio noise — for sim→real.
- [ ] **Synthetic episode driver**: wrap `sim_pick_rangefinder.py` with the SAME obs/action logger as Phase 1 → sim episodes in the SAME dataset format. Can generate **hundreds cheaply** (headless osmesa).
- [ ] **Sim-real alignment checks**: same TCP frame, same action rep, same control rates; verify sim wrist-cam vs real D405 FoV/scale; verify weld/latency parity via `sim_hal_node.py` (adds the real dead-time).

---

## 4. Phase 3 — Dataset layer

- [ ] `dataset/` package: unified loader over **real + sim** episodes (chosen format §8 Q4). Temporal windows (`To` obs, `Tp` action), per-modality normalization (image → imagenet or [-1,1]; proprio/action → dataset mean/std), rot6d handling, train/val split by episode, sim/real balancing + a `domain` flag.
- [ ] Action computation: TCP-relative deltas from the logged future TCP trajectory (with suction), horizon `Tp`.

---

## 5. Phase 4 — Four pluggable policy heads

Common module `policy/`:
```
ObsEncoder   : wrist_rgb (ResNet18/ViT + FiLM(t)) + proprio MLP → cond vector (+ obs history)
ActionHead   : swappable; consumes cond, produces an action chunk (Tp × 10)
Policy       : encode → head; .loss(obs, action_gt), .predict(obs) → chunk
```
`--policy {diffusion, umi, cfm, drift}`, `--config policy/configs/<name>.yaml`. Shared: obs history, action
chunking (Tp/Ta), EMA weights, receding-horizon inference, normalization.

- **(a) Diffusion Policy** — DDPM train / DDIM sample, conditional U-Net (1-D over action horizon) or transformer; the reference implementation.
- **(b) UMI** — *same DP head* but the **UMI representation**: wrist-cam obs + **relative-EE-pose trajectory actions** (already our default action space) + latency-matched inference. In practice a DP config with the relative-traj action space and UMI's obs/latency handling.
- **(c) Conditional Flow Matching** — learn velocity field `v_θ(a_τ, τ, cond)` (linear/OT interpolant), train with the CFM loss, sample by integrating the ODE in a few steps (π0-style action expert; faster than diffusion).
- **(d) Drift model (Yilun Du)** — a generative action head in his framework (energy/score "drift" field, iterative refinement). **Exact method pending §8 Q3.** Slots behind the same `ActionHead` interface.

Same interface ⇒ identical training/eval loop across all four for clean ablation.

---

## 6. Phase 5 — Training

- [ ] `train.py`: config-driven, one loop for all heads; AdamW + cosine, EMA, mixed precision, image aug, checkpoints + normalization stats saved with the model; W&B/CSV logging; train on **real+sim combined** (with a sim-only / real-only / mixed sweep).
- [ ] New conda env (torch) for the policy — reuse one of the existing empty diffusion/flow envs or a fresh `ilpolicy` env (keep separate from `curobo`).

---

## 7. Phase 6 — Inference / rollout harness (close the loop)

The trained policy **becomes the planner node** in the current architecture (see `current_architecture.html`):
- [ ] `rollout.py`: read obs (D405 + `/mycobot/tcp_pose` + suction) at 10 Hz → policy predicts a relative-EE-pose action chunk → convert to absolute TCP poses → **cuRobo IK/plan (`:9997`) → weld chunk → `/planner/weld_chunks` → `:9994`** (reuse `online_planner_node.py`'s welder/streaming). Suction bit → GPIO. Receding horizon (execute `Ta`, re-infer).
- [ ] **Sim rollout first** (same harness, endpoint = `sim_mujoco_node`) → score with `evaluate.py`; then real.
- [ ] Latency: the ~192 ms drive dead-time still applies — action chunks give lookahead; feedforward `--lead` compensates predictable tracking.

---

## 8. Decisions I need (see the question prompt)

1. **Action space** — relative EE-pose (rec.) vs absolute EE-pose vs joint-space.
2. **Observation** — wrist-only RGB (rec.) vs +depth vs +fixed D435.
3. **The "drift model from Yilun Du"** — exact paper/arXiv.
4. **Dataset format** — LeRobot (rec.) vs zarr (UMI-native) vs custom.

## 8b. BUILD STATUS (2026-07-05) — all modules under `pick_and_place/il/`
| module | what | status |
|---|---|---|
| `policy/{nets,heads,policy}.py` | 4 models (diffusion/umi/cfm/drift) on 1 interface | ✅ smoke-tested |
| `data/{schema,episode_writer,dataset,make_synthetic}.py` | canonical format + windowed loader + rep-aware norm | ✅ tested |
| `train.py` | one loop for all 4 presets | ✅ all 4 learn end-to-end on synthetic |
| `sim/capture_cameras.py` | inject wrist(D405)+fixed(D435) cams from real calibrations | ✅ render verified |
| `sim/capture_logger.py` + `sim_mujoco_node.py --capture` + `sim_pick_rangefinder.py --episodes` | in-process dual-cam capture, object randomization, /phase episodes | ⚙️ compiles+imports; **needs live ROS2+planner run** |
| `rollout.py` | policy→joint weld-chunk→:9994, receding horizon | ⚙️ compiles; **needs robot/ckpt** |
| `data/bags_to_dataset.py` | rosbag(mcap)→canonical episodes | ⚙️ compiles; needs `rosbags` + real bags |
| `data/to_lerobot.py` | canonical→LeRobot export | ⚙️ compiles; needs `pip install lerobot` |

Run: **sim collect** = terminal A `python3 sim_mujoco_node.py --capture outputs/il_episodes --no-video` +
B `curobo_planner_server_v2.py` (:9997) + C `python3 sim_pick_rangefinder.py --episodes 100 --capture`.
**train** = `python3 il/train.py --policy umi --data outputs/il_episodes`. **env** = torch+cv2 (dust3r works).

## 9. Risks / notes
- Kinematic→`mj_step` sim change is the biggest sim task (contacts, suction constraint tuning).
- 100 real episodes is small for vision policies — sim data + augmentation + wrist-only obs (sample-efficient) mitigate; expect sim-heavy training with real fine-tune.
- D405 contention: pick ONE owner (direct grab for collection).
- Suction is binary + slow; grasp timing labels (`/phase`) matter for the action target.
- The four heads share everything except the action head → keep that boundary clean.

## 10. Suggested new layout
```
pick_and_place/il/
  data/    bags_to_dataset.py, dataset.py, normalize.py
  policy/  encoder.py, heads/{diffusion,cfm,drift}.py, policy.py, configs/*.yaml
  sim/     (extends sim_mujoco_node: mj_step, suction weld, wrist cam, randomize)
  train.py  rollout.py  eval_policy.py
```

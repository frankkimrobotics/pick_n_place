# TODO / Next-session plan — velocity-continuous welded trajectories

## Goal

Remove inter-phase **stops / dead-time** in the touch and pick-and-place motions. The
junction ("weld") points between phases must carry **non-zero velocity**; zero velocity is
allowed **only at the suction-attach instant**. No time interval between phases.

- **A.** Weld the **touch** trajectory (pre-grasp → contact) with non-zero velocity at the
  weld point.
- **B.** Do the same for the **entire pick-and-place** task — velocity may be zero only while
  attaching the object (suction grasp), with no time interval between phases otherwise.

## Core technical problem

cuRobo plans **rest-to-rest** segments (velocity = 0 at both ends). `robot_hal`'s command is
a list of **uniform-`traj_dt` waypoints** (velocity is encoded by waypoint spacing) followed
by a B-spline tracker with a **~0.8 s follow-lag**. Concatenating rest-to-rest segments
bunches waypoints at each junction → the robot decelerates to ~0 there → that *is* the stop.

Welding therefore requires **re-timing / blending** the combined geometric path so the
junctions are not decel-to-rest. The lag is fine for smooth velocity profiles (the B-spline
tracker followed continuous trajectories cleanly); it only overshoots on abrupt mid-motion
**HOLD** commands — those must never be reintroduced.

## Subprojects

### S0 — Capability scouting *(~½ day)*
- Does cuRobo V2 trajopt accept **via-point / waypoint goals** in a single solve (a trajectory
  that passes through the pre-grasp pose **without stopping**)? Check the `MotionPlanner` /
  trajopt API for intermediate-goal costs or waypoint constraints.
- Read `mycobot_mpc/robot_hal.py`: is trajectory timing strictly **uniform `traj_dt`**, or can
  it take **per-waypoint dt / target velocities**? This decides whether we re-space waypoints
  (uniform dt) or pass velocities directly.
- **Output:** decision — native via-point planning vs. post-hoc blend/re-time.

### S1 — Trajectory weld/re-time engine *(~1–2 days) — THE ENABLER*
Build `weld_trajectories(segments, junction_vel)`:
1. Concatenate the geometric joint paths of N cuRobo segments (drop duplicate seam waypoints).
2. **Corner-smooth** each junction (C¹/C² continuous — no instantaneous direction change).
3. **Re-parametrize time** with a velocity profile that hits the requested speed at each
   junction (non-zero) and 0 only where required, respecting joint vel/acc/jerk limits
   (TOPP-style, or reuse cuRobo's interpolation).
4. Emit a `robot_hal`-compatible command (resample to uniform `traj_dt` to encode the profile,
   or per-waypoint timing if S0 allows).
- **Validate:** offline plot of joint velocity vs. time → continuous, non-zero at junctions,
  within limits; on hardware → smooth motion through the via-point, no settle.
- **Risk:** the 0.8 s lag must track the velocity profile (it did for smooth trajectories;
  only HOLDs overshot).

### S2 — Welded TOUCH (pre-grasp + contact) — *user's A (~1 day)*
- Use S1 to weld: `current → pre-grasp (non-zero v) → contact-descent`.
- Measure the **calibrated gap on-the-fly** during the slow descent (streaming works) → set the
  contact target; the **final descent decelerates to rest at contact** (gentle — the known-good
  recipe). The weld point (pre-grasp) carries non-zero velocity; only the contact endpoint is
  v = 0.
- **Validate:** one velocity-continuous descent, gentle touch, correct surface.

### S3 — Welded FULL pick-and-place — *user's B (~2–3 days)*
Structure as **two welded super-segments around the single v = 0 grasp:**
- **Seg A:** approach → pre-grasp (non-zero v) → descend-to-grasp → decelerate to **v = 0 at
  the object** (suction attach).
- **[attach]** stationary, suction ON, seal dwell — **the only v = 0**.
- **Seg B:** lift (from v = 0) → transport → place-descend → release → retract — all welded,
  non-zero junctions.
- **Mid-motion suction toggle:** release (suction OFF) fires at the release waypoint **during**
  Seg B without stopping (toggle by time/index, not a separate phase).
- **No inter-phase dead-time:** remove the per-phase settle waits; only pause is the seal dwell.
- Decisions that currently force stops (verify-held; place-target-from-grasp): pre-compute where
  possible; **fold verify-held into the welded lift** (abort only on failure).
- Reuse **S2** as the descend-to-grasp.
- **Validate:** full cycle as 2 continuous motions; velocity zero only at attach; pick success;
  gentle.

### S4 — Contact sensing: cup-deformation — *open TODO, parallel (~1 day)*
- Validate the **dome-compression** contact signal (track the black dome's depth / image-shift /
  area on contact) with **guaranteed contact** (ask the object height → descend a known amount;
  the ~1 cm cup spring bounds the force). Pick the cleanest observable.
- If robust, it becomes the **contact trigger** for the welded descend-to-grasp (stop on true
  contact rather than a pre-computed target). Note: the 0.8 s lag still requires a slow descent
  at contact — the spring absorbs the overshoot.

### S5 — Per-episode rosbag logging *(independent, ~1 day)*
Record **one rosbag2 (mcap) per object pick-and-place episode**:
`outputs/episodes/obj_NN.mcap`, written programmatically with `rosbag2_py` (the D405 is
grabbed directly, not a ROS topic, and we want clean per-episode start/stop). All messages
carry `header.stamp` (capture time) **and** the bag receive-timestamp.

| topic | type | source |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/CompressedImage` (jpeg) | D405 (raw RGB ≈ 2.5 GB/ep → compress) |
| `/camera/depth/image_raw` | `sensor_msgs/Image` 16UC1 / png | D405 aligned depth |
| `/joint_states` | `sensor_msgs/JointState` (position) | bridge `/joint_states` (URDF rad) |
| `/joint_vel` | `sensor_msgs/JointState` (velocity) | **`/mycobot/drive_feedback`** (velfb deg/s) |
| `/joint_cmd` | `std_msgs/String` (+ derived JointState) | **`/mycobot/cmd/move`** (commanded trajectory) |
| `/phase` | `std_msgs/String` | pipeline events: `pregrasp · contact · lift · move · release` |

**Joint rate:** currently **50 Hz** (`robot_hal.STREAM_RATE_HZ = 50` caps the desktop stream),
but the **LinuxCNC servo loop runs at 100 Hz** (`elerob.ini SERVO_PERIOD = 10 ms`) and the HAL
pos/vel feedback updates every 10 ms. So **log joints @ 100 Hz** by setting `STREAM_RATE_HZ = 100`
on the Pi (+ restart LinuxCNC) — matches the servo rate, no interpolation. Camera stays 30 Hz.

- **Sources already exist:** velocity on `/mycobot/drive_feedback` (no differentiation needed);
  commanded joints on `/mycobot/cmd/move`; phase labels are the `events.jsonl` events in
  `real_pipeline.py` — republish as `/phase`.
- A `rosbag2_py` writer opens at the episode start (pre-grasp) and closes at retract; a thread
  subscribes to the ROS topics and writes the directly-grabbed camera frames + phase strings.
- **Watch:** compress RGB (CompressedImage) or bags are ~GB/episode; prefer the **mcap** storage
  plugin; convert velocity deg/s → rad/s for consistency with `/joint_states`.
- Replaces (or augments) the current per-frame JPG + `states.jsonl`/`events.jsonl` recorder.

## Sequencing

```
S0 (scouting) → S1 (weld engine) → S2 (touch weld) → S3 (full weld)
                                     └── S4 (contact sensing) ── parallel, feeds S2/S3
S5 (per-episode rosbag) ── independent; lands in real_pipeline.py, useful immediately
```

## Risks / watch-items

- **0.8 s follow-lag** — smooth continuous profiles are fine; **never** re-introduce mid-motion
  HOLD/stop-on-contact (they overshoot ~1 cm = "too hard").
- `robot_hal` **uniform-dt** command may force waypoint **re-spacing** (resolve in S0/S1).
- Eye-in-hand **D405 USB drops** on large wrist swings — limit J6, add strain relief.
- Surface estimate **±1 cm** and the **near-contact gap under-read** — the weld does **not** fix
  sensing; keep the calibrated-gap + decel-to-rest + cup-spring recipe, and let **S4** improve
  the true-contact signal.

## Context / pointers

- Code: `github.com/frankkimrobotics/pick_n_place` (this package).
- Touch controller lessons: memory `real-touch-controller.md`. Calibration: TCP = 0.145 m,
  `CAM_TCP_Z_SHIFT = 0`; hand-eye `T_TCP_CAM` in `config.py`.
- Current touch = `servo_touch.py` (calibrated-gap surface → decel-to-rest, cup-mask
  `GapMonitor`). Pick-and-place = `real_pipeline.py` / `real_multi.py`.

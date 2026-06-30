# pick_and_place — MyCobot Pro 630 suction pick-and-place, touch & hand-eye calibration

Eye-in-hand (RealSense **D405**) suction pick-and-place for the **MyCobot Pro 630** (6-DOF),
plus a vision **touch/contact** controller and the **TCP + hand-eye calibration** tooling.
Started as a self-contained kinematic **simulation** (robot off); grew into the real-robot
stack. Reuses the existing repos' cuRobo V2 motion planner, URDF, D405 intrinsics, joint
conventions and ROS2 bridge (siblings `../mycobot_mpc`, `../ros2node`).

> Env: **`curobo2`** conda env for the cuRobo planner; ROS2 Humble + system python3
> (`PYTHONPATH=~/librealsense/build/release`) for the live D405/robot scripts.

---

## System / stack

```
 Desktop (this repo)                                  Raspberry Pi  (LinuxCNC)
 ┌──────────────────────────────────────┐             ┌───────────────────────────┐
 │ pick_and_place/  (detect·grasp·touch· │  /mycobot/  │ robot_hal.py              │
 │                   calibrate)          │  cmd/move   │  PID / inv-dyn controller │
 │     │  planner RPC :9997 (cuRobo V2)  │ ───────────▶│  B-spline trajectory      │
 │     │  perturb_loop.execute / state   │ /joint_     │  follower (~0.8 s lag)    │
 │     │  mycobot_ros2_bridge ───────────┤◀─ states ──┤  cmd:9998  stream:9999    │
 │     │  SAM3 :5599                     │             │  HAL pin pro600.digital_  │
 │     ▼  D405  (USB, eye-in-hand)       │             │  out00  → suction valve   │
 └──────────────────────────────────────┘             └───────────────────────────┘
   camera pose in base = FK_tcp(q) @ T_TCP_CAM   (TCP = 0.145 m suction tip, see Calibration)
```

---

## Streaming control (online, 4 ms) + touch methods

The per-command follower (~0.8 s dead-time, above) is superseded by an **online streaming**
path that drives the arm at **4 ms (250 Hz)** from cuRobo trajectory chunks, plus a set of
**touch / contact** methods. Full detail in **[TOUCH_METHODS.md](TOUCH_METHODS.md)**.

```
cuRobo planner (:9997) ── plan full traj, slice 0.4 s chunk every 0.1 s (sliding window)
  online_planner_node.py ── /planner/weld_chunks ──▶ chunk_to_pi.py ──▶ Pi :9994
  online_servo.py (Pi, 250 Hz): welds chunks → q_ref(t); target = q_ref(now+lead)
    (feed-forward lead cancels the constant dead-time) + pure-PD.  Feedback on :9999.
```

**Touch methods** — `servo_touch.py --stream`:
- `--open-touch` — open-loop descend to the detected top; the compliant cup presses. **Reliable** (controller tracks to ~1 mm).
- `--torque-stop` — joint-torque rise (`pro600.joint{i}_torqfb`, now in the `:9999` stream).
- `--gap-stop` / `--ring-px` / `--gap-descend` — depth gap; **proximity only** (fires ~+41 mm above wide tops).

> **Key finding:** depth/vision contact-sensing is impossible here — the suction cup sits in
> the D405 near-field blind spot, so its own depth is unmeasurable. Use open-loop or torque.

`touch_chunk.py` + `plot_touch_chunk.py` run touch+return via the chunk path and record an
mcap rosbag + D405/D435 frames + phase/contact, then plot trajectory/phase/contact/RGB.
`step_response.py` measures the end-to-end dead-time.

---

## A. Real pick-and-place

Clear flat tabletop objects into a 25 cm bin, one by one. `real_multi.py` does one object
per call; `real_pipeline.py` loops all objects **and records** the run.

```
 base pose
   │
   ├─ DETECT objects  ── D405 RGBD + SAM3 "everything" → base-frame clouds
   │                     filter: reachable, ≤0.13 m tall, not the bin, dedup
   │
   └─ for each object ▸
        pre-grasp 6 cm above flat-top centre   (3 cm-flat support → tolerant to ~1 cm error)
          → REFINE grasp from a close top-down capture (low-noise)
          → suction ON  → descend-to-contact (depth gap at the cup tip, floor-limited)
          → LIFT  → VERIFY held (depth right at the suction tip)
          → PLACE: object-centroid centred over the bin, released ABOVE the rim → drops in
          → back to base
```

`real_pipeline.py` additionally streams, per object, **30 Hz RGB + joint angles + suction
state + timestamps** (`obj_NN/frames/*.jpg` + `states.jsonl` + `events.jsonl`), window =
base → release.

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH=~/librealsense/build/release python3 pick_and_place/real_pipeline.py \
    --suction-host 10.0.0.27 --max-objects 8
# single object / by name:
python3 pick_and_place/real_multi.py --target "orange juice carton" --near 0.45,0.07 \
    --suction-host 10.0.0.27
```

---

## B. Welded touch + blue-dot pick-and-place (`servo_touch.py`)

The main real-robot controller. Every phase is **velocity-continuous** ("welded") — the robot
never decelerates to rest between phases except the suction-seal dwell. Contact is detected by
a **blue marker dot** on the cup's spring-loaded plunger, which physically rises when the cup
touches an object (tracked by normalized template matching, robust to shadow).

```
 base → ONE welded descent: current → pregrasp (passed at NON-ZERO velocity, no stop)
      │   → fast descent → slow last stretch → decel-to-rest at the detected surface
      → CONTACT = blue-dot plunger rises ≥ thresh (re-baselined at gate-open; gap is log-only)
   [pick-and-place mode]
      → grasp-press (settle, then press from rest to seal) → SUCTION ON → seal dwell  ← only v=0
      → WELDED carry: lift the object CLEAR of the box rim → over the box → release above rim
      │   (collision-aware: transit z = rim + clearance + object-hang, so the carried object
      │    can't clip the box; the planner has no runtime attach, so this is geometric routing)
      → WELDED return to base
```

**Hard-won constraints** (see memory `real-touch-controller.md` / `next-welded-trajectories.md`):
- ~0.8 s follow-lag → a vision-HOLD on a *fast* descent overshoots; the welded descent is
  fast then **slows to ~2 °/s in the last stretch** so the blue-dot HOLD is gentle.
- cuRobo plans **rest-to-rest** & is **torque-aware** — welds are post-hoc re-timed to a
  non-zero junction velocity, and the lift/return/carry are **capped to cuRobo's native peak**
  (forcing a flat 55 °/s races motor-cmd past feedback → following-error power-off; ~20 °/s is
  the torque-feasible ceiling lifting against gravity).
- the **gap under-reads / false-fires** near contact → blue-dot is the only contact trigger.

```bash
# touch only (records joints+RGBD with --record; plot with plot_touch.py)
PYTHONPATH=~/librealsense/build/release python3 pick_and_place/servo_touch.py --record
# full welded pick-and-place of every object into the bin at [0.1,0.4]
python3 pick_and_place/servo_touch.py --pick-place --max-objects 10 --max-foot 0.25 \
    --suction-host 10.0.0.27
# segment the blue marker dot first (once): writes outputs/blue_dot_mask.npz
python3 pick_and_place/blue_dot_mask.py
```

---

## C. Calibration

```
 TCP length      floor touch-test (base on table, base_link z=0):
                 cup tip is 0.145 m below the flange  →  URDF tcp_joint, config.PLANNER_TCP_LEN
                 (the 0.13 m it had made every grasp dive ~1.5 cm too deep)
                 ⇒ CAM_TCP_Z_SHIFT = 0  (tcp == the hand-eye calib frame)

 Hand-eye        calib_handeye.py:  base + JOINT-SPACE board views (sample J4/J5/J6 within the
 (T_TCP_CAM)     natural-elbow ranges, |J6|≤95° so the eye-in-hand cable can't drop the USB) →
                 detect ChArUco → pair FK(tcp) with board-in-cam → cv2.calibrateHandEye

 Validate        aruco_touch.py  — touch each ArUco marker at its detected base position
                 touch_objects.py — touch real object centres
```

```bash
PYTHONPATH=~/librealsense/build/release python3 pick_and_place/calib_handeye.py \
    --target 0.4,0,0.0 --square 0.035 --marker 0.026          # writes outputs/handeye_*.json
python3 pick_and_place/aruco_touch.py --max-markers 6        # validation
```

---

## D. Simulation (robot off)

Kinematic + geometric sim of the whole pick-and-place, rendered from the URDF — no ROS,
no hardware. `run_demo.py`; MuJoCo replay via `run_mujoco.sh`.

```
 base → render eye-in-hand RGBD → segment → deproject → cloud (+normals)
      → DETECT 1 cm circular suction grasp point + object 3D OBB
      → PICK (pre-grasp▸descend▸suction▸lift)
      → PLACE  segmented (waypoints)  |  planned (collision-free, OBB as collision volume)
      → release → EVALUATE → report.json/txt + demo.mp4/gif
```

```bash
conda activate curobo2 && cd /home/lisc-frank/Desktop/2026
python -m pick_and_place.run_demo --place-mode planned        # or segmented
bash pick_and_place/run_mujoco.sh planned iso                 # MuJoCo render
```

Sim evaluation PASS = all six: detection · reachability · seal · collision_free ·
released_above_rim · placed.

---

## Modules

| file | role |
|---|---|
| `config.py` | paths, **`T_TCP_CAM`**, intrinsics, TCP/standoff params, base pose, scene defaults |
| `geometry.py` | pose / wxyz-quaternion / transform helpers |
| **real robot** | |
| `real_pipeline.py` | detect-all → pick&place loop, **records 30 Hz RGB+joints+suction** |
| `real_multi.py` | one object per run; `--target`/`--near` for by-name picks; detect/grasp/place |
| `real_grasp.py` | single suction pick (no place); 1 cm suction-point detector; online viewpoint |
| `real_place.py` | place a held object into the bin (carried-OBB verified clear) |
| `robot_execute.py` | replay an exported trajectory on the Pi via the ROS2 bridge |
| `suction_test.py` | toggle the suction HAL pin (`pro600.digital_out00`), no motion |
| **touch / calibration** | |
| `touch_objects.py` | welded approach → touch object centres (no suction) |
| **`servo_touch.py`** | **welded touch + blue-dot contact + suction pick-and-place** (weld descent/return/carry, torque-feasible, collision-aware place); `GapMonitor` streams gap + plunger-dot template |
| `blue_dot_mask.py` | segment the blue plunger marker below the cup dome (contact-signal ROI) |
| `plot_touch.py` | joint pos/vel profiles + time-aligned RGBD filmstrip for a recorded episode |
| `servo_diag.py` | gap-vs-distance diagnostic at several annulus offsets/heights |
| `calib_handeye.py` | eye-in-hand `T_TCP_CAM` recalibration (ChArUco + `calibrateHandEye`) |
| `aruco_touch.py` | hand-eye validation: touch each detected ArUco marker |
| **simulation** | |
| `sim_planner.py` · `scene.py` · `perception.py` · `grasp_detection.py` · `obb.py` · `collision.py` · `simulator.py` · `pipeline.py` · `evaluate.py` · `run_demo.py` · `mujoco_export.py` · `mujoco_play.py` | the in-process sim + MuJoCo render |

## Key facts / caveats

- **TCP = 0.145 m** (suction tip below flange); `CAM_TCP_Z_SHIFT = 0`. Restart the cuRobo
  planner after any URDF tcp edit.
- **Controller follow-lag ~0.8 s** → never vision-HOLD a moving descent; descend to a
  pre-computed target and let cuRobo decelerate to rest there.
- **Surface estimates are ±~1 cm**; the **dome-arc gap under-reads** in the final ~1.5 cm.
  The soft cup's **~1 cm spring** is the only compliance (no F/T or vacuum feedback).
- **Cup contact ROI** = `outputs/cup_mask.npz` (cup is rigid to the camera ⇒ valid at every
  pose): cup mask, black-dome submask, and a top-curve monitoring ring.
- D405 **USB cable** can drop off the bus during arm motion (eye-in-hand) — needs reseat;
  large wrist swings make it worse (hence the J6 limit in calibration).
- `outputs/` (debug images, calibration JSON, recordings) is **git-ignored**.
```

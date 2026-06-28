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

## B. Touch / contact (no suction)

Bring the cup gently onto an object's centre. `touch_objects.py` (open-loop, welded
approach) and `servo_touch.py` (vision-corrected, calibration-free surface).

```
 base → DETECT object centre (interior top-face median surface — avoids edge-float bias)
      → approach to 2 cm standoff
      → measure CALIBRATED gap at the standoff   (gap = median depth[dome top-curve ring]
      │     − median depth[cup rim];  true ≈ 0.78·gap + 10.5 mm)        → surface z
      → ONE smooth decel-to-rest linear descent to  surf − margin
      │     (stop is planned in → no lag overshoot;  the ~1 cm soft-cup spring = compliance)
      → camera MONITORS the contact point (gap → 0 / dome image-shift)
```

Why this shape (hard lessons, see memory `real-touch-controller.md`): the controller has a
~0.8 s follow-lag, so any **vision-HOLD mid-descent overshoots ~1 cm** ("too hard"); the
**gap under-reads in the last ~1.5 cm**; and there is **no force/vacuum sensor** — only the
soft cup's spring. The robust recipe is therefore *measure → decel-to-rest to a target*, not
*stop-on-contact*.

```bash
PYTHONPATH=~/librealsense/build/release python3 pick_and_place/servo_touch.py \
    --standoff 0.02 --margin 0.004 --v-touch 2
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
| `servo_touch.py` | vision touch: calibrated-gap surface → decel-to-rest; cup-mask `GapMonitor` |
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

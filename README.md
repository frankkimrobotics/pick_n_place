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

---

## Controllers & sim twin (2026-08)

Three control stacks, all validated on the MuJoCo twin (`rl/scenes/*.xml`,
same model the RL trains on) with an emulated drive layer that reproduces the
real arm's velocity-mode servo behavior. Environment for all sim scripts:
the `mjwarp` conda env (`$PY = ~/miniconda3/envs/mjwarp/bin/python`).

| Script | What it runs |
|---|---|
| `sim_taskmpc.py` | closed-loop task-space MPC demo: 50 Hz QP (position + cup-vertical orientation task, wall constraint, joint limits) → 250 Hz LQR → drives; moving-object tracking, contact stop, renders mp4 + cmd-vs-actual plots |
| `compare_controllers.py` | cuRobo-v2 vs task-MPC vs pseudo-inverse on a touch task; `--trials N` randomized scenes, RMSE/time/tilt metrics |
| `pick_place_compare.py` | FULL pick-and-place in clutter (suction weld, carried-volume-aware placement, contact release); hybrid cuRobo = plan transits / servo contacts |
| `rl/eval_bench.py` | RL-policy benchmark ladder (see `rl/README.md`) |

```bash
$PY sim_taskmpc.py --out ~/pnp_rl/taskmpc          # demo + video + plots
$PY compare_controllers.py --trials 100 --workers 6
$PY pick_place_compare.py --trials 30 --workers 6 --only taskmpc
# cuRobo variants need the planner server (curobo2 env) on :9997:
cd ../frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo && \
  ~/miniconda3/envs/curobo2/bin/python curobo_planner_server_v2.py
```

Headline results (30 randomized cluttered scenes each): task-MPC 100%
(10.7 s, 0.5 cm), hybrid cuRobo 100% (19.6 s, 0.8 cm), pinv 20%.
Key rules encoded in the harnesses: reference rate strictly below the layer
beneath; anti-windup clamp to measured state; joint POSITION limits +
nullspace limit-avoidance in the task layer; planners only for free-space
transits (near-contact plan failures compound); seal welds at the CURRENT
relative pose (latch-jolt).

---

## Real robot — complete start procedure

Hardware chain: desktop ⇄ Raspberry Pi (`pi@10.0.0.27`, or Tailscale
`100.124.53.41`; password `elephant`) ⇄ LinuxCNC + `robot_hal.py` ⇄ STM32
drives. RoboFlow touchscreen login: Admin / `elephant`.

### 1. Power & hardware start
1. Flip the robot's main power switch.
2. **Press the START button on the base** and wait for the servo *click*
   (drive relay). Without it the STM32 never reports `svr_poweroned` and no
   software can move the arm.
3. The Pi boots with the stock RoboFlow stack auto-started — it must be
   stopped before launching ours.

### 2. Launch the control stack (on the Pi)
```bash
ssh pi@10.0.0.27                      # password: elephant
# stop the stock stack + clear stale state (REQUIRED before every launch):
pkill -9 -f RoboFlow; pkill -9 -f linuxcnc; pkill -9 milltask; pkill -9 rtapi_app
rm -f /tmp/linuxcnc.lock
cd ~/Desktop/mpc
linuxcnc elerob.ini                   # GUI variant (robot monitor), or:
linuxcnc elerob_headless.ini          # headless (linuxcncrsh on :5007)
```
`robot_hal.py` is auto-loaded by the HAL file and **self-initializes
everything**: drive power-on → motor init (watch for "joint 1..6 init
success") → servo enable → machine-on → command preload → "Waiting for
commands". Ports: `:9998` command (`{"target_deg":[6 lcnc deg],
"duration":s,"controller":"pid|mpc|pd_velff"}`), `:9999` 100 Hz state
stream (joints + torque). Headless variant additionally needs homing:
`set home -1` via linuxcncrsh `:5007`.

robot_hal includes the **idle hold mode** (re-servos the last target at
idle; without it the arm sags ~0.4°/s and the drives ferror-trip) and the
**lag-aware LQR `mpc` controller** (sim-tuned, zero overshoot — see
`../mycobot_mpc/README.md`).

### 3. Verify before ANY motion (from the desktop)
```bash
# probe: 1 deg on J1, confirm the stream actually moves
python3 - <<'PY'
import socket, json, time
q = json.loads(socket.create_connection(("10.0.0.27",9999),3).makefile().readline())["joints_deg"]
t = list(q); t[0] += 1.0
s = socket.create_connection(("10.0.0.27",9998),3)
s.sendall((json.dumps({"target_deg":t,"duration":2.0,"controller":"pid"})+"\n").encode())
time.sleep(3)
q2 = json.loads(socket.create_connection(("10.0.0.27",9999),3).makefile().readline())["joints_deg"]
print("moved:", round(q2[0]-q[0],2), "deg  ->", "OK" if abs(q2[0]-q[0])>0.4 else "NOT MOVING")
PY
```
**Never send large motions to an unverified stack** — commands to frozen
drives wind the PID integral into a violent-jump hazard.

### 4. Run the hardware scripts (desktop)
```bash
# cuRobo planner (desktop GPU) — needed by real_touch:
cd ../frankkimrobotics/.../curobo && ~/miniconda3/envs/curobo2/bin/python curobo_planner_server_v2.py &

python3 real_touch.py                          # dry-run plan (touch demo)
python3 real_touch.py --exec --obj X,Y,TOPZ    # execute (slow, logged, plotted)
python3 real_ctrl_validate.py --exec           # pid-vs-mpc validation protocol
PYTHONPATH=~/librealsense/build/release python3 policy/rs_shm_server.py &   # cameras
~/miniconda3/envs/mjwarp/bin/python policy/real_student.py --exec --slow 3  # vision policy
```

### 5. Shutdown
Send the arm home, then either Ctrl-C the interactive linuxcnc (robot_hal
powers the drives off cleanly) or `pkill -f linuxcnc`, then switch off the
base. **Do not leave the arm enabled and unattended raised**: the servo
enable is known to drop spontaneously (suspected 48 V path, inspection
pending) and the arm sags until the brakes catch.

### Known hardware caveats (2026-08)
* Servo-enable hold-time degrades across soft restarts; a full power cycle
  resets it. Time-box sessions.
* Fixed D435 mount was rebuilt — **re-run `calib_d435.py` before trusting
  detections** (`d435_detect.py --board` self-check).
* Drive velocity ceiling ≈ 36 °/s (STM32 firmware) — all controllers and
  cuRobo joint-velocity limits must respect it.

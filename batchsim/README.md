# batchsim — batched, latency-aware twin of the Pro 630 cell

N copies of one scene step together on a GPU (MuJoCo Warp 4.0, `mj4` env). Everything between the policy and the arm
that the scalar twin (`realsim/`) models is here per world; the two share `realsim/config.py`, so they cannot drift.

```
policy / RL agent
   │ observe()   joint feedback (10 ms sample-and-hold) + camera frames, each as OLD as the host would get them
   │ command()   joint control polygon + valve; reaches the Pi after inference + host + transport delay
   ▼
twin.BatchTwin
   ├─ latency.py   per-world delays (sampled from realsim.config.DR with dr=True), PolicyClock, RingHistory
   ├─ drive.py     Pi stream law → dead time → accel limit → inner loop + gravity ff → joint torque
   ├─ world.py     mujoco_warp model/data, per-world joint friction / damping / reflected inertia
   ├─ sensors.py   StateHistory (0.6 s), JointFeedback, BatchCamera (ray-traced batch render of the DELAYED state)
   └─ suction.py   valve onset / release delays, top-face seal, carried object
env.py     PickLiftEnv   vectorised state-based RL task acting through the same command path
eval.py    evaluate()    any policy over N worlds: task, timing and drive-limit metrics
```

## Latency budget (level `real`, defaults; measured by `tests/test_latency.py`)

| stage | value | source |
|---|---|---|
| policy inference | 150 ± 25 ms (60 ms with `--fast_infer` settings) | config `infer_*`, or the evaluator's measured wall time |
| host processing | 30 ms | `host_proc` |
| host → Pi | 3 ms, 3 % spikes of 50–125 ms | `tx_lat`, `tx_spike*` |
| Pi lead / drive dead time | 40 ms / 40 ms (30–65 ms randomised) | `lead`, `dead` |
| acceleration limit, inner loop, friction | drive.py, `joint_model=1` for the torque-fitted joints | `amax`, `kp/kd`, `JOINT_TORQUE_FIT` |
| joint feedback | 10 ms | `fb_lat`, `fb_period` |
| ZED / wrist frame | 90 / 70 ms + up to one 33 ms frame (ASSUMED, not measured on the rig) | `zed_lat`, `wrist_lat` |

Command → first joint motion: 240 ms (150 ms with fast inference); `ideal` level: 10 ms.

## Checks

```
python -m batchsim.tests.test_drive_parity cuda:0   # vs scalar twin on hardware references: max 0.0002 deg
python -m batchsim.tests.test_latency cuda:0        # command / feedback / camera delays vs the budget
python -m batchsim.tests.bench cuda:0 256,1024,4096 # throughput
python -m batchsim.eval --nworld 64 --cameras zed wrist_d405
python -m batchsim.examples.ppo --nworld 2048       # RL smoke test
```

Throughput on the RTX 2080 Ti (scene 980, 8 objects, 2 ms physics, 10 ms Pi tick, 100 ms decisions):
1,024 worlds 1,490 decisions/s state-only, 1,150 with both cameras at 432×240; 4,096 worlds 5,700 decisions/s state-only.

## Limits (not yet ported from the scalar twin)

- One scene (object set) per BatchWorld; layouts, physics and latencies vary per world. Different scenes = one batch each.
- Suction is the simplified top-face model: no rim conformance on curved tops, no sprung plunger, no seal-break force.
- The image-policy host executor (IK, guards, cross-fade, safety filter, aim assist) is still per-world CPU code in
  `realsim/eval_policy.py`; `eval.evaluate()` takes any policy that outputs joint polygons, so it needs a batched IK
  adapter before the 920M policy can be evaluated here.
- Batch-rendered images differ slightly from the GL renderer the policy was trained on (no shadows / floor reflection);
  check feature distance before trusting image-policy numbers.
- mjlab (env `mjlab`, mujoco-warp 3.11) is installed separately as an RL framework option; batchsim does not depend on it.

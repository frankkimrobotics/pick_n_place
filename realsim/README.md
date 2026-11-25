# realsim — realistic Pro 630 twin on MuJoCo 4 (`mj4`)

Closed-loop evaluation of the trained ZED visual policies with the delays and controller the
real robot actually has, instead of the old harness's "PD straight to the target, zero latency".

```
conda activate mj4
python realsim/validate.py                                  # twin vs hardware (step, sine, 21 real runs)
python realsim/eval_policy.py --ckpt ~/pnp_policies/cfm_lingbot_bspline_lat2_aug/ckpt_final.pt \
       --level real --slow 2 --scenes 90-99 --out ~/pnp_rollouts_realsim/real_slow2
```

| file | what |
|---|---|
| `config.py` | every number, with its hardware source or an **ASSUMED** tag; `make_cfg(level, dr)` |
| `twin.py` | `Pro630Twin` (Pi stream law → drive → physics) + `CameraModel` (delayed frames) |
| `plunger.py` | sprung cup for existing scenes (MuJoCo 4 `MjSpec` edit) |
| `press_test.py` | scripted 25 mm over-press, rigid vs plunger: force, compression, penetration, stall |
| `validate.py` | step / streamed sine / replay of the logged real runs, `real` vs `ideal` |
| `eval_policy.py` | the real executor's decision logic (`policy/real_rollout_zed.py`) on the twin |
| `eval_ideal_harness.py` | the old ideal harness (`/tmp/eval_zed_clean.py`), kept as the baseline |
| `host_timing_measured.json` | decision periods / plan-adoption delays extracted from the real logs |

## What is modelled

| stage | model | source |
|---|---|---|
| camera | frame captured at the last frame boundary before `now − latency` (ZED 90 ms, D405 70 ms, 30 fps, random phase), rendered from the twin's state history; 432×240 + JPEG q95 like the cam servers | latency **ASSUMED**; format = `zed_cam_server.py` |
| joint feedback | posfb at 100 Hz, sample-and-hold, 2 ms stream latency | 2026-09-17 per-hop table |
| policy compute | async inference ~N(150, 25) ms; a chunk is used from the first decision after it is ready; `lat_shift` indexing; ab_refit splice at adoption | 110–200 ms measured; the real logs show adoption 1 decision later at slow 2 |
| host loop | decision grid `t0 + k·0.1·slow`, send after 30 ms of processing, anchor = grid time | `real_rollout_zed.py`; spread ≤ 14 ms from the logs |
| executor guards | step caps 1.5 cm lateral / 3 cm vertical, dq ≤ 2.5°, graceful lead scaling (3°), tilt ≤ 30°, z floor, suction z-gate + 5-decision release hysteresis, yawlock | `real_rollout_zed.py` 2026-11-08 defaults |
| transport | 3 ms desktop→Pi, 3 % spikes of 50–125 ms | 2026-09-17 |
| Pi controller | `SplineRef` semantics (uniform cubic B-spline over the last 4 targets + 1 extrapolated point, **v_ref = 0 past the last knot**), 10 ms law `v = vff·v_ref − K0(q−q_ref) − K1(q̇−v_ref)` sampled at t + 45 ms on 1-tick-old posfb, K0 10 / K1 0.3 / vff 1, ±50 °/s | `STREAM_GAINS`, `mycobot_mpc/spline_ref.py` |
| drive | 45 ms dead-time (physics resolution), 800 °/s² accel cap, 1.5° following-error anti-windup, stiff position loop + gravity comp (drive integrator) | 2026-09-17 accel-4x calibration |
| joints | armature 0.191, viscous 0.44–0.49, Coulomb 0.28–0.31 N·m | `rl/fit_actuator.py` → `outputs/actuator_fit.json` |
| suction cup | **sprung plunger, 15 mm stroke** (`plunger.py`, injected into the generated scenes with MjSpec): 2 N preload → 6.5 N at full stroke, then a hard end stop; seal = lip seated (≥ 1 mm compression) on the object + vacuum + tilt < 25°; the object welds to the plunger tip. **No sensor**: compression/force are logged as sim-only diagnostics and never reach the policy or executor. `--rigid_tip` = the old rigid cup | stroke from the user; spring force **ASSUMED** |
| contact stiffness | cup contact and plunger end stop solref 0.004 s (scene default 0.02) | default put a rigid cup 20 mm *inside* the object on the demos' 15 mm press |
| suction | 300 ms vacuum build-up before a seal can latch, 150 ms release | **ASSUMED** |

`--level` ladder: `ideal` → `drive` → `drive+sensor` → `drive+sensor+compute` → `real`
(each adds one group); `--dr` samples dead-time, accel, vmax, armature, camera and compute latency per scene.

## Validation (`outputs/realsim_validation.json`)

| test | hardware | twin `real` | `ideal` (old assumption) |
|---|---|---|---|
| 10° step, point law K0=6: onset | 36–52 ms (+10 ms tick) | 60 ms | 10 ms |
| rise 10–90 % | 0.24–0.27 s | 0.22 s | 0.36 s |
| overshoot | 0.0–0.1 % | 0.04 % | 0 % |
| settle ±0.5° | 0.42–0.48 s | 0.39 s | 0.49 s |
| streamed 4° 0.35 Hz sine, rms | 0.28° | 0.34° | 0.015° |
| **replay of 21 real ZED-policy runs** (4082 decision pairs, logged `q_cmd` at logged timing → compare with logged `q_meas`, J1–J4) | — | **0.31° mean / 1.0° p95** | 0.82° / 3.3° |

The replay is the important number: given the exact commands the real executor sent, the twin
lands within a third of a degree of where the real joints were. The earlier free-running
Isaac-style fit (`fit_actuator.py`) was 4.6° mean on its holdout.

## Things the real logs revealed (and the twin now reproduces)

* **All hardware ZED runs used `--slow 2`**: 0.2 s decisions and 0.2 s spline knots. The policy's
  0.1 s actions are executed at half speed, and the "previous observation" fed to the policy is
  0.2 s old, not the 0.1 s it was trained with. `eval_policy.py --slow 2` is the deployed setting.
* Plans are adopted **1 decision** after their observation (inference < 0.2 s). With the
  ckpt's `lat_shift 2`, index `k − k_obs − 2` is clamped to 0, so action 0 is executed twice.
* The Pi's spline returns the end point with `v_ref = 0` once `t + lead` passes the last knot.
  The end point is `q_k + 5/6·(q_k − q_{k−1})`, which is why the real arm sits slightly *ahead* of `q_cmd`.

## Plunger press test (`press_test.py`, scene 90, 2 mm/decision, 25 mm past the grasp pose)

| tip | force at −14 mm | end state | cup inside the object |
|---|---|---|---|
| rigid (scene as generated) | — | 49 N | **19.9 mm** (soft default contact) |
| plunger, default contacts | 4.9 N | 8 N, 16.5 mm "compression" (soft end stop) | 7.6 mm |
| plunger + stiff contacts (used) | 6.2 N (13.4 mm) | bottoms out at 15 mm, then ~5 N/mm → 51 N | 0.7 mm |

## Closed-loop results: `cfm_lingbot_bspline_lat2_aug`, scenes 90-99 (47 objects), plunger, 400 decisions

| setting | sealed scenes | objects in bin | 1st object binned | velocity-saturated ticks |
|---|---|---|---|---|
| ideal, slow 2 (deployed executor) | 10/10 | 10 | 41 s | 0 % |
| +drive / +sensor / +compute, slow 2 | 10 / 10 / 9 | 10 / 10 / 8 | 41 / 42 / 37 s | — |
| **real, slow 2 (= hardware setting)** | **9/10** | **9** | 37 s | 5 % |
| real, slow 2, DR | 9/10 | 9 | 39 s | — |
| real, slow 2, rigid cup | 9/10 | 9 | 36 s | — |
| ideal, slow 1 | 10/10 | 10 | 21 s | 0 % |
| ideal, slow 1, executor caps off | 10/10 | 15 | 10 s | 0 % |
| **real, slow 1 (trained speed)** | **5/10** | 6 | 18 s | **22 %** |
| real, slow 1, caps off | 6/10 | 6 | 12 s | 45 % |
| old ideal harness (no executor, 72 s), 3 runs | 9-10/10 | 23 / 30 / 32 | — | — |

* At the policy's trained speed the measured drive cannot follow (50 deg/s saturation on 22-45 % of ticks)
  and seals halve; `--slow 2` restores 9/10. This is why the hardware runs work only at slow 2.
* The executor's step/dq caps make the carry ~2x slower (one place per 40 s episode); with caps off
  second picks appear (15/47) in the ideal twin. The remaining gap to the old harness is its looser
  execution (exact IK pose every action, latched suction state, 72 s budget).

## Pitfalls

* Replaying logs: the decision period is `0.1·slow` from the run's `args.json`. Using 0.1 s on
  slow-2 logs makes the loop look seconds late and the twin look 1 decision slow.
* Only policy-phase rows (`plan` set on both sides) are comparable. Splice and scripted legs
  move the arm without logging `q_cmd`.
* Run the evaluator in `mj4`; it imports `policy/real_rollout_zed.py` directly (no robot
  connection is made), so executor changes there flow into the sim automatically, except
  for the per-decision loop, which is mirrored in `eval_policy.py`.
* Mega (curobo/YCB) scenes keep the robot's visual meshes in **geom group 3**, which MuJoCo does not render by
  default. Their generator enables it; the evaluators now do too (`vopt.geomgroup[3] = 1`). Without it the arm is
  invisible to the policy and the mega-trained DiT scored 0/58; with it, 8/10 scenes seal at `real`, slow 2.

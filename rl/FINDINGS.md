# RL pick-and-place: findings log

myCobot Pro 630, single suction cup, `rl/env_warp.py` (mujoco_warp, batched GPU).
Written down because every item below cost hours to find and is invisible in the code.

## Status

| Artifact | What it is | Result |
|---|---|---|
| `ppo4/final.pt` | pick-and-place, table targets | 100% eval, median 5.5 mm placement |
| `ppo5/final.pt` | + full-workspace hover grid | 99.3%; 4/4 sequential multi-object clearing |
| `ppo7_ped/final.pt` | + 30 cm lift, gentle landing, pedestals | 65.4% of that spec (placement 68.8% = binding) |
| `ppo15_attach/final.pt` | attach stage, **corrected physics** | 74.6% seal+lift — the clean warm-start |
| `distill1/student_final.pt` | DAgger student, 2×RGBD only | lift 33.4 cm, landing 4.0 cm/s, placement 11.4 cm |

**Open**: no policy yet satisfies the full contact-release + tilt + verticality spec.
Runs 8–14 (~150M steps) were spent finding the defects below rather than converging.

## Physics defects found (all fixed)

1. **Free-joint `qvel` angular part is BODY-frame.** Feeding it into a world-frame
   `xfrc` torque as damping *pumps* energy at large object yaw: sealed objects
   tumbled to 90–180°. Rotate ω to world first.
2. **Peel ≠ cap saturation.** Breaking the seal when the righting spring hit its
   torque cap (reached at 5.7°) broke every lift at ~2 cm. Real peel is a sustained
   large tilt angle (`PEEL_COS`, ~28.6°).
3. **Capped spring + damping starves dissipation** (C·ω ≫ cap ⇒ undamped ringing to
   80°), but uncapped torque damping is explicit-unstable for a small box inertia
   (NaN in 92% of worlds). Use a capped spring **plus** direct exponential decay of
   angular velocity while sealed.
4. **The restoring torque had zero yaw authority.** `K_ROT × cross(objz, cupz)` is
   perpendicular to the cup axis by construction, so a sealed object could spin
   freely — only its tilt was held. Fix: store the full relative orientation at
   latch (`anchor_R`) and hold it.
5. **One torque cap cannot serve two limits.** Capping the combined vector let a
   large yaw error starve tilt correction (tilt got *worse* when the spring was
   stiffened). Cap peel (perpendicular) at `TAU_CUP` and torsion (about the axis)
   at `TAU_TORSION ≈ μ·F_MAX·r_cup` separately.

## Reward-design rules (each learned the hard way)

- **Grade every requirement on a sparse terminal.** An all-or-nothing gate added to
  a terminal recreates the never-release trap — seen 3× (release penalty, lift gate,
  contact-release).
- **Audit potentials for asymmetric freezes.** A current-height lift potential paid
  −4 for *sealed* descent while *released* descent froze it (free) — the reward
  literally bribed dropping over placing. Use potential on **max** lift.
- **Gaussian grades are cliffs when the warm start is far away** (release from 25 cm
  vs σ = 8 mm ⇒ e^−19). Use Laplace kernels sized to reach current behavior.
- **Multiplied quality grades starve jointly** (~2e−4). Combine as a geometric mean.
- **Free mistakes become strategies.** Losing the grip far from the target cost
  nothing ⇒ a stable latch–lose–retry loop. Price it.
- **Bonuses gated on `~sealed` get farmed** by delaying the latch (~+4/ep). Gate on
  `~ever_sealed`.
- **Measure grades on the warm-start policy before launching** (`diag_factors.py`
  pattern: collect `info` fields at set-down, print per-factor percentiles). A
  climbing return can hide a terminal that is exactly zero — run 8 trained 20M steps
  with `place = 0.00` throughout.

## Curriculum

- Sequential stages **catastrophically forget**: place-mode training (sealed-carry
  starts) erased lift entirely; re-initing from the lifter hit its tilt habit.
- `--mode mix` assigns each world an episode type (pnp / carry / place) at reset, so
  all skills train in one network simultaneously. Forgetting becomes impossible,
  but full-task *assembly* still did not emerge.
- `--release_mask` (+ `--mask_h` annealing 5 cm → 0.8 cm) deletes the
  drop-from-height equilibrium instead of out-bribing it; mask sealed-start worlds
  only, or it blocks discovery in pnp worlds.
- Warm-start habit surgery has a depth limit. Once the environment is economically
  coherent, from-scratch through the proven `attach → pnp` chain beats repair.

## Infrastructure

- **Render on the GPU**: `MUJOCO_GL=egl` gives ~631 render-units/s vs ~3.5 for
  single-process osmesa (~180×). `distill.py` uses `EglFarm` with a multiprocess CPU
  pool as fallback. mujoco_warp has no renderer, so images still cost a GPU→CPU
  `qpos` round-trip per step; madrona-mjx would remove that seam.
- Read episode terminals from the `info` dict, never post-step env state (auto-reset
  zeroes it).
- The Lightning studio pulls **main** — merge before every launch and verify the
  deployed code with `grep` on the studio. A stale checkout silently wasted a 20M-step
  run. It also restarts on **CPU** after a stop; request the H100 explicitly.

## Dynamics update from hardware calibration (2026-09-17)

The arm was calibrated on the real Pro 630 the same day (`mycobot_mpc/README.md`,
"Hardware calibration, agile tuning and latency"). `env_warp.py` now has a measured
drive model (`--drive real`, default) in place of the legacy stiff PD straight to the
commanded target (`--drive ideal`):

| stage (per joint) | value | source |
|---|---|---|
| decision → reference | linear ramp over the 100 ms decision (what the Pi's chunk welder does) | robot_hal chunk mode |
| outer law, every 10 ms, on 10 ms-old feedback | `u = vff·v_ref − K0·(q − q_ref) − K1·(q̇ − v_ref)`, K0 20, K1 0.3, vff 1 | deployed streaming law |
| velocity saturation | 50 °/s (DR 0.9–1.1×) | drive saturates ≈ 50 °/s |
| command → motion dead-time | 45 ms (DR 30–65 ms) | onset 36–52 ms |
| acceleration cap | 800 °/s² (DR 0.75–1.15×) at `motor_accelaration` 4× (250 at 1×) | measured 720–870 |
| drive position loop | 3× stiffer PD with damping on velocity error, |q_drive − q| ≤ 0.3° | posfb tracks to 0.04° on hardware |

`rl/drive_probe.py` replays the hardware experiments on the model:

| experiment | hardware | sim (real drive) | sim (ideal drive) |
|---|---|---|---|
| 10° step, waypoint law K0 = 6: onset / rise / overshoot / settle / peak | 36–52 ms / 0.27 s / 0.1 % / 0.43 s / 45–50 °/s | 60 ms / 0.24 s / 0.0 % / 0.43 s / 55 °/s | 10 ms / 0.12 s / 0 % / 0.17 s / 110 °/s |
| 12° 0.5 Hz streamed sine: rms / max error | 0.10–0.12° / 0.22° | 0.12° / 0.18° | 0.33° / 0.50° |

Consequences for training:

- **Observation grows 37 → 43**: `q_target − q` (the drive-lag state) is appended, so the
  policy can see how far the drive is behind its command (dead-time makes the plant
  non-Markov in `q` alone). `AC.load_state_dict` zero-pads older checkpoints.
- **Reward: two new dense terms.** `sat` (−0.1 × fraction of 10 ms command ticks with the
  velocity command saturated: the policy asked for speed the arm does not have) and
  `smooth` (−0.01 × squared decision-to-decision change of the joint delta). Both are mild
  shaping costs on the scale of `act`. A first version also counted acceleration-limited
  physics substeps as saturation and fired on ~77 % of ticks (−9/episode) — ramping at the
  cap is normal drive behaviour, not a policy fault.
- The legacy 100 ms action-delay DR is disabled for `--drive real` (the model carries the
  real latencies); the `--dr` gain / seal / obs-noise jitter stays.
- `--dq_max` (deg per decision, default 2 = 20 °/s) can be raised to 4 for agile variants;
  the real arm tracks up to ~45 °/s.
- Throughput is unchanged (~1.8k env-steps/s at 2048 worlds on the A5000).

### First A/B result and a reward correction (2026-09-17, later)

`rd_attach_ideal` vs `rd_attach_real` (attach from scratch, 2048 worlds, 4M steps): at 1.47M
steps the ideal drive was at 18.7 % seal-and-lift and climbing, the measured drive flat at
0.2 % with the return rising only through cheaper actions. `rl/seal_probe.py` (scripted
descend-press-lift, no learning) shows the seal IS reachable under the measured drive —
19–22 % latch at 0.5–1.0 °/decision vs 28–32 % ideal, same contact speeds, no breaks — so
the dynamics were not the blocker. The `sat` penalty was: with K0 = 20 and 45 ms of
dead-time, every full 2 °/decision command saturates the velocity command, so the term
punished ordinary full-speed moves and trained the policy to move less, starving the
low-probability seal discovery. `W["sat"]` is now 0 (still logged as a diagnostic);
`rd_attach_real2` is the rerun. Rule: **a feasibility penalty must not fire on the
behaviour the task needs** — measure the component on a scripted competent controller
before giving it weight (the same `diag_factors` discipline as for grades).

### Why the measured drive stalls attach-from-scratch (2026-09-17, later)

Zeroing `sat` did not help (`rd_attach_real2`: 0.1 % at 2.5M steps while the ideal run
reached 54 % at 3.9M). Three probes localised it:

1. **Anti-windup was missing.** Against a blocked joint the drive integrator ran 28° ahead
   of the joint in 0.7 s and released as a jump. The real firmware trips at its
   following-error limit instead. `DRIVE["ferror"] = 1.5°` now bounds the drive state
   (velocity state zeroed on the blocked side). Seals unaffected; it removes a violent
   artefact from contact-rich exploration.
2. **Start height is not the problem.** Random-policy discovery is 0.05–0.07 % for hover
   2–4 cm, 0.5–2 cm and 0.2–1.2 cm alike (ideal drive: 1.14 %).
3. **The binding gate is *pressing*.** Per 1000 random world-steps: near (< 12 mm) 13.4 vs
   6.3, near∧want∧pressing 0.45 vs 0.10, all gates 0.378 vs 0.043 (ideal vs real). A random
   walk's brief dips below the object top never propagate through 45 ms of dead-time and
   the acceleration ramp; the seal needs a *sustained* push of ≥ 100–200 ms. That is a real
   property of the arm, so the fix belongs in exploration/curriculum, not in the gate.

`rd_attach_real3` = PPO warm-started from the ideal-drive policy (54 %) under the measured
drive. First update: 0.34 % seal — the ideal policy's skill does not transfer as-is.

### Exploration under the measured drive: bootstrap, don't search (2026-09-17, later)

- Correlated (pink) exploration noise: ρ 0.7 → 0.10 %, ρ 0.9 → 0.16 % discovery (i.i.d. 0.05 %,
  ideal drive 1.14 %); 80-step episodes do not help. Not enough on its own.
- Warm start from the ideal-drive policy (54 %): 0.3–0.4 % under the measured drive, return
  −5.5 — its fast jerky habits are punished and never produce the sustained press.
- **`rl/bc_bootstrap.py`**: scripted descend-press-lift teacher (per-world IK direction,
  1 °/decision, σ 0.15 action noise) collected 1058 successful episodes (28–37 % per batch)
  under the measured drive; cloning them into the PPO actor gives **63.5 % deterministic
  seal+lift** — better than the noisy teacher. PPO from that init (`rd_attach_real_bc`)
  starts at 10.5 % under exploration noise (log-std −1) and is the current attach run.
  Rule: when the plant filters exploration (dead-time + ramps), inject the behaviour with a
  scripted teacher and let PPO refine; do not tune exploration noise.

### pnp under the new costs: lift-and-park (2026-09-17, later)

`rd_pnp_ideal` (ideal drive, ppo7 spec: lift_req 0.35, speed_bonus 0.3, pedestal scene, new
`smooth` cost) reached seal 98 % and full lift by 6M steps and then parked: 100-step timeouts,
object hovering 16 cm from the target, suction never released, `transport` ≈ −0.04/ep.
Economics: moving 13 cm to the target is worth +0.8 (`transport` 6 × Δm) against movement
costs of −2.3 (`act`) − 1.2 (`smooth`) − 0.7 (`tilt_pen`) per episode, and the +20 place
terminal is never experienced because releasing far away is priced (`rel_far`, `drop`).
Classic never-release trap, re-created by adding a movement cost. Rerun as `rd_pnp_ideal2`
with the proven ppo4 economy: `--lift_req 0 --speed_bonus 0 --smooth_w 0`, table scene.
Rule (again): every new dense cost must be checked against the potentials it competes with.

### pnp parks even with the ppo4 flags — the reward code moved on (2026-09-17, later)

`rd_pnp_ideal2` (`--lift_req 0 --speed_bonus 0 --smooth_w 0`, table scene) parked exactly like
the first run: seal 98 %, full lift, `transport` ≈ 0, never releases, 100-step timeouts.
`rd_pnp_real` found the other local optimum: seal, lift a little, release early anywhere
(`place` +3.9 from the graded set-down at ~10 cm, `rel_far` −0.8), 0.7 % success. A carry
probe (scripted pick, then a 20° base rotation at 10 °/s) breaks 0.6 % of seals at ≤ 17°
tilt, so physics is fine. The "ppo4 economy" no longer exists in the code: `tilt_pen`,
`rel_far`, `place_align`, `chatter` and the suction hysteresis were all added for the
contact-release spec after ppo4/5, and the plain `pnp` stage was never re-validated under
them. With those costs, moving 13 cm is worth +0.8 (`transport` 6 × Δm) against −2 of dense
costs, and the +20 terminal is never sampled. Relaunched as `rd_mix_{real,ideal}`:
`--mode mix --release_mask --mask_h 0.03 --tilt_pen -0.15 --transport_w 15 --lift_req 0
--speed_bonus 0 --smooth_w 0` from the attach checkpoints (carry/place worlds start sealed
near the target; mix keeps the pick from being forgotten). `--transport_w` is new.

### Mix curriculum: stuck a few cm above the surface (2026-09-17, later)

`rd_mix_{ideal,real}` at 8M steps: seal 97 %, transport now positive on the ideal drive, but
0 % placements. Deterministic replay in `place` worlds (start sealed 4 cm from the target,
31 cm up) shows two stalls: the ideal policy descends only to 13 cm, drifts to 9.5 cm off
target and never commands release (0/100 steps); the measured-drive policy descends to
5 cm, gets within 3.5 cm in 25 % of worlds and commands release 29/100 steps — 27 of them
blocked by the 3 cm mask (paying `rel_mask` −0.6/ep) — and never goes lower. The last
centimetres pay nothing: the descend potential's xy gate (σ = 7 cm) is ~0.4–0.6 once the
object has drifted 7–10 cm, so lowering costs more in `act` than it earns, and the
contact-release grade is only sampled below the mask. Relaunched from the 8M checkpoints
as `rd_mix2_*` with `--mask_h 0.05` (the graded Laplace contact factor takes over from
there: release at 5 cm = 0.54) and `--descend_sigma 0.15` (new flag).

### mix2 result and the recommended next move (2026-09-17, end of session)

`rd_mix2_real` (5 cm mask, 15 cm descend gate): placements appear — 1.6–2.0 % success
(0.8–1.1 % in pnp worlds), graded set-down credit 5.3/ep, episodes 42 steps — and plateau
there by 5M steps. `rd_mix2_ideal`: still 0 %, never releases (100-step timeouts). Both runs
were left to finish 16M steps; check `~/pnp_rl/rd_mix2_*/log.jsonl`.

Reading of the day: under the post-ppo4 reward code the full task no longer assembles from
the attach skill by exploration on either drive, and each dense-term fix moves the stall by
a few centimetres. The one route that worked today was the scripted-teacher bootstrap
(attach: 0 % → 63.6 %). **Recommended next step: extend `rl/bc_bootstrap.py` to the full
pick-and-place** (scripted descend → seal → lift 5 cm → IK carry to the target → lower →
release at contact, per-world IK, DR on), clone the successful episodes, then PPO in mix
mode from that init. It injects the whole behaviour, so PPO only has to refine timing and
placement accuracy — the regime in which the graded terminals are known to work
(ppo4/5/7).

### Paper reproduction track (Arafat et al. QPAIN 2026), 2026-09-18/19

`rl/env_paper.py` hosts the paper's task (reach → lift → hold at a 3-D goal, no release) on
the twin. Findings, each from a replay of the stalled policy:

1. Pure paper reward: reach learns, **no seal ever** on either drive — a gripper closing on a
   cube grasps trivially, a suction cup needs a sustained 3 mm press (`--grasp_shaping`).
2. Paper observation `[q, q̇, p_obj, p_goal, a_prev]`: 6-DoF policies plateau 10 cm from the
   object (implicit FK); `--obs_ee` adds tcp, cup axis, grasp-relative vector.
3. Reach to the object **centre** pulls a suction cup down beside the box, tilted 30–43°;
   `--reach_target grasp` (top centre + cup radius).
4. Lift indicator at 2 cm has no gradient below it: sealed policies pressed for the rest of the
   episode; `--lift_dense` ramps to the threshold.
5. **Teacher + DAgger is what made it learn** (`rl/bc_curobo.py`: cuRobo transits, slow press,
   IK-waypoint relabeling). One-shot BC = 0 % every time; 3 DAgger rounds → 11 % deterministic
   (ideal), student-driven 15 %. Clone in PPO's pre-tanh space (`mu = atanh(a)`): the squashed
   clone lost its behaviour in one update (0.3 % seals), the corrected one kept 9.9 %.
6. **The paper's weight emphasis matters**: with reach 1 / tracking 2+4 the DAgger-initialised
   run peaked at 14 % and decayed (return kept rising on lift credit); with reach 0.5 /
   tracking 4+8 (`--w_reach --w_track_c --w_track_f`) it climbed 5 → 20 % by 4M steps
   (`paper9_ideal`, 4096 worlds).
7. Measured drive: cuRobo teacher 31 % success (slow press 0.6 °/decision is what lets it seal),
   DAgger student seals 45 % at round 1 then regresses; PPO from that round (`paper9_real`) is
   the current run. Every stage is ~3× slower than the ideal drive.
9. **The paper's adaptive LR is what made every DAgger-initialised run peak and decay** (peaks
   14–20 %, then 2–4 %): the KL rule raises the rate 10× above 1e-4 once the policy settles.
   Same init, same weights, fixed lr 1e-4, entropy 0.003: **11 → 15 → 20 → 34 → 51 → 68 %**
   success by 3.9M steps, seal 92 % (`paper10_ideal`, 4096 worlds; 6/6 deterministic replay).
   `ppo.py` now saves `best.pt` at the peak success.
10. Warm-starting the measured drive from the strong ideal policy: 0 % seals (`paper11_real`),
   as for attach — cross-drive transfer of a finished policy does not work; the measured drive
   needs its own DAgger teacher (or the dynamics curriculum).
11. **The measured-drive teacher was failing at the press, not the reach.** Its IK press target
   sat 4 mm below the grasp point against a 3 mm latch requirement, so with DR and tracking error
   the cup stopped 1–2 mm short: 97 % of teacher episodes stuck pressing (34 % even on the ideal
   drive). Press-depth sweep on the measured drive (150-step episodes): 4 mm 3 %, 8 mm 65 %,
   12 mm 80 %, **20 mm 91 %** teacher success (ideal: 66 → 96 %). `--press_depth 0.02` is the default.
12. Cross-drive transfer fails for reasons the dead-time curriculum cannot fix: the 89.8 %
   ideal checkpoint scores 1–2 % in the measured-drive env even at zero dead-time and 5× accel,
   because the streamed outer law, the 100 ms ramps, the velocity ceiling and the delayed
   feedback remain. Each drive needs its own policy trained under its own dynamics.
13. **Deep-press DAgger (2026-09-19, `dagger4_*`, `--press_depth 0.02 --press_rate 0.6`)**:
   ideal teacher 90 %, student 0 → 0 → 20 → **44.5 %** deterministic over 3 rounds (one-shot BC
   still 0 %); measured drive (150-step episodes, 6 teacher batches) teacher 80 %, student
   0 → 0 (seal 68 %) → 0 → **14.5 %** — the first non-zero clone on the measured drive. The
   student-driven relabel batches (β = 0.1) succeed 34–35 % on the measured drive, so the
   student is close; the deterministic replay is what lags. PPO from the round-3 checkpoints:
   `paper13_ideal` 28 → 39 → 46 → 52 % in the first 2M steps (starts where `paper10` peaked);
   `paper13_real` from `bc_iter3.pt`, in progress.
14. Ideal-drive champion so far: `paper12_ideal` **83.1 %** at 9.8M steps with no late decay
   (entropy 0.0015, no value clip, vf_coef 0.5; `rl/weights/paper12_ideal_best.pt`, 8/8 replay).
15. **DAgger alone solves the measured drive (2026-09-19, `dagger5_real`, resumed from `dagger4_real` with
   `bc_curobo.py --resume`, beta 0 relabels)**: rounds 4-7 give 12 -> 41 -> 36 -> **84.8 %** deterministic
   student success (83-86 % on 1024 fresh episodes, `rl/weights/dagger5_real_iter7.pt`), teacher 80 %.
   PPO on top of the DAgger init never got past 24 % on this drive (paper13/14_real) and decayed even
   with the demonstration anchor (`--bc_data`, `--critic_warmup`; the anchor did hold the ideal run
   at 41-52 % instead of decaying to 40 %, but far below paper12's 83 %). Rounds 8-10 (`dagger6_real`):
   62 -> 70 -> **87.8 %** (1024 episodes; round 7 84.4 % on the same episodes) -- the rounds are noisy,
   keep the best-scoring one (`rl/weights/dagger6_real_iter10.pt`, the controller default).
16. **The BC policy needs the training observation noise at test time**: nominal sim, no noise 31 %;
   with the env's 0.005 Gaussian obs noise 78 %; full DR 85 %. Deterministic observations let the
   clone stall at a fixed point (hover/press). `real_policy_ctrl.py` adds the noise (`--obs_noise`).
17. **Deployment path** (`rl/export_trt.py`, `rl/real_policy_ctrl.py`): checkpoint -> ONNX -> TensorRT
   engine (max |trt - torch| 3e-5, 120-170 us/call on the A5000), observation rebuilt on the desktop
   exactly like `PaperPickEnv.observe()` (max diff 3e-7 on the first step, 2e-3 over an episode from
   float32 FK), closed loop through the engine reproduces the direct evaluation (80.9 % vs 78 %).
   Robot side: 10 Hz chunks into `robot_hal`'s stream welder (K0 20, K1 0.3, vff 1), reference lead
   bounded to 3 deg, torque contact guard (contact_detector thresholds), attach emulation (no object
   tracking), suction via a new `{"suction": 0/1}` robot_hal command.
18. **On the real Pro 630 (2026-09-20)**: run 1 crawled at 5.9 deg/s (the Pi's `vel_cmd_max` is in post-`vel_scale`
   units; 850 = 50 deg/s) and timed out 0.2 cm from the grasp point. Run 2 (`--touch_only --dq_max 1.0`, 10 deg/s):
   the DAgger policy brought the cup to 0.3 cm of the grasp point at 10.0 s, firm torque contact at 11.0 s, lead over
   the arm <= 1.1 deg the whole way, then retract / place pose / home (`docs/real_touch_demo_0920.png`). Scene from
   the D435 scan: table z ~ 0 (training height; config.TABLE_Z is stale), object top 0.047 (camera) / 0.058 (FK at
   contact). Run 3 at the full training speed (dq_max 2.0): touch at 5.1 s, the simulator's timing, peak joint
   speed 31 deg/s (`docs/real_touch_half_vs_full_0920.png`). Suction is not activated in demos by request.
19. **Session 2 on the robot (2026-09-20 pm)**: three more touch-only demos, including a 10 cm-tall object
   (top outside the trained 0.02-0.07 range, touched at 3.9 s) and an object at x = 0.26 near the base (6.1 s):
   the policy generalises in height and position without retraining. Two infrastructure facts: (a) the Pi's
   :9999 feedback broadcaster silently stopped while :9998 kept answering -- the controller refused to run
   (correct); a clean `launch_mpc_stack.sh` relaunch fixed it, cause not found in the log. (b) After that
   relaunch **joint 6 (wrist yaw, HAL pin `pro600.joint5_*`, 0-based) ignores commands**: 2 deg probe = no
   motion, status word 0x8637 (was 0x8237), joint 1 0x9637. Touch demos still work (cup symmetric), but treat
   as the August "deaf drive" signature on one joint: power-cycle, probe joint by joint, then CAN wiring.
20. **First suction pick-and-place series on the robot (2026-09-20 evening, residual policy, live colour tracker)**:
   10 runs, objects changed by hand between runs -> **5 full pick-and-place** (demos 3, 4, 5, 9, 10: placed 1.0-5.5 cm
   from the goal). Failures: 1-2 controller shakedown (press too hard -> hard-stop; tracker estimate walking as the cup
   entered the blob), 6-7 an object with a sloped top (no seal possible), 8 a 2 cm thin object that pressed fine but never
   sealed. Controller rules that came out of it (`rl/real_policy_ctrl.py`): soft landing in the last 3 cm; hold the
   reference at firm contact and never press past the hard level before the attach; attach = 1.0 s of contact with
   suction on, by torque OR by position (tip at the grasp height) OR lift-after-press, with the press dwell enforced;
   rolling torque baseline (gravity torque at an extended reach looked like contact); tracking frozen once the cup is
   within 15 cm vertically (the arm's shadow shifts the blob); online object-height adaptation + a z-shift of the
   observation for objects lower than the trained 4 cm (the policy descends to the ABSOLUTE trained grasp height);
   gentle place-down at the goal, tracker-verified result, always retract+home. `rl/rgb_track.py --mode diff` (Lab
   distance from the table colour) handles grey/light objects the depth and dark-blob modes lost. The Pi's :9999
   broadcaster died silently twice today (command port fine) -> clean stack relaunch each time; cause still unknown.
21. **Series 2 (2026-09-20 night, random place positions, goal 22 cm)**: 9 runs -> 7 carries (5 placed within 3.5 cm,
   2 set down ~6 cm off, 1 tipped on release before the guarded settle existed), 2 failures on an open-top cup (no
   seal possible). The 2 cm thin object sealed 1 of 8 times: below the cup's working range. Controller additions this
   series: guarded press primitive (vertical 1 cm/s until the contact metric reaches 0.11, max 15 mm below the
   calibrated top, 1 s dwell); touch height calibration before each run (camera tops were 1-6 cm low on cups);
   tracker bias measured with the cylinder centred under the cup (+0.7, -16.9 mm) and subtracted; flat "hold"
   chunks (a held reference built from past targets kept creeping); scripted 8 cm lift right after the attach
   (the policy's post-attach press is unreliable off its trained height); guarded settle before release (a can
   tipped from a 4 mm drop); stop 1 s after the object is at the goal; carry-phase action filter (the 5 Hz dither).
   Camera-based success verdict is unreliable with more than one object in view: judged by eye where it disagreed.
   Pi :9999 broadcaster died a third time (relaunch).
22. **Twin caught up with the deployed controller, and a faster residual on top (2026-09-20 night)**.
   *Twin change (`rl/env_warp.py`)*: the Pi's streaming law changed on 2026-09-20 (`real_policy_ctrl.STREAM_GAINS`,
   `mycobot_mpc/robot_hal.run_stream_loop` + `spline_ref.py`) and the twin still ran the old one. Now modelled
   exactly: **K0 20 -> 10**, a **0.045 s reference lead** (`u = vff*v_ref(t+lead) - K0*(q - q_ref(t+lead)) -
   K1*(qd - v_ref(t+lead))`), and a **uniform cubic B-spline reference** instead of the per-decision linear ramp --
   control polygon = the last 4 streamed targets + one extrapolated point `q_k + (q_k - q_{k-1})`, newest target at
   knot `t_dec + dt`, so the anchor is `t_dec - 2dt` and tick c samples `u = 2 + 0.1c + lead/dt`; same basis and
   end-clamping as `spline_ref._basis/_eval/_p`, analytic velocity (`PickEnv._spline_ref`). The twin also now applies
   the controller's **bounded lead** (`q_send = q + clip(q_virtual - q, +-3 deg)`, `LEAD_MAX_DEG`) -- the policy's own
   `q_target` keeps integrating, as on the robot. New state: `q_hist` (N,4,6), reset wherever `q_target_prev` is.
   `drive_ref="linear"` + `lead_clip=False` reproduce the pre-0920 twin (`drive_probe.py` pins them, since its
   hardware reference numbers are from 2026-09-17).
   *Validation* (`rl/validate_drive_update.py`, GPU 1, deployed `resid1_real_best.pt` run deterministically with
   obs noise 0.005 and no other DR, 256 episodes, ep_len 150, dq_max 2):

   | twin | success | seal | peak abs qd med / p90 / max (deg/s) | first seal (dec) |
   |---|---|---|---|---|
   | old: K0 20, linear, no lead clip | 97.3 % | 98.1 % | **64.7** / 65.6 / 66.0 | 51 (5.1 s) |
   | K0 10, linear, no lead clip | 97.3 % | 97.7 % | 34.8 / 36.7 / 48.4 | 51 (5.1 s) |
   | K0 10, spline, no lead clip | 97.7 % | 98.1 % | 41.1 / 43.5 / 51.1 | 50 (5.0 s) |
   | **new (deployed): K0 10, spline, lead clip 3 deg** | 97.3 % | 98.1 % | **41.1** / 43.4 / 53.7 | 51 (5.1 s) |

   Real robot: peak 22-34 deg/s (median ~27; run 3 of item 18, dq_max 2, measured 31), first contact 5.1 s on that
   same run. **The twin's timing now matches the robot exactly and the speed error fell from +135 % to +25 %.**
   Almost all of the old gap was K0 (65 -> 35 deg/s); the lead puts ~6 deg/s back, which is what it is *for* (it
   buys tracking accuracy, and the robot A/B confirmed lower ripple with it). The lead clip does NOT explain the
   remaining 41 vs 27-34 -- it changes the median by 0.0 -- so it was kept only because the controller does it, not
   as a fit. The residue is most plausibly the drive's ~36 deg/s firmware following-error ceiling, which the twin
   models as a 50 deg/s saturation (the Pi's `vel_cmd_max`); the real arm simply cannot be commanded past ~36
   without faulting, so no real trace can show 41. That is a *hardware protection*, not a controller term, so it
   was left alone and the speed penalty (`--w_speed`) is what keeps the learnt motion under it.
   *Residual for speed* (`~/pnp_rl/resid3_fast`, 8M steps, ~65 min on the A5000): three new mechanisms.
   (a) `--base_scale s` with `--dq_max 3`: executed = `clamp(s*base(obs_b) + bound*tanh(res(obs)), -1, 1)`, so the
   frozen base still moves its trained 2 deg/decision while the residual may add up to `bound*3` deg. `obs_b` is the
   observation with the six a_prev JOINT columns divided by s (the base reads a_prev in ITS action units).
   **`s` must multiply the six joint channels only**: scaling the suction logit too made `a_prev[24]` inconsistent
   with `obs_b` and dropped the zero-residual identity check from 95.9 % to 1.6 % success. With the fix the identity
   is exact -- 2.3e-5 deg per decision on random observations, and 95.90 % vs 95.90 % over 512 DR episodes.
   (b) `--residual_base` now accepts a residual checkpoint and builds the frozen base **recursively**
   (`ppo.build_frozen_policy` -> `FusedResidual`), so this run starts from resid1 (94.7 %) instead of the 87.9 %
   DAgger clone, and `export_trt` writes the whole 3-level stack as one ONNX graph.
   (c) `--w_time w`: `-w/CTRL_HZ` per decision until the object is lifted AND within `succ_tol` of the goal
   (component `time`, plus `t_seal`/`t_goal` in `info` and in the training log). Everything else in the paper reward
   is a *rate*, so finishing early paid almost nothing; this makes cycle time an explicit objective.
   Training (`--w_time 0.5 --w_speed 1.0 --v_soft 32 --critic_priv --critic_warmup 8`, rest = resid1's args):
   success 87.9 -> 89.6 % (peak, 5.9M) -> 88.6 %, `t_seal` 73.6 -> 69.1 dec, `t_goal` 98.1 -> 95.4 dec, mean peak
   qd flat at 44 deg/s, `res_mag` 0.001 -> 0.012 (the residual stays small).
   *Paired deterministic eval* (`rl/eval_residual.py`, GPU 1, 1024 episodes, DR, ep_len 150, same seed):

   | policy | dq_max | success | seal | t_seal | t_goal | peak abs qd med / p90 / max |
   |---|---|---|---|---|---|---|
   | base = resid1 | 2 deg | 95.31 % | 95.90 % | 5.71 s | 8.15 s | 38.5 / 43.4 / 55.2 |
   | **resid3 best (5.9M)** | 3 deg | **96.39 %** | 97.17 % | **5.26 s** | **7.79 s** | 38.3 / 43.1 / 54.1 |
   | resid3 final (8.06M) | 3 deg | 96.19 % | 96.88 % | 5.21 s | 7.75 s | 38.2 / 43.3 / 54.3 |

   **-7.9 % time to the seal and -4.4 % time to the goal at equal peak joint speed and +1.1 pp success.** The gain
   is real but modest: the residual is bounded at 0.3 and the base is what sets the trajectory shape, so the speed
   budget mostly goes into the approach. A bigger win needs a faster BASE (retrain the DAgger teacher at dq_max 3),
   not a bigger residual.
   *Logging fix found on the way*: with `--ep_len 150` and `--rollout 24` all worlds time out in lock-step, so one
   logging window in five contains no episode boundary and averaged only the handful of worlds desynced by an
   off-table failure -- a 3-episode 100 % froze `best.pt` at a noise peak. `ppo.py` now carries the accumulators
   into the next window unless `>= nworld/4` episodes finished, and records `n_ep`.
   *Commands*:

       CUDA_VISIBLE_DEVICES=1 $PY rl/validate_drive_update.py --nworld 256
       CUDA_VISIBLE_DEVICES=0 nohup $PY rl/ppo.py --nworld 4096 --steps 8000000 --rollout 24 --epochs 3 \
         --minibatch 24576 --gamma 0.98 --lam 0.95 --clip 0.1 --ent 0.0015 --lr 5e-5 --mode pnp --dr \
         --target_max 0.3 --out ~/pnp_rl/resid3_fast --scene rl/scenes/box_med.xml --drive real --dq_max 3.0 \
         --obs_lag -1 --hover 0.02 0.04 --init_std -1.0 --descend_sigma 0.07 --env paper --arch paper \
         --max_grad_norm 1.0 --critic_warmup 8 --lr_max 1e-3 --grasp_shaping 1 --obs_ee 1 --reach_target grasp \
         --lift_dense 1 --w_track_c 4 --w_track_f 8 --w_reach 0.5 --vf_coef 0.5 --reg_ramp 0.4 --ep_len 150 \
         --start home --residual_base ~/pnp_rl/resid1_real/best.pt --residual_bound 0.3 --base_scale 0.6667 \
         --w_time 0.5 --w_speed 1.0 --v_soft 32 --critic_priv > ~/pnp_rl/resid3_fast/run.out 2>&1 &
       CUDA_VISIBLE_DEVICES=1 $PY rl/eval_residual.py ~/pnp_rl/resid3_fast/best.pt ~/pnp_rl/resid3_fast/final.pt \
         --nworld 1024 --ep_len 150 --dq_max 3 --base_dq_max 2
       cp ~/pnp_rl/resid3_fast/best.pt rl/weights/resid3_fast_best.pt
       CUDA_VISIBLE_DEVICES=0 $PY rl/export_trt.py rl/weights/resid3_fast_best.pt --obs_dim 40 --arch paper \
         --out rl/weights/resid3_fast_best          # max|trt-torch| 2.38e-05, 208 us/call

   **Deploying this policy requires `rl/real_policy_ctrl.py DQ_MAX_DEG = 3.0`** (it is still 2.0): the exported
   graph outputs joint deltas in units of dq_max 3.
8. Planner throughput is the DAgger bottleneck (one server, ~40 % of hover goals rejected near
   the wall keep-out / camera mount → IK fallback). Table slab for the planner must clear the
   robot base (a slab through the base = every plan "no solution").

## Plan from here (2026-09-17)

Ordered by expected payoff; each step is a from-scratch or warm-chain run in the
measured dynamics, and each is gated by the previous.

1. **Re-establish the proven chain in the real drive**: `attach` from scratch (running:
   `~/pnp_rl/rd_attach_real`, control `rd_attach_ideal`), then `pnp` warm-started from it
   with the ppo4/5 spec (table targets, `--target_max 0.3`), then the ppo7 spec
   (`--lift_req 0.35 --speed_bonus 0.3`). Success criterion: ≥ 95 % `pnp` at the ppo5
   spec under the new dynamics. If the real-drive `attach` learns slower than the ideal
   one, the lag observation is doing its job; if it does *not* learn, suspect the
   dead-time (raise `--dq_max` so 2°-steps stop hiding inside the lag).
2. **Agile variant**: `--dq_max 4` (40 °/s command envelope, the arm tracks ~45) with
   `speed_bonus` — the hardware now settles a 10° move in 0.43 s, so the 3.8 s episodes
   of ppo6c have ~2× headroom. Watch the `sat` component: > 0.5/episode means the policy
   is riding the drive's saturation and the real arm will lag it.
3. **Contact-release / tilt / verticality spec** (open since run 8): do NOT resume the
   warm-chain repairs. Rerun the `diag_factors` audit on the new `attach → pnp` champion,
   then a single `--mode mix --release_mask` run from that checkpoint with the graded
   terminals as they are. The dynamics change alters the release timing (the drive now
   takes ~100 ms to stop), so `mask_h` annealing should start at 0.03, not 0.008.
4. **Sim-to-real check before distillation**: replay the champion's joint commands on
   the real arm through `ctrl_tuner /api/stream_traj` (no suction) and compare the
   measured joint trace with the sim rollout — the `drive_probe` numbers say they should
   agree to ~0.3°; a larger gap means a missing dynamics term (gravity droop of the
   real drive under load, cable drag) before any camera policy is trained.
5. **Distill** the champion to RGBD (`rl/distill.py`) only after step 4 passes.

Reward terms to leave alone: `place` (graded terminal), the max-lift potential, the
Laplace grades, `rel_far`, `chatter` — every one of them was re-derived from a failure
(see the rules above). The two new terms (`sat`, `smooth`) are shaping costs on the
scale of `act`; if a run's `sat` sum exceeds ~1/episode, lower `--dq_max` rather than
raising the weight.


23. **Cycle-time pass on the scripted phases (2026-09-20, `rl/real_policy_ctrl.py` only)**: in series 2 a run took
   35-45 s of which the policy itself is ~8 s; everything else was scripted legs run as constant-velocity ramps at
   `max_deg / 6-10 deg/s` followed by a blind `time.sleep(T + 1.2...1.6)` pad. Replaced by one helper,
   `scripted_move(link, guard, ob, q_to, label, v_peak_deg, min_T)`, used by **every** scripted leg (go_home, the three
   calibration legs, retract-to-hover, the post-attach lift, the touch-only retract/place/home, the failed-run
   retract/home, the place-down, the post-release retract/home):
   * **min-jerk time scaling** s(tau) = 10tau^3 - 15tau^4 + 6tau^5 from the *measured* q to q_to, sampled at 0.1 s into
     the same `send_path` B-spline. Zero velocity *and* zero acceleration at both ends, peak = 1.875 x average, so
     T = max(min_T, 1.875 * max_deg / v_peak). `--script_vmax` (default **25 deg/s peak** = 13.3 deg/s average; the
     CLI refuses > 34) replaces the 5-10 deg/s ramps; legs that carry the object or approach the table (post-attach
     lift, place-down) stay at 12 deg/s. Measured on the generated paths: peak 24.8 deg/s for a 45 deg leg, i.e. well
     under the ~36 deg/s following-error fault and under the 22-34 deg/s the policy itself peaked at in series 2.
   * **wait on feedback, not on a pad**: after the path's own duration, poll `link.state()` at 20 Hz until
     max|q - q_to| < 0.7 deg on 3 consecutive polls; hard timeout T + 2.0 s prints `[warn] ... not settled` with the
     residual and continues. Skipped when nothing is being sent (dry run, `SimLink`), so `--selftest` cannot hang.
   Other constants (old -> new): `ATTACH_AFTER` 1.0 -> **0.5** s; guarded press descent 1 -> **2 mm/decision**
   (2 cm/s, same TAU_PRESS 0.11 and same 15 mm floor); press fallback dwell `ATTACH_AFTER + 1.0` -> **+0.5**;
   guarded settle 1 -> **2 mm/decision** (max still 20 mm); post-settle pause 0.3 -> **0.15** s; post-release pause
   0.6 -> **0.3** s; at-goal dwell before the place 1.0 -> **0.5** s (`AT_GOAL_DWELL`). The touch calibration keeps the
   7 cm hover (camera tops are up to 6 cm low) but descends **4 cm/s down to 2 cm above the camera top, then 2 cm/s**;
   because the top can be several cm low the object may already be inside the fast stretch, so any torque rise above
   0.4 x TAU_FIRM drops the descent to 2 cm/s before the firm threshold. Soft-landing factor (0.6 in the last 3 cm),
   guard semantics, hold chunks, lead and gains are unchanged.
   Expected per-leg budget (joint distances of a typical run; "old" = ramp + fixed pad, "new" = min-jerk + ~0.4 s
   feedback confirm):

   | leg | max joint | old | new | saved |
   |---|---|---|---|---|
   | go_home at the start | 45 deg | 10.6 s | 3.8 s | 6.8 s |
   | calib home -> hover 7 cm | 45 deg | 7.0 s | 3.8 s | 3.2 s |
   | calib retract 6 cm | 10 deg | 2.6 s | 1.2 s | 1.4 s |
   | calib hover -> home | 45 deg | 7.0 s | 3.8 s | 3.2 s |
   | lift 8 cm after the attach (12 deg/s) | 12 deg | 2.7 s | 2.3 s | 0.4 s |
   | place-down ~16 cm (12 deg/s) | 25 deg | 5.6 s | 4.3 s | 1.3 s |
   | retract 8 cm after release | 10 deg | 3.1 s | 1.2 s | 1.9 s |
   | home after release | 45 deg | 8.9 s | 3.8 s | 5.1 s |
   | **scripted legs** | | **47.5 s** | **24.2 s** | **23.3 s** |
   | calib descent 7 cm | | 3.5 s | 2.2 s | 1.2 s |
   | press descent + dwell | | 3.5 s | 1.7 s | 1.8 s |
   | guarded settle (20 mm) | | 2.0 s | 1.0 s | 1.0 s |
   | settle/release pauses | | 0.9 s | 0.5 s | 0.5 s |
   | carry (at-goal dwell) | | 4.0 s | 3.5 s | 0.5 s |
   | policy approach (unchanged) | | 3.3 s | 3.3 s | 0 |

   Legs shorter than assumed save less, but the pad removal alone is 1.0-1.2 s x 8 legs ~ 9 s per run independent of
   distance; a 35-45 s run should land around **20-25 s** with the policy's ~8 s untouched.
   New output: a `[timeline]` line (calib / approach / press+attach / lift / carry / place+settle / release+retract+home
   / total) printed at the end of every run and on SIGINT, and written to `<--log>.phases.json` (the log itself is a
   list of rows, so it cannot hold a header dict).
   **Verification** (no `--exec`, no command ever sent to the robot): `--selftest --episodes 16` on the measured drive
   still passes -- obs vs `env.observe()` 2.9e-07 first step / 1.2e-03 over the episode, success 100 %, sealed 100 %.
   `python -m py_compile` clean. The dry run against the real Pi could **not** be completed: the Pi's :9999 feedback
   broadcaster is dead for the fourth time (`:9998` still answers `state: done` with the parked pose, `:9999` accepts
   the connection and then sends nothing), and relaunching the stack was out of scope with nobody at the robot. The
   dry run was instead done against a local stand-in serving the same two sockets with the arm's parked pose: it runs
   through go_home / calib hover+descent+home / episode / retract / home, prints `[dry run]` on each leg and the
   timeline, writes the `.phases.json`, exits 0, and the stand-in logged **zero** commands received (dry run really is
   read-only). `scripted_move` itself was unit-tested against a fake link with `exec=True`: settle path returns
   0.00 deg residual ~0.3 s after arrival, the never-arrives case warns at T + 2.0 s, a guard violation prints the
   same message and stops the leg, and the sampled peak speeds are 24.8 deg/s (25 requested) and 11.8 deg/s (12).
   Still unverified on hardware: the real settle time of a 25 deg/s min-jerk leg (does the 0.7 deg / 3-poll criterion
   fire before the T + 2.0 s timeout?), whether 0.5 s of vacuum dwell is enough on the cups used in series 2, and the
   4 cm/s calibration approach against a camera top that is several cm low.

24. **Series 3 (2026-09-20 night, six objects on the table, random object AND random goal, `rl/run_pick_rand.sh`)**: 6 runs
    with the committed controller -> 2 placed (2-4 cm), 4 failures, every one diagnosed: (a) the colour tracker's `diff`
    mode merged neighbouring objects and their shadows into one blob (13.7k px) so the cup landed between two cans, and
    once the D435 white balance drifted half the table failed the colour test; (b) the controller's live tracking followed
    the tracker's LARGEST blob, which switched the target to a bottle 8 cm away mid-approach (s3_04: pressed the bottle at
    the elbow limit until timeout, retract refused outside the guard box, recovered by a scripted vertical retract); (c) the
    red bottle's sloped cap cannot be sealed (2 of 6 random draws); (d) the cup attached 8-13 mm off-centre (+y) on the runs
    that did seal. Fixes: `rgb_track.py --mode depth` (default): objects = depth pixels standing > 1 cm above the table plane
    (table offset = median height over the ROI, 3-frame depth median), split into 2 cm-grid xy clusters, shadows excluded,
    up to 8 candidates published; the controller follows the candidate NEAREST its current estimate; the guarded press
    descends at the ESTIMATED centre instead of the touchdown point. Calibration check with the user watching: the cup
    hovered over the tall can at the corrected D435 estimate within ~1 cm, so the extrinsics + 17 mm bias stand; a
    torque-guarded rim-probing script (`rl/edge_calib.py`, in the session worktree) pushed the can 4 mm and gave
    inconsistent rims - not usable as is. Two-camera success monitor (`rl/success_monitor.py`, fixed D405 + wrist D405,
    before/after change detection, udp :9702) correctly flagged the two "nothing moved" runs the D435 verdict had called
    "pushed"; its fixed-cam 3-D localisation still needs the depth-first treatment with many objects. Pi :9999 broadcaster
    died a 4th time (relaunched). README on GitHub main documents the pipeline + curriculum (73e48bc).

25. **Q-Planning on the twin (2026-09-21, `rl/qplan/`, M0-M2 of `policy/PLAN_QPLANNING.md`)**: frozen
    `resid3_fast_best` + an off-policy Q over 5-decision action chunks (HL-Gauss, 51 bins, two heads,
    EMA target tau 0.005, H-step bootstrap, gamma 0.99), N chunk proposals re-ranked at 10 Hz. Full
    numbers, tables and commands in `rl/qplan/README.md`; the load-bearing findings:
    * **M0 throughput** 3 251 chunked transitions/s at 4096 worlds on the 2080 Ti = **200 k in 62 s**
      (target was 10 min). Episodes are stored as TRAJECTORIES (14.6 kB each in fp16); a 4096-world
      batch is 59 MB and carries 614 400 overlapping chunked transitions.
    * **Q_time became a copy of Q_succ** when the buffer was pi + open-loop proposals only: every
      deviation in that buffer is a deviation that FAILS (18 % success), so "chunk deviates" and
      "episode is slow" are the same feature, Q_time ranked pi's own chunk fastest everywhere, and
      the planner moved `t_goal` by **0.00 s**. Fixed by a new collector mode (`--mode scale`: a
      blanket joint scale in [0.6, 1.6] held for an episode or a 10-15 decision segment, 42 %
      success) -- episodes that still mostly succeed but finish at measurably different times. The
      scale ladder then has a real interior optimum, **1.15 x pi before the seal and 1.30 x after**,
      and post-seal Q_succ is monotone increasing in the scale up to 1.7.
    * **The proposal set must keep the speed axis clean.** The plan scales "the 16's first four"
      gaussian candidates by 0.7/1.2/1.4, which entangles speed with noise; replacing those twelve
      slots with a PURE ladder of pi's own chunk (0.7/0.85/1.15/1.3/1.5/1.7) moved `t_goal`
      **-0.76 s** the same day. `N = 16` beats `N = 32` for the same reason (at 16 the structured
      slots ARE the ladder). Best planner: **N=16, lambda=0.1, succ_frac=0.9 -> 99.02 % vs pi's
      97.17 % (+1.85 pp), t_goal 7.14 s vs 7.73 s (-0.59 s)**, reproduced on a second seed.
      Tightening the success band HURTS (0.97 -> 95.2 %): with 2-3 survivors the softmax is forced
      to trust Q_succ differences below the critic's own error.
    * **Planning quality is an inverted U in the number of TD steps while calibration rises
      monotonically.** Same fixed buffer, 2.5k/5k/10k/20k/40k/60k steps -> 26.8/90.0/95.1/**98.8**/
      97.6/95.3 % success, while the held-out Q_succ correlation goes 0.810 -> 0.835 throughout.
      This is the whole M2 story: the plan's loop warm-starts and adds 10 k steps per iteration, so
      iteration 10 is 120 k steps, and success decayed 98.4 -> 96.6 % and `t_goal` 7.59 -> 8.84 s
      across ten iterations **while the critic's calibration bias went to zero**. The useful signal
      for RANKING chunks is an early-training artefact of the value function, not its fixed point;
      select the critic by the paired eval (`rl/qplan/steps_curve.py`), never by TD loss.
    * **M2 plateaus at iteration 0 and the gate fails.** Three loops all peak at or below the offline
      critic (best iteration, success vs its own paired pi / t_goal vs pi / peak qd p90):
      **iteration 0** 98.83 % (+1.76 pp) / 7.20 s (-0.58 s) / 48.1; **v1** (plan recipe, 10 k warm
      steps per iteration) iter 2, 98.73 % (+1.76) / 7.76 s (-0.03) / 51.6, then a monotone decay to
      96.58 % / 8.84 s by iteration 10; **v3** (2.5 k steps at lr 1e-4, >=35 % fixed-pool batches,
      velocity cap, early stop) iter 4, 96.97 % (+0.20) / 8.35 s (+0.57 WORSE) / 47.9; **v5**
      (retrain from scratch, 20 k steps per iteration) iter 3, 98.73 % (+2.15) / 7.93 s (-0.14) /
      52.2, sequence 97.66/98.54/**98.73**/98.05/98.05/98.14 %. Retraining from scratch removes the
      decay but not the flatness: 20 k from scratch on M0 alone gives +1.76 pp, on M0 + four planner
      deployments +1.56 pp -- **the online episodes carry no new information**. Cause, with evidence:
      every non-placed episode of pi is a non-SEALED one (seal 97.3 %, success 96.9 %), and a seal the
      drive DR makes geometrically impossible is not recoverable by re-weighting chunks of the same
      policy. The paper's own stated limit, proposal support, is what binds -- every proposal is a
      scaled or jittered pi chunk, so the reachable set is a tube around pi.
    * **The peak-speed criterion and the cycle-time criterion are mutually exclusive here.** Two
      independent knobs trace the same front: lambda 0.05 -> 1.0 gives `t_goal` 7.14 -> 7.75 s and
      peak |qd| p90 48.5 -> 46.9, and capping the executed chunk's peak commanded speed at
      1.05/1.15/1.30 x pi's gives 8.13/7.65/7.40 s at 43.9/46.2/47.7 deg/s. p90 never goes below
      ~47 even when the planner is SLOWER than pi, because the executed chunk is a weighted MIXTURE
      that changes more between decisions than pi's own smooth output and the accel-capped drive
      turns that into peak speed. Nothing in Q prices joint speed -- the fix is a THIRD HL-Gauss head
      on a per-decision speed-excess reward (the Q analogue of `--w_speed`), not tuning. A hard
      velocity MASK on candidates is the wrong shape of fix: candidates are clamped to [-1,1], so
      where pi saturates the fast ladder entries equal pi and survive while where pi is slow they are
      dropped -- the survivor set is biased slow and `t_goal` got WORSE than pi (8.39 s). Cap the
      executed chunk after the mixing instead.
    * **Twin peak |qd| over-reads the robot by ~25 %** (FINDINGS 22: 43.6 deg/s p90 here, 22-34 on
      the arm), so the twin-side speed criterion is relative (planner <= pi p90 + 2 deg/s), reported
      alongside the absolute 36 deg/s firmware ceiling.
    * **NEW tracker-error DR** (`env_paper.PaperPickEnv(obs_obj_err=True)`, OFF by default): +-15 mm
      xy bias, +-5 mm jitter, -6..+2 cm top error on the OBSERVED object only. At the plan's
      magnitudes it is a different task, not a perturbation: pi 96.7 % -> **18.8 %** (xy only 62.3 %,
      top only 29.5 %, half magnitudes 52.0 %). The twin's seal test needs the cup tip within 12 mm
      (DR'd 9.6-14.4) of the object-top centre, so a +-15 mm bias makes a seal geometrically
      impossible on a large share of episodes, so it is not part of the gated protocol. **Training
      the critic on tracker-DR data is worse than useless**: the bias is unobservable from o_t, so
      the same observation carries wildly different outcomes, the head absorbs the variance
      (predicted success at t=0 **-40 %** against a realised 12.9 %) and the planner it drives falls
      BELOW pi (17.3 % -> 10.2 %). The CLEAN critic `q0b`, which scores the observed state as if it
      were true, still helps: pi 16.7 % -> planner 19.8 % (+3.1 pp) -> best-of-N 22.7 % (+6.0 pp),
      aggressive selection paying here because pi's approach is what is failing. So Q-planning is
      not helpless under an unobservable state error -- but the data containing the error is poison
      for the critic, and the factor really needs an observable cue (multi-frame tracker
      disagreement, force feedback, a probing motion).
    * **Scripted PLACE phase in the twin** (`place_phase=True`, OFF by default, ep_len 180): after
      `at_goal` holds 5 decisions the env descends the tcp straight down at the CURRENT xy
      (damped-least-squares on a finite-difference site Jacobian, 1.5 cm/decision, <= 1.2 deg on the
      largest joint), releases when the object bottom is within 4 mm of the table or stops
      descending, settles 5 decisions; `placed` = released AND resting within 3.5 cm of the goal XY.
      Two implementation lessons, both measured: a **per-joint** clamp on the IK step rotates the
      Cartesian direction (61 % placed, 3.0 cm median error) -- scale the whole `dq` uniformly; and
      the **orientation rows of the Jacobian are load-bearing** -- with position only the null space
      rotates the wrist and the object, welded a cup radius below the tcp, swings out (38 % placed,
      4.4 cm error) versus 85.6 % and 1.9 cm with the full 6-D Jacobian. pi places 82.9-86.6 %
      (+-1.9 pp run-to-run: a 1.9 cm median error against a 3.5 cm threshold puts many episodes on
      the boundary). Across TD budgets 20 k/40 k/80 k the planner scores -4.5/+3.3/-1.2 pp on
      PLACEMENTS -- inside the noise -- but lifts the SEAL rate 97.3 -> 99.1-99.2 % well outside it,
      and does not move `t_placed` at all (the descent is fixed-rate and starts after the dwell).
      best-of-N drops 20 pp: a hard argmax on Q_time at the moment of arrival trades away exactly the
      lateral precision the release needs. With the real end of the cycle in the loop the binding
      constraint is the lateral accuracy of the arrival, not chunk timing.
    *Commands* (GPU 1 unless noted; nothing here ever touches the robot):

        PY=~/miniconda3/envs/mjwarp/bin/python
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode pi      --tag m0  --seed 0
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode explore --p_explore 0.15 --tag m0 --seed 5
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode scale --scale_lo 0.6 --scale_hi 1.6 --p_explore 0.08 --tag m0 --seed 30
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode scale --scale_lo 0.8 --scale_hi 1.5 --seg 15 --p_explore 0.08 --tag m0b --seed 40
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/critic.py  --data ~/pnp_rl/qplan/data --steps 20000 --batch 4096 --out ~/pnp_rl/qplan/q0b
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/planner.py --eval --q ~/pnp_rl/qplan/q0b/q.pt --nworld 1024 --which full   --out ~/pnp_rl/qplan/m1_ablation_it3.json
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/planner.py --eval --q ~/pnp_rl/qplan/q0b/q.pt --nworld 1024 --which sweep  --out ~/pnp_rl/qplan/m1_sweep.json
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/planner.py --eval --q ~/pnp_rl/qplan/q0b/q.pt --nworld 1024 --which pareto --seed 1 --out ~/pnp_rl/qplan/m1_pareto.json
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/planner.py --eval --q ~/pnp_rl/qplan/q0b/q.pt --nworld 1024 --which velcap --out ~/pnp_rl/qplan/m1_velcap.json
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/diag.py       --q ~/pnp_rl/qplan/q0b/q.pt --nworld 512
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/steps_curve.py --data ~/pnp_rl/qplan/data --steps 2500 5000 10000 20000 40000 --nworld 1024
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/iterate.py --iters 10 --q ~/pnp_rl/qplan/q0b/q.pt --data ~/pnp_rl/qplan/data \
            --q_steps 10000 --n_cand 16 --lam 0.1 --vel_margin 0 --out ~/pnp_rl/qplan            # v1, the plan as written
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/iterate.py --iters 10 --q ~/pnp_rl/qplan/q0b/q.pt --data ~/pnp_rl/qplan/data \
            --fixed_data ~/pnp_rl/qplan/data_objerr --q_steps 2500 --fixed_frac 0.35 --lr 1e-4 --patience 2 \
            --vel_margin 1.2 --vel_mode cap --n_cand 16 --lam 0.1 --out ~/pnp_rl/qplan/v3        # corrected
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/iterate.py --iters 6 --q ~/pnp_rl/qplan/q0b/q.pt --data ~/pnp_rl/qplan/data \
            --q_steps 20000 --fresh --fixed_frac 0.35 --patience 3 --vel_margin 0 --n_cand 16 --lam 0.1 --out ~/pnp_rl/qplan/v5
        # tracker-error DR and the scripted place phase
        CUDA_VISIBLE_DEVICES=0 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode pi --obj_err 1 --tag oe --seed 20 --out ~/pnp_rl/qplan/data_objerr
        CUDA_VISIBLE_DEVICES=0 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --ep_len 180 --place_phase 1 --mode pi --tag p0 --seed 0 --out ~/pnp_rl/qplan/data_place
        CUDA_VISIBLE_DEVICES=0 $PY rl/qplan/steps_curve.py --data ~/pnp_rl/qplan/data_place --steps 10000 20000 40000 --nworld 1024 --ep_len 180 --place_phase 1
        CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/plot_final.py                                        # -> ~/pnp_rl/qplan/iterations.png

26. **v3: obstacles, an enlarged workspace and the suction bug that was capping env_v2 (2026-09-21, `rl/env_v3.py`,
    branch `v3-obstacles`)**.  `rl/env_v3.ObstacleEnv(env_v2.DiverseEnv)` adds, all default-off or explicit:
    spawn **x 0.20-0.56, y +-0.35** (v2: 0-0.5 / +-0.3), v3 goals (x 0.18-0.52, |y| <= 0.33, so >= 0.10 m from
    every wall, and a MINIMUM carry distance of 0.20 m so a corridor obstacle can clear both ends),
    **1-3 distractor bodies** per world (box or upright cylinder; one of them a **0.10-0.25 m tall post** with
    p = 0.5; measured 52.9 %), and obstacle failure `info["obst_hit"]` = any arm-proxy / cup-tip / held-object contact with a
    distractor OR a distractor's CENTRE displaced > 1 cm, charged exactly like `wall_hit`.  Observation
    **47 + 7 = 54**: the nearest active distractor's centre relative to the tcp, its half extents, and
    `n_active / 3`, zero-padded when there is none.
    *Scene* (`rl/build_scene_v2.py --n_dist 3` -> `rl/scenes/obstacle_v3.xml`): three FREE bodies, two geoms each
    (the env_v2 stub trick selects the shape per world through the batched `geom_size`), bottom-centre origin,
    `contype 1 / conaffinity 5` so they collide with the table plane, the object, the cup tip, the arm proxies and
    the walls and with nothing else; an inactive slot is parked at (2.0 + 0.15k, 2.0), outside the cell.
    nq 34, nv 30.  **njmax 256 / nconmax 160 (env_v2's values) are enough**: the measured max `nefc` over 200
    random-action steps at 512 worlds is 92, and 320/208 costs 22 % throughput for nothing.
    *Placement* is rejection sampling: on the table, clear of the base and the bin, >= 4 cm between distractors,
    and >= **max(6 cm, r_distractor + r_object + 3 cm)** from the target and from the goal xy.  The literal 6 cm
    of the spec is NOT enough once the bodies have size -- a 5 cm-radius distractor 6 cm from the goal leaves the
    CARRIED object overlapping it and cuRobo then rejects every carry plan.  That size-aware clearance in turn
    pushes a naive U(0.3, 0.7) corridor sample inside an end keep-out almost every time (corridor occupancy
    collapsed to **17 %**), so the fraction along the segment is clipped per world to the band that can still
    satisfy the end clearance, the carry distance is 0.20-0.32 m, and `p_corridor` defaults to **0.80** because
    about 12 % of worlds have no feasible corridor spot at all.  Measured at 1024 worlds: 1/2/3 distractors
    34.3/33.0/32.7 %, posts 52.9 %, **corridor occupancy 68.7 %** (0.70 gives 61.3 %), post-on-corridor 38.1 %,
    min |distractor - target| 9.8 cm, min |distractor - goal| 9.1 cm, carry 0.201-0.319 m.
    *THE SUCTION BUG (this is the real result of the milestone).*  The first cuRobo teacher on v3 scored only
    55-61 %, with 30-36 % of episodes "sealed but short of the goal".  Instrumenting the break: **36 seal breaks
    per 128 worlds in 12 s of rollout, every one by FORCE saturation**, the object travelling at 1.0-3.1 m/s and
    spinning at 35-230 rad/s while the arm moved at 0.1 m/s, and the rate scaling hard with mass -- **0.69 breaks
    per world at 30-60 g, 0.31 at 60-100 g, 0.05 at 100-150 g**.  Mechanism: the righting spring
    (`_apply_suction_force`) saturates at `TAU_CUP` = 0.4 N*m after 1.1 deg of orientation error, and 0.4 N*m on
    a small diverse object (I ~ 2e-5 kg m^2) injects **30 rad/s of angular velocity per 2 ms substep**; with the
    0.55 per-substep angular damping the steady state is ~40 rad/s, and because env_v2 moved the object's body
    origin to its BOTTOM FACE that spin appears in `qvel[0:3]` as a 1-3 m/s *linear* velocity, so the linear
    damper `-C*v` saturates `F_MAX` and `BREAK_STEPS` later the cup lets go.  Three scoped fixes in
    `rl/env_warp.py` (all reversible with `PNP_LEGACY_SUCTION=1`):
      * righting torque capped at `I * W_INJ_MAX / dt`, `W_INJ_MAX = 3 rad/s` -- still 5-7x the gravity peel
        torque these objects have to hold, so the cup keeps its authority;
      * `W_SEAL_MAX = 6 rad/s` ceiling on a SEALED object's spin;
      * linear damper capped at `DAMP_DT_RATIO * m / dt` (explicit-Euler stability, `C*dt/m < 2`); a no-op at and
        above the paper object's 0.05 kg.
    Result: **seal breaks 36 -> 1 per 128 worlds** (the survivor is a genuine 24 mm spring overload), and the v3
    teacher **60.9 / 55.5 % -> 88.3 / 83.6 %**.  A/B on the DEPLOYED policy (`resid3_fast_best` on box_med, 512
    paired episodes, legacy vs fixed): base 95.70 / 95.90 %, residual 96.48 / 95.90 %, seal 96.29 / 97.07 % in
    both, t_seal 5.19 s in both -- **inside the +-1.9 pp run-to-run noise, so nothing published on the paper env
    moves.**  This is almost certainly what capped `dagger_v2c` at a 50.6 % teacher / 46 % student (item 25's
    unexplained v2 plateau): env_v2's 0.03-0.15 kg mass range is exactly the band where the artefact bites.
    *Teacher* (`rl/bc_curobo.py --env v3`, class `TeacherV3`): per world and per leg the active distractors, the
    walls and the table go to the cuRobo server as cuboids (`set_world`) and cuRobo plans the leg; on the CARRY
    leg every distractor cuboid is inflated by the held object's circumradius and by its overhang below the tcp
    (`CUP_R + object height`), so a tcp-only plan keeps the OBJECT clear too.  The trajectory is retimed with
    planner_sweep's parameters (vlim 36 deg/s, alim 300 deg/s^2, applied as the uniform time dilation that
    `retime` ends with; at dq_max 2 the BINDING limit is the per-decision action clamp, 0.95*dq_max*10 Hz =
    19 deg/s).  Two shortcuts keep the ~1.7 s/plan server off the critical path: `--planner_mode corridor`
    (default) skips the RPC when the straight tcp line is already clear of every inflated obstacle, and a failed
    plan falls back to a geometric OVER-THE-TOP IK path (lift above the tallest obstacle in the way, translate,
    descend).  After the plan runs out the carry target is re-IK'd from the LIVE `tcp - object` offset every 5
    decisions (`--carry_refine`), because the offset measured once at lift-off leaves the object short.
    Three bugs found on the way, each worth a line:
      * every `set_world` was raising `TypeError: Object of type float32 is not JSON serializable`, which the
        leg's `except Exception` turned into "no plan" -- **0/21 plans** until it was caught.  RPC errors are now
        printed once and counted separately.
      * the server's BASE world keeps a 0.30 x 0.30 x 1.0 `camera_mount_d435` cuboid at (0.64, -0.05), i.e. it
        blocks x >= 0.49 up to z = 0.9 -- a slice of the v3 spawn box the MuJoCo twin has no collision geom for.
        `set_world` overrides a base cuboid BY NAME, so `TeacherV3` parks it (`--cam_mount 1` restores it):
        approach plans **17/24 -> 23/24**, carry plans **15/24 -> 23/24**.  SIM-TO-REAL: the real cell does have
        that mount; a deployment must restore the cuboid or keep the object off x > 0.49.
      * `ep_len` 150 is too short for the enlarged workspace: teacher success 34 % at 150 vs 59 % at 200 on the
        same seed (t_seal alone is 8.2-8.7 s).  **v3 runs at ep_len 200.**
    *Milestone 1 paired eval* (`rl/eval_v3.py ~/pnp_rl/dagger_v2c/bc_iter16.pt --nworld 1024 --ep_len 200 --both
    --dq_max 2`, DR on, deterministic, same seed; the 47-D env_v2 student is zero-padded into the 54-D actor, so
    it simply ignores the obstacle block -- which is exactly the baseline "what do the obstacles cost a policy
    that cannot see them"):

    | env_v3 | success | seal | obst_hit | wall_hit | off | t_seal | t_goal | peak abs qd med/p90/max |
    |---|---|---|---|---|---|---|---|---|
    | obstacles OFF (enlarged workspace only) | **56.25 %** | 72.56 % | - | 2.25 % | 1.07 % | 10.38 s | 13.98 s | 39.2/46.8/61.4 |
    | obstacles ON | **32.03 %** | 72.56 % | **31.74 %** | 2.25 % | 1.07 % | 10.43 s | 14.04 s | 39.2/47.4/61.4 |

    The enlarged workspace alone costs nothing (the same checkpoint scored 45.7 % on env_v2 with the OLD suction
    numerics); the obstacles cost **24.2 pp**, and success is flat in the NUMBER of distractors
    (1/2/3 -> 30.8/34.7/30.4 %) -- it is the corridor one that matters, not how many there are.
    *Acceptance test* `rl/test_env_v3.py` (512 worlds): obs 54, 300 random steps with no NaN in obs / reward /
    qpos / qvel at 282 env-steps/s, 0 diverged worlds, idle distractors drift <= 0.2 mm (no spurious failure),
    and the collision probe -- a 0.25 m post planted under the tcp and the arm driven straight down onto it --
    fires `obst_hit` on 8/8 probe worlds and 0/504 controls, charges the -1 `fail` component once and clears
    `placed`.  All three parts PASS.
    *Teacher result* (8 batches x 128 worlds, ep_len 200, `--press_depth 0.02 --press_rate 0.6`): success
    83.6 / 76.6 / 78.9 / 73.4 / 81.2 / 76.6 / 76.6 / 75.0 % = **77.7 % mean** (target was 60 %), seal 88-95 %,
    obst_hit 7-12 %, wall_hit 0-3 %, t_seal 8.5-9.2 s, t_goal 11-13 s for the 80-92 % that reach the goal.
    cuRobo is called on the legs whose straight tcp line is blocked and solves **43/52 approach** and
    **377/673 carry** legs; the other 1547 legs are the geometric over-the-top fallback.  ~4-5 min per batch of
    128 worlds, planner-bound.
    *DAgger* (`--teacher_batches 8 --dagger_iters 4 --dagger_batches 3`, beta 0.5 / 0.3 / 0.1 / 0, warm-started
    from the env_v2 student with `--init`, 40 epochs per fit, dataset 205k -> 512k steps):

    | round | beta | dataset | student success (128 w, in-run) | 512-world deterministic re-eval | seal | obst_hit |
    |---|---|---|---|---|---|---|
    | 0 (one-shot BC) | - | 204 800 | 41.4 % | 38.87 % | 67.2 % | 23.1 % |
    | 1 | 0.5 | 281 600 | 42.2 % | 48.83 % | 70.1 % | 16.6 % |
    | **2** | **0.3** | **358 400** | **49.2 %** | **59.18 %** | **78.7 %** | **13.9 %** |
    | 3 | 0.1 | 435 200 | 18.8 % | (collapsed: seal 28 %) | - | - |
    | 4 | 0 | 512 000 | 36.7 % | 47.66 % | 73.8 % | 14.1 % |

    Same shape as FINDINGS 15: **the rounds are noisy and one of them collapses** -- keep the best by a paired
    eval, never the last.  Round 2 is `~/pnp_rl/dagger_v3/best.pt`.  Note the one-shot BC is 38.9 % rather than
    FINDINGS 5's 0 %, because `--init` warm-starts from the env_v2 student; the DAgger rounds still buy
    +20 pp and cut obstacle hits from 23 % to 14 %.
    *Commands* (GPU 0; the cuRobo server is `curobo_planner_server_v2.py --ground-z -0.1` on :9997; nothing here
    ever touches the robot):

        PY=~/miniconda3/envs/mjwarp/bin/python
        $PY rl/build_scene_v2.py --n_dist 3                       # -> rl/scenes/obstacle_v3.xml
        CUDA_VISIBLE_DEVICES=0 $PY rl/test_env_v3.py              # acceptance: (a) (b) (c)
        CUDA_VISIBLE_DEVICES=0 $PY rl/eval_v3.py ~/pnp_rl/dagger_v2c/bc_iter16.pt --nworld 1024 --ep_len 200 \
            --both --dq_max 2 --out ~/pnp_rl/v3_m1_eval.json      # milestone 1 paired eval
        CUDA_VISIBLE_DEVICES=0 $PY rl/bc_curobo.py --env v3 --drive real --nworld 128 --teacher_batches 8 \
            --dagger_iters 4 --dagger_batches 3 --press_depth 0.02 --ep_len 200 --epochs 40 \
            --init ~/pnp_rl/dagger_v2c/bc_iter16.pt --out ~/pnp_rl/dagger_v3
        CUDA_VISIBLE_DEVICES=0 $PY rl/eval_v3.py ~/pnp_rl/dagger_v3/bc_iter{0,1,2,4}.pt --nworld 512 \
            --ep_len 200 --dq_max 2                               # pick the round; 2 wins -> best.pt
        CUDA_VISIBLE_DEVICES=0 $PY rl/ppo.py --nworld 4096 --steps 8000000 --rollout 24 --epochs 3 \
            --minibatch 24576 --gamma 0.98 --lam 0.95 --clip 0.1 --ent 0.0015 --lr 5e-5 --mode pnp --dr \
            --target_max 0.32 --out ~/pnp_rl/resid_v3 --drive real --dq_max 3.0 --obs_lag -1 \
            --hover 0.02 0.04 --init_std -1.0 --descend_sigma 0.07 --env v3 --arch paper --max_grad_norm 1.0 \
            --critic_warmup 8 --lr_max 1e-3 --grasp_shaping 1 --obs_ee 1 --reach_target grasp --lift_dense 1 \
            --w_track_c 4 --w_track_f 8 --w_reach 0.5 --vf_coef 0.5 --reg_ramp 0.4 --ep_len 200 --start home \
            --residual_base ~/pnp_rl/dagger_v3/best.pt --residual_bound 0.3 --base_scale 0.6667 \
            --w_time 0.5 --w_speed 1.0 --v_soft 32 --critic_priv
        # the run that WORKS: same command with --init_std -2.0 --ent 0.0005 --steps 4500000
        #                      --out ~/pnp_rl/resid_v3b
        CUDA_VISIBLE_DEVICES=0 $PY rl/eval_v3.py ~/pnp_rl/dagger_v3/best.pt ~/pnp_rl/resid_v3b/best.pt \
            ~/pnp_rl/resid_v3b/final.pt --nworld 1024 --ep_len 200 --out ~/pnp_rl/v3_m3b_eval.json
        CUDA_VISIBLE_DEVICES=0 $PY rl/export_trt.py rl/weights/resid_v3_best.pt --obs_dim 54 --arch paper \
            --out rl/weights/resid_v3_best
    *Milestone 3: residual PPO (resid1/resid3 recipe) FAILS on a 58 % base.*  `~/pnp_rl/resid_v3`, frozen
    student as the base, bound 0.3, `--base_scale 0.6667 --dq_max 3`, privileged critic, `--w_time 0.5
    --w_speed 1.0 --v_soft 32`, 8 M steps at 4096 worlds (3.2 h, 694 env-steps/s).  Training success (STOCHASTIC
    rollouts) climbed 24.6 -> 36.2 % and `res_mag` 0.0008 -> 0.036, i.e. PPO was optimising its own objective
    fine.  The paired deterministic eval (1024 episodes, DR on, same seed, ep_len 200) says otherwise:

    | policy | dq_max | success | seal | obst_hit | wall_hit | off | t_seal | t_goal | peak abs qd med/p90/max |
    |---|---|---|---|---|---|---|---|---|---|
    | **student** `dagger_v3/best.pt` | 2 deg | **57.62 %** | **79.79 %** | 15.14 % | 2.05 % | 3.61 % | **10.14 s** | **14.93 s** | 36.5 / 41.1 / 148.7 |
    | residual best (7.37 M) | 3 deg | 32.71 % | 48.93 % | **14.75 %** | **1.17 %** | **0.49 %** | 14.07 s | 17.52 s | 38.0 / 43.9 / 80.8 |
    | residual final (8.06 M) | 3 deg | 41.11 % | 63.57 % | 18.95 % | 0.88 % | 0.29 % | 12.15 s | 16.65 s | 37.9 / 43.8 / 71.4 |

    **The plumbing is not the problem.**  Zeroing the trained residual's output layer and re-evaluating the fused
    stack at dq_max 3 / base_scale 0.6667 reproduces the base exactly -- 58.79 % vs 59.18 %, seal 78.91 vs
    78.71 %, t_seal 10.14 vs 10.16 s, t_goal 14.99 vs 14.96 s, peak |qd| 37.0/41.2/81.7 vs 37.0/41.2/81.1 over
    512 paired episodes.  So the 54-D observation, the a_prev rescaling and the joint-only base scaling are all
    exact, and PPO really did make the policy worse.
    **Why**: the residual is trained on STOCHASTIC rollouts, and on this base the exploration noise costs far
    more than it did on the paper env.  `init_std -1.0` with bound 0.3 perturbs the executed action by only
    ~+-0.3 deg/decision, but it drops the SEAL rate from 79.8 % (deterministic) to 48 % (the first training
    window, before the actor had moved) -- the 20 mm press needs sub-millimetre precision over ~15 consecutive
    decisions and the v3 base has no margin left after the DR.  PPO therefore spent 8 M steps improving a
    ~25-36 % operating point, and what it learned there (a residual that is worth +12 pp under noise) is a
    *worse* policy without it.  The paper-env residuals did not hit this because their base was 88-95 % and its
    seal was 96 %, so the noisy and the deterministic operating points were 7 pp apart, not 25 pp.
    Note what the residual DOES buy, consistently on both checkpoints: **off-table failures 3.61 % -> 0.3-0.5 %,
    wall hits 2.05 % -> 0.9-1.2 %, and the base's 148.7 deg/s peak-speed outlier is gone (80.8)** -- the speed
    and safety terms work; it is the seal that it trades away.
    *The fix, confirmed* (`~/pnp_rl/resid_v3b`: `--init_std -2.0 --ent 0.0005`, everything else identical,
    4.5 M steps, 1.8 h).  Shrinking the exploration std from 0.37 to 0.135 moved the FIRST training window from
    24.6 % / seal 48 % to **46.1 % / seal 66 %** -- i.e. the noisy operating point is now within 12 pp of the
    deterministic one instead of 33 pp -- and training then peaked at 50.7 % stochastic.  Paired deterministic
    eval, 1024 episodes, DR on, same seed:

    | policy | dq_max | success | seal | obst_hit | wall_hit | off | t_seal | t_goal | peak abs qd med/p90/max |
    |---|---|---|---|---|---|---|---|---|---|
    | student `dagger_v3/best.pt` | 2 deg | 57.62 % | 79.88 % | 15.43 % | 2.05 % | 3.32 % | 10.14 s | 14.91 s | 36.5 / 41.1 / 144.3 |
    | **resid_v3b best (1.97 M)** | 3 deg | **58.89 %** | 79.69 % | 15.53 % | **1.76 %** | 3.32 % | **10.11 s** | **14.82 s** | 36.7 / 41.4 / 154.9 |
    | resid_v3b final (4.5 M) | 3 deg | 48.54 % | 68.65 % | 16.99 % | 1.46 % | 1.95 % | 11.54 s | 15.64 s | 37.0 / 42.4 / 140.7 |

    **+1.27 pp over its own base** -- the same order as resid3's +1.1 pp on the paper env -- and the run still
    DECAYS after its peak (58.89 -> 48.54 %), so `best.pt` is what ships.  The rule this session adds to the
    residual recipe: **match the exploration std to the base's robustness, not to the recipe.**  A useful proxy
    is the very first training window: if its success is more than ~10 pp below the base's deterministic score,
    the residual is being trained at the wrong operating point and will not transfer back.
    Shipped: `rl/weights/resid_v3_best.{pt,json,onnx}` (+ `.plan` built on GPU 0, untracked): obs_dim 54,
    max|trt-torch| 3.75e-05, 148 us/call on the A5000.  **Deploying it needs
    `rl/real_policy_ctrl.py DQ_MAX_DEG = 3.0` and an observation builder extended to the 54-D layout, plus the
    camera-mount caveat above.**

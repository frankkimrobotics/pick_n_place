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

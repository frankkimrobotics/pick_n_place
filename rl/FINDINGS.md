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


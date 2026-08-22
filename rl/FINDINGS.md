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

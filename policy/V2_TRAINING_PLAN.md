# CFM v2 training plan — consistency, latency, phase awareness

Goal: retrain CFM with six additions that target the failures observed in
sim (chunk dither, suction persistence) and on the real robot (splice
jitter, latency, OOD obs): boundary-consistency loss, previous-chunk
conditioning, within-chunk smoothness, cross-noise consistency,
train-time latency matching, phase awareness.

## 1. Dataset v2 (one new litdata conversion, ~45 min + upload)

Samples gain RAW windows so every v2 loss is computable in the loader
(spline fits are one matmul with the fixed collocation matrix):

| new field | shape | purpose |
|---|---|---|
| `q_past` | (16,6) | previous-chunk CPs (fit on any past window) |
| `q_future` | (20,6) | shifted-target fits for latency Δ∈{0..4} steps |
| `suction_future` | (20,) | shifted suction targets |
| `phase` | int | 0 approach / 1 press / 2 carry / 3 place-retreat |
| `progress` | float | (t − start)/(end − start) within the pick |

Phase boundaries come free from the `picks` records (start / latch−7
(press ≈ 0.7 s) / latch / release / end). Keep `ctrl_pts`/`ctrl_pts_eef`
for v1 compatibility. Sidecars (`goals.json`) unchanged.

## 2. Architecture (train_lit v2, flags — default off for A/B)

Context grows by two tokens (790 → 792):
- **prev-chunk token**: Linear(16·7 → d) on the previous chunk's CPs +
  suction, computed in-loader from `q_past` (fit at t−k, k sampled from
  {1,2,5,8,10} to cover deployment re-plan rates). **20% dropout to a
  learned null token** so the policy still works at episode start and
  can't ignore vision.
- **latency token**: embedding of the sampled obs-staleness Δ (see §4).

Two auxiliary heads on a learnable pooling query over the context:
- phase classifier (4-way), progress regressor (scalar).

## 3. Losses

L = L_cfm + 0.5·L_boundary + 1e-3·L_smooth + 0.5·L_xnoise
    + 0.1·(L_phase + L_progress)

- **L_cfm**: flow-matching MSE (unchanged).
- **L_boundary** (splice consistency at the source): from the denoised
  estimate x̂1 = x_t + (1−t)·v̂, decode the joint CPs and evaluate the
  spline at s=0; penalize ‖q̂(s=0) − (q_t + 0.1·q̇_t)‖² — chunks must
  launch from the current state. (Makes the executor's C1 bridge a
  no-op safety net.)
- **L_smooth** (within-chunk): ‖D₂ · x̂1_ctrl‖² — the same second-
  difference prior used in dataset fitting, regularizing OOD outputs.
- **L_xnoise** (cross-noise consistency): two noise draws x0a, x0b for
  the SAME (obs, flow-time) share one context forward (ViT cost
  unchanged; DiT runs twice ≈ +8% step time); penalize ‖x̂1a − x̂1b‖².
  Directly attacks the re-plan dither that killed 5 Hz.
- **L_phase / L_progress**: CE + MSE on the aux heads. Attacks suction
  persistence (carry = explicit concept) and gives live interpretability
  on the robot (log "policy believes: press").

## 4. Train-time latency matching (augmentation + conditioning)

Per sample draw Δ ∈ {0,1,2,3,4} frames (0–400 ms): observations stay the
sample's own (they are "stale" by construction), the action target is
refit on `q_future[Δ : Δ+16]` — i.e., supervise the actions that must
execute Δ later. Feed Δ as the latency token. At deployment, feed the
measured pipeline latency (13 ms TRT inference + capture ≈ 1 frame).
B-spline targets make the shift exact — no resampling error.

## 5. Evaluation (extends eval_mse)

- exec-aligned sampled-chunk MSE (unchanged headline, Δ=0)
- boundary error ‖q̂(0) − q_t‖ (deg) — the splice-jump predictor
- cross-noise spread ‖x̂1a − x̂1b‖ — the dither predictor
- phase accuracy + progress MAE
- stop rule: eval MSE < 0.001 after ≥100k, cap 300k

## 6. Execution order

1. Let CFM v1 finish (baseline; ~0.0011 @ 20k already) and ACT v1 (40k).
2. Run v2 litdata conversion locally; upload (~13 GB) alongside v1.
3. Implement v2 flags in train_lit; smoke-test 200 steps locally (2080 Ti).
4. Launch CFM v2 on the H100; watchers deliver video + exports as usual.
5. A/B in sim (scene 609 + rate sweep): v1 vs v2 on splice jumps at 5 Hz,
   suction holds without hysteresis, and bin rate — the three metrics the
   additions are supposed to move.

## Verification checklist before launch

- [ ] shifted-target fit: spline refit on q_future[Δ:Δ+16] matches
      time-shifted GT within fit tolerance for all Δ
- [ ] phase labels spot-checked against 3 scene videos (latch/release
      alignment)
- [ ] prev-chunk token: k-sampling covers {1..10}; dropout path returns
      null token
- [ ] x̂1 decode uses the SAME normalizer stats as targets
- [ ] one full loss step on real batch: all six terms finite, gradients
      flow to both heads and encoder

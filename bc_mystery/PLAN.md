# BC-mystery replication plan (after seohong.me/blog/behavioral-cloning-mystery)

Reproduce the blog's controlled BC phenomenology on OUR task (Pro 630 suction
pick-and-place) and OUR batched sim (mujoco_warp `PickEnv`, 37-dim state —
coincidentally the blog's dimensionality). Historical hook: our July BC line
(DP/CFM on perfect scripted demos) already exhibited Mysteries 1–2 before we
had names for them — 12× offline-MSE improvements with flat/worse closed-loop
success. The blog's diagnosis (test-time distribution shift × non-Markovian
data) is the frame we lacked.

## 1. Data collection  (`collect.py` — BUILT)

The blog's key ingredient: **human-like, non-Markovian scripted experts** —
randomized piecewise splines, not optimal controllers.

* Expert = time-parameterized **Catmull-Rom/Hermite spline in task space**
  through pick-and-place waypoints, per-episode randomized:
  - hover/press/lift/place waypoint jitter (xy ±1 cm, heights U[ranges])
  - optional mid-transport via-point (±6 cm lateral "wobble", p=0.5)
  - per-segment durations (speed variability ~2×)
  - suction timing jitter (±2 ticks)
* Tracked closed-loop by a **batched DLS + cup-vertical orientation task**
  (analytic per-world Jacobians from mjwarp `xaxis`/`xanchor` — verified vs
  `mj_jacSite` to 2.5e-8), emitting the env's native 10 Hz actions
  (Δq ±2° + suction logit). Demos therefore live exactly in the policy's
  action space and the data is non-Markovian by construction (the spline
  clock is hidden state).
* Batch-synchronized episodes: N worlds × T=100 ticks (10 s), then reset.
  All episodes recorded (success labels attached; filter at train time).
* Output shards: `obs (E,100,37) f32`, `act (E,100,7) f32`, `metrics`
  (E,8: seal setd v1 v2 v3 final_d max_lift final_spd — same ladder as
  rl/eval_bench.py, captured at each world's first done tick).
* **As built** the expert needed event gating on top of the spline clock
  (tau stalls when tracking error is large; tight near contact, loose in
  transit): pin-until-sealed at the press dwell, lift-gate hold at the mid
  via-point (ascent overlaps transport — a pure vertical 0.31 m object
  lift costs ~30 of the 100 ticks), and an event-gated release
  (rest_h < 1.2 cm), because tau-timestamped suction-off fired while the
  arm was still descending. Winners use ~89/100 ticks.
* Measured expert (512-ep smoke): **seal ~90%, setd ~55%, V1=V2=V3 ~53%**,
  d_p50 1.4 cm. Flat V1→V3 = placements that finish are clean; the losses
  are late sealers/far pedestal targets running out the 100-tick budget.

## 2. Training  (planned — NOT run yet)

State-based **flow matching** (velocity-field MLP, same family as the blog)
predicting **25-action chunks** (2.5 s at 10 Hz), conditioned on current obs.
Baselines: MSE-BC MLP. Optimizer AdamW, bs 1024, cosine; val split 5%.

Experiment matrix (one mystery per axis, all evaluated closed-loop):

| Mystery | Axis | Values |
|---|---|---|
| 1 beneficial overfit | dataset size | 1K / 10K / 50K / infinite; track val-loss vs success over epochs |
| 2 open-loop superiority | chunk exec length | 1 / 5 / 12 / 25 of a 25-chunk; + history-conditioning {0,2,8 frames} |
| 3 model size | MLP width×depth | [256]×4 → [1024]×6 → [4096]×8 (+ residual) |
| 4 feature engineering | obs scaling | raw / standardized / hand-scaled object coords (×10 on xy deltas) |

## 3. Evaluation  (`evaluate.py` — BUILT)

* Batched closed-loop rollouts in the SAME `PickEnv` (pnp mode, DR off,
  deterministic), pluggable policy interface `act(obs[N,37]) -> [N,K,7]`
  with configurable executed-chunk length and observation history stacking.
* Success = ppo4-era criterion (set-down, ≤3.5 cm, released, at rest) from
  the env's `info` terminals — plus the full V1/V2/V3 ladder for context.
* Built-in baselines to anchor the harness before any training exists:
  `expert` (the spline generator itself — upper anchor), `random`, `zero`.
* Anchors measured: expert V3 49.6% (512 eps), random 0% (MSE 0.77),
  zero 0% (MSE 0.37). Zero already shows the Mystery-1 decoupling:
  half the offline MSE of random, identical 0% closed-loop.
* Every eval reports success, place-error percentiles, and offline-proxy
  (per-step action MSE vs the expert on the same states) so the
  val-loss-vs-success divergence (Mystery 1) is measurable from day one.

## Predictions to test against the blog

1. Val loss and closed-loop success will decouple (already seen in July).
2. Chunked open-loop execution will beat single-step re-planning; history
   conditioning will hurt (it did on the real robot too — 5 Hz replan
   sealed nothing while 1 Hz worked).
3. Success will keep rising with model width long past "reasonable" sizes.
4. Feature scaling will matter at fixed loss — test via obs-scaling ablation.

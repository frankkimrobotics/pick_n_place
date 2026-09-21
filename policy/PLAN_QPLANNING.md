# Q-Planning self-improvement for the deployed pick policy (plan, 2026-09-21)

Paper: Giridhar, Khandelwal, Collins, Georgiev, Garg, *Beyond Imitation: Self-Improving Robot Policies via
Off-Policy Q-Planning* (https://q-planning.github.io/, arXiv 2608.21204). Frozen BC policy + small off-policy
Q over action chunks (HL-Gauss); at run time N chunks from the policy are scored by Q and the softmax-Q weighted
chunk is executed; after each deployment ALL episodes (incl. failures) go to a replay buffer and only Q is
retrained. LIBERO-10 90 -> 99 %, real bimanual tasks 40 -> 90 % and 25 -> 80 % in 5 iterations of 20 episodes.

Rendered plan with diagrams: the "Q-Planning for the Pro 630" artifact (claude.ai). Summary:

- Frozen policy: `rl/weights/resid3_fast_best` (deterministic, 40-D state, 3 deg bound). Chunk H = 5 decisions
  (0.5 s) = the Pi B-spline control polygon. Q: MLP [512,512,256] on (state, chunk), HL-Gauss 51 bins, two heads
  (success within 3.5 cm + graded distance; time-to-goal -0.1/decision), EMA target, H-step bootstrap.
- Proposals (N = 32): 16 chunk-correlated Gaussian perturbations of pi (sigma 0.15), 12 structured variants
  (x0.7 / x1.2 / x1.4 joint scaling, suction flip), 4 safety chunks (hold, 50 % slow). Blanket x1.3 scaling fails
  in the twin (96.7 -> 88 %); the critic picks fast chunks only where safe. Later: CFM/DP proposals (IL v2).
- Planner: drop candidates with Q_succ < 0.9 max, softmax over Q_time with lambda; ablate vs best-of-N.
- DR the critic must see: drive DR (vmax 0.9-1.1, amax 0.75-1.15, dead 30-65 ms), obs noise 0.005, action
  latency, object top 2-9 cm / footprint 3-8 cm / mass 50-400 g (env_v2), spawn x 0.17-0.50 y +-0.30, goal
  height 5-25 cm, NEW tracker error DR (+-15 mm xy bias per episode, +-5 mm jitter, top -6..+2 cm), friction/seal.
- Loop: iteration 0 = Q offline on 200k twin steps of pi; twin iterations 1-10 (deploy planner in 4096 DR worlds,
  append, retrain 10k steps; gate 99 %, t_goal <= 6.5 s, peak qd <= 36 deg/s); robot iterations 1-5 (20 random-
  object runs, monitor verdict labels, retrain on twin + robot buffer; gate >= 9/10 within 3.5 cm, no faults).
  Every iteration exports pi + Q as one batched TensorRT graph. Press/attach/place primitives keep contact.
- Milestones: M0 collector + labels (1 d), M1 critic + planner + paired eval (2 d), M2 ten twin iterations (1 d),
  M3 deployment `--qplan` + selftest (1 d), M4 robot loop (2 sessions).
- Risks: proposal support (paper's own limit), Q over-estimation (HL-Gauss + conservative term + success gate),
  label noise (monitor fixed-cam localisation still colour-based), twin gap (tracker-error DR is the missing piece).

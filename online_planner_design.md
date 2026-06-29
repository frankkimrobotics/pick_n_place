# Online (streaming) cuRobo planner — design of both options

Goal: instead of the one-shot "plan the whole rest-to-rest trajectory, send it, execute
open-loop" flow, run the planner **continuously** and feed the controller short
trajectory **chunks** (dt = 0.01) that it **welds** onto the motion it is already
executing — matching a reference ML planner that does ~10 Hz inference, 16–20 waypoints
per chunk, and never stops between chunks.

```
[online_planner_node]  --/mycobot/cmd/move {weld:true, t_anchor, traj_dt:0.01, traj:[deg]}-->  [controller]
        |  socket :9997 (cuRobo)                                                                  (sim_mujoco_node --weld
        v                                                                                          or robot_hal)
   plan / MPC-step                                                                       tracks the WELDED reference q_ref(t)
        ^----------------------------- /joint_states (state for seed / feedback) ----------------------
```

## The shared core: the velocity-continuous WELDER (`traj_weld.py`)

Every chunk is anchored at an absolute time `t_anchor = now + commit` (slightly in the
future, covering planner + comms latency, so the controller is still on the OLD reference
there). Over `[t_anchor, t_anchor+blend]` the welder cross-fades old→chunk with a
**smoothstep** `a(s)=3s²−2s³` (`a'(0)=a'(1)=0`):

```
q_ref(t) = (1−a)·q_old(t) + a·q_chunk(t),   s = (t−t_anchor)/blend
```

Because `a'`=0 at both ends, the blended reference **leaves at the OLD velocity and
arrives at the CHUNK velocity** — C1 continuous, no velocity jump = the "weld". The
controller samples `q_ref(t)` at its servo rate; the planner keeps a **mirror welder** fed
the same chunks so it plans from the junction state it has already committed.

`commit` must exceed (planner inference + comms) and chunks must extend past the next
anchor, so the weld always blends from a *live, moving* segment.

---

## Option B — receding-horizon replanning (`--mode chunk`)   ✅ exact convergence

1. Plan the **full** trajectory to the goal once (cuRobo MotionGen → proper accel/cruise/
   decel velocity profile). Resample to dt = 0.01.
2. Every cycle, slice the next `horizon`-second window (≈20 waypoints) and stream it as a
   chunk; the welder stitches them (near-identity blends, since all chunks come from one
   plan → momentum preserved, smooth & fast).
3. **Re-plan + weld only on an event** — goal change, disturbance, or cache age. The new
   full plan is welded from the *moving* junction state.

Why it's robust: the nominal motion uses ONE plan (correct velocity profile); chunking
buys reactivity without the per-cycle "restart from rest" problem.

**Result (sim):** base → goal, final tcp = exact goal `[0.35, 0.10, 0.22]`, smooth.

## Option A — reactive per-cycle replanning (`--mode mpc`)   ✅ reaches vicinity

Every cycle: predict the junction `(q,q̇)` from the mirror welder, re-plan toward the goal,
and re-time the chunk to a **constant cruise** speed (the weld absorbs the one-time
rest→cruise transition), with a deceleration zone as the goal nears. Replans every cycle →
naturally reactive to moving goals / new obstacles.

The catch: cuRobo's MotionGen is **rest-to-rest**, so naïvely re-planning every cycle and
re-timing from the *current* (tiny) speed makes the robot **creep** — each plan restarts
from rest. Constant-cruise + weld fixes the creep, but a trajopt approximation leaves a
small steady-state standoff (≈5 cm in sim) because the goal is always `commit` seconds
ahead. **The proper Option A is cuRobo's native `MpcSolver`** (`curobo/_src/solver/
solver_mpc.py`) — a velocity-aware MPPI/gradient MPC that optimizes from the live
`(q,q̇)` every cycle at ~30–50 Hz (CUDA-graph + warm-started), so momentum and convergence
are handled inside the solver. Drop-in path: add an `mpc_step(current_state, goal)` RPC to
the planner server backed by `MpcSolver.step`, and have this node call it each cycle
instead of MotionGen+retime; the welder/streaming layer is unchanged.

---

## Latency & feedback (real robot vs sim)
- **Sim (`sim_mujoco_node --weld`)**: ideal tracking, no dead-time → the mirror welder ==
  the controller welder, so prediction is exact. This is where both modes were validated.
- **Real robot**: ~0.8 s command→motion dead-time + tracking error. Two additions are
  required: (a) **forward-predict** the junction state along the committed reference by the
  measured lag (don't plan from the laggy measured `q`); (b) periodically **correct** the
  mirror welder toward `/mycobot/drive_feedback` to bound drift. `commit` and `blend` must
  be sized to the lag.

## Files
- `traj_weld.py` — `TrajectoryWelder` (smoothstep weld) + `retime_to_velocity`.
- `online_planner_node.py` — the streaming loop, `--mode chunk|mpc`, mirror welder.
- `sim_mujoco_node.py` — `--weld`-aware controller: welds incoming chunks, samples
  `q_ref(t)` at the servo rate, publishes feedback.

## Run (sim, isolated domain)
```
export ROS_DOMAIN_ID=42 MUJOCO_GL=osmesa
python3 sim_mujoco_node.py --video outputs/mujoco_sim/online.mp4 &
python3 online_planner_node.py --mode chunk --goal 0.35,0.10,0.22   # or --mode mpc
```

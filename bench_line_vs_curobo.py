#!/usr/bin/env python3
"""bench_line_vs_curobo :: planning + execution time, line-MPC vs cuRobo-MPC.

Both reach the SAME task-space goal through the SAME tuned MPC on the SAME
calibrated drive; they differ only in where the reference comes from:

  curobo_mpc : cuRobo plan_pose (collision-aware trajectory) -> velocity-cap
               retime -> MPC tracks it
  line_mpc   : local damped-least-squares IK for the goal CONFIG only
               (no planner) -> straight c-space line -> MPC tracks it

Reports wall-clock planning time, execution time, and accuracy over N goals.

    python3 bench_line_vs_curobo.py --trials 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rl"))
import warp as wp                                          # noqa: E402
import sim_touch_compare as S                              # noqa: E402

TCP_FRAME_DZ = 0.020   # MuJoCo tcp site reads 20 mm HIGHER than cuRobo tcp link
                       # (same config: mujoco_z = curobo_z + 20 mm)

GOALS = [                       # reachable down-facing touch points
    [0.34, 0.00, 0.10], [0.30, -0.10, 0.12], [0.38, 0.08, 0.11],
    [0.32, 0.12, 0.13], [0.40, -0.06, 0.10], [0.28, 0.04, 0.15],
    [0.36, -0.12, 0.12], [0.42, 0.02, 0.14],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--curobo_ctrl", default="stream", choices=["stream", "mpc"],
                    help="how the cuRobo leg is executed: stream = deployed "
                         "spline streaming law; mpc = tuned MPC")
    ap.add_argument("--v_cap", type=float, default=30.0)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/bench_line_curobo"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wp.init()
    from env_paper import PaperPickEnv
    import config as C
    import mujoco
    from mjwarp_pick_demo import ik, R_DOWN

    env = PaperPickEnv(nworld=1, drive="real", dr=False, start="home")
    env.auto_reset = False
    q_base = np.asarray(C.START_Q, float)
    # CPU model for the local IK (the line planner's only "planning" cost)
    m_cpu = env.mjm
    d_cpu = mujoco.MjData(m_cpu)

    rows = []
    for i in range(min(a.trials, len(GOALS))):
        goal = GOALS[i]
        # ---------- cuRobo: full collision-aware plan ----------
        t0 = time.perf_counter()
        traj, ndt = S.plan_pose(q_base, goal)
        t_plan_curobo = time.perf_counter() - t0
        if traj is None:
            print(f"[{i}] cuRobo plan FAILED for {goal}")
            continue
        dec_c = S.resample_decisions(traj, ndt, v_cap=a.v_cap)

        # ---------- line: local IK for the goal config only ----------
        t0 = time.perf_counter()
        # NOTE: the MuJoCo "tcp" site sits 20 mm below cuRobo's tcp link, so the
        # local IK is given the goal in the MuJoCo frame (goal_z - TCP_FRAME_DZ)
        # to land on the same physical point as the cuRobo plan.
        goal_mj = [goal[0], goal[1], goal[2] + TCP_FRAME_DZ]
        q_goal, ik_err = ik(m_cpu, d_cpu, "tcp", goal_mj, R_DOWN, q_base)
        travel = np.degrees(np.abs(q_goal - q_base)).max()
        # cosine ease: same straight c-space PATH, but accelerate/decelerate
        # (a uniform-velocity line has no terminal braking -> MPC undershoots).
        # The 1.5x duration keeps the PEAK rate at v_cap.
        n_dec = max(2, int(round(travel / a.v_cap * 10 * 1.5)) + 1)
        u = (1 - np.cos(np.pi * np.linspace(0, 1, n_dec))) / 2
        dec_l = q_base[None] + u[:, None] * (q_goal - q_base)[None]
        t_plan_line = time.perf_counter() - t0

        # ---------- execute both through the same tuned MPC ----------
        res = {}
        for name, dec, ctrl in (("curobo_mpc", dec_c, a.curobo_ctrl),
                                ("line_mpc", dec_l, "mpc")):
            S.reset_to(env, q_base)
            t0 = time.perf_counter()
            log = []
            if ctrl == "stream":
                # robot_hal chunk streaming has NO lead_max clip (that bound
                # belongs to real_policy_ctrl's incremental policy targets);
                # with it on, any leg faster than ~3 deg/decision lags badly.
                env.lead_clip = False
                for q_dec in dec:
                    env.q_target = torch.as_tensor(
                        q_dec, dtype=torch.float32, device=env.device)[None]
                    env._step_real_drive(log=log)
            else:
                S._run_mpc(env, dec, log, gains=(10.0, 0.3))
            env.lead_clip = True
            t_exec_wall = time.perf_counter() - t0
            resp = torch.stack(log).squeeze(1).cpu().numpy()
            t_exec_sim = len(resp) / 100.0          # 100 Hz command ticks
            # frame-independent accuracy: final joint error vs this
            # strategy's OWN reference endpoint (deg)
            jerr = np.degrees(np.abs(resp[-1] - dec[-1])).max()
            tt = np.linspace(0, 1, len(dec)); tr = np.linspace(0, 1, len(resp))
            tgi = np.stack([np.interp(tr, tt, dec[:, j]) for j in range(6)], 1)
            rms = float(np.degrees(np.sqrt(np.mean((resp - tgi) ** 2))))
            tcp = S.fk_np(env)
            res[name] = dict(t_exec_sim=t_exec_sim, t_exec_wall=t_exec_wall,
                             err_mm=float(jerr), rms=rms, n_dec=len(dec),
                             tcp=[float(v) for v in tcp])
        rows.append(dict(goal=goal, t_plan_curobo=t_plan_curobo,
                         t_plan_line=t_plan_line, ik_err_mm=float(ik_err * 1000),
                         **{f"{k}_{kk}": vv for k, v in res.items()
                            for kk, vv in v.items()}))
        print(f"[{i}] goal {goal}", flush=True)
        print(f"    plan:  cuRobo {t_plan_curobo*1000:7.1f} ms | "
              f"line(IK) {t_plan_line*1000:6.1f} ms  "
              f"({t_plan_curobo/max(t_plan_line,1e-9):.0f}x)")
        print(f"    exec:  cuRobo {res['curobo_mpc']['t_exec_sim']:.2f} s "
              f"({res['curobo_mpc']['n_dec']} dec) | "
              f"line {res['line_mpc']['t_exec_sim']:.2f} s "
              f"({res['line_mpc']['n_dec']} dec)")
        dtcp = np.linalg.norm(np.array(res['curobo_mpc']['tcp'])
                              - np.array(res['line_mpc']['tcp'])) * 1000
        print(f"    track rms: cuRobo {res['curobo_mpc']['rms']:.2f} deg | "
              f"line {res['line_mpc']['rms']:.2f} deg")
        print(f"    final joint err: cuRobo {res['curobo_mpc']['err_mm']:.2f} deg | "
              f"line {res['line_mpc']['err_mm']:.2f} deg | "
              f"end-pose gap {dtcp:.1f} mm")

    if not rows:
        sys.exit("no successful trials")
    summarize(rows, a.out)


def summarize(rows, out):
    import json
    pc = np.array([r["t_plan_curobo"] for r in rows]) * 1000
    pl = np.array([r["t_plan_line"] for r in rows]) * 1000
    ec = np.array([r["curobo_mpc_t_exec_sim"] for r in rows])
    el = np.array([r["line_mpc_t_exec_sim"] for r in rows])
    errc = np.array([r["curobo_mpc_err_mm"] for r in rows])
    errl = np.array([r["line_mpc_err_mm"] for r in rows])
    print("\n================ SUMMARY (n=%d) ================" % len(rows))
    print(f"{'':22} {'cuRobo+stream':>14} {'line+MPC':>14}")
    print(f"{'planning (ms)':22} {pc.mean():10.1f}±{pc.std():.0f} "
          f"{pl.mean():10.2f}±{pl.std():.2f}")
    print(f"{'execution (s)':22} {ec.mean():12.2f}   {el.mean():12.2f}")
    print(f"{'TOTAL cycle (s)':22} {(pc/1000+ec).mean():12.2f}   "
          f"{(pl/1000+el).mean():12.2f}")
    print(f"{'final joint err (deg)':22} {errc.mean():12.2f}   {errl.mean():12.2f}")
    rmc = np.array([r["curobo_mpc_rms"] for r in rows])
    rml = np.array([r["line_mpc_rms"] for r in rows])
    print(f"{'tracking rms (deg)':22} {rmc.mean():12.2f}   {rml.mean():12.2f}")
    print(f"\nplanning speedup: {pc.mean()/pl.mean():.0f}x   "
          f"total-cycle saving: {((pc/1000+ec).mean()-(pl/1000+el).mean()):.2f} s "
          f"({100*(1-(pl/1000+el).mean()/(pc/1000+ec).mean()):.0f}%)")
    json.dump(rows, open(os.path.join(out, "bench.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(rows))
    w = 0.38
    ax[0].bar(x - w/2, pc, w, label="cuRobo plan", color="#4878CF")
    ax[0].bar(x + w/2, pl, w, label="line IK", color="#8C613C")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("planning (ms, log)")
    ax[0].set_xlabel("goal #")
    ax[0].set_title("planning time")
    ax[0].legend(fontsize=8)
    ax[1].bar(x - w/2, ec, w, color="#4878CF", label="cuRobo+retime+stream")
    ax[1].bar(x + w/2, el, w, color="#8C613C", label="line+MPC")
    ax[1].set_ylabel("execution (s)")
    ax[1].set_xlabel("goal #")
    ax[1].set_title("execution time")
    ax[1].legend(fontsize=8)
    ax[2].bar(x - w/2, pc/1000 + ec, w, color="#4878CF", label="cuRobo+retime+stream")
    ax[2].bar(x + w/2, pl/1000 + el, w, color="#8C613C", label="line+MPC")
    ax[2].set_ylabel("total cycle (s)")
    ax[2].set_xlabel("goal #")
    ax[2].set_title("plan + execute")
    ax[2].legend(fontsize=8)
    fig.suptitle("Free-space touch: cuRobo+retiming+stream-law vs IK+line+MPC")
    fig.tight_layout()
    p = os.path.join(out, "bench_line_vs_curobo.png")
    fig.savefig(p, dpi=110)
    print(f"[plot] {p}")


if __name__ == "__main__":
    main()

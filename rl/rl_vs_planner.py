#!/usr/bin/env python3
"""rl_vs_planner :: the RL policies through planner_sweep's limit pipeline.

Runs the DAgger base and the fused residual policy in the SAME measured-drive
twin planner_touch --sim uses, on the SAME 5 object positions as
rl/planner_sweep.py, and pushes their joint trajectories through the identical
checker (finite differences with the same smoothing, mj_inverse torque on
rl/scenes/box_med.xml, URDF position limits + the elbow box).

The policies are sampled at the drive model's 100 Hz command rate
(`env._step_real_drive(log=...)`, monkey-patched) and cubic-resampled onto the
planner's 4 ms grid so planner_sweep._peaks / .check apply unchanged.

Two windows are reported per policy:
    approach : up to the first tick with |tip - grasp| < 1 cm (what the planner
               trajectory covers -- it stops at the grasp point)
    episode  : the whole 15 s rollout (the policy presses and lifts afterwards)

Run (mjwarp env; cuRobo is NOT needed -- the planner trajectories come from
planner_best_traj.npz, written by the curobo2 side):
    /home/lisc-frank/miniconda3/envs/mjwarp/bin/python rl/rl_vs_planner.py
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import planner_sweep as PS  # noqa: E402

SCENE = os.path.join(HERE, "scenes", "box_med.xml")
OBJ_Z = 0.020                      # box half height 0.020 -> centre 0.020, top 0.040
CMD_DT = 0.010                     # DRIVE["cmd_dt"], the 100 Hz log period
TRACE_CASE = 1                     # the case whose J2/J3 velocity traces are plotted


# ------------------------------------------------------------------ torque
class Torque:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d0 = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d0)
        self.dinv = mujoco.MjData(self.m)

    def peaks(self, q, qd, qdd, every=5):
        a = PS.decimate(q, qd, qdd, every)       # [N, 3, 6], qdd box-smoothed
        tau = np.zeros((len(a), 6))
        for k in range(len(a)):
            self.dinv.qpos[:] = self.d0.qpos
            self.dinv.qvel[:] = 0
            self.dinv.qacc[:] = 0
            self.dinv.qpos[:6] = a[k, 0]
            self.dinv.qvel[:6] = a[k, 1]
            self.dinv.qacc[:6] = a[k, 2]
            self.mj.mj_inverse(self.m, self.dinv)
            tau[k] = self.dinv.qfrc_inverse[:6]
        return np.abs(tau).max(axis=0), tau


def _natural_spline(y, h):
    """Second derivatives of the natural cubic spline through uniformly spaced y."""
    n = len(y)
    M = np.zeros_like(y)
    if n < 3:
        return M
    rhs = 6.0 * (y[:-2] - 2.0 * y[1:-1] + y[2:]) / (h * h)
    m = n - 2
    c = np.zeros(m)
    d = np.zeros_like(rhs)
    b = 4.0
    c[0] = 1.0 / b
    d[0] = rhs[0] / b
    for i in range(1, m):                                   # Thomas, sub/super = 1
        den = 4.0 - c[i - 1]
        c[i] = 1.0 / den
        d[i] = (rhs[i] - d[i - 1]) / den
    M[m] = d[m - 1]
    for i in range(m - 2, -1, -1):
        M[i + 1] = d[i] - c[i] * M[i + 2]
    return M


def to_4ms(q, dt_src):
    """C2 natural-cubic resample of q(t) onto planner_sweep's 4 ms grid (no scipy:
    the mjwarp env's scipy is ABI-broken against its numpy)."""
    q = np.asarray(q, float)
    if abs(dt_src - PS.DT) < 1e-9:
        return q
    h = dt_src
    M = _natural_spline(q, h)
    T = (len(q) - 1) * h
    tn = np.arange(0.0, T + 1e-9, PS.DT)
    i = np.clip((tn / h).astype(int), 0, len(q) - 2)
    u = (tn - i * h) / h
    u = u[:, None]
    return (q[i] + u * (q[i + 1] - q[i])
            - (h * h / 6.0) * u * (1.0 - u) * ((2.0 - u) * M[i] + (1.0 + u) * M[i + 1]))


def evaluate(q100, dt_src, tq, jlim=2000.0, vlim=36.0, alim=600.0):
    """planner_sweep's own verdict on a trajectory."""
    if len(q100) < 12:
        return None
    q = to_4ms(np.asarray(q100, float), dt_src)
    chk, qd, qdd = PS.check(q, jlim, vlim, alim)
    tau, _ = tq.peaks(q, qd, qdd)
    viol = list(chk["viol"])
    if (tau > PS.TAU_LIM * 1.005).any():
        viol.append("torque")
    return {"dur": float(chk["dur"]),
            "vpk": chk["vpk"].round(2).tolist(),
            "apk": chk["apk"].round(1).tolist(),
            "jpk": chk["jpk"].round(0).tolist(),
            "tau": tau.round(2).tolist(),
            "viol": ",".join(viol), "feasible": len(viol) == 0}


# ------------------------------------------------------------------ env
def make_env(ep_len, dq_max_deg, seed):
    import torch
    import warp as wp
    wp.init()
    from env_paper import PaperPickEnv
    torch.manual_seed(seed)
    env = PaperPickEnv(nworld=1, device="cuda:0", xml=SCENE, dr=False, drive="real",
                       ep_len=ep_len, grasp_shaping=True, obs_ee=True,
                       reach_target="grasp", lift_dense=True, w_reach=0.5,
                       w_track_c=4, w_track_f=8, seed=seed, dq_max_deg=dq_max_deg)
    env.auto_reset = False
    env.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env.reset(torch.ones(1, dtype=torch.bool, device=env.device))
    return env, torch


def place(env, torch, cx, cy):
    """Park the object at (cx, cy, OBJ_Z) and the arm exactly at START_Q."""
    import env_warp as E
    qa = env.jadr_obj
    T = lambda v: torch.tensor(v, dtype=torch.float32, device=env.device)  # noqa: E731
    env.qpos[0, qa:qa + 2] = T([cx, cy])
    env.qpos[0, qa + 2] = OBJ_Z
    env.qpos[0, qa + 3] = 1.0
    env.qpos[0, qa + 4:qa + 7] = 0.0
    env.qpos[0, :6] = T(PS.START_Q)
    env.qvel[0, :] = 0.0
    E.mjw.forward(env.m, env.d)
    for nm in ("q_target", "q_target_prev", "q_drive", "q_meas_lag"):
        getattr(env, nm)[0] = T(PS.START_Q)
    for nm in ("v_drive", "v_buf", "qd_meas_lag"):
        getattr(env, nm)[0] = 0.0
    obj = env._obj_pos()[0].cpu().numpy()
    return obj, env._grasp_point()[0].cpu().numpy()


def logged(env, sink):
    """Monkey-patch _step_real_drive so every 10 ms command tick lands in `sink`."""
    raw = env._step_real_drive

    def wrapped(log=None, _raw=raw, _sink=sink):
        return _raw(log=_sink)
    env._step_real_drive = wrapped
    return raw


def roll_policy(env, torch, act_fn, grasp, steps, obs_noise, seed):
    sink = []
    logged(env, sink)
    g = torch.Generator(device=env.device); g.manual_seed(seed)
    d10 = []
    for _ in range(steps):
        with torch.no_grad():
            o = env.observe()
            if obs_noise > 0:
                o = o + torch.randn(o.shape, generator=g, device=env.device) * obs_noise
            a = act_fn(o)
        env.step(a)
        tip = env._tcp()[0][0].cpu().numpy()
        d10.append(float(np.linalg.norm(tip - grasp)))
    q100 = torch.cat(sink, dim=0).cpu().numpy() if sink else np.zeros((0, 6))
    return q100, np.asarray(d10)


def roll_plan(env, torch, q_knots, grasp, steps, hold):
    """Feed the planner's 100 ms knots as 10 Hz joint-delta actions (planner_touch)."""
    sink = []
    logged(env, sink)
    targets = list(q_knots) + [q_knots[-1]] * hold
    d10, n_clip = [], 0
    for q_k in targets[1:][:steps]:
        need = torch.as_tensor(q_k, dtype=torch.float32, device=env.device)[None] - env.q_target
        a6 = need / env.dq_max
        if float(a6.abs().max()) > 1.0 + 1e-6:
            n_clip += 1
        a = torch.cat([a6.clamp(-1, 1), -torch.ones(1, 1, device=env.device)], dim=-1)
        env.step(a)
        d10.append(float(np.linalg.norm(env._tcp()[0][0].cpu().numpy() - grasp)))
    return torch.cat(sink, dim=0).cpu().numpy(), np.asarray(d10), n_clip


def t_1cm(d10, tol=0.01):
    hit = np.nonzero(d10 < tol)[0]
    return (float(hit[0] + 1) / 10.0) if len(hit) else None


def load_policies(env, torch, base_path, res_path, bound):
    from ppo import AC
    obs_dim = env.observe().shape[-1]
    dev = env.device

    def _ac(path, extra=0):
        ac = AC(obs_dim=obs_dim, arch="paper", critic_extra=extra).to(dev)
        ck = torch.load(path, map_location=dev, weights_only=False)
        ac.load_state_dict(ck["ac"] if "ac" in ck else ck)
        ac.eval()
        return ac, ck
    base, _ = _ac(base_path)
    ckr = torch.load(res_path, map_location=dev, weights_only=False)
    res, _ = _ac(res_path, extra=(5 if ckr.get("critic_priv") else 0))
    bd = float(bound if bound is not None else ckr.get("residual_bound", 0.3))
    return ({"dagger": lambda o: torch.tanh(base.pi(o)),
             "residual": lambda o: (torch.tanh(base.pi(o))
                                    + bd * torch.tanh(res.pi(o))).clamp(-1.0, 1.0)}, bd)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/planner_sweep"))
    ap.add_argument("--base", default="/home/lisc-frank/pnp_rl/dagger6_real/bc_iter10.pt")
    ap.add_argument("--residual", default="/home/lisc-frank/pnp_rl/resid1_real/best.pt")
    ap.add_argument("--bound", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--dq_max", type=float, default=2.0)
    ap.add_argument("--plan_dq_max", type=float, default=6.0)
    ap.add_argument("--obs_noise", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traces_only", action="store_true",
                    help="re-run only the trace case and rewrite the overlay plots")
    ap.add_argument("--summary_only", action="store_true",
                    help="re-do the table/plots from an existing rl_vs_planner.json")
    a = ap.parse_args()
    a.out = os.path.expanduser(a.out)

    if a.traces_only:
        d = json.load(open(os.path.join(a.out, "rl_vs_planner.json")))
        tq = Torque()
        plan_q = np.load(os.path.join(a.out, "planner_best_traj.npz"))
        cx, cy = PS.CASES[TRACE_CASE]
        qp = np.asarray(plan_q[f"case{TRACE_CASE}"], float)
        tr = {"planner_plan": {"dt": PS.DT, "q": qp.tolist()}}
        env, torch = make_env(a.steps, a.plan_dq_max, a.seed)
        acts, _ = load_policies(env, torch, a.base, a.residual, a.bound)
        obj, grasp = place(env, torch, cx, cy)
        _, q_k = PS.to_knots(np.arange(len(qp)) * PS.DT, qp)
        q100, _, _ = roll_plan(env, torch, q_k, grasp, a.steps, 20)
        tr["planner_exec"] = {"dt": CMD_DT, "q": q100.tolist()}
        del env
        for tag, fn in acts.items():
            env, torch = make_env(a.steps, a.dq_max, a.seed)
            obj, grasp = place(env, torch, cx, cy)
            q100, _ = roll_policy(env, torch, fn, grasp, a.steps, a.obs_noise, a.seed)
            tr[tag] = {"dt": CMD_DT, "q": q100.tolist()}
            del env
        d["traces"], d["traces_case"] = tr, TRACE_CASE
        json.dump(d, open(os.path.join(a.out, "rl_vs_planner.json"), "w"), default=float)
        summarise(a, d["rows"], tr)
        return
    if a.summary_only:
        d = json.load(open(os.path.join(a.out, "rl_vs_planner.json")))
        summarise(a, d["rows"], d["traces"])
        return

    tq = Torque()
    plan_q = np.load(os.path.join(a.out, "planner_best_traj.npz"))
    plan_meta = json.load(open(os.path.join(a.out, "planner_best_meta.json")))

    env, torch = make_env(a.steps, a.dq_max, a.seed)
    acts, bound = load_policies(env, torch, a.base, a.residual, a.bound)
    print(f"[rl] fused residual bound {bound}", flush=True)
    del env

    rows, traces = [], {}
    for ci, (cx, cy) in enumerate(PS.CASES):
        # ---- planner: the PLAN itself (what planner_sweep judged) ----
        qp = np.asarray(plan_q[f"case{ci}"], float)
        ev = evaluate(qp, PS.DT, tq)
        ev.update(policy="planner_plan", case=ci, cx=cx, cy=cy, window="approach",
                  t_1cm=plan_meta[f"case{ci}"]["dur"])
        rows.append(ev)

        # ---- planner: the SAME plan executed through the measured drive ----
        env, torch = make_env(a.steps, a.plan_dq_max, a.seed)
        obj, grasp = place(env, torch, cx, cy)
        t_k, q_k = PS.to_knots(np.arange(len(qp)) * PS.DT, qp)
        q100, d10, n_clip = roll_plan(env, torch, q_k, grasp, a.steps, 20)
        q100_plan = q100
        t1 = t_1cm(d10)
        n_app = int(round((t1 or (len(d10) / 10.0)) / CMD_DT))
        for win, seg in (("approach", q100[:max(n_app, 12)]), ("episode", q100)):
            e = evaluate(seg, CMD_DT, tq)
            e.update(policy="planner_exec", case=ci, cx=cx, cy=cy, window=win,
                     t_1cm=t1, d_min_cm=100 * float(d10.min()), n_clip=n_clip)
            rows.append(e)
        del env

        # ---- the two policies ----
        for tag, fn in acts.items():
            env, torch = make_env(a.steps, a.dq_max, a.seed)
            obj, grasp = place(env, torch, cx, cy)
            q100, d10 = roll_policy(env, torch, fn, grasp, a.steps, a.obs_noise, a.seed)
            t1 = t_1cm(d10)
            n_app = int(round((t1 or (len(d10) / 10.0)) / CMD_DT))
            for win, seg in (("approach", q100[:max(n_app, 12)]), ("episode", q100)):
                e = evaluate(seg, CMD_DT, tq)
                e.update(policy=tag, case=ci, cx=cx, cy=cy, window=win, t_1cm=t1,
                         d_min_cm=100 * float(d10.min()))
                rows.append(e)
            if ci == TRACE_CASE:
                traces[tag] = {"dt": CMD_DT, "q": q100.tolist()}
            del env
        if ci == TRACE_CASE:
            traces["planner_exec"] = {"dt": CMD_DT, "q": q100_plan.tolist()}
            traces["planner_plan"] = {"dt": PS.DT, "q": qp.tolist()}
        print(f"[rl] case {ci} ({cx},{cy}) done", flush=True)

    json.dump({"rows": rows, "bound": bound, "obj_z": OBJ_Z,
               "note": "object centre z=0.020 (top 0.040); planner rows re-planned "
                       "at that top", "traces_case": TRACE_CASE, "traces": traces},
              open(os.path.join(a.out, "rl_vs_planner.json"), "w"), default=float)
    summarise(a, rows, traces)


def summarise(a, rows, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pols = ["planner_plan", "planner_exec", "dagger", "residual"]
    tbl = {}
    for p in pols:
        for win in ("approach", "episode"):
            sel = [r for r in rows if r["policy"] == p and r["window"] == win and r]
            if not sel:
                continue
            t1 = [r["t_1cm"] for r in sel if r.get("t_1cm") is not None]
            tbl[(p, win)] = {
                "n": len(sel), "n_reached": len(t1),
                "t_1cm_mean": float(np.mean(t1)) if t1 else None,
                "t_1cm_worst": float(np.max(t1)) if t1 else None,
                "vpk_mean": np.mean([r["vpk"] for r in sel], axis=0).round(2).tolist(),
                "vpk_worst": np.max([r["vpk"] for r in sel], axis=0).round(2).tolist(),
                "apk_worst": np.max([r["apk"] for r in sel], axis=0).round(1).tolist(),
                "jpk_worst": np.max([r["jpk"] for r in sel], axis=0).round(0).tolist(),
                "tau_worst": np.max([r["tau"] for r in sel], axis=0).round(2).tolist(),
                "n_feasible": int(sum(r["feasible"] for r in sel)),
                "viol": sorted({v for r in sel for v in
                                (r["viol"].split(",") if r["viol"] else [])})}
    json.dump({f"{p}|{w}": v for (p, w), v in tbl.items()},
              open(os.path.join(a.out, "rl_vs_planner_summary.json"), "w"),
              indent=2, default=str)

    show = [p for p in pols if (p, "approach") in tbl]
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    w, x = 0.8 / len(show), np.arange(6)
    cols = {"planner_plan": "#2f6fdb", "planner_exec": "#7aa7f0",
            "dagger": "#e0a458", "residual": "#2a9d8f"}
    for k, (key, lim, ttl) in enumerate([("vpk_worst", 36, "peak |qd| (deg/s)"),
                                         ("apk_worst", 600, "peak |qdd| (deg/s^2)"),
                                         ("jpk_worst", np.degrees(2000.0), "peak |jerk| (deg/s^3)"),
                                         ("tau_worst", None, "peak |tau| (Nm)")]):
        for i, p in enumerate(show):
            ax[k].bar(x + i * w - 0.4, tbl[(p, "approach")][key], w, label=p, color=cols[p])
        if lim:
            ax[k].axhline(lim, color="#d1495b", ls="--", lw=1, label="limit")
        if key == "tau_worst":
            for j, tl in enumerate(PS.TAU_LIM):
                ax[k].plot([j - 0.4, j + 0.4], [tl, tl], color="#d1495b", ls="--", lw=1)
        ax[k].set_xticks(x); ax[k].set_xticklabels([f"j{j + 1}" for j in range(6)])
        ax[k].set_title(ttl, fontsize=10)
        ax[k].set_yscale("log")
        if k == 0:
            ax[k].legend(fontsize=8)
    fig.suptitle("RL policies vs the best planner config -- same limit pipeline, worst of the "
                 "5 sweep positions, APPROACH window (object top 0.040)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(a.out, "rl_vs_planner.png"), dpi=110)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.2))
    for p, tr in traces.items():
        q = np.asarray(tr["q"], float)
        dt = float(tr["dt"])
        qd = np.degrees(np.gradient(q, dt, axis=0))
        t = np.arange(len(q)) * dt
        ax2[0].plot(t, qd[:, 1], lw=1, color=cols.get(p), label=p)
        ax2[1].plot(t, qd[:, 2], lw=1, color=cols.get(p), label=p)
    for k, nm in enumerate(("joint2", "joint3")):
        ax2[k].axhline(36, color="#d1495b", ls="--", lw=1)
        ax2[k].axhline(-36, color="#d1495b", ls="--", lw=1)
        ax2[k].set_xlabel("t (s)"); ax2[k].set_ylabel("deg/s")
        cx, cy = PS.CASES[TRACE_CASE]
        ax2[k].set_title(f"{nm} velocity, case {TRACE_CASE} ({cx}, {cy})")
    ax2[0].legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(a.out, "rl_vs_planner_traces.png"), dpi=110)
    print(f"[rl] {a.out}/rl_vs_planner.png + _traces.png")
    for (p, win), v in tbl.items():
        print(f"[rl] {p:14s} {win:9s} t1cm mean {v['t_1cm_mean']} worst {v['t_1cm_worst']} "
              f"reached {v['n_reached']}/{v['n']}  feasible {v['n_feasible']}/{v['n']} "
              f"viol {v['viol']}")
        print(f"        vpk {v['vpk_worst']}  apk {v['apk_worst']}")
        print(f"        jpk {v['jpk_worst']}  tau {v['tau_worst']}")


if __name__ == "__main__":
    main()

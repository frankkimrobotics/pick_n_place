#!/usr/bin/env python3
"""bc_curobo :: cuRobo-planned teacher + DAgger for the paper-style pick-and-place env.

Recommendation 1+2 (2026-09-19): the scripted IK teacher was crude (fixed 2 deg/decision
descent) and one-shot BC of it collapsed under PPO on the measured drive. Here:
  * transit phases (home -> hover above the object, lifted -> carry to the goal) come from the
    cuRobo planner server (127.0.0.1:9997, curobo2 env): collision-free, velocity-profiled,
    time-scaled to the env's action clamp;
  * the contact phase is a SLOW scripted press (--press_rate deg/decision) with suction on,
    held until the seal latches (the part the measured drive needs);
  * DAgger: after the initial teacher set, the cloned student drives the worlds and an
    IK-waypoint expert (same phase logic, no planner) relabels every visited state; the
    aggregated set is re-fit each iteration. Per-iteration metrics -> metrics.json.
Frames: cuRobo's `tcp` link and the sim's cup-tip site differ by a fixed tool-frame offset
(measured 2.0 cm); goals sent to the planner are corrected for it. The planner world gets the
sim table as a cuboid (top at z = 0.01) for the session.

  $PY rl/bc_curobo.py --drive real --out ~/pnp_rl/dagger_real
"""
import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np
import torch
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from env_paper import PaperPickEnv  # noqa: E402
from env_warp import CTRL_HZ  # noqa: E402

R_DOWN_QUAT = [0.0, 0.7071, 0.7071, 0.0]      # wxyz of the demo's R_DOWN (cup down, +90 yaw)


def rpc(req, timeout=120):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=timeout)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        d = s.recv(1 << 20)
        if not d:
            break
        buf += d
    s.close()
    return json.loads(buf.split(b"\n")[0])


def mat2quat(M):
    w = math.sqrt(max(0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2
    x = math.sqrt(max(0, 1 + M[0, 0] - M[1, 1] - M[2, 2])) / 2
    y = math.sqrt(max(0, 1 - M[0, 0] + M[1, 1] - M[2, 2])) / 2
    z = math.sqrt(max(0, 1 - M[0, 0] - M[1, 1] + M[2, 2])) / 2
    return [w, math.copysign(x, M[2, 1] - M[1, 2]), math.copysign(y, M[0, 2] - M[2, 0]), math.copysign(z, M[1, 0] - M[0, 1])]


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def plan_to_targets(start_q, goal_pos_sim, tool_off, dq_max):
    """cuRobo plan_pose -> list of per-decision joint targets (rad), time-scaled so the largest
    per-decision joint move <= 0.9*dq_max. Returns None if planning fails."""
    R = quat2mat(R_DOWN_QUAT)
    goal_curobo = np.asarray(goal_pos_sim) - R @ tool_off
    res = rpc({"type": "plan_pose", "start_q": [float(x) for x in start_q],
               "goal_pose": [*[float(x) for x in goal_curobo], *R_DOWN_QUAT], "max_attempts": 4})
    if not res.get("success"):
        return None
    traj = np.asarray(res["trajectory"], float)
    if len(traj) < 2:
        return None
    dt = float(res["dt"])
    peak = np.abs(np.diff(traj, axis=0)).max() / dt                 # rad/s
    T = dt * (len(traj) - 1) * max(1.0, peak / (0.9 * dq_max * CTRL_HZ))
    n = max(2, int(math.ceil(T * CTRL_HZ)) + 1)
    ts = np.linspace(0, dt * (len(traj) - 1), n)
    src = np.arange(len(traj)) * dt
    return np.stack([np.interp(ts, src, traj[:, j]) for j in range(6)], 1)


class Teacher:
    """Phase controller; cuRobo for transits when available, IK waypoints otherwise."""

    def __init__(self, env, demo, dik, use_planner, press_rate, tool_off, dq_max, press_depth=0.012):
        self.env, self.demo, self.dik = env, demo, dik
        self.press_depth = press_depth       # IK press target below the grasp point (m); 4 mm was marginal vs the 3 mm latch
        self.use_planner, self.tool_off, self.dq_max = use_planner, tool_off, dq_max
        self.press_rate = math.radians(press_rate)
        self.rate = 0.9 * dq_max
        N = env.nworld
        self.phase = torch.zeros(N, dtype=torch.long, device=env.device)
        self.plans = [None] * N          # per world: list of targets for the current transit
        self.q_hover = None
        self.q_grasp = None
        self.q_carry = [None] * N
        self.n_plan_fail = 0

    def ik(self, pos, seed):
        q, err = self.demo["ik"](self.env.mjm, self.dik, "tcp", [float(x) for x in pos], self.demo["R_DOWN"], seed)
        return q if err < 0.01 else None

    def new_episode(self):
        env, N = self.env, self.env.nworld
        self.phase[:] = 0
        gp = env._grasp_point().cpu().numpy()
        q0 = env.qpos[:, :6].cpu().numpy()
        self.q_hover = np.zeros_like(q0); self.q_grasp = np.zeros_like(q0)
        self.q_carry = [None] * N
        for w in range(N):
            qh = self.ik(gp[w] + [0, 0, 0.04], q0[w]); self.q_hover[w] = qh if qh is not None else q0[w]
            qg = self.ik(gp[w] + [0, 0, -self.press_depth], self.q_hover[w]); self.q_grasp[w] = qg if qg is not None else self.q_hover[w]
            self.plans[w] = None
            if self.use_planner:
                p = plan_to_targets(q0[w], gp[w] + [0, 0, 0.04], self.tool_off, self.dq_max)
                if p is None:
                    self.n_plan_fail += 1
                else:
                    self.plans[w] = list(p[1:])

    def act(self):
        """Expert action for every world from its CURRENT state (used for teacher rollouts
        and for DAgger relabeling alike)."""
        env, N, dev = self.env, self.env.nworld, self.env.device
        tcp, R = env._tcp()
        gpt = env._grasp_point()
        d_hover = torch.norm(tcp - (gpt + torch.tensor([0, 0, 0.04], device=dev)), dim=-1)
        lifted = (env._obj_pos()[:, 2] - float(env.half[2])) > env.paper["h_min"] + 0.01
        # phase transitions from state
        self.phase = torch.where((self.phase == 0) & (d_hover < 0.012), torch.ones_like(self.phase), self.phase)
        self.phase = torch.where((self.phase == 1) & env.sealed, torch.full_like(self.phase, 2), self.phase)
        self.phase = torch.where((self.phase >= 1) & ~env.sealed & ~env.ever_sealed, torch.ones_like(self.phase), self.phase)
        self.phase = torch.where((self.phase == 2) & lifted, torch.full_like(self.phase, 3), self.phase)
        self.phase = torch.where((self.phase == 3) & ~env.sealed, torch.zeros_like(self.phase), self.phase)  # lost it: start over
        q_now = env.q_target.cpu().numpy()
        qh_t = torch.tensor(self.q_hover, device=dev, dtype=torch.float32)
        qg_t = torch.tensor(self.q_grasp, device=dev, dtype=torch.float32)
        tgt = torch.where((self.phase == 1)[:, None], qg_t, qh_t)      # phase 0 default: hover; 2: back to hover (lift)
        rate = torch.full((N, 1), self.rate, device=dev)
        rate[self.phase == 1] = self.press_rate
        # carry targets (IK once per episode per world, when entering phase 3)
        ph = self.phase.cpu().numpy()
        goal = env.goal.cpu().numpy(); obj = env._obj_pos().cpu().numpy(); tcpn = tcp.cpu().numpy()
        for w in np.nonzero(ph == 3)[0]:
            if self.q_carry[w] is None:
                target_tcp = goal[w] + (tcpn[w] - obj[w])
                if self.use_planner:
                    p = plan_to_targets(env.qpos[w, :6].cpu().numpy(), target_tcp, self.tool_off, self.dq_max)
                    if p is not None:
                        self.plans[w] = list(p[1:])
                qc = self.ik(target_tcp, q_now[w])
                self.q_carry[w] = qc if qc is not None else q_now[w]
        qc_t = torch.tensor(np.stack([q if q is not None else q_now[i] for i, q in enumerate(self.q_carry)]), device=dev, dtype=torch.float32)
        tgt = torch.where((self.phase == 3)[:, None], qc_t, tgt)
        # planned transits override the straight-line target in phases 0 and 3
        for w in range(N):
            if self.plans[w] and ph[w] in (0, 3):
                tgt[w] = torch.tensor(self.plans[w].pop(0), device=dev, dtype=torch.float32)
                if not self.plans[w]:
                    self.plans[w] = None
            elif ph[w] not in (0, 3):
                self.plans[w] = None
        dq = tgt - env.q_target
        sc = (rate / dq.abs().max(-1, keepdim=True).values.clamp(min=1e-6)).clamp(max=1.0)
        act6 = (dq * sc / env.dq_max).clamp(-1, 1)
        suc = torch.where((self.phase >= 1), 1.0, -1.0)[:, None]
        return torch.cat([act6, suc], -1)


def rollout(env, teacher, policy, noise, EP, beta):
    """One batch of episodes. policy=None -> teacher drives; else the student drives with the
    teacher labelling (DAgger, beta = prob of executing the teacher action)."""
    N, dev = env.nworld, env.device
    env.reset(torch.ones(N, dtype=torch.bool, device=dev))
    teacher.new_episode()
    O, A = [], []
    for k in range(EP):
        obs = env.observe()
        a_exp = teacher.act()
        if policy is None:
            a = a_exp.clone(); a[:, :6] += torch.randn(N, 6, device=dev) * noise
        else:
            with torch.no_grad():
                a = torch.tanh(policy(obs))
            use_t = torch.rand(N, device=dev) < beta
            a = torch.where(use_t[:, None], a_exp, a)
        O.append(obs.clone()); A.append(a_exp.clone())
        _, r, done, info = env.step(a.clamp(-1, 1))
    ok = info["placed"]
    return torch.stack(O, 1), torch.stack(A, 1), ok, env.ever_sealed.clone()


def pre_tanh(a):
    """PPO executes tanh(mu + noise): clone mu = atanh(a) so the deterministic policy reproduces a."""
    return torch.atanh(a.clamp(-0.97, 0.97))


def fit(ac, X, Y, epochs, dev):
    Y = pre_tanh(Y)
    opt = torch.optim.Adam(ac.pi.parameters(), lr=1e-3)
    idx = torch.arange(X.shape[0], device=dev)
    for ep in range(epochs):
        perm = idx[torch.randperm(idx.numel(), device=dev)]
        for i in range(0, perm.numel(), 4096):
            j = perm[i:i + 4096]
            loss = ((ac.pi(X[j]) - Y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


def evaluate(env, ac, EP):
    N, dev = env.nworld, env.device
    env.reset(torch.ones(N, dtype=torch.bool, device=dev))
    for k in range(EP):
        with torch.no_grad():
            a = torch.tanh(ac.pi(env.observe()))
        _, r, done, info = env.step(a)
    return float(info["placed"].float().mean()), float(env.ever_sealed.float().mean()), float(info["ep_comp"].sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--nworld", type=int, default=128)
    ap.add_argument("--teacher_batches", type=int, default=6)
    ap.add_argument("--dagger_iters", type=int, default=3)
    ap.add_argument("--dagger_batches", type=int, default=3)
    ap.add_argument("--press_rate", type=float, default=0.6, help="deg/decision during the press (slow for the real drive)")
    ap.add_argument("--press_depth", type=float, default=0.02, help="press target below the grasp point (m); sweep 2026-09-19: 4 mm 3 %, 8 mm 65 %, 12 mm 80 %, 20 mm 91 % teacher success on the measured drive")
    ap.add_argument("--ep_len", type=int, default=100, help="episode length (decisions); 150 for the measured drive (slower arm)")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no_planner", action="store_true", help="IK waypoints for transits too (no cuRobo)")
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/dagger_real"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wp.init()
    import mujoco
    env = PaperPickEnv(nworld=a.nworld, xml=a.scene, dr=True, drive=a.drive, ep_len=a.ep_len)
    env.auto_reset = False
    demo = {"__file__": os.path.join(os.path.dirname(HERE), "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    dik = mujoco.MjData(env.mjm)
    dev, EP = env.device, env.paper["ep_len"]
    use_planner = not a.no_planner
    tool_off = np.zeros(3)
    if use_planner:
        # planner world = sim table (top z=0.01) + the server's base cuboids; tool-frame offset from FK
        print(rpc({"type": "set_world", "cuboids": [{"name": "sim_table", "dims": [0.56, 0.9, 0.02], "pose": [0.42, 0.0, 0.0, 1, 0, 0, 0]}]}), flush=True)
        env.reset(torch.ones(env.nworld, dtype=torch.bool, device=dev))
        q0 = env.qpos[0, :6].cpu().numpy().tolist(); tcp, R = env._tcp()
        fk = rpc({"type": "fk", "q": q0}); p = np.array(fk["pos"][0]); Rm = R[0].cpu().numpy()
        tool_off = Rm.T @ (tcp[0].cpu().numpy() - p)          # sim cup tip in cuRobo tool frame
        print(f"[dagger] tool offset (tool frame) {np.round(tool_off, 4).tolist()} ({1000 * np.linalg.norm(tool_off):.1f} mm)", flush=True)
    teacher = Teacher(env, demo, dik, use_planner, a.press_rate, tool_off, env.dq_max, press_depth=a.press_depth)
    from ppo import AC
    metrics = []
    X, Y = [], []
    t0 = time.time()
    # ---- iteration 0: teacher demonstrations ----
    n_ok = n_ep = 0
    for b in range(a.teacher_batches):
        O, Aexp, ok, es = rollout(env, teacher, None, a.noise, EP, 1.0)
        X.append(O.reshape(-1, O.shape[-1])); Y.append(Aexp.reshape(-1, 7))      # DAgger keeps ALL states
        n_ok += int(ok.sum()); n_ep += env.nworld
        print(f"[dagger] teacher batch {b}: success {100 * ok.float().mean():.1f} %  sealed {100 * es.float().mean():.0f} %  plan fails {teacher.n_plan_fail}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ac = AC(obs_dim=X[0].shape[1], arch="paper").to(dev)
    for it in range(a.dagger_iters + 1):
        Xc, Yc = torch.cat(X), torch.cat(Y)
        mse = fit(ac, Xc, Yc, a.epochs, dev)
        with torch.no_grad():
            ac.log_std.fill_(-1.5)
        succ, seal, ret = evaluate(env, ac, EP)
        torch.save({"X": Xc.cpu(), "Y": Yc.cpu()}, os.path.join(a.out, "dataset.pt"))
        rec = dict(iter=it, dataset=int(Xc.shape[0]), mse=mse, student_success=succ, student_seal=seal, student_return=ret,
                   teacher_success=n_ok / max(1, n_ep), minutes=(time.time() - t0) / 60)
        metrics.append(rec)
        print(f"[dagger] iter {it}: dataset {Xc.shape[0]} mse {mse:.4f} | student success {100 * succ:.1f} % seal {100 * seal:.0f} % return {ret:.2f}", flush=True)
        json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
        torch.save({"ac": ac.state_dict(), "step": 0}, os.path.join(a.out, f"bc_iter{it}.pt"))
        if it == a.dagger_iters:
            break
        beta = max(0.0, 0.5 - 0.2 * it)                    # mixing: 0.5, 0.3, 0.1
        for b in range(a.dagger_batches):
            O, Aexp, ok, es = rollout(env, teacher, ac.pi, 0.0, EP, beta)
            X.append(O.reshape(-1, O.shape[-1])); Y.append(Aexp.reshape(-1, 7))
            print(f"[dagger]   relabel batch {b} (beta {beta:.1f}): student-driven success {100 * ok.float().mean():.1f} %  sealed {100 * es.float().mean():.0f} %", flush=True)
    torch.save({"ac": ac.state_dict(), "step": 0}, os.path.join(a.out, "bc_init.pt"))
    print(f"[dagger] saved {a.out}/bc_init.pt", flush=True)


if __name__ == "__main__":
    main()

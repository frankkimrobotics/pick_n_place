#!/usr/bin/env python3
"""sim_touch_compare :: 'touch a point and return' under four trajectory-
execution strategies, in the mujoco_warp twin with the hardware-calibrated
Pro 630 drive model (rl/env_warp DRIVE: 50 deg/s, 800 deg/s^2, 45 ms
dead-time, spline streaming law k0=10/k1=0.3/vff=1, 45 ms lead).

All four drive the SAME measured plant (dead-time ring buffer -> accel cap ->
drive position loop -> physics); they differ only in the controller/reference:

  1 curobo_native   cuRobo plan resampled at its OWN TOPP-RA timing -> spline stream law
  2 curobo_retime   same geometry, uniform velocity-cap timing        -> spline stream law
  3 rl              the trained policy's decision targets (closed loop)-> spline stream law
  4 mpc             cuRobo geometry as setpoints, tracked by the lag-aware LQR-clamp MPC

No object / no suction: the "touch" is the arm reaching the target pose.
Records per-command-tick joint response vs the commanded decision targets.

    python3 sim_touch_compare.py --which 1,2,4 --repeats 1
"""
import argparse
import json
import os
import socket
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rl"))
import warp as wp                                          # noqa: E402

TOUCH_TCP = [0.34, 0.00, 0.10]     # a low reachable point (down-facing)
DOWN = [0.0, 1.0, 0.0, 0.0]
K_LQR = np.array([53.4798, 4.7131])  # lag-aware DARE gains (deployed mpc_solve)


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=90)
    s.sendall((json.dumps(d) + "\n").encode())
    b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close()
    return json.loads(b)


def plan_pose(q0, xyz, attempts=16):
    for _ in range(3):
        r = rpc({"type": "plan_pose", "start_q": [float(v) for v in q0],
                 "goal_pose": list(xyz) + DOWN, "max_attempts": attempts})
        if r.get("success"):
            return np.array(r["trajectory"]), float(r.get("dt") or 0.025)
    return None, None


def plan_joint(q0, qg, attempts=12):
    for _ in range(3):
        r = rpc({"type": "plan_joint", "start_q": [float(v) for v in q0],
                 "goal_q": [float(v) for v in qg], "max_attempts": attempts})
        if r.get("success"):
            return np.array(r["trajectory"]), float(r.get("dt") or 0.025)
    return None, None


def resample_decisions(traj, native_dt, v_cap=None, ctrl_hz=10):
    """Resample a joint trajectory onto the 10 Hz decision grid.

    v_cap None  -> keep native TOPP-RA timing.
    v_cap set   -> stretch/compress so the fastest joint moves at v_cap deg/s.
    Returns (T, 6) decision targets."""
    M = len(traj)
    t_native = np.arange(M) * native_dt
    if v_cap is not None:
        travel = np.degrees(np.abs(np.diff(traj, axis=0)).sum(0)).max()
        dur = max(travel / v_cap, 0.2)
    else:
        dur = t_native[-1] if t_native[-1] > 1e-3 else (M - 1) * 0.025
    t_scaled = t_native / t_native[-1] * dur if t_native[-1] > 0 else \
        np.linspace(0, dur, M)
    n_dec = max(2, int(round(dur * ctrl_hz)) + 1)
    t_dec = np.linspace(0, dur, n_dec)
    return np.stack([np.interp(t_dec, t_scaled, traj[:, j])
                     for j in range(6)], 1)


# ---------- plant: measured drive with a pluggable per-tick controller ----------
def run_drive(env, targets, controller):
    """Drive the calibrated plant through a sequence of decision targets.

    controller: 'stream' (spline law, env._step_real_drive) or 'mpc'
    (per-joint lag-aware LQR-clamp producing v_cmd into the same plant).
    Logs joint response at every 10 ms command tick. Returns (L, 6) rad."""
    log = []
    if controller == "stream":
        for q_dec in targets:
            env.q_target = torch.as_tensor(q_dec, dtype=torch.float32,
                                           device=env.device)[None]
            env._step_real_drive(log=log)
    elif controller == "mpc":
        # tuned for the 45 ms dead-time: bandwidth-matched position gain +
        # feedforward, MINIMAL velocity feedback (K1 amplifies measurement
        # noise on a dead-time plant -> the deployed K_LQR=[53,4.7] rings)
        _run_mpc(env, targets, log, gains=(10.0, 0.3))
    elif controller == "mpc_deployed":
        _run_mpc(env, targets, log)                      # K_LQR, rings (shown for contrast)
    return torch.stack(log).squeeze(1).cpu().numpy()


def _run_mpc(env, targets, log, gains=None):
    """Lag-aware LQR-clamp tracking each setpoint into the measured plant.
    Mirrors _step_real_drive's plant (dead-time, accel, position loop) but
    replaces the streaming law with v_cmd = vff - clip(K0 e + K1 (qd-vff))."""
    import mujoco_warp as mjw
    from env_warp import CTRL_HZ
    K = K_LQR if gains is None else np.asarray(gains)
    dt_phys = float(env.mjm.opt.timestep)
    sub_per_cmd = max(1, env.substeps // env.n_cmd)
    ar = torch.arange(env.nworld, device=env.device)
    vmax = env.vmax_w
    dev = env.device
    prev = torch.as_tensor(targets[0], dtype=torch.float32, device=dev)[None]
    for q_dec in targets:
        qset = torch.as_tensor(q_dec, dtype=torch.float32, device=dev)[None]
        # feedforward = setpoint velocity over the decision (the lag model's
        # role in _mpc_solve_lag); the LQR-clamp then only corrects the error
        v_ff = (qset - prev) * CTRL_HZ
        prev = qset
        for _c in range(env.n_cmd):
            e = env.q_meas_lag - qset
            qd = env.qd_meas_lag
            v_cmd = v_ff - (K[0] * e + K[1] * (qd - v_ff))
            v_cmd = torch.maximum(torch.minimum(v_cmd, vmax), -vmax)
            env.v_buf[:, env.buf_i] = v_cmd
            v_eff = env.v_buf[ar, (env.buf_i - env.dead_ticks) % env.dead_max]
            env.buf_i = (env.buf_i + 1) % env.dead_max
            for _ in range(sub_per_cmd):
                dv = (v_eff - env.v_drive).clamp(-env.amax_w * dt_phys,
                                                 env.amax_w * dt_phys)
                env.v_drive = env.v_drive + dv
                env.q_drive = env.q_drive + env.v_drive * dt_phys
                qj = env.qpos[:, :6]
                lead = env.q_drive - qj
                from env_warp import DRIVE
                over = lead.abs() > DRIVE["ferror"]
                env.q_drive = torch.where(over, qj + lead.clamp(
                    -DRIVE["ferror"], DRIVE["ferror"]), env.q_drive)
                env.v_drive = torch.where(
                    over & (torch.sign(env.v_drive) == torch.sign(lead)),
                    torch.zeros_like(env.v_drive), env.v_drive)
                tau = (env.gain_scale * (env.kp * (env.q_drive - env.qpos[:, :6])
                       - env.kd * (env.qvel[:, :6] - env.v_drive))).clamp(
                    -env.tau_max, env.tau_max)
                env.ctrl[:, :6] = tau
                mjw.step(env.m, env.d)
            env.q_meas_lag = env.qpos[:, :6].clone()
            env.qd_meas_lag = env.qvel[:, :6].clone()
            log.append(env.qpos[:, :6].clone())


def reset_to(env, q):
    """Park the arm at joint config q (rad) with the drive state settled."""
    qt = torch.as_tensor(q, dtype=torch.float32, device=env.device)
    env.qpos[:, :6] = qt
    env.qvel[:, :6] = 0.0
    env.q_target[:] = qt
    env.q_target_prev[:] = qt
    env.q_hist[:] = qt[None, None]
    env.q_drive[:] = qt
    env.v_drive[:] = 0.0
    env.q_meas_lag[:] = qt
    env.qd_meas_lag[:] = 0.0
    import mujoco_warp as mjw
    mjw.step(env.m, env.d)


def fk_np(env):
    p, _ = env._tcp()
    return p[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="1,2,4")
    ap.add_argument("--v_cap", type=float, default=30.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--policy", default=os.path.join(
        HERE, "rl", "weights", "dagger6_real_iter10.pt"))
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/sim_touch_compare"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wp.init()
    from env_paper import PaperPickEnv
    import config as C

    env = PaperPickEnv(nworld=1, drive="real", dr=False, start="home")
    env.auto_reset = False
    q_base = np.asarray(C.START_Q, float)
    names = {"1": "curobo_native", "2": "curobo_retime", "4": "mpc",
             "3": "rl", "5": "mpc_deployed", "6": "mpc_line"}
    order = [names[c] for c in a.which.split(",")]

    # plan the touch stroke once (shared geometry for 1/2/4)
    reset_to(env, q_base)
    down, ndt = plan_pose(q_base, TOUCH_TCP)
    if down is None:
        sys.exit("touch plan failed")
    print(f"touch plan: {len(down)} wps, native dt {ndt:.3f}s, "
          f"tcp target {TOUCH_TCP}")
    dec_native = resample_decisions(down, ndt, v_cap=None)
    dec_retime = resample_decisions(down, ndt, v_cap=a.v_cap)
    print(f"decisions: native {len(dec_native)} (~{len(dec_native)/10:.1f}s), "
          f"retime@{a.v_cap} {len(dec_retime)} (~{len(dec_retime)/10:.1f}s)")

    ac = None
    if "rl" in order:
        from ppo import AC
        ck = torch.load(a.policy, map_location=env.device, weights_only=False)
        sd = ck["ac"] if "ac" in ck else ck
        obs_dim = sd["pi.0.weight"].shape[1]
        # paper arch = mlp_paper [256,128,64]; default = wider mlp
        arch = "paper" if sd["pi.0.weight"].shape[0] == 256 else "default"
        ac = AC(obs_dim=obs_dim, arch=arch).to(env.device)
        torch.nn.Module.load_state_dict(ac, sd, strict=False)
        ac.eval()

    logs = {}
    for name in order:
        if name == "rl":
            # closed-loop policy: object at the touch xy, suction forced off,
            # log the approach-descend (10 Hz decisions) as the RL 'touch'
            reset_to(env, q_base)
            env.reset(torch.ones(1, dtype=torch.bool, device=env.device))
            reset_to(env, q_base)                      # arm back to base
            op = env._obj_pos().clone()
            op[0, 0], op[0, 1] = TOUCH_TCP[0], TOUCH_TCP[1]
            env.qpos[0, env.jadr_obj] = TOUCH_TCP[0]
            env.qpos[0, env.jadr_obj + 1] = TOUCH_TCP[1]
            import mujoco_warp as mjw
            mjw.step(env.m, env.d)
            resp10 = []
            obs = env.observe()
            for _ in range(40):
                with torch.no_grad():
                    act = torch.tanh(ac.pi(obs))
                act[:, 6] = -1.0                       # suction OFF
                obs, _, _, _ = env.step(act)
                resp10.append(env.qpos[:, :6].clone())
                if env._tcp()[0][0, 2].item() < TOUCH_TCP[2] + 0.01:
                    break
            resp = torch.stack(resp10).squeeze(1).cpu().numpy()
            tcp = fk_np(env)
            logs["rl"] = dict(resp=resp, targ=resp, tcp=tcp.tolist(),
                              z_err_mm=float((tcp[2] - TOUCH_TCP[2]) * 1000),
                              fin_deg=0.0, rms_deg=0.0, ticks=len(resp),
                              hz=10)
            print(f"[rl] policy descended to tcp z {tcp[2]*1000:.1f} mm "
                  f"in {len(resp)} decisions ({len(resp)/10:.1f}s)")
            continue
        reset_to(env, q_base)
        if name == "curobo_native":
            resp = run_drive(env, dec_native, "stream")
            targ = dec_native
        elif name == "curobo_retime":
            resp = run_drive(env, dec_retime, "stream")
            targ = dec_retime
        elif name == "mpc":
            resp = run_drive(env, dec_retime, "mpc")
            targ = dec_retime
        elif name == "mpc_deployed":
            resp = run_drive(env, dec_retime, "mpc_deployed")
            targ = dec_retime
        elif name == "mpc_line":
            # NO cuRobo path: straight line in c-space from base to the goal
            # joint config (cuRobo's endpoint), velocity-capped, tracked by
            # the same tuned MPC. Feedback adjusts for the drive dynamics.
            q_goal = down[-1]
            travel = np.degrees(np.abs(q_goal - q_base)).max()
            n_dec = max(2, int(round(travel / a.v_cap * 10)) + 1)
            dec_line = np.linspace(q_base, q_goal, n_dec)
            resp = run_drive(env, dec_line, "mpc")
            targ = dec_line
        tcp = fk_np(env)
        z_err = tcp[2] - TOUCH_TCP[2]
        # final tracking error (last decision vs response)
        fin = np.degrees(np.abs(resp[-1] - targ[-1])).max()
        # rms cmd-vs-response over the run (resample targets to tick grid)
        tt = np.linspace(0, 1, len(targ))
        tr = np.linspace(0, 1, len(resp))
        tgi = np.stack([np.interp(tr, tt, targ[:, j]) for j in range(6)], 1)
        rms = np.degrees(np.sqrt(np.mean((resp - tgi) ** 2)))
        logs[name] = dict(resp=resp, targ=targ, tcp=tcp.tolist(),
                          z_err_mm=float(z_err * 1000), fin_deg=float(fin),
                          rms_deg=float(rms), ticks=len(resp))
        print(f"[{name}] reached tcp z {tcp[2]*1000:.1f} mm "
              f"(target {TOUCH_TCP[2]*1000:.0f}), final err {fin:.2f} deg, "
              f"rms {rms:.2f} deg, {len(resp)} ticks ({len(resp)/100:.1f}s)")

    plot(logs, a.out)
    json.dump({k: {kk: vv for kk, vv in v.items()
                   if kk not in ("resp", "targ")}
               for k, v in logs.items()},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"[done] {a.out}")


def plot(logs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(curobo_native="#D65F5F", curobo_retime="#4878CF",
                  mpc="#6ACC64", rl="#B47CC7", mpc_deployed="#EE8544", mpc_line="#8C613C")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for name, d in logs.items():
        resp = d["resp"]
        hz = float(d.get("hz", 100))
        t = np.arange(len(resp)) / hz
        lbl = f"{name} ({len(resp)/hz:.1f}s"
        lbl += f", rms {d['rms_deg']:.2f}°)" if d["rms_deg"] > 0 else ")"
        for j in range(6):
            axes[j // 2, j % 2].plot(t, np.degrees(resp[:, j]), lw=1.1,
                                     color=colors.get(name), label=lbl)
    for j in range(6):
        ax = axes[j // 2, j % 2]
        ax.set_title(f"J{j+1}", fontsize=10)
        ax.set_ylabel("deg")
        if j == 0:
            ax.legend(fontsize=8)
    axes[2, 0].set_xlabel("t (s)")
    axes[2, 1].set_xlabel("t (s)")
    fig.suptitle("Sim touch-and-return (mujoco_warp, measured drive): "
                 "joint response by planner/transport")
    fig.tight_layout()
    p = os.path.join(out, "sim_touch_compare.png")
    fig.savefig(p, dpi=110)
    print(f"[plot] {p}")


if __name__ == "__main__":
    main()

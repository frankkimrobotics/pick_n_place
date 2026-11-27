"""End-to-end latency check of BatchTwin: command a step at t = 0 and time when (a) the joint really starts to move,
(b) the host's joint feedback shows it, (c) each camera image shows it. Compared with the configured budget.
  python -m batchsim.tests.test_latency [cuda:0] [scene.xml]        (mj4 env)"""
import json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from batchsim.twin import BatchTwin

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
xml = sys.argv[2] if len(sys.argv) > 2 else "~/pnp_dataset_mega/scene_980/scene.xml"
out = {}
for name, kw in (("real", dict(level="real")), ("real, fast inference", dict(level="real", infer_mean=0.06, infer_sd=0.0, infer_min=0.045)),
                 ("ideal", dict(level="ideal")), ("real, randomised", dict(level="real", dr=True)),
                 ("5 ms stack", dict(level="real", timing="stack_5ms", stream="fast5")),
                 ("5 ms stack, randomised", dict(level="real", timing="stack_5ms", stream="fast5", dr=True))):
    N = 16
    tw = BatchTwin(xml, N, dev, cameras=("zed", "wrist_d405"), tx_spike_p=0.0, infer_sd=kw.pop("infer_sd", 0.0), **kw)
    q0 = tw.world.qpos0.clone(); q0[:6] = torch.tensor([0.0, -0.35, 1.40, 0.17, -1.57, 0.0]); tw.reset(qpos=q0[None].expand(N, -1))
    hold = tw.world.qpos[:, :6].clone()
    for _ in range(60):                                            # settle, and fill the sensor history
        tw.command(hold[:, None, :].expand(-1, 2, -1), 1.0, compute=0.0); tw.advance(1)
    base = tw.observe(); q_start = tw.world.qpos[:, :6].clone(); t0 = tw.t
    tgt = hold.clone(); tgt[:, 0] += 0.35; tgt[:, 1] += 0.20       # 20 deg base turn + 11 deg shoulder: visible in both cameras
    ramp = hold[:, None, :] + (tgt - hold)[:, None, :] * torch.linspace(0, 1, 8, device=dev)[None, :, None]
    lat_cmd = tw.command(ramp, 0.1)                                # knot 0 at the observation time: latency eats into the plan
    seen = {k: torch.full((N,), float("nan"), device=dev) for k in ("true", "fb")}
    qlog, cam_err, cam_age = [tw.world.qpos[:, :6].clone()], {c: 0.0 for c in tw.cams}, {c: [] for c in tw.cams}
    for i in range(120):
        tw.advance(1); o = tw.observe(); t = tw.t - t0
        qlog.append(tw.world.qpos[:, :6].clone())
        moved = dict(true=(tw.world.qpos[:, :6] - q_start).abs().max(1).values > np.radians(0.05),
                     fb=(o["q"] - base["q"]).abs().max(1).values > np.radians(0.05))
        for k, m in moved.items():
            seen[k] = torch.where(m & torch.isnan(seen[k]), torch.full_like(seen[k], t), seen[k])
        if i >= 70:                                                # the rendered state must be the true state `age` ago
            for c, cam in tw.cams.items():
                u = cam.age / tw.cfg["cmd_dt"]                     # ages fall between ticks: interpolate the true log
                k0 = torch.floor(u + 1e-4).long(); w_ = (u - k0).clamp(0, 1)[:, None]; Q = torch.stack(qlog); ar = torch.arange(N, device=dev)
                past = (1 - w_) * Q[len(qlog) - 1 - k0, ar] + w_ * Q[len(qlog) - 2 - k0, ar]
                cam_err[c] = max(cam_err[c], float((cam.qpos[:, :6] - past).abs().max()))
                cam_age[c].append(cam.age.clone())
    ms = lambda x: [round(1000 * float(x.min())), round(1000 * float(x.mean())), round(1000 * float(x.max()))]
    L = tw.lat
    out[name] = dict(cmd_latency_ms=ms(lat_cmd), pi_lead_ms=ms(L["lead"]), drive_dead_ms=ms(L["dead"]), zed_lat_ms=ms(L["zed_lat"]), wrist_lat_ms=ms(L["wrist_lat"]),
                     measured_motion_onset_ms=ms(seen["true"]), measured_fb_minus_motion_ms=ms(seen["fb"] - seen["true"]),
                     zed_frame_age_ms=ms(torch.stack(cam_age["zed"])), wrist_frame_age_ms=ms(torch.stack(cam_age["wrist_d405"])),
                     rendered_state_error_rad=max(cam_err.values()),
                     healthy=bool(tw.healthy().all()))
    print(name, json.dumps(out[name]), flush=True)
    del tw; torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(__file__), "latency.json"), "w"), indent=1)

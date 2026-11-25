"""The vectorised drive chain must reproduce the scalar twin (realsim.twin.Pro630Twin) and, through it, the arm:
replay a hardware reference (1 Hz sine on all joints, tuned gains) in both and compare joint angles tick by tick.
  python -m batchsim.tests.test_drive_parity [cuda:0]        (mj4 env)"""
import glob, json, os, sys
import numpy as np, torch, mujoco
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path[:0] = [ROOT, ROOT + "/realsim/sysid"]
from realsim.twin import Pro630Twin
from realsim.config import make_cfg
from realsim.validate import settle, XML
from batchsim.world import BatchWorld
from batchsim.drive import DriveChain
from batchsim import latency

D = os.path.expanduser("~/pnp_real_runs/sysid"); OFF = np.array([0, 90, 0, 90, 0, 0.0])


def run(tag="sine_012345_fast_new", joint_model=0, dev="cuda:0", N=4):
    f = sorted(glob.glob(f"{D}/*{tag}_pi.csv"))[-1]; ref = np.load(f[:-7] + "_ref.npy"); g = json.load(open(f[:-7] + "_meta.json"))["gains"]
    over = dict(k0=g.get("k0", 10.0), k1=g.get("k1", 0.3), vff=g.get("vff", 1.0), lead=g.get("lead", 0.0), lead_p=g.get("lead_p"),
                vfric=np.radians(g.get("vfric", 0.0)), vmax=np.radians(g.get("vmax", 40.0)), vgain=1.025 * g.get("vel_scale", 17.0) / 17.0,
                tx_spike_p=0.0, joint_model=joint_model)
    cfg = make_cfg("real", **over)
    q_start = np.radians(ref[0] + OFF)
    # ---- scalar twin
    m_ = mujoco.MjModel.from_xml_path(XML); d_ = mujoco.MjData(m_); d_.qpos[:6] = q_start; mujoco.mj_forward(m_, d_)
    tw = Pro630Twin(m_, d_, cfg); settle(tw)
    q0 = tw.d.qpos[:6].copy(); pts = q0[None] + np.radians(ref - ref[0]); t_a = tw.t + 0.4
    tw.send_spline(pts, t_a, dt=0.01); tw.log = []; n_tick = int(round((0.4 + len(ref) * 0.01 + 0.3) / cfg["cmd_dt"]))
    tw.advance(n_tick * cfg["cmd_dt"])
    qs = np.array([r["q"] for r in tw.log]); taus = np.array([r["tau"] for r in tw.log])
    # ---- batched
    A = lambda v: np.full(6, v) if np.isscalar(v) else np.asarray(v, float)
    w = BatchWorld(XML, N, dev, joint_physics=dict(armature=A(cfg["armature"]), damping=A(cfg["damping"]), frictionloss=A(cfg["frictionloss"])))
    qp = w.qpos0.clone(); qp[:6] = torch.tensor(q_start, dtype=torch.float32); w.reset(qpos=qp[None].expand(N, -1))
    lat = latency.sample(cfg, N, dev)
    dr = DriveChain(w, cfg, lat, max_pts=len(ref))
    hold = w.qpos[:, :6].clone()                                   # settle like the scalar twin: hold the start pose 0.5 s
    for _ in range(50):
        dr.send(hold[:, None, :].expand(-1, 2, -1), dr.t, 1.0, dr.t); dr.tick()
    q0b = w.qpos[:, :6].clone()
    ptsb = q0b[:, None, :] + torch.tensor(np.radians(ref - ref[0]), dtype=torch.float32, device=dev)[None]
    dr.send(ptsb, dr.t + 0.4, 0.01, dr.t + cfg["tx_lat"])
    qb, tb = [], []
    for _ in range(n_tick):
        dr.tick(); qb.append(w.qpos[:, :6].clone()); tb.append(dr.tau.clone())
    qb = torch.stack(qb).cpu().numpy(); tb = torch.stack(tb).cpu().numpy()
    dq = np.degrees((qb[:, 0] - q0b[0].cpu().numpy()) - (qs - q0))
    spread = np.degrees(np.abs(qb - qb[:, :1]).max())
    amp = np.degrees(np.abs(qs - q0).max(0))
    return dict(tag=tag, joint_model=joint_model, rms_deg=np.sqrt((dq ** 2).mean(0)).round(4).tolist(), max_deg=np.abs(dq).max(0).round(4).tolist(),
                motion_amp_deg=amp.round(2).tolist(), world_spread_deg=float(spread), tau_rms_Nm=np.sqrt(((tb[:, 0] - taus) ** 2).mean(0)).round(3).tolist(),
                finite=bool(np.isfinite(qb).all()))


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    out = []
    for jm in (0, 1):
        for tag in ("sine_012345_fast_new", "sine_012345_fast_old", "check_j1_stepnew"):
            r = run(tag, jm, dev); out.append(r); print(r, flush=True)
    ok = all(r["finite"] and max(r["rms_deg"]) < 0.02 for r in out)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "drive_parity.json"), "w"), indent=1)
    print("PARITY", "OK" if ok else "FAIL")

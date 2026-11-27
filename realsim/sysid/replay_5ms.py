"""Replay envelope J1 30 deg/s runs through realsim twin under a timing preset (read-only on ~/pnp_real_runs)."""
import glob, json, os, sys
import numpy as np
ROOT = "/home/lisc-frank/Desktop/2026/pick_and_place"; sys.path[:0] = [ROOT, ROOT + "/realsim/sysid"]
from look import load
import mujoco
from realsim.validate import settle, XML, Q_START
from realsim.twin import Pro630Twin
from realsim.config import make_cfg
D = os.path.expanduser("~/pnp_real_runs/envelope"); j = 0   # usage: replay_5ms.py STAMP[,STAMP] timing [json-over]
runs = sys.argv[1].split(","); timing = sys.argv[2]; extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
res = []
for stamp in runs:
    f = glob.glob(f"{D}/{stamp}_vel_j0_30_pi.csv")[0]; Lg = load(f); ref = np.load(f[:-7] + "_ref.npy"); meta = json.load(open(f[:-7] + "_meta.json"))
    g = meta["gains"]
    over = dict(k0=g["k0"], k1=g["k1"], vff=g["vff"], lead=g["lead"], lead_p=g.get("lead_p"), vfric=np.radians(g.get("vfric", 0)),
                vmax=np.radians(g.get("vmax", 40)), taff=g.get("taff", 0.0), vgain=1.025 * g["vel_scale"] / 17.0, tx_spike_p=0.0, timing=timing, **extra)
    m_ = mujoco.MjModel.from_xml_path(XML); d_ = mujoco.MjData(m_); d_.qpos[:6] = Q_START; mujoco.mj_forward(m_, d_)
    cfg = make_cfg("real", **over); tw = Pro630Twin(m_, d_, cfg); settle(tw)
    q0 = tw.d.qpos[:6].copy(); pts = q0[None] + np.radians(ref - ref[0]); t_a = tw.t + 0.4
    tw.send_spline(pts, t_a, dt=0.01); tw.log = []; tw.advance(0.4 + len(ref) * 0.01 + 0.3)
    ts = np.array([r["t"] for r in tw.log]) - t_a; qs = np.degrees(np.array([r["q"][j] for r in tw.log]) - q0[j])
    tr = np.arange(len(ref)) * 0.01; r = ref[:, j] - ref[0, j]
    th = Lg["t"] - meta["t_anchor"]; qh = Lg["q"][:, j] - ref[0, j]
    kh = (th > 0) & (th < tr[-1]); ks = (ts > 0) & (ts < tr[-1])
    eh = qh[kh] - np.interp(th[kh], tr, r); es = qs[ks] - np.interp(ts[ks], tr, r)
    diff = float(np.sqrt(((qs[ks] - np.interp(ts[ks], th, qh)) ** 2).mean()))
    res.append(diff)
    print(f"{stamp} period {meta['done']['period_ms']} lead {g['lead']}: hw err rms {np.sqrt((eh**2).mean()):.3f} max {np.abs(eh).max():.3f} | "
          f"twin[{timing} dt {tw.dt_phys*1e3:.3f} dead {cfg['dead']}] err rms {np.sqrt((es**2).mean()):.3f} max {np.abs(es).max():.3f} | twin-vs-hw rms {diff:.3f} deg", flush=True)
print("mean twin-vs-hw", round(float(np.mean(res)), 4))

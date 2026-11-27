"""batchsim.latency :: every delay between the policy and the arm, per world.

    observation side   camera capture -> host   (frame-period quantised + transport: zed_lat / wrist_lat)
                       joint feedback -> host   (100 Hz sample-and-hold + fb_lat)
    compute            policy inference + host processing (infer_mean / infer_sd / infer_min, host_proc)
    command side       host -> Pi transport (tx_lat, rare tx_spike), Pi lead, drive dead time (in drive.py)

Values come from realsim.config (measured on the arm where noted there). With dr=True each world draws its own.
"""
import numpy as np
import torch


class RingHistory:
    """The last `length` per-tick snapshots of an [N, width] tensor; `at(k)` returns, per world, the snapshot k[n]
    ticks old (k = 0 is the newest)."""

    def __init__(self, nworld, width, length, device):
        self.buf = torch.zeros(length, nworld, width, device=device)
        self.L, self.i, self.n = length, -1, 0
        self.idx = torch.arange(nworld, device=device)

    def fill(self, x, mask=None):
        if mask is None:
            self.buf[:] = x[None]
        else:
            self.buf[:, mask] = x[mask][None]

    def push(self, x):
        self.i = (self.i + 1) % self.L
        self.buf[self.i] = x
        self.n += 1

    def at(self, k):
        k = torch.as_tensor(k, device=self.buf.device).long().clamp(0, self.L - 1).expand(self.idx.shape[0])
        return self.buf[(self.i - k) % self.L, self.idx]


def sample(cfg, nworld, device, dr=False, gen=None):
    """Per-world latency / drive parameters as [N] or [N, 1] tensors from a realsim config dict.
    dr=True spreads them over realsim.config.DR (the same ranges the scalar twin draws from)."""
    from realsim.config import DR
    g = gen or torch.Generator(device="cpu").manual_seed(0)
    U = lambda lo, hi: (lo + (hi - lo) * torch.rand(nworld, generator=g)).to(device)
    C = lambda v: torch.full((nworld,), float(v), device=device)
    p = dict(dead=C(cfg["dead"]), lead=C(cfg["lead"]), amax=C(cfg["amax"]), vmax=C(cfg["vmax"]),
             zed_lat=C(cfg["zed_lat"]), wrist_lat=C(cfg["wrist_lat"]), infer_mean=C(cfg["infer_mean"]),
             tx_lat=C(cfg["tx_lat"]), fb_lat=C(cfg["fb_lat"]), phys_scale=C(1.0),
             cam_phase=torch.zeros(nworld, device=device))
    lead_follows = cfg.get("lead_p") is None and abs(cfg["lead"] - cfg["dead"]) < 0.011
    if dr:
        p["dead"] = U(*cfg.get("dr_dead", DR["dead"]))
        if lead_follows:
            p["lead"] = p["dead"].clone()                   # the Pi lead is tuned to the dead time
        p["amax"] = p["amax"] * U(*DR["amax_mult"])
        p["vmax"] = p["vmax"] * U(*DR["vmax_mult"])
        p["zed_lat"] = U(*cfg.get("dr_zed_lat", DR["zed_lat"]))        # timing preset ranges (realsim.config.TIMING)
        p["wrist_lat"] = U(*cfg.get("dr_wrist_lat", DR["wrist_lat"]))
        p["infer_mean"] = U(*DR["infer_mean"]) if cfg["infer_mean"] > 0.1 else p["infer_mean"] * U(0.8, 1.3)
        p["phys_scale"] = U(DR["armature"][0] / 0.191, DR["armature"][1] / 0.191)
        p["cam_phase"] = U(0.0, 1.0)                       # fraction of a frame period: cameras free-run
    return p


class PolicyClock:
    """Compute + transport delay of one decision: the command computed from the observation taken at t reaches the
    Pi at t + inference + host processing + transport (occasionally a transport spike)."""

    def __init__(self, cfg, lat, nworld, device, seed=0):
        self.c, self.lat, self.N, self.dev = cfg, lat, nworld, device
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def delay(self):
        c = self.c
        z = torch.randn(self.N, generator=self.g).to(self.dev)
        infer = (self.lat["infer_mean"] + c["infer_sd"] * z).clamp(min=c["infer_min"]) if c["infer_mean"] > 0 else torch.zeros(self.N, device=self.dev)
        tx = self.lat["tx_lat"].clone()
        if c["tx_spike_p"] > 0:
            u = torch.rand(self.N, 2, generator=self.g).to(self.dev)
            spike = c["tx_spike"][0] + (c["tx_spike"][1] - c["tx_spike"][0]) * u[:, 1]
            tx = torch.where(u[:, 0] < c["tx_spike_p"], spike, tx)
        return infer + c["host_proc"], tx

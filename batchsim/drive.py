"""batchsim.drive :: the deployed command chain of the Pro 630, vectorised over worlds (port of realsim.twin).

    host polygon --(arrives after compute + transport delay)--> Pi stream law @ cmd_dt on its joint feedback
      v = vff * v_ref(t + lead) - K0 * (q_fb - q_ref(t + lead_p)) - K1 * (qd_fb - v_ref), +friction comp, clip vmax
    --> drive: gain, breakaway band, dead time, acceleration limit, integrated drive position (following-error
        clamp) --> stiff inner PD + gravity feed-forward --> joint torque --> physics @ timestep

Per-world: dead time, lead, amax, vmax (latency.sample). Shared: stream gains and inner-loop gains (cfg).
One `tick()` = one Pi period (cfg['cmd_dt']) = cmd_dt / timestep physics steps.
"""
import torch

NJ = 6


def spline_eval(pts, npts, t_anchor, dt, t):
    """Uniform cubic B-spline over per-world control polygons. pts [N, P, 6] (rows beyond npts[n] ignored),
    knot i at t_anchor + i * dt, evaluated at t [N]. Mirrors realsim.twin.spline_eval (clamped ends)."""
    N = pts.shape[0]
    u = (t - t_anchor) / dt
    last = (npts - 1).clamp(min=1)
    i = torch.minimum(torch.floor(u).long().clamp(min=0), last - 1)
    s = (u - i).clamp(0.0, 1.0)
    om = 1.0 - s
    b = torch.stack([om ** 3, 3 * s ** 3 - 6 * s * s + 4, -3 * s ** 3 + 3 * s * s + 3 * s + 1, s ** 3], 1) / 6.0
    db = torch.stack([-0.5 * om * om, (3 * s * s - 4 * s) * 0.5, (-3 * s * s + 2 * s + 1) * 0.5, 0.5 * s * s], 1)
    idx = torch.stack([i - 1, i, i + 1, i + 2], 1).clamp(min=0)
    idx = torch.minimum(idx, last[:, None])
    P = pts[torch.arange(N, device=pts.device)[:, None], idx]                 # [N, 4, 6]
    return (b[:, :, None] * P).sum(1), (db[:, :, None] * P).sum(1) / dt[:, None]


class DriveChain:
    Q = 4                                                                     # commands that can be in flight per world

    def __init__(self, world, cfg, lat, max_pts=16):
        self.w, self.c, self.lat = world, cfg, lat
        N, dev = world.N, world.device
        self.N, self.dev, self.P = N, dev, max_pts
        self.sub = max(1, int(round(cfg["cmd_dt"] / world.dt)))
        f = lambda v: torch.as_tensor(v, dtype=torch.float32, device=dev)
        self.kp, self.kd = f(cfg["kp"]), f(cfg["kd"])
        self.n_dead = torch.round(lat["dead"] / world.dt).long()             # dead time at physics resolution
        self.Lb = int(self.n_dead.max().item()) + 2
        self.vbuf = torch.zeros(self.Lb, N, NJ, device=dev)
        self.bi = 0
        self.ar = torch.arange(N, device=dev)
        self.t = 0.0
        self.q_drive = torch.zeros(N, NJ, device=dev)
        self.v_drive = torch.zeros(N, NJ, device=dev)
        self.q_prev_fb = torch.zeros(N, NJ, device=dev)
        self.qd_fb = torch.zeros(N, NJ, device=dev)
        self.ref_pts = torch.zeros(N, max_pts, NJ, device=dev)
        self.ref_n = torch.full((N,), 2, dtype=torch.long, device=dev)
        self.ref_t0 = torch.full((N,), -1e9, device=dev)
        self.ref_dt = torch.full((N,), 0.1, device=dev)
        self.in_pts = torch.zeros(N, self.Q, max_pts, NJ, device=dev)
        self.in_n = torch.full((N, self.Q), 2, dtype=torch.long, device=dev)
        self.in_t0 = torch.zeros(N, self.Q, device=dev)
        self.in_dt = torch.full((N, self.Q), 0.1, device=dev)
        self.in_arr = torch.full((N, self.Q), float("inf"), device=dev)
        self.stats = dict(vsat=torch.zeros(N, device=dev), ferror=torch.zeros(N, device=dev), late=torch.zeros(N, device=dev),
                          dropped=torch.zeros(N, device=dev), ticks=0)
        self.tau = torch.zeros(N, NJ, device=dev)
        self.v_cmd = torch.zeros(N, NJ, device=dev)
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self, mask=None):
        """Re-initialise drive and reference at the CURRENT joint pose of the masked worlds."""
        m = torch.ones(self.N, dtype=torch.bool, device=self.dev) if mask is None else mask
        q = self.w.qpos[:, :NJ]
        self.q_drive[m] = q[m]
        self.v_drive[m] = 0.0
        self.vbuf[:, m] = 0.0
        self.q_prev_fb[m] = q[m]
        self.qd_fb[m] = 0.0
        self.ref_pts[m] = q[m][:, None, :]
        self.ref_n[m] = 2
        self.ref_t0[m] = -1e9
        self.ref_dt[m] = 0.1
        self.in_arr[m] = float("inf")

    # ---------------------------------------------------------------- host -> Pi
    def send(self, pts, t_anchor, dt, arrive, mask=None, npts=None):
        """Queue a control polygon per world: pts [N, p, 6] (p <= max_pts), knot 0 at t_anchor [N] (sim time),
        knot spacing dt [N] or float, reaching the Pi at `arrive` [N] (sim time). Only masked worlds send."""
        N, p = self.N, pts.shape[1]
        m = torch.ones(N, dtype=torch.bool, device=self.dev) if mask is None else mask
        free = torch.isinf(self.in_arr)
        slot = torch.where(free.any(1), free.float().argmax(1), self.in_arr.argmin(1))   # full: overwrite the oldest
        self.stats["dropped"] += (m & ~free.any(1)).float()
        a, s = self.ar[m], slot[m]
        self.in_pts[a, s, :p] = pts[m]
        if p < self.P:
            self.in_pts[a, s, p:] = pts[m][:, -1:, :]
        self.in_n[a, s] = p if npts is None else npts[m]
        self.in_t0[a, s] = torch.as_tensor(t_anchor, dtype=torch.float32, device=self.dev).expand(N)[m]
        self.in_dt[a, s] = torch.as_tensor(dt, dtype=torch.float32, device=self.dev).expand(N)[m]
        self.in_arr[a, s] = torch.as_tensor(arrive, dtype=torch.float32, device=self.dev).expand(N)[m]

    # ---------------------------------------------------------------- one Pi period
    def tick(self, on_substep=None):
        c, w = self.c, self.w
        t = self.t
        arrived = self.in_arr <= t + 1e-6
        if arrived.any():                                         # newest arrival wins (the Pi re-welds)
            key = torch.where(arrived, self.in_arr, torch.full_like(self.in_arr, -float("inf")))
            j = key.argmax(1)
            has = arrived.any(1)
            a = self.ar[has]
            self.ref_pts[a] = self.in_pts[a, j[has]]
            self.ref_n[a] = self.in_n[a, j[has]]
            self.ref_t0[a] = self.in_t0[a, j[has]]
            self.ref_dt[a] = self.in_dt[a, j[has]]
            self.in_arr[arrived] = float("inf")
        lead = self.lat["lead"]
        q_ref, v_ref = spline_eval(self.ref_pts, self.ref_n, self.ref_t0, self.ref_dt, t + lead)
        end = self.ref_t0 + (self.ref_n - 1) * self.ref_dt
        past = (t + lead) > end                                   # ran past the polygon: the Pi holds
        v_ref = torch.where(past[:, None], torch.zeros_like(v_ref), v_ref)
        self.stats["late"] += past.float()
        lead_p = c.get("lead_p")
        if lead_p is None:
            q_ref_p, v_ref_p = q_ref, v_ref
        else:
            tp = torch.full_like(lead, t + float(lead_p))
            q_ref_p, v_ref_p = spline_eval(self.ref_pts, self.ref_n, self.ref_t0, self.ref_dt, tp)
            v_ref_p = torch.where((tp > end)[:, None], torch.zeros_like(v_ref_p), v_ref_p)
        q_fb = w.qpos[:, :NJ]                                     # robot_hal reads the driver pins directly (fb_ticks = 0)
        v_cmd = c["vff"] * v_ref - c["k0"] * (q_fb - q_ref_p) - c["k1"] * (self.qd_fb - v_ref_p)
        if c.get("vfric", 0.0) > 0:                               # breakaway / Coulomb compensation
            v_cmd = v_cmd + c["vfric"] * torch.tanh(v_cmd / c.get("veps", 0.005236))
        vmax = self.lat["vmax"][:, None]
        self.stats["vsat"] += (v_cmd.abs() >= 0.98 * vmax).any(1).float()
        v_cmd = torch.maximum(torch.minimum(v_cmd, vmax), -vmax) * c.get("vgain", 1.0)
        brk = c.get("breakaway", 0.0)
        if brk > 0:                                               # a resting joint ignores commands below the band
            v_cmd = torch.where((self.v_drive.abs() < 0.5 * brk) & (v_cmd.abs() < brk), torch.zeros_like(v_cmd), v_cmd)
        self.v_cmd = v_cmd
        da = (self.lat["amax"] * w.dt)[:, None]
        fe = c["ferror"]
        for _ in range(self.sub):
            self.bi = (self.bi + 1) % self.Lb
            self.vbuf[self.bi] = v_cmd
            v_eff = self.vbuf[(self.bi - self.n_dead) % self.Lb, self.ar]         # written n_dead physics steps ago
            self.v_drive = self.v_drive + torch.maximum(torch.minimum(v_eff - self.v_drive, da), -da)
            self.q_drive = self.q_drive + self.v_drive * w.dt
            qj = w.qpos[:, :NJ]
            ld = self.q_drive - qj
            over = ld.abs() > fe
            if over.any():                                        # blocked joint: following-error clamp
                self.stats["ferror"] += over.any(1).float()
                self.q_drive = torch.where(over, qj + ld.clamp(-fe, fe), self.q_drive)
                self.v_drive = torch.where(over & (torch.sign(self.v_drive) == torch.sign(ld)), torch.zeros_like(self.v_drive), self.v_drive)
            tau = self.kp * (self.q_drive - qj) - self.kd * (w.qvel[:, :NJ] - self.v_drive)
            if c["gravity_comp"]:                                 # the drive's integrator removes static sag
                tau = tau + w.qfrc_bias[:, :NJ]
            self.tau = tau.clamp(-c["tau_max"], c["tau_max"])
            w.ctrl[:, :NJ] = self.tau
            if on_substep is not None:
                on_substep()
            w.step()
        self.t += c["cmd_dt"]
        q = w.qpos[:, :NJ]
        self.qd_fb = (q - self.q_prev_fb) / c["cmd_dt"]           # robot_hal differentiates posfb
        self.q_prev_fb = q.clone()
        self.stats["ticks"] += 1

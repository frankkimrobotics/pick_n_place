"""realsim.config :: measured Pro 630 latency / drive / controller numbers for the MuJoCo 4 twin.

Every value is either a hardware measurement (source noted) or marked ASSUMED (not yet
measured on this robot; chosen from vendor specs / typical values, exposed so it can be
swept). Units: seconds, radians, metres unless noted.

Signal chain modelled (one decision = 100 ms, one command tick = 10 ms, physics = 2 ms):

  cameras --(exposure->host latency, frame period)--> host encode
  joints  --(posfb stream to desktop)--> host FK
  host: policy inference (async, latency L) -> chunk adopted at the next decision >= t_obs+L
        -> receding-horizon index i = k - k_obs - lat_shift -> guards -> IK -> dq clamp
        -> spline segment (last 4 targets + 1 extrapolated) --(TCP transport)--> Pi
  Pi robot_hal (10 ms): B-spline ref sampled at t+lead, law
        v = vff*v_ref - K0*(q_fb - q_ref) - K1*(qd_fb - v_ref), clip +-vmax,
        on ONE-TICK-OLD posfb
  drive: dead-time -> accel-limited velocity -> integrated drive position (anti-windup
        ferror) -> stiff position loop (+ gravity comp = the drive's integrator) -> joint
        with fitted armature / viscous / Coulomb friction
  suction: pump vacuum build-up before a contact can latch, valve release delay
"""
import copy

import numpy as np

D = np.radians

REAL = dict(
    # ---------------- Pi robot_hal streaming controller ----------------
    cmd_dt=0.010,          # HAL thread / command period (elerob.hal, 10 ms default; 5 ms opt-in)
    k0=10.0, k1=0.3, vff=1.0,     # real_policy_ctrl.STREAM_GAINS (deployed 2026-09-20)
    lead=0.045,            # Pi samples the reference at t + lead (= drive dead-time)
    lead_p=None,           # position-error lead; None = same as `lead` (the law before 2026-11-14)
    vlag=0.0,              # s, drive first-order velocity lag (twin drive model); 0 = dead time + amax only (fit of 2026-11-14)
    taff=0.0,              # acceleration feedforward u += taff * a_ref (robot_hal; 0.015 in STREAM_FAST)
    vfric=0.0, veps=D(0.3),       # breakaway compensation u += vfric*tanh(u/veps) (robot_hal 2026-11-14)
    vmax=D(50.0),          # vel_cmd_max 850 / vel_scale 17 = 50 deg/s
    vgain=1.025,           # measured velocity / commanded at vel_scale 17 (bare-drive runs 2026-11-14: 1.01-1.04)
    breakaway=D(0.3),      # a resting joint ignores velocity commands below this (bare ramps 2026-11-14)
    fb_ticks=0,            # age of the posfb the law uses, in 10 ms ticks. 0 since 2026-11-14: with 1 the twin led the
                           # reference 10 ms more than the arm on every run (robot_hal reads the driver pins directly)
    # ---------------- drive (STM32 velocity-mode servo) ----------------
    dead=0.040,            # bare-drive fit 2026-11-14: 32-36 ms dead + 10-20 ms lag; 40 ms matches old AND tuned
                           # gains on sines and ramp-stops (was 0.045: HAL write -> first motion 36-52 ms, 2026-09-17);
                           # step battery median actuation latency 61 ms (2026-11-07, incl. the 10 ms tick)
    amax=D(1200.0),        # 800 (10 deg steps, 2026-09-17: 720-870) overshoots the 30 deg/s ramp-stops of 2026-11-14
                           # by 2x; 1200 matches their lag and overshoot under both gain sets
    ferror=D(1.5),         # following-error anti-windup (env_warp DRIVE.ferror; drive trips beyond)
    # inner position loop of the drive (env_warp real drive: 3x stiffer PD, posfb tracks poscmd 0.04 deg)
    kp=np.array([600, 900, 600, 200, 80, 40], float) * 3.0,
    kd=np.array([40, 60, 40, 12, 4, 2], float) * 1.7,
    tau_max=100.0,
    gravity_comp=True,     # the drive's integrator removes static sag; modelled as qfrc_bias feedforward
    # joint physics, fitted to real traces (rl/fit_actuator.py -> outputs/actuator_fit.json, 2026-11-07)
    armature=0.191,
    damping=[0.493, 0.443, 0.472, 0.462, 0.462, 0.479],
    frictionloss=[0.285, 0.289, 0.312, 0.281, 0.298, 0.282],
    # ---------------- transport ----------------
    # measured 2026-11-27 on the cable, clocks chrony-synced over the cable (realsim/sysid/stream_slack.py, 670 spline
    # commands through PiLink, ~/pnp_real_runs/stream_slack/): send -> Pi t_recv median 0.8-1.0 ms, p99 2.9-4.8, max 11.9;
    # TCP RTT 0.49 ms. The 50-125 ms spikes of September (WiFi era, old robot_hal) were not seen again.
    tx_lat=0.001,
    tx_spike_p=0.01,
    tx_spike=(0.004, 0.012),
    fb_lat=0.001,          # Pi -> desktop joint stream (one-way ~0.25 ms + ack/parse); 2026-11-27 realsim/sysid/fb_rate.py
    fb_period=0.010,       # 10 ms stack: posfb/torqfb pins change every 10.0 ms per joint (sampled on the Pi at 770 Hz,
                           # 2026-11-27); 5 ms stack: timing="stack_5ms"
    # ---------------- cameras: policy/cam_server60.py, every camera at 60 fps (measured 2026-11-27) ----------------
    # *_lat = sensor capture time -> frame published in /dev/shm, from the cameras' own capture stamps (ZED
    # TIME_REFERENCE.IMAGE, D405 global time); 60 s, all three cameras running. Frame-period sample-and-hold is
    # modelled separately by *_period. Not yet checked against an external flash -> DR below is skewed upward.
    zed_lat=0.030, zed_period=1.0 / 60,      # ZED 2i HD720@60: median 29.6 ms, p95 39.8, max 47.2
    wrist_lat=0.011, wrist_period=1.0 / 60,  # D405 424x240@60: median 10.9 ms, p95 11.1, max 17.0
    # the old policy/zed_cam_server.py is NOT this: make_cfg(timing="old_cam_server")
    cam_noise=0.0,         # additive gaussian pixel noise (0-255 scale); 0 = off
    # ---------------- policy compute ----------------
    infer_mean=0.150, infer_sd=0.025, infer_min=0.09,  # UNet/CFM 110-200 ms measured (2026-10-15, 11-07);
                           # real logs: plans adopted 1 decision (0.2 s at slow 2) after the observation
    host_proc=0.030,       # obs read -> segment send inside one decision (2 encodes + IK); spread <= 14 ms
                           # measured from steps.jsonl, absolute offset ESTIMATED
    jpeg_q=95,             # zed_cam_server / rs_shm_server write JPEG q95 to /dev/shm
    # ---------------- suction (ASSUMED) ----------------
    suction_onset=0.30,    # vacuum build-up after the valve opens before a seal can hold
    suction_release=0.15,  # residual vacuum after the valve closes
    # ---------------- host executor (policy/real_rollout_zed.py defaults 2026-11-08) ----------------
    dq_max=D(2.5), step_max=0.015, step_max_z=0.030, lead_max=D(3.0),
    max_tilt=30.0, z_floor=0.005, suction_z=0.09, suction_off_n=5, n_exec=8, orient="yawlock",
)

# stream gains tuned on the arm 2026-11-14 (~/pnp_real_runs/sysid/stream_gains_new.json): feedforward leads the
# drive delay, the position error does not; hardware 1 Hz 3 deg sine rms 0.45 -> 0.045 deg, lag -42 -> 0 ms.
# vgain: the same drive commanded with vel_scale 16.4 instead of 17.
# k0 10, not the 15 used in the tracking tests: in the policy runs K0 15 put the velocity command on its limit for
# 15-22 % of ticks (all while the executor holds a blocked arm), K0 10 for 4-8 %, with the same task result and
# 0.033 vs 0.031 deg tracking rms on the arm.
STREAM_TUNED = dict(k0=10.0, k1=0.3, vff=1.0, lead=0.040, lead_p=0.0, vfric=D(0.35), veps=D(0.3),
                    vgain=1.025 * 16.4 / 17.0)
# 2026-11-25 envelope (realsim/sysid/envelope.py): tuned + acceleration feedforward, clamp 60 deg/s. On the arm, J1 step
# 30 deg/s err max 0.59 -> 0.23 deg; no drive fault at 60.7 deg/s actual. real_policy_ctrl.STREAM_GAINS_FAST.
STREAM_FAST = dict(STREAM_TUNED, taff=0.015, vmax=D(60.0))
# 2026-11-27 5 ms stack (launch_mpc_stack_5ms.sh): same law, lead 0.030 (lead 0.040 is 2x worse there).
# real_policy_ctrl.STREAM_GAINS_FAST_5MS / real_rollout --stream fast5. Use with timing="stack_5ms".
STREAM_FAST_5MS = dict(STREAM_FAST, lead=0.030)
STREAMS = {"tuned": STREAM_TUNED, "fast": STREAM_FAST, "fast5": STREAM_FAST_5MS}

# domain randomisation ranges (multipliers / absolute), env_warp DRIVE_DR + latency spreads
DR = dict(
    dead=(0.030, 0.065), amax_mult=(0.75, 1.15), vmax_mult=(0.9, 1.1),
    # cameras 2026-11-27 (cam_server60): measured p5..max plus room for the unverified capture-stamp offset
    zed_lat=(0.025, 0.060), wrist_lat=(0.008, 0.030), infer_mean=(0.110, 0.200),
    armature=(0.15, 0.23),
)

# Timing presets, make_cfg(timing=...). Each overrides config values and, under "dr", the camera DR ranges
# (stored in the cfg as dr_<key> so the batched twin draws from the same ranges).
TIMING = {
    "measured": {},        # REAL + DR as above: 10 ms stack, cam_server60 at 60 fps, cable transport (2026-11-27)
    # policy/zed_cam_server.py (all real runs up to 2026-11-27): ZED throttle aliased to 10.7 Hz (frames 100 ms apart),
    # 30 fps grab + 17 ms publish; wrist frames bunched by the GIL every ~100 ms (median gap 7 ms, p95 92 ms)
    "old_cam_server": dict(zed_lat=0.049, zed_period=0.100, wrist_lat=0.011, wrist_period=0.100,
                           dr=dict(zed_lat=(0.040, 0.080), wrist_lat=(0.008, 0.030))),
    # the assumptions every twin result before 2026-11-27 used -- reproduce old numbers with this
    "legacy_1126": dict(tx_lat=0.003, tx_spike_p=0.03, tx_spike=(0.050, 0.125), fb_lat=0.002,
                        zed_lat=0.090, zed_period=1.0 / 30, wrist_lat=0.070, wrist_period=1.0 / 30,
                        dr=dict(zed_lat=(0.060, 0.130), wrist_lat=(0.040, 0.100))),
    # 5 ms HAL stack (elerob_5ms.ini, CAN IRQ pinned cpu3 fifo99), 2026-11-27: :9999 196 Hz, drive pins every 5 ms,
    # dead time 46-50 -> 31 ms. Pair with stream="fast5" (lead 0.030). Physics runs at 5/3 ms (phys_substeps).
    "stack_5ms": dict(cmd_dt=0.005, fb_period=0.005, dead=0.031, lead=0.030,
                      dr=dict(dead=(0.025, 0.045))),
}


def phys_substeps(cmd_dt, dt0):
    """Physics steps per Pi tick and the timestep to use: the largest step <= the scene's dt0 that divides cmd_dt
    (10 ms on 2 ms physics -> 5 x 2 ms; the 5 ms stack -> 3 x 1.667 ms rather than silently running 4 ms ticks)."""
    n = max(1, int(np.ceil(cmd_dt / dt0 - 1e-6)))
    return n, (dt0 if abs(n * dt0 - cmd_dt) < 1e-9 else cmd_dt / n)      # keep the scene's step bit-exact when it divides

# "ideal" = the old eval harness's assumptions: no delays, stiff drive, no transport/compute time
IDEAL_OVERRIDES = dict(
    dead=0.0, lead=0.0, amax=D(1e5), vmax=D(1e4), fb_ticks=0, k0=60.0, k1=0.0, vgain=1.0, breakaway=0.0,
    tx_lat=0.0, tx_spike_p=0.0, fb_lat=0.0, zed_lat=0.0, wrist_lat=0.0,
    infer_mean=0.0, infer_sd=0.0, infer_min=0.0, suction_onset=0.0, suction_release=0.0,
    host_proc=0.0, jpeg_q=0,
    armature=0.15, damping=[1.0] * 6, frictionloss=[0.0] * 6,
)

# realism ablation ladder: each level switches one group of effects on top of the previous one
LEVELS = {
    "ideal": dict(IDEAL_OVERRIDES),
    "drive": {k: v for k, v in IDEAL_OVERRIDES.items()
              if k not in ("dead", "lead", "amax", "vmax", "fb_ticks", "k0", "k1", "vgain", "breakaway",
                           "armature", "damping", "frictionloss")},
    "drive+sensor": {k: v for k, v in IDEAL_OVERRIDES.items()
                     if k in ("infer_mean", "infer_sd", "infer_min", "host_proc",
                              "suction_onset", "suction_release")},
    "drive+sensor+compute": dict(suction_onset=0.0, suction_release=0.0),
    "real": {},
}


# Joint physics fitted from the drives' torque feedback (motor-side N m), 2026-11-25:
# ~/pnp_real_runs/sysid/friction/{fit,twin_check}.py. Coulomb + viscous friction and reflected inertia at the joint =
# motor-side fit x gear ratio (joints 2, 3: ratio 250 / 204 from the gravity load; joint 1 assumed = joint 2;
# wrist joints ASSUMED 100). The larger friction needs the stiffer inner loop below to keep the measured tracking
# (kp x4 on joints 1-3: twin-vs-hardware 0.05-0.06 deg as before, motor torque R2 0.9 instead of 0.1-0.2).
# Select with make_cfg(..., joint_model=1) / eval_policy --cfg joint_model=1. Default stays the 2026-11-07 fit.
JOINT_TORQUE_FIT = dict(
    frictionloss=[5.272, 5.687, 5.047, 0.273, 0.293, 0.216],
    damping=[3.978, 2.415, 0.431, 0.191, 0.319, 0.326],
    armature=[0.738, 1.189, 0.754, 0.109, 0.099, 0.093],
    kp=np.array([7200, 10800, 7200, 600, 240, 120], float),
    kd=np.array([200, 300, 200, 20.4, 6.8, 3.4], float),
    gear=[250, 250, 204, 100, 100, 100],
)


def make_cfg(level="real", dr=False, rng=None, **over):
    joint_model = int(over.pop("joint_model", 0))
    timing = over.pop("timing", "measured")
    if timing not in TIMING:
        raise ValueError(f"timing={timing!r}: choose one of {list(TIMING)}")
    cfg = copy.deepcopy(REAL)
    t_over = copy.deepcopy(TIMING[timing])
    t_dr = t_over.pop("dr", {})
    cfg.update(t_over)
    cfg["timing"] = timing
    for k in ("zed_lat", "wrist_lat", "dead"):
        cfg[f"dr_{k}"] = tuple(t_dr.get(k, DR[k]))
    cfg.update(LEVELS[level])
    if dr:
        rng = rng or np.random.default_rng()
        if level in ("drive", "drive+sensor", "drive+sensor+compute", "real"):
            cfg["dead"] = rng.uniform(*cfg["dr_dead"])
            if "lead" not in over:
                cfg["lead"] = cfg["dead"]      # the Pi lead is tuned to the dead-time
            cfg["amax"] *= rng.uniform(*DR["amax_mult"])
            cfg["vmax"] *= rng.uniform(*DR["vmax_mult"])
            cfg["armature"] = rng.uniform(*DR["armature"])
        if level in ("drive+sensor", "drive+sensor+compute", "real"):
            cfg["zed_lat"] = rng.uniform(*cfg["dr_zed_lat"])
            cfg["wrist_lat"] = rng.uniform(*cfg["dr_wrist_lat"])
        if level in ("drive+sensor+compute", "real"):
            cfg["infer_mean"] = rng.uniform(*DR["infer_mean"])
    if joint_model:
        scale = cfg["armature"] / REAL["armature"] if dr and np.isscalar(cfg["armature"]) else 1.0   # keep the DR spread
        cfg.update(copy.deepcopy(JOINT_TORQUE_FIT))
        cfg["armature"] = (np.array(cfg["armature"]) * scale).tolist()
    cfg.update(over)
    return cfg

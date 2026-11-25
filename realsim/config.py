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
    tx_lat=0.003,          # desktop -> Pi TCP 2-5 ms typical (2026-09-17 per-hop table)
    tx_spike_p=0.03,       # Pi GIL-contention spikes 50-125 ms ("arrived late once" at 20 Hz chunks)
    tx_spike=(0.050, 0.125),
    fb_lat=0.002,          # Pi -> desktop joint stream 0.2-0.8 ms + 100 Hz cadence (sample-and-hold below)
    fb_period=0.010,
    # ---------------- cameras (ASSUMED: capture->host age not yet measured on this rig) ----------------
    zed_lat=0.090, zed_period=1.0 / 30,      # ZED 2i HD720@30 typical 2-3 frames glass-to-host
    wrist_lat=0.070, wrist_period=1.0 / 30,  # D405 USB3 typical ~2 frames
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

# domain randomisation ranges (multipliers / absolute), env_warp DRIVE_DR + latency spreads
DR = dict(
    dead=(0.030, 0.065), amax_mult=(0.75, 1.15), vmax_mult=(0.9, 1.1),
    zed_lat=(0.060, 0.130), wrist_lat=(0.040, 0.100), infer_mean=(0.110, 0.200),
    armature=(0.15, 0.23),
)

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
    cfg = copy.deepcopy(REAL)
    cfg.update(LEVELS[level])
    if dr:
        rng = rng or np.random.default_rng()
        if level in ("drive", "drive+sensor", "drive+sensor+compute", "real"):
            cfg["dead"] = rng.uniform(*DR["dead"])
            if "lead" not in over:
                cfg["lead"] = cfg["dead"]      # the Pi lead is tuned to the dead-time
            cfg["amax"] *= rng.uniform(*DR["amax_mult"])
            cfg["vmax"] *= rng.uniform(*DR["vmax_mult"])
            cfg["armature"] = rng.uniform(*DR["armature"])
        if level in ("drive+sensor", "drive+sensor+compute", "real"):
            cfg["zed_lat"] = rng.uniform(*DR["zed_lat"])
            cfg["wrist_lat"] = rng.uniform(*DR["wrist_lat"])
        if level in ("drive+sensor+compute", "real"):
            cfg["infer_mean"] = rng.uniform(*DR["infer_mean"])
    if joint_model:
        scale = cfg["armature"] / REAL["armature"] if dr and np.isscalar(cfg["armature"]) else 1.0   # keep the DR spread
        cfg.update(copy.deepcopy(JOINT_TORQUE_FIT))
        cfg["armature"] = (np.array(cfg["armature"]) * scale).tolist()
    cfg.update(over)
    return cfg

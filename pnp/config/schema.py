"""Schema for pnp.config: type, unit and range per key, plus cross-checks. `validate(cfg)` returns a
list of problems (empty = valid). Keys not listed are allowed but unchecked, so adding a tunable to
defaults.toml never breaks validation; adding it here makes it checked."""
from __future__ import annotations

NUM = (int, float)
# key: (type, unit, (lo, hi) or None, note)
SCHEMA = {
    "robot.joint_names":        (list, "", None, "6 URDF joint names"),
    "robot.home_q":             (list, "rad", None, "6 joint angles"),
    "robot.base_q":             (list, "rad", None, "6 joint angles, pick-and-place start pose"),
    "robot.joint_limits_deg":   (list, "deg", None, "6 x [lo, hi]"),
    "robot.handeye_tcp_len":    (NUM, "m", (0.05, 0.30), "tcp length when the hand-eye was calibrated"),
    "robot.planner_tcp_len":    (NUM, "m", (0.05, 0.30), "URDF/planner tcp length"),
    "suction.cup_lip_radius":   (NUM, "m", (0.003, 0.03), "rim_seal ring radius"),
    "suction.plunger_travel":   (NUM, "m", (0.005, 0.03), "spring stroke"),
    "suction.plunger_k":        (NUM, "N/m", (50, 5000), "ASSUMED until calibrated"),
    "pi.cmd_port":              (int, "", (1024, 65535), ""),
    "pi.feedback_port":         (int, "", (1024, 65535), ""),
    "pi.period_ms":             (int, "ms", (1, 50), "robot_hal control period"),
    "pi.vel_cmd_max":           (int, "drive units", (50, 2000), "17 units per deg/s -> 850 = 50 deg/s"),
    "pi.stream.k0":             (NUM, "", (0, 50), ""),
    "pi.stream.k1":             (NUM, "", (0, 5), ""),
    "pi.stream.vmax":           (NUM, "deg/s", (1, 60), "firmware following-error cap is ~36 deg/s steady"),
    "pi.stream.vel_scale":      (NUM, "units per deg/s", (10, 25), "measured 17 (2026-09-17), 16.4 (2026-11-14)"),
    "pi.stream.lead":           (NUM, "s", (0, 0.2), "drive dead-time compensation"),
    "cameras.fps":              (NUM, "Hz", (1, 60), ""),
    "cameras.wrist.width":      (int, "px", (64, 2000), ""),
    "cameras.wrist.height":     (int, "px", (64, 2000), ""),
    "task.table_z":             (NUM, "m", (-0.5, 0.5), ""),
    "task.lift_height":         (NUM, "m", (0.01, 0.5), ""),
}
REQUIRED_MACHINE = ["paths.repo2026", "paths.datasets", "paths.policies", "paths.runs_real", "paths.mcap_runs",
                    "interpreters.ilpolicy", "interpreters.mjwarp", "interpreters.curobo2", "interpreters.system",
                    "pi.host", "cameras.wrist.serial", "cameras.fixed.serial"]


def validate(cfg):
    flat = cfg.flat()
    bad = []
    for key, (typ, unit, rng, _note) in SCHEMA.items():
        if key not in flat:
            bad.append(f"missing {key}"); continue
        v = flat[key]
        if not isinstance(v, typ) or isinstance(v, bool) and typ is not bool:
            bad.append(f"{key}: expected {getattr(typ, '__name__', typ)}, got {type(v).__name__} ({cfg.origin(key)})"); continue
        if rng and not (rng[0] <= v <= rng[1]):
            bad.append(f"{key} = {v} {unit} outside [{rng[0]}, {rng[1]}] ({cfg.origin(key)})")
    for key in REQUIRED_MACHINE:
        if key not in flat:
            bad.append(f"missing machine value {key} (configs/machine/<hostname>.toml)")
    # cross-checks
    for k in ("robot.home_q", "robot.base_q", "robot.joint_names"):
        if k in flat and len(flat[k]) != 6:
            bad.append(f"{k} must have 6 entries, has {len(flat[k])}")
    lim = flat.get("robot.joint_limits_deg")
    if lim and len(lim) == 6 and "robot.base_q" in flat:
        import math
        for i, (q, (lo, hi)) in enumerate(zip(flat["robot.base_q"], lim)):
            if not (lo <= math.degrees(q) <= hi):
                bad.append(f"robot.base_q[{i}] = {math.degrees(q):.1f} deg outside joint limit [{lo}, {hi}]")
    if "robot.handeye_tcp_len" in flat and "robot.planner_tcp_len" in flat:
        if abs(flat["robot.handeye_tcp_len"] - flat["robot.planner_tcp_len"]) > 0.05:
            bad.append("handeye_tcp_len and planner_tcp_len differ by more than 5 cm")
    for p in ("paths.repo2026", "paths.datasets", "paths.policies"):
        import os
        if p in flat and not os.path.isdir(os.path.expanduser(flat[p])):
            bad.append(f"{p} = {flat[p]} is not a directory ({cfg.origin(p)})")
    for name in ("ilpolicy", "mjwarp", "curobo2", "system"):
        k = f"interpreters.{name}"
        import os
        if k in flat and not os.path.exists(os.path.expanduser(flat[k])):
            bad.append(f"{k} = {flat[k]} does not exist ({cfg.origin(k)})")
    return bad

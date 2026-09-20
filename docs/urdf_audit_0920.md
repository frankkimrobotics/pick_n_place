# URDF / MuJoCo model audit -- myCobot Pro 630

URDF : `/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/urdf/mycobot_pro_630.urdf` (the path in mycobot_pro_630.yml `urdf_path`)  
MuJoCo: `/home/lisc-frank/Desktop/2026/pick_and_place/rl/scenes/box_med.xml` (compiled; inertia shown rotated back into the body frame from `body_inertia` (principal) + `body_iquat`)

## Per-link mass / inertia / COM

| link | src | mass (kg) | ixx iyy izz ixy ixz iyz (kg m^2) | COM xyz (m) |
|---|---|---|---|---|
| base | URDF | 3.4225 | 1.129e-02 1.123e-02 9.902e-03 5.521e-08 -1.427e-04 -2.052e-07 | -0.01340 +0.00001 +0.06584 |
| base | MuJoCo | 3.4225 | 1.129e-02 1.123e-02 9.902e-03 5.521e-08 -1.427e-04 -2.052e-07 | -0.01340 +0.00001 +0.06584 |
| link1 | URDF | 0.5434 | 5.669e-04 5.118e-04 4.049e-04 -3.760e-12 9.896e-13 -4.414e-05 | +0.00000 +0.00756 -0.01455 |
| link1 | MuJoCo | 0.5434 | 5.669e-04 5.118e-04 4.049e-04 0.000e+00 0.000e+00 -4.414e-05 | +0.00000 +0.00756 -0.01455 |
| link2 | URDF | 1.8125 | 1.229e-03 1.587e-02 1.590e-02 -1.272e-06 -2.471e-08 -9.612e-09 | +0.13444 +0.00002 -0.07533 |
| link2 | MuJoCo | 1.8125 | 1.229e-03 1.587e-02 1.590e-02 -1.272e-06 -2.471e-08 -9.612e-09 | +0.13444 +0.00002 -0.07533 |
| link3 | URDF | 2.2651 | 2.065e-03 1.967e-02 1.904e-02 -1.733e-06 -3.722e-04 -9.829e-08 | +0.12248 -0.00023 +0.00100 |
| link3 | MuJoCo | 2.2651 | 2.065e-03 1.967e-02 1.904e-02 -1.733e-06 -3.722e-04 -9.829e-08 | +0.12248 -0.00023 +0.00100 |
| link4 | URDF | 0.3447 | 2.681e-04 1.882e-04 2.438e-04 -2.279e-08 4.082e-09 -2.695e-05 | -0.00000 -0.01491 +0.00910 |
| link4 | MuJoCo | 0.3447 | 2.681e-04 1.882e-04 2.438e-04 -2.279e-08 4.082e-09 -2.695e-05 | +0.00000 -0.01491 +0.00910 |
| link5 | URDF | 0.4204 | 2.502e-04 3.887e-04 3.526e-04 1.986e-12 -2.156e-06 -3.526e-12 | +0.00078 -0.00000 -0.00654 |
| link5 | MuJoCo | 0.4204 | 2.502e-04 3.887e-04 3.526e-04 -2.158e-20 -2.156e-06 0.000e+00 | +0.00078 +0.00000 -0.00654 |
| link6 | URDF | 0.0376 | 1.680e-05 8.708e-06 8.709e-06 -1.013e-13 8.044e-10 6.140e-13 | +0.00543 +0.00000 -0.00000 |
| link6 | MuJoCo | 0.0376 | 1.680e-05 8.708e-06 8.709e-06 0.000e+00 8.044e-10 0.000e+00 | +0.00543 +0.00000 +0.00000 |
| camera_mount | URDF | 0.0500 | 3.000e-05 3.000e-05 3.000e-05 0.000e+00 0.000e+00 0.000e+00 | +0.00000 +0.00000 +0.00000 |
| camera_mount | MuJoCo | 0.0500 | 3.000e-05 3.000e-05 3.000e-05 0.000e+00 0.000e+00 0.000e+00 | +0.00000 +0.00000 +0.00000 |
| suction_cup | URDF | 0.1537 | 1.216e-04 1.216e-04 3.597e-05 2.032e-13 -3.219e-13 3.146e-10 | +0.00000 -0.00000 -0.03428 |
| suction_cup | MuJoCo | 0.1537 | 1.216e-04 1.216e-04 3.597e-05 0.000e+00 0.000e+00 3.146e-10 | +0.00000 +0.00000 -0.03428 |

**Mass totals** -- URDF link1..link6 = **5.424 kg**; + base (3.422) + camera_mount + suction_cup = 9.050 kg. MuJoCo: 5.424 / 9.050 kg (max |delta| per link 0.00e+00 kg).

## URDF joint limits

| joint | effort (Nm) | lower (rad/deg) | upper (rad/deg) | velocity (rad/s -> deg/s) | damping | friction |
|---|---|---|---|---|---|---|
| joint1 | 1000.0 | -3.1416 / -180.0 | +3.1416 / +180.0 | 1.50 -> 85.9 | 0.5 | 0.1 |
| joint2 | 1000.0 | -3.1416 / -180.0 | +3.1416 / +180.0 | 1.50 -> 85.9 | 0.5 | 0.1 |
| joint3 | 1000.0 | -2.6100 / -149.5 | +2.6180 / +150.0 | 1.50 -> 85.9 | 0.5 | 0.1 |
| joint4 | 1000.0 | -2.9670 / -170.0 | +2.9670 / +170.0 | 1.50 -> 85.9 | 0.3 | 0.1 |
| joint5 | 1000.0 | -2.9300 / -167.9 | +2.9321 / +168.0 | 1.50 -> 85.9 | 0.3 | 0.1 |
| joint6 | 1000.0 | -3.0300 / -173.6 | +3.0368 / +174.0 | 1.50 -> 85.9 | 0.2 | 0.1 |

## Verdicts

- **(ii) inertia validity** -- all 9 URDF inertia tensors positive-definite: YES; principal-moment triangle inequality (Ia+Ib >= Ic): YES.
- **(iii) placeholders** -- identical inertia on several links: NO; 1e-6-class diagonals: none; **effort = 1000.0 Nm on every joint -> PLACEHOLDER**; **velocity = 1.5 rad/s = 85.9 deg/s on every joint, 2.4x the 36 deg/s firmware ceiling and 1.7x the 50 deg/s drive saturation -> PLACEHOLDER**.
- **(iii, cont.)** two hand-entered placeholders survive in the tool chain, both harmless: `camera_mount`
  (0.0500 kg, perfectly isotropic 3.0e-5 kg m^2 -- a round guess, not a CAD value) and the massless
  `tcp` frame link (0.001 kg, isotropic 1e-6). Every ACTUATED link (link1..link6) and `base` carries
  real CAD-derived, fully populated tensors.
- **(i) mass plausibility** -- URDF link1..link6 = **5.424 kg**; with `base` (3.4225 kg) = **8.847 kg**,
  and with the wrist `camera_mount` + `suction_cup` = **9.050 kg**. The vendor's ~8-9 kg figure for the
  arm therefore matches the URDF only if it INCLUDES the base casting; the moving links alone are
  5.42 kg. The repo's own cross-check agrees: `mycobot_mpc/controller_params.yaml` says
  "dynamics model with DH-derived inertias (total ~8.7 kg), 100:1 harmonic drives". **Verdict: plausible.**
- **URDF vs MuJoCo consistency** -- mass and COM are IDENTICAL to machine precision on all 9 links;
  the inertia tensors agree to <= 4.7e-6 kg m^2 (pure round-trip error through MuJoCo's principal
  frame + `body_iquat`). `rl/scenes/box_med.xml` is a faithful export of this URDF's `<inertial>` data.
  **Verdict: fine.**
- **(iv) armature / damping (sim only)** -- `box_med.xml` carries `armature="0.15"` and `damping="1.0"`
  on all six hinges (the reflected harmonic-drive rotor inertia the warp twin needs for stable
  discrete-time PD, cf. the memory note "armature=0.15 needed for stable torque PD"). The URDF has
  NEITHER (it declares `damping` 0.5/0.5/0.5/0.3/0.3/0.2 and no rotor inertia at all), so cuRobo's
  dynamics-aware trajopt sees a lighter, undamped machine.
  **The sweep's torque check used the MuJoCo model, i.e. WITH armature and damping.**
  On the best config's worst case the armature adds at most 0.53 Nm to a peak (j4) and
  |tau_full - tau_no_armature_no_damping| <= 1.26 Nm; at the sweep's design caps the bounds are
  armature 0.15 * 600 deg/s^2 = **1.57 Nm** and damping 1.0 * 36 deg/s = **0.63 Nm**.
  Against peaks of 7.4 Nm (j2) and limits of 186/50 Nm this is noise -- it does not change any verdict.
- **Tool frames** -- URDF `tcp_joint` puts `tcp` **0.135 m** below the flange along the suction-cup axis
  (re-calibrated 2026-06-30 by a J2-torque floor touch-test, i.e. with the cup COMPRESSED).
  `sim_robot_mjcf.SIM_TIP_Z` overrides the sim body to **0.115 m** ("visible cup surface, not the
  planner's calibrated tcp"), which is exactly the **20.0 mm** offset measured by
  `planner_touch.measure_tcp_corr` / `planner_sweep` (MuJoCo site 20 mm ABOVE the cuRobo tcp when the
  tool points down). `real_pnp_servo.py` commands the cuRobo tcp to `btop - TIP_BELOW_TCP (0.019)` and
  expects the cup to sit on the box, i.e. the physical tip is 19 mm **above** the cuRobo tcp
  (~0.116 m below the flange). **The MuJoCo `tcp` site (0.115 m) is the one that matches the physical
  cup tip, to ~1 mm; the cuRobo/URDF `tcp` is a deliberately over-extended press frame.** Correcting
  cuRobo goals by +20 mm, as planner_touch and planner_sweep do, is therefore the right direction.

## Summary

| item | verdict |
|---|---|
| link masses / COM / inertias (URDF) | **fine** -- physically consistent, positive-definite, triangle inequality holds, no duplicated or 1e-6 placeholder tensors |
| total mass | **fine** -- 8.85 kg with base, 9.05 kg with wrist hardware; matches the vendor ~8-9 kg and the repo's ~8.7 kg note |
| MuJoCo `box_med.xml` inertials | **fine** -- identical to the URDF |
| joint `effort="1000.0"` (all 6) | **PLACEHOLDER** -- use ±186 Nm (J1-3) / ±50 Nm (J4-6) from `mycobot_mpc/controller_params.yaml` |
| joint `velocity="1.5"` rad/s (85.9 deg/s, all 6) | **PLACEHOLDER / WRONG for this robot** -- 2.4x the 36 deg/s STM32 following-error ceiling and 1.7x the 50 deg/s drive saturation; any planner trusting it produces unexecutable plans |
| `armature` (rotor inertia) | **missing from the URDF** -- present only in the sim (0.15); worth <= 1.6 Nm here, but it is why the twin and cuRobo disagree on dynamics |
| joint damping | **inconsistent** -- URDF 0.5/0.5/0.5/0.3/0.3/0.2 vs sim 1.0 on every joint; worth <= 0.63 Nm |
| `tcp` frame | **two different frames** -- cuRobo/URDF 0.135 m, MuJoCo 0.115 m below the flange; the MuJoCo one is the physical cup tip |

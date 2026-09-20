# Policy weights

Curated checkpoints (small enough for git). Load with `weights_only=False`;
PPO files hold `{"ac": state_dict, "step": int}`, the student holds
`{"student": state_dict}`.

| File | Mode | Trained on | Result |
|---|---|---|---|
| `ppo4_pnp_table.pt` | `pnp` | table targets, DR | 100% eval, median **5.5 mm** placement |
| `ppo5_workspace.pt` | `pnp` | + 63-cell hover grid | 99.3%; **4/4** sequential multi-object clearing |
| `ppo7_ped_teacher.pt` | `pnp` | + 30 cm lift, gentle landing, pedestals | 65.4% of that spec; used as the DAgger teacher |
| `ppo15_attach_fixedphys.pt` | `attach` | **corrected suction physics** | 74.6% seal+lift — warm-start for new runs |
| `distill1_student_rgbd.pt` | vision | DAgger from `ppo7_ped` | lift 33.4 cm, landing 4.0 cm/s, placement 11.4 cm |
| `paper10_ideal_best.pt` | `--env paper` (Arafat et al. 2026 task: reach/lift/hold at goal) | ideal drive, cuRobo+DAgger init, fixed LR, 4096 worlds | **70.5 %** at 8.4M steps, 6/6 deterministic replay |
| `paper12_ideal_best.pt` | `--env paper` | same init; entropy 0.0015, no value clipping, vf_coef 0.5 | **83.1 %** at 9.8M steps, no late decay (final 82.2 %) |
| `paper14_real_best.pt` | `--env paper`, **measured drive** | DAgger round-3 init + PPO with demo anchor (`--bc_data`) | 24 % deterministic (PPO never exceeded this on the measured drive) |
| `dagger6_real_iter10.pt` (+ `.onnx`, `.plan`) | `--env paper`, **measured drive** | DAgger continued to round 10 | **87.8 %** deterministic on 1024 episodes; `rl/real_policy_ctrl.py` default |
| `resid3_fast_best.pt` (+ `.onnx`, `.plan`, `.json`) | `--env paper`, **measured drive (0920 controller: K0 10, B-spline reference, 45 ms lead)** | residual-on-residual: PPO `--base_scale 0.6667 --dq_max 3 --residual_bound 0.3 --w_time 0.5 --w_speed 1 --v_soft 32` on `resid1_real_best` | **96.4 %** deterministic on 1024 paired episodes (base 95.3 %); **time to seal 5.71 -> 5.26 s, to goal 8.15 -> 7.79 s** at the same peak joint speed (38 deg/s median). Fused 3-level graph (resid3 + resid1 + dagger6). **Needs `real_policy_ctrl.DQ_MAX_DEG = 3.0`** |
| `resid1_real_best.pt` (+ `.onnx`, `.plan`) | `--env paper`, **measured drive** | residual PPO (`--residual_base dagger6_real/bc_iter10.pt --residual_bound 0.3 --critic_priv`) | 93.5 % at 7.9M steps; 95.3 % deterministic on 1024 episodes; the base of `resid3_fast_best` |
| `dagger5_real_iter7.pt` (+ `.onnx`, `.plan`) | `--env paper`, **measured drive** | pure DAgger, 7 rounds, 20 mm press teacher (80 %), `bc_curobo.py --resume` | **84.8 %** deterministic (83-86 % on 1024 episodes); the TensorRT engine `.plan` is what `rl/real_policy_ctrl.py` runs on the robot (rebuild with `rl/export_trt.py` on another GPU) |

`ppo*` observe the privileged 37-D state (`rl/env_warp.py: observe()`).
`distill1_student_rgbd` observes only 2×RGBD 96×96 + proprio + goal
(`rl/distill.py: Student`) — no object state.

Obs-dim note: older checkpoints predate observation growth; loaders in
`ppo.py`/`distill.py`/`demo_video.py` zero-pad mismatched rows.

Replay any of them:

    python rl/demo_video.py --actor rl/weights/ppo7_ped_teacher.pt \
        --algo ppo --mode pnp --scene rl/scenes/box_med_ped.xml \
        --out /tmp/demo

## Not in git

BC/export artifacts (~9.4 GB: DP/CFM/ACT checkpoints, traced TorchScript,
ONNX, TensorRT engines) exceed GitHub's 100 MB/file limit — 25 files do
individually. They live at `~/pnp_export/` and `~/pnp_runs_studio/` and are
regenerable via `policy/export_models.py` from the training checkpoints.
Full RL run history (all intermediate `ac.pt` + `log.jsonl`) is at `~/pnp_rl/`.

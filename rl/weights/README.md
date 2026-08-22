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

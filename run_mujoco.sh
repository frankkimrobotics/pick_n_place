#!/usr/bin/env bash
# run_mujoco.sh :: full MuJoCo pick-and-place demo in one command.
#   1) export the planned trajectory + baked robot meshes + MJCF   (curobo2 env)
#   2) render the kinematic replay to mp4 (headless OSMesa)         (base env, mujoco)
#
# Usage:  bash pick_and_place/run_mujoco.sh [planned|segmented] [camera]
set -e
MODE="${1:-planned}"
CAM="${2:-iso}"
ROOT="/home/lisc-frank/Desktop/2026"
CUROBO_PY="/home/lisc-frank/miniconda3/envs/curobo2/bin/python"
MJ_PY="/home/lisc-frank/miniconda3/bin/python"      # base env has mujoco 3.2.5

cd "$ROOT"
echo "== [1/2] exporting MuJoCo package (mode=$MODE) using curobo2 =="
"$CUROBO_PY" pick_and_place/mujoco_export.py --place-mode "$MODE"

echo "== [2/2] rendering MuJoCo replay (camera=$CAM, headless OSMesa) =="
MUJOCO_GL=osmesa "$MJ_PY" pick_and_place/mujoco_play.py --camera "$CAM"

echo "== done -> pick_and_place/outputs/mujoco/mujoco_demo.mp4 =="

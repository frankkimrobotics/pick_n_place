#!/usr/bin/env bash
# sim_launch :: run the full ROS2 simulated pick-and-place.
#
#   cuRobo planner socket (curobo2, :9997)  <-- must already be running
#     |socket
#   [sim_planner_node]  --ROS2 plan_request/result-->  [sim_orchestrator]
#   [sim_orchestrator]  --/mycobot/cmd/move-->         [sim_mujoco_node]  --/joint_states-->
#
# Args after the script name are passed to the orchestrator (e.g. --pick 0.3,0,0.06).
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source /opt/ros/humble/setup.bash 2>/dev/null
export MUJOCO_GL=osmesa
export ROS_DOMAIN_ID="${SIM_ROS_DOMAIN:-42}"   # isolate sim from the real robot bridge (domain 0)
LOGD="${SIM_LOGDIR:-/tmp/sim_ros2}"; mkdir -p "$LOGD"
VIDEO="${SIM_VIDEO:-$HERE/outputs/mujoco_sim/episode.mp4}"

if ! (echo >/dev/tcp/127.0.0.1/9997) 2>/dev/null; then
  echo "ERROR: cuRobo planner socket :9997 is down. Start it first (curobo2 env)."; exit 1
fi

echo "[launch] starting planner node + mujoco robot node..."
python3 sim_planner_node.py            >"$LOGD/planner_node.log" 2>&1 &  PP=$!
python3 sim_mujoco_node.py --video "$VIDEO" >"$LOGD/mujoco_node.log" 2>&1 &  PM=$!

cleanup() { kill -INT "$PM" 2>/dev/null; sleep 2; kill "$PP" "$PM" 2>/dev/null; }
trap cleanup EXIT

echo "[launch] waiting for nodes to come up (mujoco load + discovery)..."
sleep 8
echo "[launch] running orchestrator..."
python3 sim_orchestrator.py "$@"        2>&1 | tee "$LOGD/orch.log"

echo "[launch] orchestrator done; flushing mujoco video..."
kill -INT "$PM" 2>/dev/null
sleep 3
echo "[launch] video -> $VIDEO ; logs -> $LOGD"

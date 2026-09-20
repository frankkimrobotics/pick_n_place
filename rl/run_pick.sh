#!/bin/bash
# usage: run_pick.sh <n> [goal_x goal_y]
N=$1; GX=${2:-0.35}; GY=${3:--0.12}
cd ~/Desktop/2026/pick_and_place
D=$(python3 -c "
import socket, json, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', 9701)); s.settimeout(2); last=None; t0=time.time()
while time.time()-t0 < 0.6:
    try:
        m=json.loads(s.recv(4096))
        if m.get('cx') is not None: last=m
    except socket.timeout: break
print(last['cx'], last['cy'], last['top']) if last else print('NONE')")
[ "$D" = "NONE" ] && { echo "no object detected"; exit 1; }
read CX CY TOP <<< "$D"; HZ=$(python3 -c "print(round(($TOP+0.005)/2,4))")
read BX BY <<< $(python3 -c "import json; d=json.load(open('/home/lisc-frank/pnp_rl/tracker_bias.json')); print(d['bias_x'], d['bias_y'])" 2>/dev/null || echo "0 0")
CX=$(python3 -c "print(round($CX-$BX,4))"); CY=$(python3 -c "print(round($CY-$BY,4))")
if [ -z "$2" ]; then read GX GY <<< $(python3 -c "
import math, random; c=(float('$CX'), float('$CY'))
while True:
    g=(round(random.uniform(0.28, 0.48), 3), round(random.uniform(-0.24, 0.24), 3))
    if math.hypot(g[0]-c[0], g[1]-c[1]) > 0.12 and not (g[1] > 0.18 and g[0] < 0.30): break
print(g[0], g[1])"); fi
echo "object $CX $CY top $TOP -> half z $HZ; goal ($GX, $GY)"
CUDA_VISIBLE_DEVICES=0 timeout 240 ~/miniconda3/envs/mjwarp/bin/python rl/real_policy_ctrl.py --policy rl/weights/resid1_real_best --obj $CX $CY $HZ --half 0.04 0.04 $HZ --goal $GX $GY 0.22 --steps 200 --dq_max 2.0 --track --track_bias $BX $BY --touch_calib --go_home --force --exec --log ~/pnp_rl/real_pick_$N.json 2>&1 | grep "^\[ctrl\] guarded\|^\[calib\]\|^\[ctrl\] step\|^\[ctrl\] episode\|^\[ctrl\] place\|^\[ctrl\] suction\|^\[guard\]\|^\[result\]\|^\[home\]\|^\[warn\]\|no feedback\|abort\|Traceback"

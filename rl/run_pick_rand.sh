#!/bin/bash
# usage: run_pick_rand.sh <n>  -- like run_pick.sh but picks a RANDOM tracked object and asks the two-camera monitor for the verdict
N=$1
cd ~/Desktop/2026/pick_and_place
python3 -c "
import socket,json
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(3); s.sendto(json.dumps({'cmd':'snap'}).encode(),('127.0.0.1',9702)); print('[monitor] snap', s.recv(4096)[:60])" 2>/dev/null || echo "[monitor] not running"
D=$(python3 -c "
import socket, json, time, random
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', 9701)); s.settimeout(2); cands=[]; t0=time.time()
while time.time()-t0 < 0.6:
    try:
        m=json.loads(s.recv(4096))
        if m.get('cands'): cands=m['cands']
    except socket.timeout: break
ok=[c for c in cands if 0.18 < c['cx'] < 0.50 and abs(c['cy']) < 0.30 and c['n'] >= 300]
if not ok: print('NONE')
else:
    c=random.choice(ok); print(c['cx'], c['cy'], c['top'], len(ok))")
[ "$D" = "NONE" ] && { echo "no object detected"; exit 1; }
read CX CY TOP NC <<< "$D"; HZ=$(python3 -c "print(round(($TOP+0.005)/2,4))")
read BX BY <<< $(python3 -c "import json; d=json.load(open('/home/lisc-frank/pnp_rl/tracker_bias.json')); print(d['bias_x'], d['bias_y'])")
CX=$(python3 -c "print(round($CX-$BX,4))"); CY=$(python3 -c "print(round($CY-$BY,4))")
read GX GY <<< $(python3 -c "
import math, random; c=(float('$CX'), float('$CY'))
while True:
    g=(round(random.uniform(0.28, 0.48), 3), round(random.uniform(-0.24, 0.24), 3))
    if math.hypot(g[0]-c[0], g[1]-c[1]) > 0.12 and not (g[1] > 0.18 and g[0] < 0.30): break
print(g[0], g[1])")
echo "chosen object $CX $CY top $TOP (of $NC candidates) -> half z $HZ; goal ($GX, $GY)"
CUDA_VISIBLE_DEVICES=0 timeout 240 ~/miniconda3/envs/mjwarp/bin/python rl/real_policy_ctrl.py --policy rl/weights/resid1_real_best --obj $CX $CY $HZ --half 0.04 0.04 $HZ --goal $GX $GY 0.22 --steps 200 --dq_max 2.0 --track --track_bias $BX $BY --touch_calib --go_home --force --exec --log ~/pnp_rl/real_pick_$N.json 2>&1 | grep "^\[ctrl\] guarded\|^\[calib\]\|^\[ctrl\] step\|^\[ctrl\] episode\|^\[ctrl\] place\|^\[ctrl\] suction\|^\[guard\]\|^\[result\]\|^\[home\]\|^\[warn\]\|no feedback\|abort\|Traceback"
python3 -c "
import socket,json
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(10); s.sendto(json.dumps({'cmd':'verdict','goal':[$GX,$GY],'obj':[$CX,$CY],'top':max(0.02,$TOP)}).encode(),('127.0.0.1',9702)); r=json.loads(s.recv(65536))
print('[monitor]', r['verdict'], '(d_goal %s cm; agree %s)' % (None if r['d_goal'] is None else round(100*r['d_goal'],1), r['agree']))
print('[monitor] fixed appeared', [(round(c['cx'],3),round(c['cy'],3),c['src']) for c in r['fixed']['appeared']], 'vanished', [(round(c['cx'],3),round(c['cy'],3),c['src']) for c in r['fixed']['vanished']], 'wrist app/van', len(r['wrist']['appeared']), len(r['wrist']['vanished']))" 2>/dev/null || echo "[monitor] no verdict"

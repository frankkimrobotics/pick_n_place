"""Close the loop: a trained policy becomes the planner node — obs → policy → joint weld-chunk →
:9994 welder (receding horizon). The policy outputs joint targets directly (no IK).

Real:  D405+D435 via pyrealsense, joints via :9999, chunks to :9994, suction via GPIO.
Sim :  --sim uses ROS image topics /sim/wrist_rgb,/sim/fixed_rgb + /joint_states, chunks to /mycobot/cmd/move.

  python3 il/rollout.py --ckpt outputs/il_ckpt/umi_final.pt --source real --exec
"""
import argparse
import json
import os
import socket
import sys
import time
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "policy"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "mycobot_mpc")))
from policy import build_policy
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg   # noqa

PI = "10.0.0.27"; CMD_PORT = 9994; FB_PORT = 9999
WRIST_SER = "218622271300"; FIXED_SER = "043422070101"


def read_joints_rad():
    s = socket.create_connection((PI, FB_PORT), timeout=3); b = b""
    while b"\n" not in b:
        b += s.recv(4096)
    s.close(); d = json.loads(b.split(b"\n")[0])
    return np.array(linuxcnc_deg_to_rad(d["joints_deg"]), float)


def send_weld(traj_rad, dt):
    deg = [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj_rad]
    m = {"trajectory": deg, "traj_dt": dt, "target_deg": deg[-1], "weld": True, "t_anchor": time.time() + 0.1}
    k = socket.create_connection((PI, CMD_PORT), timeout=3)
    k.sendall((json.dumps(m) + "\n").encode()); k.close()


class RealCams:
    """Direct pyrealsense D405 (wrist) + D435 (fixed) → latest RGB."""
    def __init__(self, img=96):
        import pyrealsense2 as rs
        self.img = img; self.pipes = {}
        for name, ser in (("wrist", WRIST_SER), ("fixed", FIXED_SER)):
            p = rs.pipeline(); c = rs.config(); c.enable_device(ser)
            c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            p.start(c); self.pipes[name] = p
    def get(self, name):
        f = self.pipes[name].wait_for_frames(1000).get_color_frame()
        im = cv2.resize(np.asanyarray(f.get_data()), (self.img, self.img))
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def img_tensor(rgb):
    return torch.from_numpy(rgb).float().permute(2, 0, 1) / 127.5 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", default="real", choices=["real"])   # sim path: see module docstring
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--Ta", type=int, default=8, help="actions executed per inference (receding horizon)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--exec", action="store_true", help="actually stream to the robot (else dry-run)")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=dev)
    pol = build_policy(ck["policy"], action_dim=7, proprio_dim=7,
                       Tp=ck["cfg"]["Tp"], To=ck["cfg"]["To"]).to(dev)
    pol.load_state_dict(ck["ema"]); pol.eval()
    pol.a_mean.copy_(ck["norm"][0].to(dev)); pol.a_std.copy_(ck["norm"][1].to(dev))
    To = ck["cfg"]["To"]; print(f"[rollout] {ck['policy']} To={To} Tp={ck['cfg']['Tp']} Ta={args.Ta} exec={args.exec}")

    cams = RealCams(args.img_size)
    hist = []                                              # rolling obs history
    suction = 0
    for step in range(args.steps):
        q = read_joints_rad()
        prop = np.concatenate([q, [suction]]).astype(np.float32)   # (7)
        frame = {"wrist_rgb": img_tensor(cams.get("wrist")), "fixed_rgb": img_tensor(cams.get("fixed")),
                 "proprio": torch.from_numpy(prop)}
        hist.append(frame); hist = hist[-To:]
        if len(hist) < To:
            time.sleep(1.0 / args.fps); continue
        obs = {k: torch.stack([h[k] for h in hist], 0)[None].to(dev) for k in frame}   # (1,To,...)
        with torch.no_grad():
            a = pol.predict(obs)[0].cpu().numpy()          # (Tp,7)
        chunk = a[:args.Ta]; joints = chunk[:, :6]; suc_cmd = int(chunk[-1, 6] > 0.5)
        dt = 1.0 / args.fps
        if args.exec:
            send_weld(joints, dt)
            if suc_cmd != suction:
                try:
                    sys.path.insert(0, HERE.replace("/il", "")); import suction_test
                    suction_test.set_pin(bool(suc_cmd))
                except Exception as e:
                    print("  suction set failed:", e)
        suction = suc_cmd
        print(f"step {step:4d}  q0={q[0]:+.2f} suction={suction}  |a|={np.abs(chunk).mean():.3f}")
        time.sleep(args.Ta / args.fps)                     # execute Ta, then re-infer (receding horizon)
    print("done")


if __name__ == "__main__":
    main()

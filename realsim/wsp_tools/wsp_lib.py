import os, sys, glob, numpy as np, torch
P = os.path.expanduser("~/Desktop/2026/pick_and_place")
sys.path[:0] = [P, P + "/policy"]
import real_rollout_zed as RZ
RZ.DEV = os.environ.get("DEV", "cuda:1")
import train_policy_zed as T
from spline_action import ctrl_to_waypoints
CK = os.path.expanduser("~/pnp_policies/cfm_cradio_dit_w2048_mega_dit2048/ckpt_final_ema.pt")
def load():
    tl = torch.load
    torch.load = lambda *a, **k: tl(*a, **{**k, "map_location": "cpu"})
    pol = RZ.ZedPolicy(CK); torch.load = tl
    return pol
def traces(d):
    out = {}
    for f in sorted(glob.glob(os.path.expanduser(d) + "/trace*.npz")):
        out[int(f[-7:-4])] = dict(np.load(f, allow_pickle=True))
    return out
def run(pol, tok, prop, goal):
    """tok [4,1280] np, prop [2,10], goal [9] -> waypoints [16,10] (tool frame of obs)"""
    a = pol(torch.from_numpy(tok.astype(np.float32)).to(RZ.DEV), prop, goal)
    return ctrl_to_waypoints(a) if pol.action_mode == "bspline" else a

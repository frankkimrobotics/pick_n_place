#!/usr/bin/env python3
"""real_rollout_zed :: run a UMI-style visual policy (policy/train_policy_zed.py, trained on
~/pnp_dataset_zed with frozen lingbot features of the wrist D405 + the fixed ZED 2i) on the REAL
myCobot Pro 630.

Observation (identical to UmiDataset.__getitem__):
    tokens  [4, 196, 768]  lingbot patch tokens of wrist(t-1), wrist(t), zed(t-1), zed(t)
                           (224x224 centre crop of the 432x240 frame, policy/extract_features.frames_224)
    proprio [2, 10]        pose at t-1 and at t, each RELATIVE to the pose at t:
                           dp (tcp frame), rot6d(dR), suction state
Action  [16, 10]           tcp pose at t+1..t+16 (10 Hz) relative to the pose at t + suction.
The tcp pose is the MuJoCo "tcp" site of rl/scenes/box_med.xml at the measured joints
(the dataset's eef.npz is the same site of outputs/mujoco_sim/robot_warp.xml; verified identical).

Transport: rl/real_policy_ctrl.PiLink (joints from Pi :9999, 10 Hz reference segments to :9998,
suction through robot_hal {"suction":0/1}). Cameras: policy/zed_cam_server.py (/dev/shm jpegs).

Execution: receding horizon. Every --n_exec decisions a new chunk is inferred in a background thread
from the newest observation; its actions are absolute targets computed at observation time, so the
inference latency only shifts which action index is executed first. Each decision: IK (damped least
squares seeded at the last command) -> guards -> one 0.1 s reference segment.

SAFETY (defaults conservative; nothing moves without --exec):
  --slow S          every action step takes S x 0.1 s (policy time runs at 1/S)
  --step_max 0.02   max cartesian step per decision (m); larger predicted steps are clipped
  --dq_max 3.0      max joint step per decision (deg) -> 30 deg/s at --slow 1
  --max_tilt 20     max tilt of the commanded tool axis from straight down (deg)
  workspace/joint guard: rl/real_policy_ctrl.Guard (elbow box, LinuxCNC soft limits, wall keep-out,
                    tip z in [--z_floor, 0.60])
  torque contact guard: J2/J3 torque rise over a rolling baseline (frozen below --tau_freeze_z);
                    >= TAU_HARD holds any further descent (lift/lateral still allowed),
                    >= --tau_abort -> suction off, stop
  lead guard: a reference more than LEAD_MAX_DEG ahead of the measured joints holds the reference
  SIGINT: suction off, stop streaming (the Pi holds the last target)

    PY=~/miniconda3/envs/ilpolicy/bin/python
    $PY policy/real_rollout_zed.py --ckpt ~/pnp_policies/cfm_lingbot_dit/ckpt_final.pt --go_home   # dry run
    $PY policy/real_rollout_zed.py --ckpt ... --go_home --exec --slow 2
"""
import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import zed_mask
import pick_verify as _pv            # testable pick verification
from pick_verify import inspect_xy
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "rl"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "mycobot_mpc")))

import train_policy_zed as T  # noqa: E402
from real_policy_ctrl import (PiLink, Guard, ObsBuilder, scripted_move, START_Q, CTRL_DT,  # noqa: E402
                              LEAD_MAX_DEG, TAU_FIRM, TAU_HARD, W_J3)
from joint_conventions import linuxcnc_deg_to_rad, LINUXCNC_SOFT_LIMITS_DEG  # noqa: E402
import mujoco  # noqa: E402

# LinuxCNC soft limits (with the Guard's 2 deg margin + 3 deg) expressed in URDF radians, used as IK joint
# bounds: the sim demos' IK only knew the model's +-170 deg wrist range, and the real joint 4 (lcnc) drifted
# to -178 deg while pressing (runs 2 and 3: every press target was rejected by the guard)
_lo_lc = np.array([lo + 5.0 for lo, hi in LINUXCNC_SOFT_LIMITS_DEG][:6])
_hi_lc = np.array([hi - 5.0 for lo, hi in LINUXCNC_SOFT_LIMITS_DEG][:6])
_a, _b = np.asarray(linuxcnc_deg_to_rad(_lo_lc.tolist())), np.asarray(linuxcnc_deg_to_rad(_hi_lc.tolist()))
LC_LO, LC_HI = np.minimum(_a, _b), np.maximum(_a, _b)

SHM = "/dev/shm"
DEV = "cuda:0"

# tool orientation used by every demo (mjwarp_pick_demo.R_DOWN): z down, yawed +90 deg about world z
R_FLAT = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
_RZ90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
R_DOWN = _RZ90 @ R_FLAT
ELBOW_LO = np.radians([-180, -70, -145, -170, -167, -173])
ELBOW_HI = np.radians([180, 70, 145, 170, 167, 173])


# ----------------------------------------------------------------- maths
def rot6d_to_R(v):
    a1, a2 = np.asarray(v[:3], float), np.asarray(v[3:6], float)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - b1 * (b1 @ a2)
    b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def rot6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def tilt_deg(R):
    return math.degrees(math.acos(float(np.clip(-R[2, 2], -1.0, 1.0))))   # angle of the tool z axis from straight down


def ik(m, d, sid, target_p, target_R, q0, iters=200):
    """Damped least-squares 6D IK on the tcp site (mjwarp_pick_demo._ik_once, seeded only)."""
    d.qpos[:6] = q0
    perr = rerr = None
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.site_xpos[sid].copy()
        R = d.site_xmat[sid].reshape(3, 3).copy()
        perr = target_p - p
        rq = np.zeros(4)
        mujoco.mju_mat2Quat(rq, (target_R @ R.T).ravel())
        rerr = np.zeros(3)
        mujoco.mju_quat2Vel(rerr, rq, 1.0)
        err = np.concatenate([perr, 0.5 * rerr])
        if np.linalg.norm(perr) < 1e-4 and np.linalg.norm(rerr) < 1e-3:
            break
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, :6], 0.5 * jr[:, :6]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        d.qpos[:6] = np.clip(d.qpos[:6] + dq, np.maximum(np.maximum(m.jnt_range[:6, 0], ELBOW_LO), LC_LO),
                             np.minimum(np.minimum(m.jnt_range[:6, 1], ELBOW_HI), LC_HI))
    return d.qpos[:6].copy(), float(np.linalg.norm(perr)), float(np.linalg.norm(rerr))


GOAL_BASE = None

# ----------------------------------------------------------------- policy
class ZedPolicy:
    def __init__(self, ckpt, euler=16, frozen_noise=True):
        ck = torch.load(ckpt, map_location=DEV, weights_only=False)
        self.model, self.arch = ck["model"], ck.get("arch", "unet")
        self.action_mode = ck.get("action", "waypoint")   # "waypoint" | "bspline" (control points)
        self.lat_shift = int(ck.get("lat_shift") or 0)    # latency-matched chunks start at t+1+L
        if self.model not in ("dp", "cfm", "ode"):
            raise SystemExit(f"model {self.model}: only dp/cfm/ode checkpoints are supported here")
        kw = {}
        _sd = ck.get("ema") or ck["net"]
        if self.arch == "mlp":
            kw = {"d": int(ck["net"]["inp.weight"].shape[0])}
        elif self.arch == "dit":
            # infer width/depth from the checkpoint (mega DiT is d=2048, depth=12)
            kw = {"d": int(_sd["inp.weight"].shape[0]),
                  "depth": 1 + max(int(k.split(".")[1]) for k in _sd if k.startswith("blocks."))}
        if self.model == "ode":           # ODE policy (train_policy_zed.ODEField): H-step integrated path + suction head
            self.net = T.ODEField(cond_dim=512, H=int(ck.get("ode_H") or 8)).to(DEV)
        else:
            self.net = {"unet": T.CondUnet1D, "dit": T.DiT1D, "mlp": T.MLPDenoiser}[self.arch](cond_dim=512, **kw).to(DEV)
        self.use_goal = bool(ck.get("goal"))
        self.goal_se3 = bool(ck.get("goal_se3"))
        _fd = int(_sd["proj.0.weight"].shape[1]) if "proj.0.weight" in _sd else 768
        self.obs_enc = T.ObsEncoder(_fd, use_goal=self.use_goal,
                                    goal_dim=(9 if self.goal_se3 else 3)).to(DEV)
        w = ck.get("ema") or {}
        # strict=False: ckpts trained before the graspability head lack grasp_head.*
        self.net.load_state_dict({k: v for k, v in w.items() if k in self.net.state_dict()},
                                 strict=False)
        self.obs_enc.load_state_dict({k: v for k, v in w.items() if k in self.obs_enc.state_dict()},
                                     strict=False)
        self.net.eval()
        self.obs_enc.eval()
        self.lo, self.hi = ck["lo"].to(DEV), ck["hi"].to(DEV)
        self.euler = euler
        self.frozen_noise = frozen_noise
        self.fast = False                # True: fp16 encoder + half-weight compiled denoiser (set before the first call,
        self._fnet = None                # after any weight patch); same chunks to ~0.1 mm, about 2.4x faster
        self.decode_c0 = None            # set by the executor before each inference (bspline)
        self.gen = torch.Generator(device=DEV)
        self.gen.manual_seed(0)
        self.noise0 = torch.randn(1, T.ADIM, T.TA, device=DEV, generator=self.gen)
        if self.model == "dp":
            from diffusers import DDIMScheduler
            self.sched = DDIMScheduler(num_train_timesteps=50, beta_schedule="squaredcos_cap_v2",
                                       prediction_type="epsilon")
            self.sched.set_timesteps(16)
        # frozen encoder
        import extract_features as _EF
        _EF.DEV = DEV
        self._fwd, _m, _s = _EF.load_encoder(ck.get("enc", "lingbot"))
        self._emean, self._estd = _m.to(DEV), _s.to(DEV)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=DEV).view(1, 3, 1, 1)
        print(f"[policy] {os.path.basename(os.path.dirname(ckpt))}: {self.model}/{self.arch} step {ck.get('step')} "
              f"euler {euler} {'frozen' if frozen_noise else 'fresh'} noise", flush=True)

    def _basis(self):
        from spline_action import BASIS
        return torch.from_numpy(np.asarray(BASIS, np.float32)).to(DEV)

    @torch.no_grad()
    def encode(self, bgr):
        """BGR frame (240x424 or 240x432) -> [196, 768] patch tokens, exactly extract_features.frames_224."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != 432 or rgb.shape[0] != 240:
            rgb = cv2.resize(rgb, (432, 240), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(DEV) / 255.0
        t = torch.nn.functional.interpolate(t, size=(224, 404), mode="bilinear", align_corners=False)
        x0 = (404 - 224) // 2
        px = (t[:, :, :, x0:x0 + 224] - self._emean) / self._estd
        if self.fast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._fwd(px)[0].float()
        return self._fwd(px)[0]

    def _fast_net(self):
        """the 920M DiT reads 3.7 GB of fp32 weights per flow step: half weights + a compiled CUDA graph"""
        if self._fnet is None:
            self.net.half()
            self._fnet = torch.compile(self.net, mode="reduce-overhead")
        return self._fnet

    @torch.no_grad()
    def __call__(self, tok4, prop, goal=None, rtc=None):
        """tok4 [4,196,768] torch, prop [2,10] np, goal [3] np -> act [16,10].
        rtc = (Y [16,10] waypoints in this observation's frame, w [16,10] in 0..1): real-time-chunking style
        conditioning (cfm only). At every flow step the predicted clean chunk is pulled toward Y where w > 0
        (w = 1: the waypoint is the old chunk's), so the new chunk continues the one being executed."""
        g = (torch.from_numpy(np.asarray(goal, np.float32))[None].to(DEV)
             if self.use_goal else None)
        cond = self.obs_enc(tok4[None], torch.from_numpy(np.asarray(prop, np.float32))[None].to(DEV), g)
        if self.model == "ode":
            # integrate from the identity action; the chunk is H (< TA) waypoints, padded by holding the last one
            # (the executor re-plans every n_exec = 8 decisions, so the padding is only reached on a late plan)
            lo9, hi9 = self.lo[:9], self.hi[:9]
            a0 = ((torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0.0], device=DEV) - lo9) / (hi9 - lo9) * 2 - 1)[None]
            traj = self.net.rollout(a0, cond)[0]                       # [H, 9] normalized
            pose = (traj + 1) / 2 * (hi9 - lo9) + lo9
            suc = torch.sigmoid(self.net.suc_head(cond))[0]            # [H]
            act = torch.cat([pose, suc[:, None]], 1)
            act = torch.cat([act, act[-1:].expand(T.TA - act.shape[0], -1)], 0)
            return act.cpu().numpy()
        x = self.noise0.clone() if self.frozen_noise else torch.randn(1, T.ADIM, T.TA, device=DEV, generator=self.gen)
        if self.model == "cfm":
            n = self.euler
            if rtc is not None:
                # waypoints = BASIS @ control points; BASIS rows sum to 1, so the per-dim normalisation commutes
                _B = torch.eye(T.TA, device=DEV) if self.action_mode != "bspline" else self._basis()
                _Bp = torch.linalg.pinv(_B, rtol=1e-3)
                Yn = (torch.from_numpy(np.asarray(rtc[0], np.float32)).to(DEV) - self.lo) / (self.hi - self.lo) * 2 - 1
                Wt = torch.from_numpy(np.asarray(rtc[1], np.float32)).to(DEV)
            fnet = self._fast_net() if (self.fast and rtc is None) else None
            for i in range(n):
                t = torch.full((1,), i / n, device=DEV)
                if fnet is not None:
                    with torch.autocast("cuda", dtype=torch.float16):
                        v = fnet(x, t * 50, cond)
                    v = v.float().clone()
                else:
                    v = self.net(x, t * 50, cond)
                if rtc is not None:
                    x1 = (x + (1 - i / n) * v)[0].T                    # predicted clean chunk [TA, ADIM]
                    dC = _Bp @ (Wt * (Yn - _B @ x1))                    # smallest control-point change that meets it
                    x = x + v / n + (i + 1) / n * dC.T[None]
                else:
                    x = x + v / n
        else:
            for t in self.sched.timesteps:
                eps = self.net(x, t.expand(1).to(DEV), cond)
                x = self.sched.step(eps, t, x).prev_sample
        act = (x.transpose(1, 2) + 1) / 2 * (self.hi - self.lo) + self.lo
        a = act[0].cpu().numpy()
        # bspline ckpts: return RAW control points; the executor decodes at plan-ADOPTION
        # time with the then-current commanded state (c0) and velocity (v0), so chunk
        # boundaries splice smoothly (decoding here would anchor to a ~300 ms stale pose).
        return a


# ----------------------------------------------------------------- cameras
def read_jpg(path):
    for _ in range(3):
        try:
            with open(path, "rb") as f:
                b = f.read()
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except OSError:
            pass
        time.sleep(0.003)
    return None


def cam_stamp():
    try:
        with open(f"{SHM}/pnp_cam_stamp.json") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"wrist": 0.0, "zed": 0.0}


# ----------------------------------------------------------------- main
def rel_pose(p_ref, R_ref, p, R, suc):
    dp = R_ref.T @ (p - p_ref)
    dR = R_ref.T @ R
    return np.concatenate([dp, rot6d(dR), [float(suc)]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/pnp_policies/cfm_lingbot_dit/ckpt_final.pt"))
    ap.add_argument("--pi", default="192.168.50.2")
    ap.add_argument("--exec", action="store_true", help="actually send commands (default: dry run)")
    ap.add_argument("--go_home", action="store_true", help="scripted move to the training start pose first")
    ap.add_argument("--steps", type=int, default=400, help="max decisions (0.1 s policy time each)")
    ap.add_argument("--n_exec", type=int, default=8, help="actions executed per chunk before re-planning (<= 16)")
    ap.add_argument("--euler", type=int, default=16)
    ap.add_argument("--fast_infer", type=int, default=0, help="1 = fp16 encoder + half-weight compiled denoiser (cfm, no --rtc)")
    ap.add_argument("--fresh_noise", action="store_true")
    ap.add_argument("--slow", type=float, default=1.0)
    ap.add_argument("--step_max", type=float, default=0.015, help="max LATERAL cartesian step per decision (m)")
    ap.add_argument("--step_max_z", type=float, default=0.030,
                    help="separate vertical step cap (m); 0 = use step_max. Descent is protected by the "
                         "contact guard, so it can be looser than lateral motion")
    ap.add_argument("--dq_max", type=float, default=2.5,
                    help="max joint step per decision (deg). Measured 2026-11-08: 2.5 gave cmd-step "
                         "p95 4.14->2.81 deg and lead-guard lurches 1->0 vs the old 3.0")
    ap.add_argument("--dq_mid", type=float, default=0.0,
                    help="separate joint step cap (deg/decision) for joints 2-4, used only while the commanded tool height is above "
                         "--dq_mid_z; 0 = off. Twin 2026-11-19: 3.75 (25 deg/s at --slow 1.5) with the goal time-out boxed 127 vs 104 "
                         "objects on 55 held-out scenes, at twice the time above 36 deg/s. Not yet run on the arm.")
    ap.add_argument("--dq_mid_z", type=float, default=0.15, help="FK tool height (m) above which --dq_mid applies (twin used 0.12, table at z=0)")
    ap.add_argument("--goal_to", type=float, default=0.0,
                    help="s without an attached object before the episode gives up on this --goal (0 = off); prints '[result] goal_giveup'")
    ap.add_argument("--zed_mask", type=int, default=0, help="1 = the policy sees the table-only ZED view (zed_mask.py); needs a patch fitted on masked features")
    ap.add_argument("--aim", type=float, default=0.0,
                    help="m; final-approach assist: within this lateral distance of --goal (and below --aim_z above it) the lateral "
                         "target is put on the goal xy. 0 = off. Twin 2026-11-21 (with --aim_down 0.06 and the goal time-out): 146 vs 104 "
                         "objects on 55 held-out scenes, 132 with an 8 mm position error; 15 mm error without the descent was worse than off.")
    ap.add_argument("--aim_z", type=float, default=0.15, help="m above the goal below which --aim acts")
    ap.add_argument("--aim_down", type=float, default=0.0,
                    help="m/s; with --aim, once centred within 1 cm the executor descends at this rate, opens the valve 5 cm above the "
                         "object and presses to 10 mm below the goal height; the carry hand-off waits for the bottom or torque contact")
    ap.add_argument("--aim_press", type=float, default=2.0, help="s at the bottom without a hand-off before the descent assist lets go for 5 s")
    ap.add_argument("--box_real", type=float, nargs=5, default=None, metavar=("X0", "X1", "Y0", "Y1", "H"),
                    help="outer footprint (base frame, m) and rim height of the real place box: four walls added to the safety "
                         "filter / scripted-path check. Measured from the ZED 2026-11-21: -0.05 0.21 0.24 0.49 0.23")
    ap.add_argument("--max_tilt", type=float, default=30.0, help="deg from straight down (START_Q itself is 20 deg)")
    ap.add_argument("--orient", default="yawlock", choices=["yawlock", "fixed", "policy"],
                    help="yawlock: predicted tool tilt, yaw locked to the demos' 90 deg (run 2: the policy's relative "
                         "rotations random-walked the wrist yaw into the joint-4 soft limit); fixed: R_DOWN for every "
                         "target (rotates the 20 deg start tilt out at once); policy: the predicted rotation as is")
    ap.add_argument("--x_min", type=float, default=0.05, help="tip x keep-out (m); the sim bin is at x 0.10, the wall at x -0.30")
    ap.add_argument("--z_floor", type=float, default=0.005, help="lowest allowed cup-tip z (m); table top is z=0")
    ap.add_argument("--tau_abort", type=float, default=0.25)
    ap.add_argument("--tau_freeze_z", type=float, default=0.15, help="torque baseline frozen when the tip is below this z")
    ap.add_argument("--no_suction", action="store_true", help="never send suction")
    ap.add_argument("--goal_dz", type=float, default=-0.024,
                    help="ZED->FK z correction applied to --goal (m). Measured 2026-11-08 by "
                         "torque-guarded touch at 4 points: the ZED depth map reports surfaces "
                         "23.9 mm +-6.9 HIGHER than the robot finds them, so an uncorrected goal "
                         "makes the cup hover above the object and never seal.")
    ap.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X","Y","Z"),
                    help="base-frame grasp target (goal-conditioned ckpts): object top + cup radius. "
                         "Static for the whole episode (training semantics: goal stays at the original "
                         "grasp location through the carry).")
    ap.add_argument("--suction_z", type=float, default=0.09, help="suction predictions are ignored above this tip z unless the torque guard sees contact")
    ap.add_argument("--place_xy", type=float, nargs=2, default=None, metavar=("X","Y"),
                    help="RUNTIME box position: once the policy has the object attached, hand off to a "
                         "scripted carry to this base-frame xy and release. Decouples the place target "
                         "from the bin baked into the policy, and runs at full joint speed.")
    ap.add_argument("--place_z", type=float, default=0.22, help="release height above the table (m)")
    ap.add_argument("--place_v", type=float, default=25.0, help="scripted carry peak joint speed (deg/s)")
    ap.add_argument("--stop_after_place", type=int, default=0,
                    help="end the episode this many decisions after a carried object is released "
                         "(0 = run to --steps). The 2026-10-16 run placed at k120 but ran to k220.")
    ap.add_argument("--suction_off_n", type=int, default=5, help="consecutive 'off' predictions before releasing")
    ap.add_argument("--script_vmax", type=float, default=15.0, help="peak joint speed of the scripted home legs (deg/s)")
    ap.add_argument("--lead_max", type=float, default=LEAD_MAX_DEG)
    ap.add_argument("--log", default=None, help="directory for the per-step log + observation frames")
    ap.add_argument("--force", action="store_true", help="skip the start-pose check")
    ap.add_argument("--seal_ckpt", default=None,
                    help="RL seal controller (rl/seal_sac Actor, 10 Hz-trained, e.g. ~/pnp_rl/seal_sac13/"
                         "actor_best.pt). Enables the contact-phase splice: when the tip hovers near "
                         "--goal, scripted align + press/seal/lift by the seal policy instead of the CFM.")
    ap.add_argument("--handoff_r", type=float, default=0.06, help="xy radius around the goal that arms the splice (m)")
    ap.add_argument("--handoff_dz", type=float, default=0.12, help="tip height above goal z that arms the splice (m)")
    ap.add_argument("--seal_retries", type=int, default=2, help="seal-policy attempts before falling back to the CFM")
    ap.add_argument("--tau_scale", type=float, default=6.7,
                    help="real J2/J3 torque rise -> sim torque-proxy units (sim firm contact ~0.7, real ~0.105)")
    ap.add_argument("--table_z", type=float, default=0.030, help="FK z of the table top (seal obs tipz offset)")
    ap.add_argument("--seal_dry", action="store_true", help="splice prints actions but does not execute them")
    ap.add_argument("--lat_shift_override", type=int, default=-1,
                    help="override the ckpt's lat_shift at inference (match MEASURED latency; "
                         "e.g. the 920M DiT runs ~430 ms = 4-5 decisions vs its trained 2)")
    ap.add_argument("--place_r", type=float, default=0.15,
                    help="radius for the place-arrival check (m). Must cover how far a released "
                         "object bounces/rolls: dropping from --place_z scatters it well beyond "
                         "the release xy, and a tight radius reports a delivered object as missing")
    ap.add_argument("--splice_tau", type=float, default=0.16,
                    help="descent-refusal torque rise INSIDE the splice (crush protection; "
                         "separate from --tau_abort, which must stay above carry/payload torque)")
    ap.add_argument("--stream", default="deployed", choices=["deployed", "tuned"],
                    help="Pi stream gains: deployed (arm runs ~42 ms ahead of its reference; all runs before 2026-11-14) "
                         "or tuned (real_policy_ctrl.STREAM_GAINS_TUNED: on the reference, 10x lower tracking error)")
    ap.add_argument("--safe", type=int, default=0, help="1 = realsim/safety.py filter on every streamed target (speed, accel, joint limits, "
                                                        "arm vs table / back wall)")
    ap.add_argument("--mast_r", type=float, default=0.050, help="m, radius of the ZED mast keep-out pillar in the filter model (twin uses the 1 cm mast; "
                                                                    "here it also has to cover the ZED body and the wrist camera, which the meshes lack)")
    ap.add_argument("--safe_wall", type=float, default=0.05, help="m, clearance the filter keeps to the back wall (x < -0.30)")
    ap.add_argument("--ws_xmin", type=float, default=-0.05, help="m, lowest tool-target x; the policy cannot steer the tool behind it")
    ap.add_argument("--safe_margin", type=float, default=0.010, help="m, arm-to-table/wall clearance kept by the filter")
    ap.add_argument("--safe_vlim", type=float, default=30.0, help="deg/s joint speed limit of the filter")
    ap.add_argument("--safe_alim", type=float, default=0.0, help="deg/s^2 joint acceleration limit of the filter (0 = off)")
    ap.add_argument("--blend", type=int, default=0, help="cross-fade a new chunk over this many decisions (0 = switch)")
    ap.add_argument("--vmax", type=float, default=0.0, help="deg/s Pi velocity-command clamp (0 = keep the stream gains' 50)")
    ap.add_argument("--patch", default="", help="weight-space patch (.pt, realsim/wsp.py) applied on top of --ckpt")
    a = ap.parse_args()
    if a.stream == "tuned":
        import real_policy_ctrl as _rpc
        _rpc.STREAM_GAINS.update(_rpc.STREAM_GAINS_TUNED)       # PiLink reads the module dict at send time
        print(f"[stream] tuned gains {_rpc.STREAM_GAINS}", flush=True)
    if a.vmax > 0:
        import real_policy_ctrl as _rpc
        _rpc.STREAM_GAINS["vmax"] = float(a.vmax)
        _rpc.VEL_CMD_MAX = int(round(a.vmax * _rpc.STREAM_GAINS["vel_scale"]))
        print(f"[stream] Pi velocity clamp {a.vmax} deg/s (vel_cmd_max {_rpc.VEL_CMD_MAX})", flush=True)
    assert 1 <= a.n_exec <= T.TA
    dt = CTRL_DT * a.slow

    log_dir = a.log or os.path.expanduser(time.strftime("~/pnp_real_runs/zed_%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(os.path.join(log_dir, "steps.jsonl"), "w")
    json.dump(vars(a), open(os.path.join(log_dir, "args.json"), "w"), indent=1)

    policy = ZedPolicy(a.ckpt, euler=a.euler, frozen_noise=not a.fresh_noise)
    policy.fast = bool(a.fast_infer)
    if a.patch:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from realsim.wsp import apply_patch
        apply_patch(policy, os.path.expanduser(a.patch))
    if policy.use_goal and a.goal is None:
        raise SystemExit("[abort] goal-conditioned checkpoint needs --goal X Y Z (object top + cup radius)")
    if policy.fast:                                  # the first fast call compiles the denoiser (~8 s): do it before any motion
        _t = time.time()
        _tok = policy.encode(np.zeros((240, 432, 3), np.uint8))
        _prop = np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32), (2, 1))
        for _ in range(3):
            policy(torch.stack([_tok] * 4), _prop, np.zeros(9 if getattr(policy, "goal_se3", False) else 3, np.float32))
        print(f"[policy] fast inference warmed up in {time.time() - _t:.1f} s", flush=True)
    global GOAL_BASE
    GOAL_BASE = np.asarray(a.goal, float) if a.goal is not None else None
    if GOAL_BASE is not None:
        _cf = os.path.join(ROOT, "outputs", "zed_fk_correction.json")
        if os.path.exists(_cf):
            _c = json.load(open(_cf))["coef"]            # spatially-varying plane fit
            _dz = _c[0] * GOAL_BASE[0] + _c[1] * GOAL_BASE[1] + _c[2]
            GOAL_BASE = GOAL_BASE + np.array([0.0, 0.0, _dz])
            print(f"[goal] ZED->FK plane correction {1000*_dz:+.1f} mm -> {np.round(GOAL_BASE,4).tolist()}", flush=True)
        elif a.goal_dz:
            GOAL_BASE = GOAL_BASE + np.array([0.0, 0.0, a.goal_dz])
            print(f"[goal] ZED->FK constant correction {1000*a.goal_dz:+.0f} mm", flush=True)
    ob = ObsBuilder(os.path.join(ROOT, "rl", "scenes", "box_med.xml"), [0.025, 0.025, 0.02])
    m, dik = ob.m, mujoco.MjData(ob.m)
    FILT = None
    if a.safe:
        # static world the real cell has: table top z = 0 (wide slab; the scene's sim bin is not there), back wall x < -0.30
        sys.path.insert(0, ROOT)
        from realsim.safety import SafetyFilter
        _sp = mujoco.MjSpec.from_file(os.path.join(ROOT, "rl", "scenes", "box_med.xml"))
        for _g in list(_sp.worldbody.geoms):
            if _g.name.startswith("box_"):
                _sp.delete(_g)
        _sp.worldbody.add_geom(name="real_table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.55, 0.0, -0.01], size=[0.40, 0.60, 0.01])
        if a.box_real is not None:
            _x0, _x1, _y0, _y1, _bh = a.box_real
            _w = 0.01
            for _n, _pos, _sz in (("real_box_xm", [_x0, (_y0 + _y1) / 2, _bh / 2], [_w, (_y1 - _y0) / 2, _bh / 2]),
                                  ("real_box_xp", [_x1, (_y0 + _y1) / 2, _bh / 2], [_w, (_y1 - _y0) / 2, _bh / 2]),
                                  ("real_box_ym", [(_x0 + _x1) / 2, _y0, _bh / 2], [(_x1 - _x0) / 2, _w, _bh / 2]),
                                  ("real_box_yp", [(_x0 + _x1) / 2, _y1, _bh / 2], [(_x1 - _x0) / 2, _w, _bh / 2])):
                _sp.worldbody.add_geom(name=_n, type=mujoco.mjtGeom.mjGEOM_BOX, pos=_pos, size=_sz, rgba=[0.6, 0.5, 0.3, 0.0], group=5)
        from realsim.mast import add_zed_mast            # the ZED mast broke the wrist camera mount on 2026-11-15
        _zp = np.array(json.load(open(os.path.join(ROOT, "outputs/extrinsics_zed.json")))["T_base_camzed"])[:3, 3]
        add_zed_mast(_sp, cam_pos=_zp, radius=a.mast_r)
        print(f"[safe] ZED mast pillar at ({_zp[0]:.3f},{_zp[1]:+.3f}), z up to {_zp[2]:.3f}, radius {1000 * a.mast_r:.0f} mm", flush=True)
        FILT = SafetyFilter(_sp.compile(), margin=a.safe_margin, v_lim=math.radians(a.safe_vlim), a_lim=math.radians(a.safe_alim),
                            wall_margin=a.safe_wall)
        print(f"[safe] filter on: margin {1000 * a.safe_margin:.0f} mm, v {a.safe_vlim} deg/s, a {a.safe_alim} deg/s^2, "
              f"world {[FILT.m.geom(g).name for g in FILT.world]} + wall x<-0.30", flush=True)

    class ZedGuard(Guard):
        """real_policy_ctrl.Guard with a configurable tip-x keep-out (its 0.12 blocks the sim bin at x 0.10)."""
        def check_q(self, q):
            v = [s for s in Guard.check_q(self, q) if not s.startswith("tip x ")]
            tcp, _ = self.ob.fk(q)
            if tcp[0] < a.x_min:
                v.append(f"tip x {tcp[0]:.3f} < {a.x_min} (keep-out)")
            if FILT is not None:
                # scripted legs (lift / inspect / carry / home) are joint-space paths that never pass the per-decision
                # filter: check every path point against the same static world (table, back wall, ZED mast)
                _c, _ = FILT.clearance(np.asarray(q, float))
                if _c < a.safe_margin:
                    v.append(f"arm clearance {1000 * _c:.0f} mm < {1000 * a.safe_margin:.0f} mm (table / wall / ZED mast)")
            return v
    guard = ZedGuard(ob, a.z_floor + 0.03, force=a.force)   # Guard.z_floor = obj_top - 0.03

    # cameras
    st = cam_stamp()
    if time.time() - max(st.get("wrist", 0), st.get("zed", 0)) > 1.0:
        raise SystemExit("camera frames are stale -- start policy/zed_cam_server.py (system python3)")
    if time.time() - st.get("zed", 0) > 1.0 or time.time() - st.get("wrist", 0) > 1.0:
        raise SystemExit(f"one camera is stale: {st}")

    # robot
    link = PiLink(a.pi, a.exec, ref_mode="spline")
    t_wait = time.time()
    while not link.ok() and time.time() - t_wait < 6:
        time.sleep(0.1)
    q, qd, age = link.state()
    if q is None:
        raise SystemExit(f"no feedback from {a.pi}:9999 -- is robot_hal running on the Pi?")
    print(f"[robot] q (URDF deg) {np.round(np.degrees(q), 2).tolist()}  fb age {1000 * age:.0f} ms  "
          f"cmd {'ok' if link.cmd_sock else 'DOWN'}  {'EXEC' if a.exec else 'DRY RUN'}", flush=True)
    tcp, R = ob.fk(q)
    print(f"[robot] cup tip {np.round(tcp, 4).tolist()}  tilt {tilt_deg(R):.1f} deg", flush=True)

    state = {"suction": 0, "stop": False}

    def stop(*_):
        print("\n[ctrl] interrupted -> suction off, stop streaming (Pi holds)", flush=True)
        state["stop"] = True
        try:
            link.suction(False)
        except Exception:  # noqa: BLE001
            pass
        link.close()
        sys.exit(1)
    signal.signal(signal.SIGINT, stop)

    import real_policy_ctrl as RPC
    RPC.SCRIPT_VMAX_DEG = float(a.script_vmax)
    if a.go_home:
        q_m = scripted_move(link, guard, ob, START_Q, "home (START_Q)", v_peak_deg=a.script_vmax, min_T=1.0)
        if q_m is None:
            raise SystemExit("[abort] path to START_Q violates the guard")
        q = q_m
    dev = float(np.degrees(np.abs(q - START_Q)).max())
    if dev > 6.0 and not a.force:
        raise SystemExit(f"[abort] arm is {dev:.1f} deg from the training start pose; use --go_home or --force")
    v0 = guard.check_q(q)
    if v0:
        raise SystemExit("[abort] start pose violates: " + "; ".join(v0))

    # PRE-GRASP REFERENCE, taken now while the arm is at home and the view is clear.
    # The lifted-vs-dragged test needs to know what the neighbourhood looked like BEFORE
    # the grasp -- a drag is only detectable as occupancy appearing where there was none.
    nbr_ref = [None]
    place_ref = [None]
    try:
        _Tz = np.array(json.load(open(os.path.join(ROOT, "outputs/extrinsics_zed.json")))["T_base_camzed"])
        if GOAL_BASE is not None:
            _gxy = (float(GOAL_BASE[0]), float(GOAL_BASE[1]))
            nbr_ref[0] = (_pv.occupancy_cells(_gxy, a.table_z, _Tz, shm=SHM),
                          _pv.occupancy_cells(_gxy, a.table_z, _Tz, shm=SHM, R=0.04),
                          _pv.table_coverage(_gxy, a.table_z, _Tz, shm=SHM))
            print(f"[verify] pre-grasp reference at ({_gxy[0]:.3f},{_gxy[1]:+.3f}): "
                  f"{len(nbr_ref[0][0])} cells within 12 cm, "
                  f"{len(nbr_ref[0][1])} in the object footprint", flush=True)
            if len(nbr_ref[0][1]) == 0:
                # The same reference that powers the drag test also says whether there is
                # anything to pick. Saying so now beats a descent onto bare table followed
                # by an unexplainable "lifted" from the seal controller.
                print(f"[verify] WARNING: no object-height surface at the goal -- the goal "
                      f"looks EMPTY (stale coordinates?); the pick will have nothing to seal on",
                      flush=True)
        if a.place_xy is not None:
            place_ref[0] = _pv.occupancy_cells((a.place_xy[0], a.place_xy[1]), a.table_z,
                                               _Tz, shm=SHM, R=a.place_r)
            print(f"[verify] place-site reference: {len(place_ref[0])} cells occupied "
                  f"within {100 * a.place_r:.0f} cm", flush=True)
    except Exception as _e:
        print(f"[verify] pre-grasp reference unavailable ({_e})", flush=True)

    tau_base = link.torque_baseline(1.0) if a.exec else np.zeros(6)
    print(f"[contact] torque baseline {np.round(tau_base, 3).tolist()}  firm {TAU_FIRM} hard {TAU_HARD} abort {a.tau_abort}", flush=True)
    tq_hist = deque(maxlen=10)

    # ---- receding-horizon loop
    hist = deque(maxlen=4)          # per decision: dict(t, wrist_tok, zed_tok, p, R, suc, wrist_img, zed_img)
    plan = None                     # dict(k_obs, p_abs [16,3], R_abs [16,3,3], suc [16], latency)
    pending = {}                    # inference in flight: thread + result slot
    q_cmd = np.asarray(q, float).copy()
    p_cmd, R_cmd = ob.fk(q_cmd)
    off_count = 0
    attached_since = None
    n_hold = 0
    t0 = time.time() + 0.3
    k = 0
    seq = 0
    released_at = [None]          # decision index at which a carried object was released
    aim_block, aim_press, aim_n = 0, 0, 0

    _T_ZED = None

    def zed_site_state(xy, table_z, r=0.03):
        """Delegates to policy.pick_verify so the executor and the standalone test
        (/tmp/verify_test.py) exercise the identical code path."""
        nonlocal _T_ZED
        if _T_ZED is None:
            try:
                _T_ZED = np.array(json.load(open(os.path.join(ROOT, "outputs/extrinsics_zed.json")))["T_base_camzed"])
            except Exception as e:
                return "unknown", f"extrinsics unavailable ({e})"
        return _pv.site_state(xy, table_z, _T_ZED, shm=SHM, r=r)

    # unloaded torque reference captured at a known joint pose, for payload differencing
    probe_ref = [None]

    def torque_median(dur=0.3):
        s = []
        t_end = time.time() + dur
        while time.time() < t_end:
            s.append(link.torque()); time.sleep(0.01)
        return np.median(np.array(s), axis=0) if s else np.zeros(6)

    prev_p_cmd = None
    if a.lat_shift_override >= 0:
        print(f"[policy] lat_shift {policy.lat_shift} -> {a.lat_shift_override} (inference override)", flush=True)
        policy.lat_shift = a.lat_shift_override
    splice = None
    splice_tries = [0]
    if a.seal_ckpt:
        from seal_splice import SealSplice
        splice = SealSplice(a.seal_ckpt, tau_scale=a.tau_scale, table_z=a.table_z, dry=a.seal_dry)

    def infer_worker(obs, k_obs, slot):
        t1 = time.time()
        tok4 = torch.stack([obs[0]["wrist_tok"], obs[1]["wrist_tok"], obs[0]["zed_tok"], obs[1]["zed_tok"]])
        p_t, R_t = obs[1]["p"], obs[1]["R"]
        prop = np.stack([rel_pose(p_t, R_t, obs[0]["p"], obs[0]["R"], obs[0]["suc"]),
                         rel_pose(p_t, R_t, p_t, R_t, obs[1]["suc"])])
        goal_rel = None
        if GOAL_BASE is not None:
            goal_rel = R_t.T @ (GOAL_BASE - p_t)
            if getattr(policy, "goal_se3", False):
                _dRg = R_t.T @ R_DOWN
                goal_rel = np.concatenate([goal_rel, _dRg[:, 0], _dRg[:, 1]])
        act = policy(tok4, prop, goal_rel)
        p_abs = np.stack([p_t + R_t @ act[i, :3] for i in range(T.TA)])
        R_abs = np.stack([R_t @ rot6d_to_R(act[i, 3:9]) for i in range(T.TA)])
        slot["plan"] = dict(k_obs=k_obs, p_abs=p_abs, R_abs=R_abs, suc=act[:, 9].copy(), act=act,
                            latency=time.time() - t1, p_t=p_t.copy(), R_t=R_t.copy())

    print(f"[ctrl] loop: dt {dt:.2f} s, n_exec {a.n_exec}, step_max {a.step_max} m, dq_max {a.dq_max} deg, "
          f"max tilt {a.max_tilt} deg, z floor {a.z_floor}", flush=True)
    t_free0 = time.time()
    last_print = 0.0
    try:
        while k < a.steps and not state["stop"]:
            if (released_at[0] is not None
                    and k - released_at[0] >= a.stop_after_place):
                print(f"[ctrl] placed at k {released_at[0]} -> early stop at k {k}", flush=True)
                break
            if attached_since is not None or released_at[0] is not None:
                t_free0 = time.time()                      # goal time-out counts only time with nothing attached
            elif a.goal_to > 0 and time.time() - t_free0 > a.goal_to:
                if state["suction"]:
                    link.suction(False); state["suction"] = 0
                print(f"[ctrl] goal not picked within {a.goal_to:.0f} s -> giving up at k {k}", flush=True)
                print("[result] goal_giveup", flush=True)
                break
            t_k = t0 + k * dt
            # scripted legs (splice align/seal, place carry, recovery) take 10-30 s without
            # advancing k; without a re-base the loop then free-runs ~30 s behind its grid at
            # ~0.03 s/decision (policy played ~3x its trained speed, plans adopted 4-5 decisions
            # late, every segment anchored in the past -- 2026-11-07/08 logs). Re-base the grid
            # and restart the spline stream so the first segment does not bend back to stale targets.
            if time.time() - t_k > 2 * dt:
                late = time.time() - t_k
                t0 = time.time() - k * dt
                t_k = t0 + k * dt
                if hasattr(link, "hist"):
                    link.hist.clear()
                    link.spline_t0 = None
                print(f"[ctrl] decision grid re-based at k {k} ({late:.1f} s late)", flush=True)
            while time.time() < t_k:
                time.sleep(0.001)
            # ---- observe
            q_m, qd_m, age = link.state()
            if age > 0.5:
                print(f"[abort] joint feedback stale ({age:.2f} s)", flush=True)
                break
            p_m, R_m = ob.fk(q_m)
            wrist_img, zed_img = read_jpg(f"{SHM}/pnp_wrist.jpg"), read_jpg(f"{SHM}/pnp_zed.jpg")
            if wrist_img is None or zed_img is None:
                print("[abort] camera frame missing", flush=True)
                break
            st = cam_stamp()
            if time.time() - min(st.get("wrist", 0), st.get("zed", 0)) > 0.5:
                print(f"[abort] camera stale {st}", flush=True)
                break
            entry = dict(t=time.time(), p=p_m, R=R_m, suc=state["suction"], wrist_img=wrist_img, zed_img=zed_img,
                         wrist_tok=policy.encode(wrist_img),
                         zed_tok=policy.encode(zed_mask.apply(zed_img) if a.zed_mask else zed_img))
            hist.append(entry)
            # ---- torque contact
            tq = link.torque()
            tq_hist.append(tq)
            if p_m[2] > a.tau_freeze_z and state["suction"] == 0 and len(tq_hist) >= 5:
                tau_base = np.median(np.array(tq_hist)[-5:], axis=0)
            tau = float(abs(tq[1] - tau_base[1]) + W_J3 * abs(tq[2] - tau_base[2])) if a.exec else 0.0
            attached = attached_since is not None and (time.time() - attached_since) >= 1.0
            if tau >= (0.40 if attached else a.tau_abort):
                print(f"[abort] torque contact {tau:.3f} >= abort -> suction off, stop", flush=True)
                break
            # ---- contact-phase splice: scripted align + RL seal controller
            if (splice is not None and GOAL_BASE is not None and state["suction"] == 0
                    and splice_tries[0] < a.seal_retries
                    and np.hypot(p_m[0] - GOAL_BASE[0], p_m[1] - GOAL_BASE[1]) < a.handoff_r
                    and p_m[2] < GOAL_BASE[2] + a.handoff_dz):
                splice_tries[0] += 1
                print(f"[splice] handoff #{splice_tries[0]} at k {k}: tip {np.round(p_m, 3).tolist()} "
                      f"-> align over goal {np.round(GOAL_BASE, 3).tolist()}", flush=True)
                # 1) scripted align 25 mm above the claimed top (the seal policy owns the press)
                align = np.array([GOAL_BASE[0], GOAL_BASE[1], GOAL_BASE[2] + 0.025])
                qg, err, _ = ik(m, dik, ob.sid, align, R_DOWN, q_cmd)
                if err > 0.005 or guard.check_q(qg):
                    print(f"[splice] align IK/guard fail (err {err:.3f}) -> back to policy", flush=True)
                else:
                    scripted_move(link, guard, ob, qg, "splice-align", v_peak_deg=10.0)
                    q_cmd = qg
                    p_cmd, R_cmd = ob.fk(q_cmd)
                    # UNLOADED REFERENCE: settled here, the cup is in free air 25 mm above the
                    # object top. Returning to this exact joint configuration after the grasp
                    # makes the gravity term cancel, so the torque difference is pure payload --
                    # a vision-free "am I actually holding something" that no occlusion can fool.
                    time.sleep(0.35)
                    tau_base_s = torque_median(0.3)
                    probe_ref[0] = (np.asarray(q_cmd, float).copy(), tau_base_s.copy())
                    print(f"[verify] unloaded ref at align pose: J2 {tau_base_s[1]:+.4f} "
                          f"J3 {tau_base_s[2]:+.4f}", flush=True)

                    class _IO:
                        def read(io_self):
                            qm, qdm, _ = link.state()
                            tq_now = link.torque()
                            pm, _Rm = ob.fk(qm)
                            return dict(q=qm, qd=qdm, terr=q_cmd - qm,
                                        tau_rise23=[abs(tq_now[1] - tau_base_s[1]),      # press-positive,
                                                    abs(tq_now[2] - tau_base_s[2])],     # like the sim proxy
                                        tip=pm)

                        def move_dz(io_self, dz):
                            nonlocal q_cmd, p_cmd, R_cmd
                            # hard torque guard: the main loop's abort is paused while the
                            # splice runs, so descent itself must refuse on a torque spike
                            if dz < 0:
                                tq_now = link.torque()
                                rise = (abs(tq_now[1] - tau_base_s[1])
                                        + W_J3 * abs(tq_now[2] - tau_base_s[2]))
                                if rise >= a.splice_tau:
                                    print(f"[splice] torque {rise:.3f} >= splice_tau -> descent refused", flush=True)
                                    return
                            tgt = p_cmd + np.array([0.0, 0.0, dz])
                            tgt[2] = max(tgt[2], a.z_floor)
                            qi, pe, _ = ik(m, dik, ob.sid, tgt, R_DOWN, q_cmd)
                            if pe > 0.004 or guard.check_q(qi):
                                return
                            link.send_segment(q_cmd, qi, time.time(), dt=0.02, seq=k, tag="splice", hold=False)
                            q_cmd = qi
                            p_cmd, R_cmd = ob.fk(q_cmd)

                        def suction(io_self, on):
                            if not on and state["suction"] == 1:
                                # one-way during the press: vacuum must build to attach; the
                                # actor's off-flickers would kill it (cleanup forces off later)
                                print("[splice] suction-off ignored during press", flush=True)
                                return
                            link.suction(bool(on))
                            state["suction"] = int(on)
                            if on:
                                io_self.suc_t = time.time()

                        def lift(io_self, dz):
                            nonlocal q_cmd, p_cmd, R_cmd
                            # real pump needs ~1 s to pull vacuum; the 50 Hz-trained actor
                            # fires lift ~0.6 s after suction (sim seal is faster). Hold the
                            # press until the pump has had a fair dwell.
                            dwell = time.time() - getattr(io_self, "suc_t", time.time())
                            if dwell < 1.0:
                                time.sleep(1.0 - dwell)
                            tgt = p_cmd + np.array([0.0, 0.0, dz])
                            qi, pe, _ = ik(m, dik, ob.sid, tgt, R_DOWN, q_cmd)
                            if pe > 0.005 or guard.check_q(qi):
                                return False
                            scripted_move(link, guard, ob, qi, "splice-lift", v_peak_deg=10.0)
                            q_cmd = qi
                            p_cmd, R_cmd = ob.fk(q_cmd)
                            return True

                    res = splice.run(_IO())
                    print(f"[splice] result: {res}", flush=True)
                    # The actor can report "timeout" while the cup is ACTUALLY holding -- e.g. the
                    # crush guard refused the descent its state machine was waiting on. Suction
                    # being ON is the signal worth testing; the place block lifts and ZED-verifies,
                    # which is the authoritative check. Never discard a possible grasp here.
                    if not a.seal_dry and (res == "lifted" or state["suction"] == 1):
                        if res != "lifted":
                            print(f"[splice] result {res} but suction is ON -> trying lift+verify anyway",
                                  flush=True)
                        attached_since = time.time() - 1.1    # place block fires next iteration
                    else:
                        if state["suction"]:
                            link.suction(False)
                            state["suction"] = 0
                        qh, eh, _ = ik(m, dik, ob.sid,
                                       np.array([GOAL_BASE[0], GOAL_BASE[1], GOAL_BASE[2] + 0.08]),
                                       R_DOWN, q_cmd)
                        if eh < 0.005 and not guard.check_q(qh):
                            scripted_move(link, guard, ob, qh, "splice-recover", v_peak_deg=10.0)
                            q_cmd = qh
                            p_cmd, R_cmd = ob.fk(q_cmd)
                    plan = None                                # CFM plans are stale after the splice
                    pending.clear()
                    hist.clear()
                    k += 1
                    continue
            # ---- plan: start inference every n_exec decisions (or when none in flight / none yet)
            need = plan is None or (k - plan["k_obs"]) >= a.n_exec
            if need and "thread" not in pending:
                obs = [hist[-2] if len(hist) >= 2 else hist[-1], hist[-1]]
                pending["slot"] = {}
                pending["k_obs"] = k
                pending["thread"] = threading.Thread(target=infer_worker, args=(obs, k, pending["slot"]), daemon=True)
                pending["thread"].start()
                pending["t"] = time.time()
            if "thread" in pending and not pending["thread"].is_alive():
                new = pending["slot"].get("plan")
                pending.clear()
                if new is not None:
                    _old = plan
                    plan = new
                    plan["k_adopt"] = k
                    plan["old"] = None if _old is None else dict(p=_old["p_abs"], R=_old["R_abs"], k_obs=_old["k_obs"])
                    if policy.action_mode == "bspline":
                        # ABPolicy adoption-time splice: pin the curve to the CURRENT commanded
                        # state and velocity relative to the plan's obs frame -> smooth boundary
                        from spline_action import ctrl_to_waypoints, ab_refit
                        pT, RT = plan["p_t"], plan["R_t"]
                        dp0 = RT.T @ (p_cmd - pT)
                        dR0 = RT.T @ R_cmd
                        c0 = np.concatenate([dp0, dR0[:, 0], dR0[:, 1], [float(state["suction"])]])
                        v0 = np.zeros(10)
                        if prev_p_cmd is not None:
                            v0[:3] = RT.T @ (p_cmd - prev_p_cmd) / dt     # cmd velocity, obs frame
                        acts = ctrl_to_waypoints(ab_refit(plan["act"], v0=v0, c0=c0)).astype(np.float32)
                        plan["p_abs"] = np.stack([pT + RT @ acts[i, :3] for i in range(T.TA)])
                        plan["R_abs"] = np.stack([RT @ rot6d_to_R(acts[i, 3:9]) for i in range(T.TA)])
                        plan["suc"] = acts[:, 9].copy()
                    seq += 1
                    i0 = k - plan["k_obs"]
                    dp = plan["act"]
                    cv2.imwrite(os.path.join(log_dir, f"obs_{seq:03d}_wrist.jpg"), hist[-1]["wrist_img"])
                    cv2.imwrite(os.path.join(log_dir, f"obs_{seq:03d}_zed.jpg"), hist[-1]["zed_img"])
                    print(f"[plan {seq}] k_obs {plan['k_obs']} latency {1000 * plan['latency']:.0f} ms -> start at action {i0}; "
                          f"dp[0] {np.round(dp[0, :3], 3).tolist()} dp[15] {np.round(dp[15, :3], 3).tolist()} (tcp frame, +z = down) "
                          f"tip_end {np.round(plan['p_abs'][-1], 3).tolist()} suction {np.round(dp[:, 9], 1).tolist()}", flush=True)
            # ---- pick the target for this decision
            holding = "no plan" if plan is None else None
            tgt_p, tgt_R, suc_pred = p_cmd, R_cmd, 0.0
            if plan is not None:
                i = max(0, k - plan["k_obs"] - policy.lat_shift)   # lat-matched chunks start late
                if i >= T.TA:
                    holding = "plan exhausted (inference late?)"
                else:
                    i = max(i, 0)
                    tgt_p, tgt_R, suc_pred = plan["p_abs"][i].copy(), plan["R_abs"][i].copy(), float(plan["suc"][i])
                    jb = k - plan.get("k_adopt", k)               # cross-fade: old chunk's remaining steps -> new chunk
                    if a.blend and plan.get("old") is not None and jb < a.blend:
                        io = k - plan["old"]["k_obs"] - policy.lat_shift
                        if 0 <= io < T.TA:
                            wb = (jb + 1.0) / (a.blend + 1.0)
                            wb = wb * wb * (3.0 - 2.0 * wb)
                            tgt_p = (1.0 - wb) * plan["old"]["p"][io] + wb * tgt_p
                            _U, _, _V = np.linalg.svd((1.0 - wb) * plan["old"]["R"][io] + wb * tgt_R)
                            tgt_R = _U @ _V
                    if a.orient == "fixed":
                        tgt_R = R_DOWN.copy()
                    elif a.orient == "yawlock":
                        # keep the predicted tool axis (the demos roll the 20 deg start tilt out over ~1.6 s)
                        # but lock the yaw to the demos' constant 90 deg: x axis = R_DOWN x projected onto the
                        # plane normal to the predicted z axis (prevents the yaw random walk of run 2)
                        z = tgt_R[:, 2] / (np.linalg.norm(tgt_R[:, 2]) + 1e-9)
                        x = R_DOWN[:, 0] - z * (z @ R_DOWN[:, 0])
                        if np.linalg.norm(x) > 1e-3:
                            x /= np.linalg.norm(x)
                            tgt_R = np.stack([x, np.cross(z, x), z], axis=1)
            # ---- final-approach assist (twin: realsim/eval_policy.py --aim/--aim_down): near the goal the executor puts the
            # lateral target on it; with --aim_down it also brings the cup down and opens the valve. GOAL_BASE z = object top + cup radius.
            if (a.aim > 0 and GOAL_BASE is not None and plan is not None and holding is None and released_at[0] is None
                    and np.hypot(p_cmd[0] - GOAL_BASE[0], p_cmd[1] - GOAL_BASE[1]) < a.aim and p_cmd[2] < GOAL_BASE[2] + a.aim_z):
                tgt_p[:2] = GOAL_BASE[:2]
                aim_n += 1
                if a.aim_down > 0 and k >= aim_block and np.hypot(p_cmd[0] - GOAL_BASE[0], p_cmd[1] - GOAL_BASE[1]) < 0.010:
                    _flo = GOAL_BASE[2] - 0.010
                    tgt_p[2] = min(tgt_p[2], max(_flo, p_cmd[2] - a.aim_down * dt))
                    if p_cmd[2] < GOAL_BASE[2] + 0.042:
                        suc_pred = 1.0
                    if state["suction"] == 1 and attached_since is not None and p_m[2] > _flo + 0.006 and tau < TAU_FIRM:
                        attached_since = time.time()           # no hand-off to the carry before the cup is down or in contact
                    aim_press = aim_press + 1 if p_cmd[2] < _flo + 0.003 else 0
                    if aim_press * dt > a.aim_press:
                        aim_block, aim_press = k + int(5.0 / dt), 0
                        print(f"[aim] {a.aim_press:.0f} s at the bottom without a hand-off -> assist off for 5 s", flush=True)
            # ---- suction (hysteresis on release)
            if plan is not None and not a.no_suction:
                # physical sanity gate: a seal needs contact -- the cup must be near the table (object tops
                # <= 7 cm in training) or the torque guard must see contact; run 1 switched suction on at
                # z 0.20 in mid air after the wrist view made the (large, real) objects look close
                # run 5: posture torque after the frozen baseline (0.083 at z 0.128, cup in mid air) passed the
                # torque branch -> height only
                can_seal = p_m[2] < a.suction_z
                if suc_pred > 0.5 and not can_seal and state["suction"] == 0:
                    if off_count == 0:
                        print(f"[suction] predicted ON at k {k} but tip z {p_m[2]:.3f} >= {a.suction_z} and no contact -> ignored", flush=True)
                    off_count = 1
                elif suc_pred > 0.5:
                    off_count = 0
                    if state["suction"] == 0:
                        print(f"[suction] ON at k {k} (pred {suc_pred:.2f}) tip {np.round(p_m, 3).tolist()} tau {tau:.3f}", flush=True)
                        link.suction(True)
                        state["suction"] = 1
                        attached_since = time.time()
                else:
                    off_count += 1
                    if state["suction"] == 1 and off_count >= a.suction_off_n:
                        print(f"[suction] OFF at k {k} tip {np.round(p_m, 3).tolist()}", flush=True)
                        link.suction(False)
                        state["suction"] = 0
                        attached_since = None
                        if a.stop_after_place and released_at[0] is None:
                            # only a release NEAR the place target (or clearly carried aloft) counts
                            # as a place; a failed-grasp release at the pick site must fall through
                            # so the policy can retry (bspline_run1 was cut short by this).
                            near_place = (a.place_xy is not None
                                          and np.hypot(p_m[0] - a.place_xy[0],
                                                       p_m[1] - a.place_xy[1]) < 0.08)
                            if near_place or (a.place_xy is None and p_m[2] > 0.15):
                                released_at[0] = k    # placed -> wind the episode down
                            else:
                                print(f"[suction] release at pick site (not a place) -> retry", flush=True)
            # ---- runtime place: policy picks, scripted carry puts it in the CURRENT box
            if (a.place_xy is not None and state["suction"] == 1
                    and attached_since is not None
                    and time.time() - attached_since > 1.0):
                px, py = a.place_xy
                print(f"[place] attached -> scripted carry to ({px:.3f},{py:.3f},{a.place_z:.2f})", flush=True)
                ok = True
                pick_xy = (p_m[0], p_m[1])     # where the grasp happened
                # Legs: a PROBE leg first when an unloaded reference exists. It returns to the
                # exact joint configuration measured before the press, which lifts the object
                # ~25 mm clear of the table and lets the payload torque be read against a
                # same-pose baseline. Verifying HERE means an empty cup is caught in ~2 s
                # instead of carrying air all the way to the box.
                legs = []
                legs.append(("lift", [p_m[0], p_m[1], max(a.place_z, p_m[2] + 0.10)], None))
                # INSPECT leg: step laterally off the pick site before judging it. The lift
                # ends directly ABOVE the pick xy, where the arm's own body makes the ZED
                # unusable -- counted, it reads as "object still there"; filtered out by
                # height, its occlusion shadow reads as "site empty". Neither is a
                # measurement. 35 % of the way to the box is ~10 cm clear and costs nothing,
                # because it is motion the carry had to make anyway.
                _ix, _iy = inspect_xy(pick_xy, (px, py))
                legs.append(("inspect", [_ix, _iy, a.place_z], None))
                legs.append(("carry", [px, py, a.place_z], None))
                for lbl, tgt_xyz, tgt_q in legs:
                    if tgt_q is not None:
                        qg = np.asarray(tgt_q, float).copy()
                        if guard.check_q(qg):
                            print(f"[place] probe pose fails the guard -> skipping the load probe", flush=True)
                            continue
                    else:
                        qg, err, _ = ik(m, dik, ob.sid, tgt_xyz, R_cmd, q_cmd)
                        if err > 0.01:
                            print(f"[place] IK fail ({lbl}, err {err:.3f}) -> leaving it to the policy", flush=True)
                            ok = False; break
                    scripted_move(link, guard, ob, qg, f"place-{lbl}", v_peak_deg=a.place_v)
                    q_cmd = qg; p_cmd, R_cmd = ob.fk(q_cmd)

                    if lbl == "inspect":
                        # Poll the map instead of trusting one frame: at 3 Hz a single read can
                        # land mid-publish or on a dropout. Take the first DEFINITE verdict.
                        st, info = "unknown", "no reads"
                        for _try in range(6):
                            time.sleep(0.35)
                            st, info = zed_site_state(pick_xy, a.table_z)
                            if st != "unknown":
                                break
                        # record the payload torque for the log only -- calibration (see
                        # /tmp/calib_load.json) measured same-pose drift of 0.048 pin against
                        # just 0.021 pin for a 0.3 kg object, so this number CANNOT gate
                        # anything on this hardware. It is kept because it is free to read.
                        if probe_ref[0] is not None:
                            ref = probe_ref[0][1]; tq_now = torque_median(0.2)
                            load = float(abs(tq_now[1] - ref[1]) + W_J3 * abs(tq_now[2] - ref[2]))
                            print(f"[verify] (payload torque {load:.4f} pin -- below the "
                                  f"0.048 noise floor, informational only)", flush=True)
                        print(f"[verify] pick site {st}: {info}", flush=True)
                        # LIFTED vs DRAGGED: a cup that shoves the object sideways also
                        # empties the site. Only a neighbourhood that stayed empty proves
                        # the object actually left the table.
                        dragged = False
                        if st == "cleared" and nbr_ref[0] is not None and _T_ZED is not None:
                            now = _pv.occupancy_cells(pick_xy, a.table_z, _T_ZED, shm=SHM)
                            dragged, n_new, need = _pv.drag_verdict(nbr_ref[0][0], now, nbr_ref[0][1])
                            cov = _pv.table_coverage(pick_xy, a.table_z, _T_ZED, shm=SHM)
                            cov_ref = max(nbr_ref[0][2], 1)
                            cov_frac = cov / cov_ref
                            print(f"[verify] neighbourhood: {n_new} newly-occupied cells "
                                  f"(drag threshold {need}), table coverage "
                                  f"{100 * cov_frac:.0f}% of reference", flush=True)
                            if not dragged and cov_frac < 0.6:
                                # The arm is shadowing the neighbourhood, so "nothing appeared"
                                # is not evidence that nothing is there.
                                st = "unknown"
                                print(f"[verify] drag test INCONCLUSIVE (neighbourhood "
                                      f"{100 * (1 - cov_frac):.0f}% occluded)", flush=True)
                        if dragged:
                            print(f"[place] VERIFY FAILED: site is clear but the object "
                                  f"re-appeared nearby -> dragged, not lifted; release and retry",
                                  flush=True)
                            link.suction(False); state["suction"] = 0; attached_since = None
                            if GOAL_BASE is not None:
                                qh, eh, _ = ik(m, dik, ob.sid,
                                               np.array([GOAL_BASE[0], GOAL_BASE[1], GOAL_BASE[2] + 0.08]),
                                               R_DOWN, q_cmd)
                                if eh < 0.005 and not guard.check_q(qh):
                                    scripted_move(link, guard, ob, qh, "verify-recover", v_peak_deg=12.0)
                                    q_cmd = qh; p_cmd, R_cmd = ob.fk(q_cmd)
                            plan = None; pending.clear(); hist.clear()
                            ok = False; break
                        if st == "unknown":
                            # Honest fallback: proceed with the place, but do NOT claim a
                            # verified pick. Guessing "cleared" here is what carried air to
                            # the box and reported success.
                            print(f"[verify] UNVERIFIED (site not visible) -> placing anyway, "
                                  f"outcome decided by the final check", flush=True)
                        elif st == "occupied":
                            print(f"[place] VERIFY FAILED: object still at "
                                  f"({pick_xy[0]:.3f},{pick_xy[1]:.3f}) -> empty cup, "
                                  f"release and retry", flush=True)
                            link.suction(False); state["suction"] = 0; attached_since = None
                            if GOAL_BASE is not None:
                                qh, eh, _ = ik(m, dik, ob.sid,
                                               np.array([GOAL_BASE[0], GOAL_BASE[1], GOAL_BASE[2] + 0.08]),
                                               R_DOWN, q_cmd)
                                if eh < 0.005 and not guard.check_q(qh):
                                    scripted_move(link, guard, ob, qh, "verify-recover", v_peak_deg=12.0)
                                    q_cmd = qh; p_cmd, R_cmd = ob.fk(q_cmd)
                            plan = None; pending.clear(); hist.clear()
                            ok = False; break
                        else:
                            print(f"[verify] pick CONFIRMED (site cleared with a clear view) "
                                  f"-> carrying to the box", flush=True)
                if ok:
                    link.suction(False); state["suction"] = 0; attached_since = None
                    print(f"[place] released at ({px:.3f},{py:.3f})", flush=True)
                    # FINAL outcome: retreat to home first. "The object left the pick site" is
                    # not success -- it also happens when the cup shoves the object off. Success
                    # is the object ARRIVING at the box, and judging that needs the arm out of
                    # the way of the drop site, which only a retreat gives.
                    qh_m = scripted_move(link, guard, ob, START_Q, "place-retreat",
                                         v_peak_deg=a.script_vmax, min_T=1.0)
                    if qh_m is not None:
                        q_cmd = qh_m; p_cmd, R_cmd = ob.fk(q_cmd)
                    time.sleep(1.2)
                    st, info = zed_site_state(pick_xy, a.table_z)
                    arrived, n_new = None, 0
                    if place_ref[0] is not None and _T_ZED is not None:
                        now = _pv.occupancy_cells((px, py), a.table_z, _T_ZED, shm=SHM, R=a.place_r)
                        n_new = len(now - place_ref[0])
                        arrived = n_new >= 3
                    print(f"[place] OUTCOME: pick site {st} ({info}); "
                          f"place site {'+' if arrived else ''}{n_new} new cells"
                          + ("  <-- OBJECT DELIVERED" if arrived
                             else "  <-- NOTHING ARRIVED at the box" if arrived is False
                             else ""), flush=True)
                    released_at[0] = k
                    if a.stop_after_place: break
            # ---- guards on the target
            clips = []
            step = tgt_p - p_cmd
            # split lateral / vertical so a tight lateral cap cannot throttle descent
            lat = np.array([step[0], step[1], 0.0]); vert = np.array([0.0, 0.0, step[2]])
            nl = float(np.linalg.norm(lat)); nv = abs(float(vert[2]))
            zcap = a.step_max_z if a.step_max_z > 0 else a.step_max
            if nl > a.step_max:
                lat *= a.step_max / nl
                clips.append(f"lat {100 * nl:.1f} cm")
            if nv > zcap:
                vert *= zcap / nv
                clips.append(f"vert {100 * nv:.1f} cm")
            tgt_p = p_cmd + lat + vert
            if tgt_p[2] < a.z_floor:
                tgt_p[2] = a.z_floor
                clips.append("z floor")
            if tgt_p[0] < a.ws_xmin:                       # workspace limit toward the back wall
                tgt_p[0] = min(a.ws_xmin, p_cmd[0] + a.step_max) if p_cmd[0] < a.ws_xmin else a.ws_xmin
                clips.append("x min")
            if tilt_deg(tgt_R) > a.max_tilt:
                tgt_R = R_cmd
                clips.append("tilt")
            if tau >= TAU_HARD and not attached and tgt_p[2] < p_m[2] - 0.002:
                tgt_p[2] = max(p_m[2], a.z_floor)          # hard contact: no further descent, lift/lateral allowed
                clips.append(f"hard contact {tau:.2f}")
            lead = float(np.degrees(np.abs(q_cmd - q_m)).max())
            # graceful degradation: scale the step down as lead grows instead of freezing
            # outright (a binary hold lets the chunk advance without motion, then releases
            # in one lurch -- measured 25 deg on the DiT run). Full speed below lead_soft,
            # linearly to zero at lead_max.
            # LEAD_MAX_DEG=3 is NOT arbitrary: K0*3deg saturates the Pi law at vmax, so a larger
            # lead buys no speed and only lets the reference drift ahead (-> catch-up lurch).
            # Ramp inside the ceiling: full step to 60%, linearly to zero at the cap.
            lead_soft = 0.6 * a.lead_max
            lead_scale = 1.0
            if lead > lead_soft:
                lead_scale = max(0.0, (a.lead_max - lead) / max(a.lead_max - lead_soft, 1e-6))
            if lead_scale <= 0.0:
                holding = f"lead {lead:.1f} deg"
            q_tgt = q_cmd.copy()
            if holding is None:
                q_ik, pe, re = ik(m, dik, ob.sid, tgt_p, tgt_R, q_cmd)
                if pe > 0.003 or re > 0.05:
                    holding = f"ik residual {1000 * pe:.1f} mm {math.degrees(re):.1f} deg"
                else:
                    dq = q_ik - q_cmd
                    lim = np.full(6, a.dq_max)
                    if a.dq_mid > 0 and p_cmd[2] > a.dq_mid_z:     # joints 2-4 faster, only well above the table
                        lim[1:4] = a.dq_mid
                    sc = float((np.degrees(np.abs(dq)) / lim).max())   # uniform scaling keeps the direction
                    if sc > 1.0:
                        mx = float(np.degrees(np.abs(dq)).max())
                        dq /= sc
                        clips.append(f"dq {mx:.1f} deg")
                    if lead_scale < 1.0:
                        dq *= lead_scale               # ease off instead of stalling
                        clips.append(f"lead-scale {lead_scale:.2f}")
                    q_tgt = q_cmd + dq
                    if FILT is not None:
                        q_tgt, finfo = FILT(q_cmd, q_tgt, dt)
                        if finfo["why"]:
                            clips.append(f"filt {finfo['why']} a{finfo['alpha']:.2f} clr {1000 * finfo['clr']:.0f}mm")
                    v = guard.check_q(q_tgt)
                    if v:
                        holding = "guard: " + "; ".join(v)
                        q_tgt = q_cmd.copy()
            if holding is not None:
                n_hold += 1
                if FILT is not None:
                    FILT.reset()
            # ---- send
            link.send_segment(q_cmd, q_tgt, t_k, dt=dt, seq=k, tag=f"zed:{k}", hold=holding is not None)
            q_cmd = q_tgt
            prev_p_cmd = p_cmd.copy()
            p_cmd, R_cmd = ob.fk(q_cmd)
            rec = dict(k=k, t=time.time(), q_meas=np.round(q_m, 5).tolist(), q_cmd=np.round(q_cmd, 5).tolist(),
                       tip=np.round(p_m, 4).tolist(), tip_cmd=np.round(p_cmd, 4).tolist(), tgt=np.round(tgt_p, 4).tolist(),
                       tau=round(tau, 4), suction=state["suction"], suc_pred=round(suc_pred, 3), hold=holding,
                       clips=clips, plan=None if plan is None else plan["k_obs"], lead=round(lead, 2))
            log_f.write(json.dumps(rec) + "\n")
            if time.time() - last_print > 1.0 or holding or clips:
                print(f"[k {k:3d}] tip {np.round(p_m, 3).tolist()} -> {np.round(tgt_p, 3).tolist()} tau {tau:.3f} "
                      f"suc {state['suction']} lead {lead:.1f}{'  HOLD: ' + holding if holding else ''}"
                      f"{'  clip: ' + ', '.join(clips) if clips else ''}", flush=True)
                last_print = time.time()
            k += 1
    finally:
        log_f.close()
    if a.aim > 0:
        print(f"[aim] lateral assist active in {aim_n} decisions", flush=True)
    print(f"[ctrl] episode end at k {k}: holds {n_hold}, suction {state['suction']}, log {log_dir}", flush=True)
    if state["suction"]:
        print("[ctrl] releasing", flush=True)
        link.suction(False)
        time.sleep(1.0)
    # A path sent within a few seconds of the policy stream's end is lost on the Pi (runs 1 and 2: acked but
    # no motion; a resend ~10 s later moved). robot_hal ends a stream 0.5 s after its reference runs out and
    # then clears the reference lists, so: wait for that, send the leg through a fresh link, verify that the
    # joints actually start moving within 2 s, and resend otherwise.
    link.close()

    def fresh_link():
        lk = PiLink(a.pi, a.exec, ref_mode="spline")
        t_w = time.time()
        while not lk.ok() and time.time() - t_w < 6:
            time.sleep(0.1)
        return lk

    def verified_leg(lk, q_to, label, v_peak, tries=3):
        if not a.exec:
            return lk, scripted_move(lk, guard, ob, q_to, label, v_peak_deg=v_peak, min_T=1.0)
        for attempt in range(tries):
            q_before, _, _ = lk.state()
            if float(np.degrees(np.abs(np.asarray(q_before) - q_to)).max()) < 1.0:
                return lk, np.asarray(q_before)
            # send the min-jerk path (scripted_move waits the full duration + settle; we watch the start)
            done = {}

            def _run():
                done["q"] = scripted_move(lk, guard, ob, q_to, label, v_peak_deg=v_peak, min_T=1.0)
            th = threading.Thread(target=_run, daemon=True)
            th.start()
            t_w = time.time()
            moved = False
            while th.is_alive() and time.time() - t_w < 3.0:
                q_now, _, _ = lk.state()
                if float(np.degrees(np.abs(np.asarray(q_now) - q_before)).max()) > 0.3:
                    moved = True
                    break
                time.sleep(0.05)
            if moved:
                th.join()
                return lk, done.get("q")
            print(f"[warn] {label}: no motion 3 s after the path was sent (attempt {attempt + 1}) -> reconnect and resend", flush=True)
            th.join(timeout=15.0)
            lk.close()
            time.sleep(2.0)
            lk = fresh_link()
        return lk, None

    time.sleep(1.5)
    link = fresh_link()
    q_m, _, _ = link.state()
    p_m, R_m = ob.fk(q_m)
    if p_m[2] < 0.25:
        q_up, pe, _ = ik(m, dik, ob.sid, [p_m[0], p_m[1], 0.25], R_m, q_m)
        if pe < 0.005:
            link, _ = verified_leg(link, q_up, "retract up", min(a.script_vmax, 12.0))
        else:
            print(f"[warn] retract IK residual {1000 * pe:.1f} mm -> homing directly", flush=True)
    link, q_end = verified_leg(link, START_Q, "home (START_Q)", a.script_vmax)
    q_m, _, _ = link.state()
    if q_m is not None:
        print(f"[home] final deviation from START_Q {np.round(np.degrees(np.asarray(q_m) - START_Q), 2).tolist()} deg", flush=True)
    link.close()


if __name__ == "__main__":
    main()

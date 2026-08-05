"""distill :: DAgger distillation of the state-based PPO teacher into a
two-RGBD-camera student policy.

Student obs:  wrist D405 + fixed D435, RGB-D each (4ch, 96x96), proprio
(q, qd, suction), numeric goal (place target x,y,h). NO object state --
the student must find the object in pixels.
Teacher:      rl/ppo.py AC on the privileged 37-D state (frozen).

Loop: iter 0 collects under the TEACHER's actions (BC seed); later iters
collect under the STUDENT (DAgger -- states include the student's own
mistakes), always labeled with the teacher's action. Visual DR per
episode per world: object color, light position, camera pose jitter.

Physics: PickEnv (mjwarp, GPU, nworld small). Rendering: one CPU MuJoCo
model per world slot, re-randomized at that world's reset.

    python rl/distill.py --teacher ~/pnp_rl/ppo8_wrench/final.pt \
        --iters 4 --steps_per_iter 60000
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warp as wp                                          # noqa: E402

IMG = 96
ACT_DIM = 7
PROP_DIM = 13          # q6 + qd6 + suction1
GOAL_DIM = 3           # place target x, y, h


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        def tower():
            return nn.Sequential(
                nn.Conv2d(4, 32, 5, 2, 2), nn.SiLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(128, 128, 3, 2, 1), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cam1 = tower()
        self.cam2 = tower()
        self.head = nn.Sequential(
            nn.Linear(128 * 2 + PROP_DIM + GOAL_DIM, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, ACT_DIM))

    def forward(self, im1, im2, prop, goal):
        f = torch.cat([self.cam1(im1), self.cam2(im2), prop, goal], -1)
        return self.head(f)                    # pre-tanh action


_W = {}          # per-process render state (set by _worker_init)


def _worker_init(xml):
    m = mujoco.MjModel.from_xml_path(xml)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0
    _W.update(
        m=m, d=mujoco.MjData(m),
        ren=mujoco.Renderer(m, height=IMG, width=IMG), vopt=vopt,
        gid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object0"),
        cid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_d435"))


def _worker_render(job):
    """job = (qpos, mocap, params); params = dict(rgba, light, cam)."""
    qpos, mocap, p = job
    m, d, ren = _W["m"], _W["d"], _W["ren"]
    m.geom_rgba[_W["gid"], :3] = p["rgba"]
    if m.nlight > 0:
        m.light_pos[0] = p["light"]
    m.cam_pos[_W["cid"]] = p["cam"]
    d.qpos[:len(qpos)] = qpos
    if mocap is not None and m.nmocap > 0:
        d.mocap_pos[:] = mocap
    mujoco.mj_forward(m, d)
    out = []
    for cam in ("wrist_d405", "fixed_d435"):
        ren.update_scene(d, camera=cam, scene_option=_W["vopt"])
        rgb = ren.render()
        ren.enable_depth_rendering()
        ren.update_scene(d, camera=cam, scene_option=_W["vopt"])
        dep = ren.render()
        ren.disable_depth_rendering()
        rgbd = np.concatenate([rgb.astype(np.float32) / 255.0,
                               np.clip(dep, 0, 2.0)[..., None] / 2.0], -1)
        out.append(rgbd.transpose(2, 0, 1))              # (4, H, W)
    return out


class RenderFarm:
    """Multiprocess CPU renderer; main process owns the DR RNG and ships
    explicit per-world params with each request (workers are stateless)."""
    def __init__(self, xml, nworld, seed=0, workers=32):
        import multiprocessing as mp
        self.rng = np.random.default_rng(seed)
        m0 = mujoco.MjModel.from_xml_path(xml)
        cid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_d435")
        self._cam0 = m0.cam_pos[cid].copy()
        self.params = [None] * nworld
        for i in range(nworld):
            self.randomize(i)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(min(workers, nworld), _worker_init, (xml,))

    def randomize(self, i):
        self.params[i] = dict(
            rgba=self.rng.uniform(0.15, 0.95, 3),
            light=np.array([self.rng.uniform(-0.6, 1.1),
                            self.rng.uniform(-0.8, 1.0),
                            self.rng.uniform(0.9, 1.7)]),
            cam=self._cam0 + self.rng.normal(0, 0.01, 3))

    def render_batch(self, qp, mocap):
        """qp (N, nq); mocap (N, nmocap, 3) or None -> two lists of (4,H,W)."""
        jobs = [(qp[i], mocap[i] if mocap is not None else None, self.params[i])
                for i in range(len(qp))]
        res = self.pool.map(_worker_render, jobs)
        return [r[0] for r in res], [r[1] for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.path.expanduser("~/pnp_rl/ppo8_wrench/final.pt"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med_ped.xml"))
    ap.add_argument("--nworld", type=int, default=16)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--steps_per_iter", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lift_req", type=float, default=0.35)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/distill1"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    torch.manual_seed(0)
    wp.init()
    from env_warp import PickEnv
    from ppo import AC
    env = PickEnv(nworld=a.nworld, mode="pnp", xml=a.scene, dr=True,
                  lift_req=a.lift_req)
    teacher = AC().to(dev)
    ck = torch.load(a.teacher, map_location=dev, weights_only=False)
    sd = ck["ac"]
    own = teacher.state_dict()
    for k in list(sd.keys()):                  # obs-dim growth: zero-pad
        if k in own and own[k].shape != sd[k].shape:
            pad = torch.zeros_like(own[k])
            sl = tuple(slice(0, s) for s in sd[k].shape)
            pad[sl] = sd[k]
            sd[k] = pad
    teacher.load_state_dict(sd)
    teacher.eval()
    student = Student().to(dev)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    farm = RenderFarm(a.scene, a.nworld)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    # dataset buffers (grow across iters -- DAgger aggregation)
    D_im1, D_im2, D_prop, D_goal, D_act = [], [], [], [], []

    def proprio_goal():
        prop = torch.cat([env.qpos[:, :6], env.qvel[:, :6],
                          env.sealed.float()[:, None]], -1)
        goal = torch.cat([env.place_target,
                          env.target_h[:, None]], -1)
        return prop, goal

    def student_act(im1, im2, prop, goal):
        with torch.no_grad():
            return torch.tanh(student(im1, im2, prop, goal))

    for it in range(a.iters):
        beta_teacher = 1.0 if it == 0 else 0.0
        collected = 0
        t0 = time.time()
        obs = env.observe()
        # track per-world mocap for the render (pedestal)
        while collected < a.steps_per_iter:
            ims1, ims2 = [], []
            mocap = None
            if env.mocap_pos is not None:
                mocap = env.mocap_pos.cpu().numpy()
            qp = env.qpos.cpu().numpy()
            for i in range(a.nworld):
                r1, r2 = farm.render(i, qp[i], mocap[i] if mocap is not None else None)
                ims1.append(r1)
                ims2.append(r2)
            im1 = torch.tensor(np.stack(ims1), device=dev)
            im2 = torch.tensor(np.stack(ims2), device=dev)
            prop, goal = proprio_goal()
            with torch.no_grad():
                t_act = torch.tanh(teacher.pi(obs))
            if beta_teacher >= 1.0:
                act = t_act
            else:
                act = student_act(im1, im2, prop, goal)
            D_im1.append((im1.cpu().numpy() * 255).astype(np.uint8))
            D_im2.append((im2.cpu().numpy() * 255).astype(np.uint8))
            D_prop.append(prop.cpu().numpy())
            D_goal.append(goal.cpu().numpy())
            D_act.append(t_act.cpu().numpy())
            obs, r, done, info = env.step(act)
            if done.any():
                for i in torch.nonzero(done).squeeze(-1).tolist():
                    farm.randomize(i)
            collected += a.nworld
        sps = a.steps_per_iter / (time.time() - t0)
        print(f"[dagger] iter {it}: collected {collected} frames "
              f"({sps:.0f} fps), dataset {sum(x.shape[0] for x in D_prop)}",
              flush=True)

        # ---- train ----
        X1 = np.concatenate(D_im1)
        X2 = np.concatenate(D_im2)
        P = torch.tensor(np.concatenate(D_prop), device=dev)
        G = torch.tensor(np.concatenate(D_goal), device=dev)
        Y = torch.tensor(np.concatenate(D_act), device=dev)
        n = len(P)
        for ep in range(a.epochs):
            perm = np.random.permutation(n)
            tot = 0.0
            nb = 0
            for k in range(0, n, a.batch):
                mb = perm[k:k + a.batch]
                b1 = torch.tensor(X1[mb], device=dev).float() / 255.0
                b2 = torch.tensor(X2[mb], device=dev).float() / 255.0
                # light photometric aug
                b1 = (b1 + torch.randn_like(b1[:, :1, :1, :1]) * 0.03).clamp(0, 1)
                b2 = (b2 + torch.randn_like(b2[:, :1, :1, :1]) * 0.03).clamp(0, 1)
                pred = torch.tanh(student(b1, b2, P[mb], G[mb]))
                loss = Fn.mse_loss(pred, Y[mb])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
            print(f"[dagger] iter {it} epoch {ep}: loss {tot/nb:.5f}", flush=True)

        # ---- student-driven eval ----
        succ, n_ep = 0, 0
        obs = env.observe()
        done_ct = 0
        while done_ct < 64:
            mocap = env.mocap_pos.cpu().numpy() if env.mocap_pos is not None else None
            qp = env.qpos.cpu().numpy()
            ims1 = []; ims2 = []
            for i in range(a.nworld):
                r1, r2 = farm.render(i, qp[i], mocap[i] if mocap is not None else None)
                ims1.append(r1); ims2.append(r2)
            im1 = torch.tensor(np.stack(ims1), device=dev)
            im2 = torch.tensor(np.stack(ims2), device=dev)
            prop, goal = proprio_goal()
            act = student_act(im1, im2, prop, goal)
            obs, r, done, info = env.step(act)
            if done.any():
                idx = torch.nonzero(done).squeeze(-1)
                succ += int(info["placed"][idx].sum())
                done_ct += idx.numel()
                for i in idx.tolist():
                    farm.randomize(i)
        rec = dict(iter=it, dataset=int(n), succ=succ / max(1, done_ct),
                   eval_eps=done_ct)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[dagger] iter {it}: STUDENT success {rec['succ']:.1%} "
              f"({done_ct} eps)", flush=True)
        torch.save(dict(student=student.state_dict(), iter=it),
                   os.path.join(a.out, f"student_it{it}.pt"))
    torch.save(dict(student=student.state_dict()),
               os.path.join(a.out, "student_final.pt"))
    print("[dagger] done", flush=True)


if __name__ == "__main__":
    main()

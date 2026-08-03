"""train_lit :: DP / CFM / ACT on the litdata streaming dataset (Lightning AI).

Reads the litdata dir produced by convert_litdata.py plus a goals.json
sidecar ({scene: {pick: {pos, quat}}} = executed grasp pose per episode).

Conditioning (all three heads, cross-attention):
  context tokens = per-image CLIP ViT-B/16 tokens (CLS + 196 patches, 4
  images: 2 cams x 2 obs steps) + 1 proprio token + 1 GOAL token (the pick's
  grasp pose expressed relative to the current tcp pose, pos3 + rot6d).
  DP / CFM denoiser = DiT: 16 action tokens, AdaLN-Zero timestep modulation,
  [self-attn -> cross-attn(context) -> MLP] x depth.
  ACT = CVAE transformer whose decoder cross-attends the same context.

Action = 16 B-spline control points (+ per-step suction), q-space by default
(--action eef for the UMI-style relative-EEF chunk). Min-max normalized.

UMI settings kept: obs horizon 2, DDIM 50 train / 16 inference steps, head
lr 3e-4, vision fine-tuned at 3e-5 (10x lower), EMA power 0.75.

Stopping: --steps 500k, OR early-stop once eval action-MSE (sampled chunk
vs ground truth, normalized space) drops below 0.005 (checked every
--val_every, only after 25k steps). Val split = scenes >= 590.

Deps: torch, litdata, open_clip_torch, diffusers, opencv-python-headless.
Run:  python train_lit.py --model dp --action q --data /path/pnp_litdata
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from litdata import StreamingDataset

TO = 2
TA = 16
NCAM = 2
NIMG = NCAM * TO
PDIM = 6 + 6 + 1 + 1          # q, qd, suction, rangefinder per obs step
GDIM = 9                      # goal: rel pos3 + rot6d
ADIM = 7                      # 6 ctrl-pt dims + suction
VAL_SCENE_MIN = 590
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]],
        dtype=np.float32)


def decode(jpg):
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return ((img.astype(np.float32) / 255.0) - CLIP_MEAN) / CLIP_STD


class Transform:
    """sample -> (imgs [4,3,224,224], prop [2,14], goal [9], act [16,7])."""
    def __init__(self, action, goals):
        self.action = action
        self.goals = goals

    def __call__(self, s):
        imgs = np.stack([decode(s["d405_jpg"][0]), decode(s["d405_jpg"][1]),
                         decode(s["d435_jpg"][0]), decode(s["d435_jpg"][1])])
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        prop = np.concatenate([s["q"], s["qd"],
                               s["suction"][:, None].astype(np.float32),
                               s["rangefinder"][:, None]], -1)
        g = self.goals[str(int(s["scene"]))][str(int(s["pick"]))]
        R_t = quat_to_R(s["eef_quat"][1])
        R_g = quat_to_R(np.asarray(g["quat"], np.float32))
        dR = R_t.T @ R_g
        goal = np.concatenate([
            R_t.T @ (np.asarray(g["pos"], np.float32) - s["eef_pos"][1]),
            dR[:, 0], dR[:, 1]]).astype(np.float32)
        ctrl = s["ctrl_pts"] if self.action == "q" else s["ctrl_pts_eef"]
        act = np.concatenate(
            [ctrl, s["suction_cmd"][:, None].astype(np.float32)], -1)
        return imgs, torch.from_numpy(prop), torch.from_numpy(goal), \
            torch.from_numpy(act)


class IndexedDS(torch.utils.data.Dataset):
    """Random-access view over a local litdata dir restricted to indices.
    Each worker lazily opens its own StreamingDataset handle."""
    def __init__(self, data_dir, indices, transform):
        self.data_dir = data_dir
        self.indices = indices
        self.transform = transform
        self._ds = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._ds is None:
            self._ds = StreamingDataset(self.data_dir)
        return self.transform(self._ds[int(self.indices[i])])


def split_indices(data_dir, cache):
    if os.path.exists(cache):
        d = json.load(open(cache))
        return d["train"], d["val"]
    ds = StreamingDataset(data_dir)
    tr, va = [], []
    for i in range(len(ds)):
        (va if int(ds[i]["scene"]) >= VAL_SCENE_MIN else tr).append(i)
    json.dump({"train": tr, "val": va}, open(cache, "w"))
    return tr, va


# ----------------------------------------------------------------- encoder
class VisionEncoder(nn.Module):
    """CLIP ViT-B/16; emits per-image token sequences (CLS + patches)."""
    def __init__(self, d_model):
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-quickgelu", pretrained="openai")
        self.visual = model.visual
        self.visual.output_tokens = True
        try:                       # activation memory: 4 imgs/sample x batch
            self.visual.set_grad_checkpointing(True)
        except Exception:
            pass
        self.proj_cls = nn.Linear(512, d_model)
        self.proj_patch = nn.Linear(768, d_model)

    def forward(self, imgs):                        # [B,4,3,224,224]
        B = imgs.shape[0]
        pooled, tokens = self.visual(imgs.flatten(0, 1))
        cls = self.proj_cls(pooled).view(B, NIMG, 1, -1)
        pat = self.proj_patch(tokens).view(B, NIMG, tokens.shape[1], -1)
        return torch.cat([cls, pat], 2).flatten(1, 2)  # [B, 4*(1+196), d]


class Context(nn.Module):
    """image tokens + proprio token + goal token -> context sequence."""
    def __init__(self, d_model):
        super().__init__()
        self.venc = VisionEncoder(d_model)
        self.propp = nn.Linear(PDIM * TO, d_model)
        self.goalp = nn.Linear(GDIM, d_model)

    def forward(self, imgs, prop, goal):
        return torch.cat([self.venc(imgs),
                          self.propp(prop.flatten(1))[:, None],
                          self.goalp(goal)[:, None]], 1)


# ----------------------------------------------------------------- DiT
class SinEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)


class DiTBlock(nn.Module):
    """AdaLN-Zero self-attn + cross-attn(context) + MLP."""
    def __init__(self, d, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, ctx, temb):
        s1, b1, g1, s2, b2, g2 = self.ada(temb)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.sa(h, h, h, need_weights=False)[0]
        x = x + self.ca(self.n2(x), ctx, ctx, need_weights=False)[0]
        h = self.n3(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class DiT(nn.Module):
    def __init__(self, d=512, depth=6):
        super().__init__()
        self.inp = nn.Linear(ADIM, d)
        self.pos = nn.Parameter(torch.randn(TA, d) * 0.02)
        self.temb = nn.Sequential(SinEmb(128), nn.Linear(128, d), nn.Mish(),
                                  nn.Linear(d, d))
        self.blocks = nn.ModuleList([DiTBlock(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.ada = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.out = nn.Linear(d, ADIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, ctx):                  # x [B,TA,ADIM]
        temb = self.temb(t)
        h = self.inp(x) + self.pos
        for blk in self.blocks:
            h = blk(h, ctx, temb)
        s, b = self.ada(temb)[:, None].chunk(2, -1)
        return self.out(self.norm(h) * (1 + s) + b)


# ----------------------------------------------------------------- ACT
class ACT(nn.Module):
    """CVAE transformer; decoder cross-attends [context tokens + z]."""
    def __init__(self, d=512, z=32):
        super().__init__()
        self.zdim = z
        enc_layer = nn.TransformerEncoderLayer(d, 8, 2048, 0.1, batch_first=True)
        self.cvae_enc = nn.TransformerEncoder(enc_layer, 4)
        self.act_in = nn.Linear(ADIM, d)
        self.z_head = nn.Linear(d, z * 2)
        self.zp = nn.Linear(z, d)
        self.mem_enc = nn.TransformerEncoder(enc_layer, 4)
        dec_layer = nn.TransformerDecoderLayer(d, 8, 2048, 0.1, batch_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, 7)
        self.query = nn.Parameter(torch.randn(TA, d) * 0.02)
        self.out = nn.Linear(d, ADIM)
        self.cls = nn.Parameter(torch.randn(1, d) * 0.02)

    def forward(self, ctx, act=None):
        B = ctx.shape[0]
        if act is not None:
            seq = torch.cat([self.cls.expand(B, 1, -1), self.act_in(act)], 1)
            mu, logv = self.z_head(self.cvae_enc(seq)[:, 0]).chunk(2, -1)
            zs = mu + torch.randn_like(mu) * (0.5 * logv).exp()
        else:
            mu = logv = None
            zs = torch.zeros(B, self.zdim, device=ctx.device)
        mem = self.mem_enc(torch.cat([ctx, self.zp(zs)[:, None]], 1))
        h = self.dec(self.query.expand(B, -1, -1), mem)
        return self.out(h), mu, logv


# ----------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dp", "cfm", "act"])
    ap.add_argument("--action", default="q", choices=["q", "eef"])
    ap.add_argument("--data", default=os.path.expanduser("~/pnp_litdata"))
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_runs"))
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--mse_stop", type=float, default=0.005)
    ap.add_argument("--min_steps", type=int, default=25_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val_every", type=int, default=5_000)
    ap.add_argument("--val_batches", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=25_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vision_lr", type=float, default=3e-5)
    ap.add_argument("--dit_depth", type=int, default=6)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    run = f"{a.model}_{a.action}"
    rdir = os.path.join(a.out, run)
    os.makedirs(rdir, exist_ok=True)
    spec = json.load(open(os.path.join(a.data, "dataset_spec.json")))
    goals = json.load(open(os.path.join(a.data, "goals.json")))

    tf = Transform(a.action, goals)
    tr_idx, va_idx = split_indices(a.data, os.path.join(a.out, "split.json"))
    print(f"[{run}] train {len(tr_idx)} val {len(va_idx)}", flush=True)
    dl = torch.utils.data.DataLoader(
        IndexedDS(a.data, tr_idx, tf), batch_size=a.batch, shuffle=True,
        num_workers=a.workers, pin_memory=True, drop_last=True,
        persistent_workers=True)
    dv = torch.utils.data.DataLoader(
        IndexedDS(a.data, va_idx, tf), batch_size=a.batch,
        num_workers=4, pin_memory=True)

    # normalizer over a fixed subsample of training action chunks
    base = StreamingDataset(a.data)
    rng = np.random.default_rng(0)
    acts = np.stack([tf(base[int(tr_idx[i])])[3].numpy()
                     for i in rng.integers(0, len(tr_idx), 2000)])
    lo = torch.tensor(acts.reshape(-1, ADIM).min(0) - 1e-4, device=dev).float()
    hi = torch.tensor(acts.reshape(-1, ADIM).max(0) + 1e-4, device=dev).float()
    del base
    norm = lambda x: ((x - lo) / (hi - lo)) * 2 - 1

    d_model = 512
    ctxnet = Context(d_model).to(dev)
    if a.model in ("dp", "cfm"):
        net = DiT(d_model, a.dit_depth).to(dev)
        if a.model == "dp":
            from diffusers import DDIMScheduler, DDPMScheduler
            nsched = DDPMScheduler(num_train_timesteps=50,
                                   beta_schedule="squaredcos_cap_v2",
                                   prediction_type="epsilon")
            isched = DDIMScheduler(num_train_timesteps=50,
                                   beta_schedule="squaredcos_cap_v2",
                                   prediction_type="epsilon")
            isched.set_timesteps(16)
    else:
        net = ACT(d_model).to(dev)
    vparams = list(ctxnet.venc.visual.parameters())
    vset = set(id(p) for p in vparams)
    heads = [p for p in list(net.parameters()) + list(ctxnet.parameters())
             if id(p) not in vset]
    opt = torch.optim.AdamW([{"params": heads, "lr": a.lr},
                             {"params": vparams, "lr": a.vision_lr}],
                            weight_decay=1e-6)
    mods = [net, ctxnet]
    ema = ({i: {k: v.detach().clone().float() for k, v in m.state_dict().items()}
            for i, m in enumerate(mods)} if a.model in ("dp", "cfm") else None)

    warm = 2000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm) * 0.5 *
        (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, a.steps - warm)))))
    log = open(os.path.join(rdir, "log.jsonl"), "a")
    json.dump(dict(args=vars(a), spec=spec, train=len(tr_idx), val=len(va_idx)),
              open(os.path.join(rdir, "run.json"), "w"), indent=1)

    def save_ckpt(tag, step):
        torch.save(dict(model=a.model, action=a.action, step=step,
                        lo=lo.cpu(), hi=hi.cpu(), spec=spec,
                        net=net.state_dict(), ctx=ctxnet.state_dict(), ema=ema),
                   os.path.join(rdir, f"ckpt_{tag}.pt"))

    @torch.no_grad()
    def eval_mse():
        """action-chunk MSE in normalized space, sampled (not loss proxy)."""
        errs, nb = [], 0
        for imgs, prop, goal, act in dv:
            imgs, prop, goal = (imgs.to(dev), prop.to(dev).float(),
                                goal.to(dev).float())
            x1 = norm(act.to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx = ctxnet(imgs, prop, goal)
                if a.model == "dp":
                    x = torch.randn_like(x1)
                    for t in isched.timesteps:
                        tb = t.expand(x.shape[0]).to(dev)
                        x = isched.step(net(x, tb, ctx).float(), t, x).prev_sample
                    pred = x
                elif a.model == "cfm":
                    x = torch.randn_like(x1)
                    for k in range(16):
                        tb = torch.full((x.shape[0],), k / 16, device=dev)
                        x = x + net(x, tb * 50, ctx).float() / 16
                    pred = x
                else:
                    pred, _, _ = net(ctx, act=None)
            errs.append(float(F.mse_loss(pred.float(), x1)))
            nb += 1
            if nb >= a.val_batches:
                break
        return float(np.mean(errs))

    step, t0, done = 0, time.time(), False
    while step < a.steps and not done:
        for imgs, prop, goal, act in dl:
            if step >= a.steps:
                break
            imgs = imgs.to(dev, non_blocking=True)
            prop = prop.to(dev, non_blocking=True).float()
            goal = goal.to(dev, non_blocking=True).float()
            x1 = norm(act.to(dev, non_blocking=True))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx = ctxnet(imgs, prop, goal)
                if a.model == "dp":
                    t = torch.randint(0, 50, (x1.shape[0],), device=dev)
                    eps = torch.randn_like(x1)
                    loss = F.mse_loss(net(nsched.add_noise(x1, eps, t), t, ctx),
                                      eps.to(torch.bfloat16))
                elif a.model == "cfm":
                    t = torch.rand(x1.shape[0], device=dev)
                    x0 = torch.randn_like(x1)
                    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
                    loss = F.mse_loss(net(xt, t * 50, ctx),
                                      (x1 - x0).to(torch.bfloat16))
                else:
                    pred, mu, logv = net(ctx, act=x1)
                    l1 = F.l1_loss(pred, x1)
                    kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
                    loss = l1 + 10.0 * kl
            opt.zero_grad()
            loss.float().backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    decay = min(0.9999, (1 + step) / (10 + step)) ** 0.75
                    for i, m in enumerate(mods):
                        cur = m.state_dict()
                        for k in ema[i]:
                            ema[i][k].mul_(decay).add_(cur[k].float(),
                                                       alpha=1 - decay)
            if step % 100 == 0:
                sps = (step + 1) / (time.time() - t0)
                log.write(json.dumps({"step": step, "loss": float(loss),
                                      "sps": sps}) + "\n")
                log.flush()
            if step > 0 and step % a.val_every == 0:
                v = eval_mse()
                log.write(json.dumps({"step": step, "eval_mse": v}) + "\n")
                log.flush()
                sps = (step + 1) / (time.time() - t0)
                eta_h = (a.steps - step) / max(sps, 1e-9) / 3600
                print(f"[{run}] step {step}/{a.steps} loss {float(loss):.4f} "
                      f"eval_mse {v:.5f} {sps:.1f} it/s eta {eta_h:.1f}h",
                      flush=True)
                if v < a.mse_stop and step >= a.min_steps:
                    print(f"[{run}] early stop: eval_mse {v:.5f} < {a.mse_stop}",
                          flush=True)
                    done = True
                    break
            if step > 0 and step % a.ckpt_every == 0:
                save_ckpt(f"{step//1000}k", step)
            step += 1

    save_ckpt("final", step)
    print(f"[{run}] done {step} steps -> {rdir}", flush=True)


if __name__ == "__main__":
    main()

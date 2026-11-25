"""realsim.policy_server :: one copy of the policy on the GPU, shared by every evaluation worker.

The scalar evaluator used to load the policy (6.4 GB for the 920M DiT) in each worker process, which capped it at 3
parallel scenes although the workers are CPU-bound (twin physics, OSMesa rendering, IK, safety queries: ~0.5 s per
decision against ~45 ms of GPU work). Here the workers keep the exact twin and host executor and send their images
and inference inputs to this server, which batches whatever is waiting:

    encode   BGR uint8 [240, 432, 3]  ->  pooled image token [1, D]   (the policy's ObsEncoder mean-pools patch tokens,
                                                                         so the pooled token is all a worker needs)
    infer    tok4 [4, 1, D], prop [2, 10], goal [9]  ->  action chunk [16, 10]

Simulated latency is unaffected: the twin's inference delay is drawn from its config, not from wall time.
"""
import os
import queue
import sys
import time

import numpy as np


def serve(a, req_q, resp_qs, ready):
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    sys.path[:0] = [ROOT, os.path.join(ROOT, "policy")]
    sys.path.append(os.path.expanduser("~/Desktop/lingbot-vision"))
    import torch
    import real_rollout_zed as RZ
    import train_policy_zed as T
    dev = "cuda:" + a["gpus"].split(",")[0]
    RZ.DEV = dev
    _tl = torch.load
    torch.load = lambda *x, **kw: _tl(*x, **{**kw, "map_location": "cpu"})
    pol = RZ.ZedPolicy(a["ckpt"], euler=a["euler"], frozen_noise=True)
    torch.load = _tl
    if a["patch"]:
        from realsim.wsp import apply_patch
        apply_patch(pol, a["patch"])
    fast = bool(a["fast_infer"])
    if fast:
        pol.net.half()
    meta = dict(use_goal=pol.use_goal, goal_se3=pol.goal_se3, action_mode=pol.action_mode, lat_shift=pol.lat_shift)
    ready.put(meta)
    st = dict(enc=0, inf=0, enc_b=0, inf_b=0, t_enc=0.0, t_inf=0.0)
    x0 = (404 - 224) // 2
    MAXB = int(a.get("serve_batch", 96))

    @torch.no_grad()
    def encode(imgs):
        t = torch.from_numpy(np.stack(imgs)[..., ::-1].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0      # BGR -> RGB
        t = torch.nn.functional.interpolate(t, size=(224, 404), mode="bilinear", align_corners=False)
        px = (t[:, :, :, x0:x0 + 224] - pol._emean) / pol._estd
        with torch.autocast("cuda", dtype=torch.float16, enabled=fast):
            f = pol._fwd(px)
        return f.float().mean(1).cpu().numpy()                    # [B, D]

    @torch.no_grad()
    def infer(tok, prop, goal):
        B = len(tok)
        g = torch.from_numpy(np.stack(goal).astype(np.float32)).to(dev) if pol.use_goal else None
        cond = pol.obs_enc(torch.from_numpy(np.stack(tok)).to(dev), torch.from_numpy(np.stack(prop).astype(np.float32)).to(dev), g)
        x = pol.noise0.expand(B, -1, -1).clone()
        n = pol.euler
        for i in range(n):
            t = torch.full((B,), i / n * 50, device=dev)
            if fast:
                with torch.autocast("cuda", dtype=torch.float16):
                    v = pol.net(x, t, cond).float()
            else:
                v = pol.net(x, t, cond)
            x = x + v / n
        return ((x.transpose(1, 2) + 1) / 2 * (pol.hi - pol.lo) + pol.lo).cpu().numpy()

    while True:
        batch = [req_q.get()]
        t_end = time.time() + 0.004                               # let concurrent workers' requests join the batch
        while len(batch) < MAXB:
            try:
                batch.append(req_q.get(timeout=max(0.0, t_end - time.time())))
            except queue.Empty:
                break
        if any(b[1] == "stop" for b in batch):
            ready.put(st)
            return
        enc = [b for b in batch if b[1] == "encode"]
        inf = [b for b in batch if b[1] == "infer"]
        if enc:
            t0 = time.time()
            out = encode([b[2] for b in enc])
            for b, o in zip(enc, out):
                resp_qs[b[0]].put(o[None])
            st["enc"] += len(enc); st["enc_b"] += 1; st["t_enc"] += time.time() - t0
        if inf:
            t0 = time.time()
            out = infer([b[2][0] for b in inf], [b[2][1] for b in inf], [b[2][2] for b in inf])
            for b, o in zip(inf, out):
                resp_qs[b[0]].put(o)
            st["inf"] += len(inf); st["inf_b"] += 1; st["t_inf"] += time.time() - t0


class RemotePolicy:
    """Worker-side stand-in for real_rollout_zed.ZedPolicy (encode + call), talking to serve()."""

    def __init__(self, wid, req_q, resp_q, meta):
        self.wid, self.req, self.resp = wid, req_q, resp_q
        self.use_goal, self.goal_se3 = meta["use_goal"], meta["goal_se3"]
        self.action_mode, self.lat_shift = meta["action_mode"], meta["lat_shift"]
        self.fast = False

    def _ask(self, kind, payload):
        self.req.put((self.wid, kind, payload))
        return self.resp.get(timeout=900)

    def encode(self, bgr):
        import cv2
        import torch
        if bgr.shape[1] != 432 or bgr.shape[0] != 240:
            bgr = cv2.resize(bgr, (432, 240), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(self._ask("encode", np.ascontiguousarray(bgr)))          # [1, D]: already pooled

    def __call__(self, tok4, prop, goal=None, rtc=None):
        if rtc is not None:
            raise NotImplementedError("--rtc is not supported with --serve")
        g = np.zeros(9, np.float32) if goal is None else np.asarray(goal, np.float32)
        return self._ask("infer", (tok4.numpy().astype(np.float32), np.asarray(prop, np.float32), g))

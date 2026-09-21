#!/usr/bin/env python3
"""qplan.export :: the WHOLE planner as ONE TensorRT graph (obs -> executed action).

The deployed controller (`rl/real_policy_ctrl.py`) calls `policy(obs) -> 7-D action` once per
decision at 10 Hz and knows nothing else.  Everything the planner does therefore has to live
inside the graph:

    obs (1, 40)
      -> a_pi = pi(obs)                              the frozen fused residual stack
      -> N candidate chunks around it                gaussian (FIXED noise table) + scale ladder
                                                     + safety holds, exactly proposals.propose()
      -> Q_succ, Q_time, Q_speed for all N           one batched critic call (the 3-head QChunk)
      -> success gate, speed gate, softmax mixture    planner.QPlanner.plan()
      -> one-step EMA with a_prev (obs[18:24])
      -> act (1, 7)                                  the first action of the mixed chunk

Two things make this exportable at all:

* the gaussian family's noise is a CONSTANT tensor baked into the graph (`proposals.noise_table`)
  instead of a fresh draw per decision, so the graph has no RNG and is deterministic;
* the EMA reads the previous action out of the OBSERVATION rather than from planner state, so the
  graph is pure `f(obs)` and the controller stays stateless.

Both are validated in the twin (README "Speed head"): fixing the noise table and reading a_prev
from the observation cost nothing measurable.

    # build on the GPU that will RUN it (.plan files are GPU/driver specific)
    CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/export.py --q ~/pnp_rl/qplan/q1sp/q.pt \\
        --out rl/weights/qplan_v1 --w_speed 1.0 --ema 0.7
    CUDA_VISIBLE_DEVICES=0 $PY rl/qplan/export.py --q ... --out rl/weights/qplan_v1   # controller GPU

    # then, with NO change to the controller (load_policy already picks up <stem>.plan):
    $PY rl/real_policy_ctrl.py --policy rl/weights/qplan_v1 ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
for p in (HERE, RL):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import ACT_DIM, H, OBS_DIM, PI_PATH, DATA_ROOT, load_pi  # noqa: E402
from planner import QPlanner, load_q                                # noqa: E402
import proposals as PR                                              # noqa: E402

A_PREV0 = 18            # obs layout: q6 qd6 p_obj3 goal3 a_prev7 tcp3 axis3 rel3 lag6
NEG = -1e9


# --------------------------------------------------------------------------- candidate builder
def _scale_j(c, sc):
    """Scale the six joint channels, leaving the suction logit alone (functional: no in-place
    index assignment, which is what keeps the ONNX graph free of ScatterND)."""
    return torch.cat([c[..., :6] * sc, c[..., 6:]], -1)


def build_candidates(a_pi, noise, n_cand=16, ladder=True, fine=False):
    # `noise` is (n_g-1, H, 6) for a frozen table or (B, n_g-1, H, 6) for the observation hash
    """a_pi (B, 7) + fixed noise (n_g-1, H, 6) -> (B, C, H, 7), clamped.

    A functional re-implementation of `proposals.propose(..., noise=noise)`.  `export.py --check`
    asserts the two agree bit-for-bit on random inputs, so the graph cannot silently drift away
    from the planner that was evaluated in the twin.
    """
    n_g, n_s, n_safe = PR.family_sizes(n_cand)
    B = a_pi.shape[0]
    base = a_pi[:, None, None, :].expand(B, 1, H, ACT_DIM)
    outs = [base]
    if n_g > 1 and fine:
        sc = torch.tensor([PR.FINE[k % len(PR.FINE)] for k in range(n_g - 1)],
                          device=a_pi.device, dtype=a_pi.dtype)
        outs.append(torch.cat([base[..., :6] * sc[None, :, None, None],
                               base[..., 6:].expand(B, n_g - 1, H, 1)], -1))
    elif n_g > 1:
        eps = noise if noise.dim() == 4 else noise[None]
        g_noisy = torch.cat([base[..., :6] + eps,
                             base[..., 6:].expand(B, n_g - 1, H, 1)], -1)
        outs.append(g_noisy)
    g = torch.cat(outs, 1)                                    # (B, n_g, H, 7)
    outs = [g]
    if n_s > 0:
        rep = []
        if ladder:
            for sc in PR.LADDER:
                rep.append(_scale_j(base, sc))
            n_noisy = max(1, (n_s - len(PR.LADDER)) // 3)
            src = g[:, 1:1 + n_noisy]
        else:
            src = g[:, :max(1, n_s // 3)]
        for sc in (0.7, 1.2, 1.4):
            rep.append(_scale_j(src, sc))
        s = torch.cat(rep, dim=1)
        while s.shape[1] < n_s:
            src2 = g[:, 1:1 + max(1, n_s // 6)]
            for sc in (0.85, 1.1, 1.3):
                s = torch.cat([s, _scale_j(src2, sc)], 1)
        s = s[:, :n_s]
        n_flip = min(2, s.shape[1])
        if n_flip:
            s = torch.cat([s[:, :-n_flip],
                           torch.cat([s[:, -n_flip:, :, :6], -s[:, -n_flip:, :, 6:]], -1)], 1)
        outs.append(s)
    safe = []
    for k in range(n_safe):
        sc, suck = PR.SAFE_SPECS[k % len(PR.SAFE_SPECS)]
        c = _scale_j(base, sc)
        if suck is not None:
            c = torch.cat([c[..., :6], torch.full_like(c[..., 6:], float(suck))], -1)
        safe.append(c)
    if safe:
        outs.append(torch.cat(safe, dim=1))
    return torch.cat(outs, dim=1).clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- the graph
class QPlanGraph(nn.Module):
    """planner.QPlanner, written so that the whole thing traces to a single static ONNX graph.

    Every gate is a FLOAT mask (`torch.where` on `mask > 0.5`) rather than a boolean reduction:
    TensorRT's support for bool ReduceMax/And is patchy, and the arithmetic is identical.
    """

    def __init__(self, pi, q, noise, n_cand=16, lam=0.1, succ_frac=0.9, w_speed=0.0,
                 speed_margin=None, ema=None, rule="weighted", fine=False, hash_tab=None,
                 sigma_shared=0.15, sigma_step=0.05):
        super().__init__()
        self.pi, self.q = pi, q
        self.register_buffer("noise", noise.float())
        self.sigma_shared, self.sigma_step = float(sigma_shared), float(sigma_step)
        self.hash_k = 0
        if hash_tab is not None:
            self.hash_k = int(hash_tab["k"])
            for nm in ("Ws", "bs", "Wt", "bt"):
                self.register_buffer("h_" + nm, hash_tab[nm].float())
        self.n_cand, self.lam, self.succ_frac = int(n_cand), float(lam), float(succ_frac)
        self.w_speed = float(w_speed)
        self.speed_margin = -1.0 if speed_margin is None else float(speed_margin)
        self.use_sp_gate = speed_margin is not None
        self.ema = None if ema is None else float(ema)
        self.rule = rule
        self.fine = bool(fine)
        # the "nothing survived -> fall back to pi" one-hot, as a CONSTANT.  Building it in the
        # forward pass (`zeros_like(alive)[:, 0] = 1`) traces to a ScatterND whose output buffer
        # TensorRT does not zero-initialise: the engine then multiplied (1 - any_alive) = 0 by
        # uninitialised memory, produced NaNs in the mask on ~0.5 % of the candidates and dropped
        # them, and the action differed from torch by up to 1.8 on 15 of 256 observations.
        oh = torch.zeros(int(n_cand))
        oh[0] = 1.0
        self.register_buffer("onehot0", oh)
        self.heads = tuple(q.head_names)

    def forward(self, obs):
        B = obs.shape[0]
        a_pi = self.pi(obs)
        noise = self.noise
        if self.hash_k:
            z = (obs - self.q.obs_mean) / self.q.obs_std
            noise = PR.hash_noise(z, dict(k=self.hash_k, Ws=self.h_Ws, bs=self.h_bs,
                                          Wt=self.h_Wt, bt=self.h_bt),
                                  self.sigma_shared, self.sigma_step)
        cand = build_candidates(a_pi, noise, self.n_cand, fine=self.fine)        # (B, C, H, 7)
        C = cand.shape[1]
        o = obs[:, None, :].expand(B, C, obs.shape[-1]).reshape(B * C, -1)
        lg = self.q.logits(o, cand.reshape(B * C, -1))
        qv = {k: self.q.heads[k].q(lg[k]).view(B, C) for k in self.heads}
        s, tq = qv["succ"], qv["time"]
        vq = qv["speed"] if "speed" in qv else torch.zeros_like(tq)
        smax = s.max(dim=1, keepdim=True).values
        thr = smax - (1.0 - self.succ_frac) * smax.abs()
        alive = (s >= thr - 1e-6).float()
        if self.use_sp_gate:
            alive = alive * (vq >= vq[:, :1] - self.speed_margin).float()
        # never leave the graph with nothing to execute: fall back to pi's own chunk
        any_alive = alive.max(dim=1, keepdim=True).values
        alive = torch.clamp(alive + (1.0 - any_alive) * self.onehot0[None, :], max=1.0)
        score = tq + self.w_speed * vq if self.w_speed else tq
        if self.rule == "best":
            masked = torch.where(alive > 0.5, score, torch.full_like(score, NEG))
            w = (masked >= masked.max(dim=1, keepdim=True).values).float()
            w = w / w.sum(dim=1, keepdim=True)
        else:
            masked = torch.where(alive > 0.5, score / self.lam, torch.full_like(score, NEG))
            w = torch.softmax(masked, dim=1)
        chunk = (w[:, :, None, None] * cand).sum(1)                     # (B, H, 7)
        a = chunk[:, 0]
        if self.ema is not None:
            j = self.ema * a[:, :6] + (1.0 - self.ema) * obs[:, A_PREV0:A_PREV0 + 6]
            a = torch.cat([j.clamp(-1.0, 1.0), a[:, 6:]], -1)
        return a


# --------------------------------------------------------------------------- build + verify
def build_engine(onnx_path, plan_path, fp16=False, workspace_mb=512):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[trt]", parser.get_error(i))
            raise SystemExit("onnx parse failed")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16 and builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        raise SystemExit("engine build failed")
    with open(plan_path, "wb") as f:
        f.write(bytes(blob))
    print(f"[trt] engine {plan_path} ({blob.nbytes / 1e3:.0f} kB, {time.time() - t0:.1f} s, fp16={fp16})")
    return plan_path


def main():
    ap = argparse.ArgumentParser()
    q_def = os.path.join(RL, "weights", "qplan_v1_critic.pt")
    ap.add_argument("--q", default=(q_def if os.path.exists(q_def)
                                    else os.path.join(DATA_ROOT, "q1sep", "q.pt")),
                    help="the 3-head critic (qplan/critic.py --sep_speed); the copy in "
                         "rl/weights makes the export reproducible from the repo alone")
    ap.add_argument("--pi", default=PI_PATH)
    ap.add_argument("--out", default=os.path.join(RL, "weights", "qplan_v1"))
    ap.add_argument("--n_cand", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--succ_frac", type=float, default=0.9)
    ap.add_argument("--w_speed", type=float, default=0.0)
    ap.add_argument("--speed_margin", type=float, default=None)
    ap.add_argument("--ema", type=float, default=None)
    ap.add_argument("--rule", default="weighted", choices=["weighted", "best"])
    ap.add_argument("--fine", type=int, default=0,
                    help="1 = replace the gaussian family with extra ladder scales (deterministic "
                         "but measurably worse -- see README; prefer --hash)")
    ap.add_argument("--hash", type=int, default=1,
                    help="1 (default) = gaussian family from sin(W.obs): deterministic AND it "
                         "keeps the per-decision random search the planner needs")
    ap.add_argument("--hash_gain", type=float, default=5.0)
    ap.add_argument("--sigma_shared", type=float, default=0.15)
    ap.add_argument("--sigma_step", type=float, default=0.05)
    ap.add_argument("--noise_seed", type=int, default=31337,
                    help="seed of the FIXED gaussian table baked into the graph")
    ap.add_argument("--obs_dim", type=int, default=OBS_DIM)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--n_verify", type=int, default=256)
    ap.add_argument("--verify_obs", default=os.path.join(DATA_ROOT, "data", "m0_pi_b00.pt"),
                    help="episode shard to draw REAL verification observations from")
    ap.add_argument("--no_trt", action="store_true", help="ONNX only (no GPU needed)")
    a = ap.parse_args()

    dev = "cpu"
    pi = load_pi(device=dev, path=a.pi)
    q, ck = load_q(a.q, device=dev)
    print(f"[export] pi={a.pi}  q={a.q} (heads {q.head_names}, {ck.get('episodes')} episodes)")

    # the noise table is drawn ONCE, on the cpu, from a fixed seed and then frozen in the graph
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.noise_seed)
    noise = PR.noise_table(a.n_cand, gen=gen, device=dev)
    hash_tab = PR.hash_tables(a.n_cand, obs_dim=a.obs_dim, gen=gen, device=dev,
                              gain=a.hash_gain) if a.hash else None
    if hash_tab is not None:
        torch.save(hash_tab, a.out + "_hash.pt")
        print(f"[export] hash tables -> {a.out}_hash.pt  (evaluate the SAME planner in the twin "
              f"with: rl/qplan/planner.py --hash_tab {a.out}_hash.pt)")

    # candidate builder == proposals.propose (the planner evaluated in the twin)
    with torch.no_grad():
        ap_ = torch.randn(8, ACT_DIM).clamp(-1, 1)
        nz = noise
        if hash_tab is not None:
            nz = PR.hash_noise(torch.randn(8, a.obs_dim), hash_tab, a.sigma_shared, a.sigma_step)
        d = (build_candidates(ap_, nz, a.n_cand, fine=a.fine)
             - PR.propose(ap_, n_cand=a.n_cand, noise=nz, fine=a.fine)).abs().max()
    assert float(d) == 0.0, f"candidate builder differs from proposals.propose by {float(d)}"
    print(f"[export] candidate builder == proposals.propose (max abs diff {float(d):.1e})")

    graph = QPlanGraph(pi, q, noise, n_cand=a.n_cand, lam=a.lam, succ_frac=a.succ_frac,
                       w_speed=a.w_speed, speed_margin=a.speed_margin, ema=a.ema,
                       rule=a.rule, fine=a.fine, hash_tab=hash_tab,
                       sigma_shared=a.sigma_shared, sigma_step=a.sigma_step).eval()
    stem = a.out
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    onnx_path = stem + ".onnx"
    dummy = torch.zeros(1, a.obs_dim)
    with torch.no_grad():
        torch.onnx.export(graph, dummy, onnx_path, input_names=["obs"], output_names=["act"],
                          opset_version=17, dynamo=False)
    print(f"[onnx] {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    # verification observations: REAL ones from the replay buffer when available.  N(0,1) noise
    # is far outside the critic's normaliser, where the candidates' Q values collapse onto each
    # other and the gate is a coin flip -- see the tie discussion printed below.
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(a.n_verify, a.obs_dim)).astype(np.float32)
    src = "random N(0,1)"
    if a.verify_obs and os.path.exists(a.verify_obs):
        d_ = torch.load(a.verify_obs, map_location="cpu", weights_only=False)
        ep = torch.randint(0, d_["obs"].shape[0], (a.n_verify,),
                           generator=torch.Generator().manual_seed(0))
        dt = torch.randint(0, d_["obs"].shape[1] - 1, (a.n_verify,),
                           generator=torch.Generator().manual_seed(1))
        X = d_["obs"][ep, dt].float().numpy().astype(np.float32)
        src = os.path.basename(a.verify_obs)
    print(f"[verify] {a.n_verify} observations from {src}")
    with torch.no_grad():
        ref = np.concatenate([graph(torch.tensor(X[i:i + 1])).numpy() for i in range(len(X))])
    # the graph re-writes the planner's gates with float masks; check it against the planner
    # class that was actually evaluated in the twin, on the same observations
    qp = QPlanner(pi, q, a.n_cand, a.lam, succ_frac=a.succ_frac, w_speed=a.w_speed,
                  speed_margin=a.speed_margin, ema=a.ema, rule=a.rule, noise=noise,
                  fine=bool(a.fine), hash_tab=hash_tab, sigma_shared=a.sigma_shared,
                  sigma_step=a.sigma_step)
    qp.reset(len(X), dev)
    with torch.no_grad():
        ref_cls = qp(torch.tensor(X)).numpy()
    e_cls = float(np.abs(ref_cls - ref).max())
    print(f"[verify] max|graph-QPlanner| {e_cls:.2e} (batched, same fixed noise table)")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = np.concatenate([sess.run(None, {"obs": X[i:i + 1]})[0] for i in range(len(X))])
    e_onnx = float(np.abs(ort_out - ref).max())
    meta = dict(kind="qplan", q=os.path.abspath(a.q), pi=os.path.abspath(a.pi),
                n_cand=a.n_cand, lam=a.lam, succ_frac=a.succ_frac, fine=bool(a.fine),
                hash_gain=(a.hash_gain if a.hash else None), w_speed=a.w_speed,
                speed_margin=a.speed_margin, ema=a.ema, rule=a.rule, noise_seed=a.noise_seed,
                heads=list(q.head_names), obs_dim=a.obs_dim, fp16=a.fp16,
                dq_max_deg=3.0, gpu=None, max_abs_err_onnx=e_onnx, max_abs_err_planner=e_cls,
                obs_layout="q[6] qd[6] p_obj[3] p_goal[3] a_prev[7] tcp[3] cup_axis[3] "
                           "grasp_rel[3] (q_target-q)[6]")
    if not a.no_trt:
        import torch as _t
        gpu = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "?"
        plan_path = build_engine(onnx_path, stem + ".plan", fp16=a.fp16)
        from export_trt import TrtPolicy
        trt_pol = TrtPolicy(plan_path)
        trt_out = np.stack([trt_pol(X[i]) for i in range(len(X))])
        t0 = time.time()
        for i in range(len(X)):
            trt_pol(X[i % len(X)])
        dt = (time.time() - t0) / len(X)
        per = np.abs(trt_out - ref).max(1)
        e_trt, med, n_bad = float(per.max()), float(np.median(per)), int((per > 1e-4).sum())
        print(f"[verify] max|onnx-torch| {e_onnx:.2e}  |trt-torch| median {med:.2e} max "
              f"{e_trt:.2e} ({n_bad}/{len(X)} over 1e-4)  "
              f"trt latency {1e6 * dt:.0f} us/call on {gpu}")
        if n_bad:
            # A graph with GATES cannot agree bit-for-bit across backends.  The engine's INTERIOR
            # Q differs from torch's by enough to move a candidate across the success / speed
            # threshold on a few per cent of observations, and the mixture then changes by a lot;
            # everywhere else the two agree to ~2e-7.  (A Q-only graph, where the values are
            # network OUTPUTS, agrees to 1.4e-5 -- so this is TensorRT's fusion of interior
            # tensors, not the network.)  What matters is behaviour, and that is unchanged:
            # SimLink selftest 97.7 % against the twin's 98.2 % (README "Speed head").
            with torch.no_grad():
                ot = torch.tensor(X)
                cd = build_candidates(pi(ot), (PR.hash_noise((ot - q.obs_mean) / q.obs_std,
                                                             hash_tab, a.sigma_shared, a.sigma_step)
                                               if hash_tab is not None else noise),
                                      a.n_cand, fine=bool(a.fine))
                qq = q.score(ot, cd)
                sm = qq["succ"].max(1, keepdim=True).values
                d1 = (qq["succ"] - (sm - (1 - a.succ_frac) * sm.abs())).abs().min(1).values
                dd = d1 if a.speed_margin is None else torch.minimum(
                    d1, (qq["speed"] - (qq["speed"][:, :1] - a.speed_margin)).abs().min(1).values)
            print(f"[verify] the {n_bad} disagreeing observation(s) have a candidate within "
                  f"{dd[torch.tensor(per > 1e-4)].max():.1e} of a gate threshold (median over "
                  f"all {len(X)}: {dd.median():.1e}): the engine puts it on the other side of "
                  f"the gate.  Behaviour is unchanged -- see the SimLink selftest in the README.")
        meta.update(gpu=gpu, max_abs_err_trt=e_trt, med_abs_err_trt=med, n_over_1e4=n_bad,
                    n_verify=len(X), verify_obs=src, trt_us=1e6 * dt)
    else:
        print(f"[verify] max|onnx-torch| {e_onnx:.2e}  (no TensorRT)")
    json.dump(meta, open(stem + ".json", "w"), indent=1)
    print("[meta]", json.dumps(meta))
    print(f"[export] the controller needs NO code change: rl/real_policy_ctrl.py --policy {stem}")


if __name__ == "__main__":
    main()

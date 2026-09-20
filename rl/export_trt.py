#!/usr/bin/env python3
"""export_trt :: PPO/DAgger checkpoint -> ONNX -> TensorRT engine for the real-robot controller.

The exported graph is the DETERMINISTIC policy: obs (1, obs_dim) -> tanh(pi(obs)) in [-1, 1]^7
(6 joint deltas in units of dq_max, suction command sign). The critic and log_std are dropped.

  $PY rl/export_trt.py rl/weights/paper14_real_best.pt --obs_dim 40 --out rl/weights/paper14_real_best
  -> paper14_real_best.onnx, paper14_real_best.plan (+ .json meta), verified against torch on random obs

Needs tensorrt-cu12, onnx, onnxruntime-gpu (pip, mjwarp env). The engine is GPU/driver specific:
rebuild on the machine that runs the controller (rl/real_policy_ctrl.py rebuilds automatically
if the .plan is missing or fails to deserialize).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ppo import AC  # noqa: E402


class DetPolicy(torch.nn.Module):
    def __init__(self, ac):
        super().__init__()
        self.pi = ac.pi

    def forward(self, obs):
        return torch.tanh(self.pi(obs))


class ResidualDetPolicy(torch.nn.Module):
    """Fused residual policy: clamp(tanh(base.pi(o)) + bound*tanh(res.pi(o)), -1, 1).
    Exported as ONE graph, so the controller sees the same interface as a plain policy."""

    def __init__(self, base_ac, res_ac, bound):
        super().__init__()
        self.base = base_ac.pi
        self.res = res_ac.pi
        self.bound = float(bound)

    def forward(self, obs):
        return torch.clamp(torch.tanh(self.base(obs)) + self.bound * torch.tanh(self.res(obs)), -1.0, 1.0)


def build_engine(onnx_path, plan_path, fp16=False, workspace_mb=256):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)          # TRT >= 10: explicit batch is the default
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


class TrtPolicy:
    """Synchronous single-batch inference on a serialized engine (numpy in, numpy out).
    Device buffers and the stream come from torch, so no cuda-python dependency."""

    def __init__(self, plan_path, device="cuda:0"):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.ERROR)
        with open(plan_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("engine deserialize failed")
        self.ctx = self.engine.create_execution_context()
        self.in_name, self.out_name = self.engine.get_tensor_name(0), self.engine.get_tensor_name(1)
        self.dev = torch.device(device)
        self.d_in = torch.zeros(tuple(self.engine.get_tensor_shape(self.in_name)), dtype=torch.float32, device=self.dev)
        self.d_out = torch.zeros(tuple(self.engine.get_tensor_shape(self.out_name)), dtype=torch.float32, device=self.dev)
        self.stream = torch.cuda.Stream(device=self.dev)
        self.ctx.set_tensor_address(self.in_name, self.d_in.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.d_out.data_ptr())

    def __call__(self, obs):
        with torch.cuda.stream(self.stream):
            self.d_in.copy_(torch.as_tensor(np.asarray(obs, np.float32)).reshape(self.d_in.shape), non_blocking=True)
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            out = self.d_out.to("cpu", non_blocking=True)
        self.stream.synchronize()
        return out.numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--obs_dim", type=int, default=40, help="40 = paper env on the measured drive (34 ideal)")
    ap.add_argument("--arch", default="paper")
    ap.add_argument("--out", default=None, help="output stem (default: ckpt path without .pt)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--residual_base", default=None, help="override the checkpoint's residual_base path")
    ap.add_argument("--residual_bound", type=float, default=None, help="override the checkpoint's residual_bound")
    a = ap.parse_args()
    stem = a.out or os.path.splitext(a.ckpt)[0]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ac = AC(obs_dim=a.obs_dim, arch=a.arch, critic_extra=(5 if ck.get("critic_priv") else 0))
    ac.load_state_dict(ck["ac"])
    ac.eval()
    base_path = a.residual_base or ck.get("residual_base")
    if base_path:                       # residual checkpoint -> export the FUSED policy
        bound = float(a.residual_bound if a.residual_bound is not None else ck.get("residual_bound", 0.3))
        base = AC(obs_dim=a.obs_dim, arch=a.arch)
        ckb = torch.load(base_path, map_location="cpu", weights_only=False)
        base.load_state_dict(ckb["ac"] if "ac" in ckb else ckb)
        base.eval()
        pol = ResidualDetPolicy(base, ac, bound).eval()
        print(f"[export] residual policy: base {base_path} bound {bound}")
    else:
        pol = DetPolicy(ac).eval()
    dummy = torch.zeros(1, a.obs_dim)
    onnx_path = stem + ".onnx"
    torch.onnx.export(pol, dummy, onnx_path, input_names=["obs"], output_names=["act"], opset_version=17, dynamo=False)
    print(f"[onnx] {onnx_path}")
    plan_path = build_engine(onnx_path, stem + ".plan", fp16=a.fp16)
    # ---- verify: torch vs onnxruntime vs tensorrt on random observations ----
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(256, a.obs_dim)).astype(np.float32)
    with torch.no_grad():
        ref = pol(torch.tensor(X)).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = np.concatenate([sess.run(None, {"obs": X[i:i + 1]})[0] for i in range(len(X))])
    trt_pol = TrtPolicy(plan_path)
    t0 = time.time()
    trt_out = np.stack([trt_pol(X[i]) for i in range(len(X))])
    dt = (time.time() - t0) / len(X)
    print(f"[verify] max|onnx-torch| {np.abs(ort_out - ref).max():.2e}  max|trt-torch| {np.abs(trt_out - ref).max():.2e}  trt latency {1e6 * dt:.0f} us/call")
    meta = dict(ckpt=os.path.abspath(a.ckpt), obs_dim=a.obs_dim, arch=a.arch, step=ck.get("step"), success=ck.get("success"),
                residual_base=(os.path.abspath(base_path) if base_path else None),
                residual_bound=(bound if base_path else None),
                fp16=a.fp16, max_abs_err_trt=float(np.abs(trt_out - ref).max()), trt_us=1e6 * dt,
                obs_layout="q[6] qd[6] p_obj[3] p_goal[3] a_prev[7] tcp[3] cup_axis[3] grasp_rel[3] (q_target-q)[6]" if a.obs_dim == 40
                else "q[6] qd[6] p_obj[3] p_goal[3] a_prev[7] tcp[3] cup_axis[3] grasp_rel[3]")
    json.dump(meta, open(stem + ".json", "w"), indent=1)
    print("[meta]", json.dumps(meta))


if __name__ == "__main__":
    main()

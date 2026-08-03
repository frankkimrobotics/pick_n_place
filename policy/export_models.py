"""export_models :: compile a train_lit checkpoint for deployment.

Per checkpoint, exports TWO modules at batch 1 (EMA weights preferred):
  context : (imgs [1,4,3,224,224], prop [1,2,14], goal [1,9]) -> ctx [1,790,512]
  net     : dp/cfm  (x [1,16,7], t [1], ctx) -> eps/velocity [1,16,7]
            act     (ctx) -> action [1,16,7]   (CVAE prior branch, z = 0)

Formats, written to --out/<model>_<tag>/:
  *_traced.pt   torch.jit.trace (TorchScript)
  *.onnx        opset 17, static shapes
  *.engine      TensorRT serialized engine (fp16), built for the LOCAL GPU
                (arch-specific -- rebuild on the deployment machine if it
                differs). Numerical check traced-vs-eager printed per module.

Run in sam3 env:  python export_models.py --ckpt ~/pnp_runs_studio/dp_q/ckpt_final.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_lit as tl                                     # noqa: E402


class ACTInfer(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, ctx):
        return self.net(ctx, act=None)[0]


class DenoiseInfer(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, ctx):
        return self.net(x, t, ctx)


def build_trt(onnx_path, engine_path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  trt-parse:", parser.get_error(i))
            raise RuntimeError(f"onnx parse failed: {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"trt build failed: {onnx_path}")
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"  [trt] {engine_path} ({os.path.getsize(engine_path)//2**20} MB)")


def export_module(mod, example, name, odir):
    mod.eval()
    with torch.no_grad():
        ref = mod(*example)
        traced = torch.jit.trace(mod, example)
        out = traced(*example)
        err = (out - ref).abs().max().item()
    tpath = os.path.join(odir, f"{name}_traced.pt")
    traced.save(tpath)
    print(f"  [trace] {tpath} (max dev {err:.2e})")
    opath = os.path.join(odir, f"{name}.onnx")
    torch.onnx.export(mod, example, opath, opset_version=17,
                      input_names=[f"in{i}" for i in range(len(example))],
                      output_names=["out"], dynamo=False)
    print(f"  [onnx] {opath} ({os.path.getsize(opath)//2**20} MB)")
    build_trt(opath, os.path.join(odir, f"{name}.engine"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_export"))
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = ck["model"]
    tag = os.path.splitext(os.path.basename(a.ckpt))[0].replace("ckpt_", "")
    odir = os.path.join(a.out, f"{model}_{tag}")
    os.makedirs(odir, exist_ok=True)
    print(f"[export] {model} ({a.ckpt}) -> {odir}")

    ctxnet = tl.Context(512).to(dev)
    net = (tl.DiT(512) if model in ("dp", "cfm") else tl.ACT(512)).to(dev)
    if ck.get("ema") is not None:
        ema = {int(i): st for i, st in ck["ema"].items()}
        net.load_state_dict(ema[0])
        ctxnet.load_state_dict(ema[1])
        print("[export] loaded EMA weights")
    else:
        net.load_state_dict(ck["net"])
        ctxnet.load_state_dict(ck["ctx"])
    try:                                       # incompatible with trace/onnx
        ctxnet.venc.visual.set_grad_checkpointing(False)
    except Exception:
        pass

    imgs = torch.randn(1, 4, 3, 224, 224, device=dev)
    prop = torch.randn(1, tl.TO, tl.PDIM, device=dev)
    goal = torch.randn(1, tl.GDIM, device=dev)
    export_module(ctxnet, (imgs, prop, goal), "context", odir)

    with torch.no_grad():
        ctx = ctxnet(imgs, prop, goal)
    if model in ("dp", "cfm"):
        x = torch.randn(1, tl.TA, tl.ADIM, device=dev)
        t = torch.full((1,), 25.0, device=dev)
        export_module(DenoiseInfer(net), (x, t, ctx), "net", odir)
    else:
        export_module(ACTInfer(net), (ctx,), "net", odir)

    json.dump(dict(model=model, ckpt=os.path.abspath(a.ckpt), step=ck["step"],
                   action=ck["action"], lo=ck["lo"].tolist(),
                   hi=ck["hi"].tolist(), spec=ck["spec"],
                   io=dict(context="imgs[1,4,3,224,224] prop[1,2,14] goal[1,9]"
                                   " -> ctx[1,790,512]",
                           net=("x[1,16,7] t[1] ctx -> out[1,16,7]"
                                if model in ("dp", "cfm")
                                else "ctx -> action[1,16,7]"))),
              open(os.path.join(odir, "export_meta.json"), "w"), indent=1)
    print(f"[export] done -> {odir}")


if __name__ == "__main__":
    main()

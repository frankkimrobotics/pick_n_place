"""Throughput of BatchTwin: physics + drive chain + suction per world count, and the cost of the delayed cameras.
  python -m batchsim.tests.bench [cuda:0] [256,1024,4096]        (mj4 env)"""
import json, os, sys, time
import torch, warp as wp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from batchsim.twin import BatchTwin

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
Ns = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "256,1024,4096").split(",")]
xml = "~/pnp_dataset_mega/scene_980/scene.xml"; out = {}
for N in Ns:
    r = {}
    for cams in ((), ("zed", "wrist_d405")):
        try:
            tw = BatchTwin(xml, N, dev, dr=True, cameras=cams)
            hold = tw.world.qpos[:, :6].clone()[:, None, :].expand(-1, 2, -1)
            tw.command(hold, 1.0); tw.advance(10); tw.observe(); wp.synchronize()
            t0 = time.time()
            for _ in range(5):                                     # 5 decisions of 100 ms: observe, command, advance
                tw.observe(); tw.command(hold, 1.0); tw.advance(10)
            wp.synchronize(); torch.cuda.synchronize(); dt = (time.time() - t0) / 5
            k = "with_cameras" if cams else "state_only"
            r[k] = dict(decisions_per_s=round(N / dt), sim_seconds_per_s=round(N * 0.1 / dt, 1), physics_steps_per_s=round(N * 50 / dt),
                        gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30 + 0, 2), healthy=float(tw.healthy().float().mean()))
            if cams:
                t0 = time.time()
                for _ in range(5): tw.observe()
                wp.synchronize(); torch.cuda.synchronize(); r[k]["observe_ms"] = round(1000 * (time.time() - t0) / 5, 1); r[k]["images_per_s"] = round(2 * N / ((time.time() - t0) / 5))
            del tw
        except Exception as ex:
            r["with_cameras" if cams else "state_only"] = dict(error=repr(ex)[:200])
        torch.cuda.empty_cache()
    out[N] = r; print(N, json.dumps(r), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), f"bench_{dev.replace(':', '')}.json"), "w"), indent=1)

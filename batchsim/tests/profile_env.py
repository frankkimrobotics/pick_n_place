"""Where does a PickLiftEnv decision spend its time?   python -m batchsim.tests.profile_env [cuda:0] [N]"""
import os, sys, time, torch, warp as wp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from batchsim.env import PickLiftEnv
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
env = PickLiftEnv("~/pnp_dataset_mega/scene_980/scene.xml", N, dev, dr=True)
a = torch.zeros(N, 7, device=dev)
for _ in range(3): env.step(a)
tw = env.tw; T = {}
def tm(name, f, n=1):
    wp.synchronize(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    wp.synchronize(); torch.cuda.synchronize(); T[name] = 1000 * (time.time() - t0) / n
tm("env.step (1 decision)", lambda: env.step(a), 3)
tm("world.step x50", lambda: [tw.world.step() for _ in range(50)])
tm("drive.tick x10 (incl. physics)", lambda: [tw.drive.tick() for _ in range(10)])
tm("drive.tick x10 + suction.apply", lambda: [tw.drive.tick(on_substep=tw.suction.apply) for _ in range(10)])
tm("suction.apply x50", lambda: [tw.suction.apply() for _ in range(50)])
tm("suction.update x10", lambda: [tw.suction.update(tw.t) for _ in range(10)])
tm("hist.record x10", lambda: [tw.hist.record() for _ in range(10)])
tm("observe", lambda: env.observe(), 5)
for k, v in T.items(): print(f"{k:36s} {v:8.1f} ms")
print("decisions/s", round(N / (T["env.step (1 decision)"] / 1000)), " healthy", float(tw.healthy().float().mean()))

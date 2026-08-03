"""End-to-end test of the ML data path: synthetic episodes → dataset → train each preset a few
steps → verify the loss drops (obs→action is learnable in the fixture). No robot/sim needed.
Run in an env with torch + cv2 (e.g. dust3r):  python3 il/test_data_train.py"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "policy"))
sys.path.insert(0, os.path.join(HERE, "data"))
import torch
from torch.utils.data import DataLoader
from make_synthetic import gen
from dataset import EpisodeDataset
from policy import build_policy, PRESETS


def train_a_bit(name, ds, steps=100, bs=8, dev="cpu"):
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    pol = build_policy(name, Tp=ds.Tp, To=ds.To).to(dev)
    pol.fit_norm(loader, n_batches=10, dev=dev)
    opt = torch.optim.AdamW(pol.parameters(), 1e-3)
    losses = []; it = iter(loader); s = 0
    while s < steps:
        try:
            obs, act = next(it)
        except StopIteration:
            it = iter(loader); obs, act = next(it)
        obs = {k: v.to(dev) for k, v in obs.items()}; act = act.to(dev)
        loss = pol.loss(obs, act); opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item()); s += 1
    return losses


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(tempfile.gettempdir(), "il_synth_test")
    shutil.rmtree(root, ignore_errors=True)
    gen(root, 8, 40, seed=1)
    ds = EpisodeDataset(root, To=2, Tp=16, img_size=96)
    print(f"dataset: {len(ds.eps)} episodes, {len(ds)} windows  dev={dev}")
    print(f"{'preset':10} {'loss[0]':>9} {'loss[-1]':>9} {'drop%':>7}  ok")
    allok = True
    for name in PRESETS:
        L = train_a_bit(name, ds, steps=120, bs=8, dev=dev)
        s0 = sum(L[:10]) / 10; s1 = sum(L[-10:]) / 10; drop = 100 * (s0 - s1) / max(s0, 1e-6)
        ok = s1 < s0 * 0.9
        allok = allok and ok
        print(f"{name:10} {s0:9.3f} {s1:9.3f} {drop:6.1f}%  {'OK' if ok else '??'}")
    print("ALL LEARNING OK" if allok else "some preset did not reduce loss >10%")


if __name__ == "__main__":
    main()

"""EpisodeDataset — reads canonical episodes (schema.py) into windowed (obs history To, action
horizon Tp) samples for training. Works uniformly over sim + real episodes. Action target = future
absolute joints+suction (DP-style); the umi preset re-expresses it as Δjoint inside the policy."""
import glob
import json
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from schema import CAMS

_SUB = {"wrist_rgb": "wrist", "fixed_rgb": "fixed"}


class EpisodeDataset(Dataset):
    def __init__(self, root, To=2, Tp=16, img_size=96, sources=None, cams=CAMS):
        self.To, self.Tp, self.img, self.cams = To, Tp, img_size, cams
        self.eps = []
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            mp, ap = os.path.join(d, "meta.json"), os.path.join(d, "arrays.npz")
            if not (os.path.isfile(mp) and os.path.isfile(ap)):
                continue
            meta = json.load(open(mp))
            if sources and meta.get("source") not in sources:
                continue
            arr = np.load(ap, allow_pickle=True)
            joints = arr["joints"].astype(np.float32); suc = arr["suction"].astype(np.float32)
            prop = np.concatenate([joints, suc[:, None]], 1)          # (N,7)
            self.eps.append({"dir": d, "n": len(joints), "proprio": prop})
        self.index = [(ei, k) for ei, e in enumerate(self.eps)
                      for k in range(e["n"]) if k >= self.To - 1 and k + self.Tp < e["n"]]
        allp = (np.concatenate([e["proprio"] for e in self.eps], 0)
                if self.eps else np.zeros((1, 7), np.float32))
        self._mean = allp.mean(0).astype(np.float32); self._std = (allp.std(0) + 1e-4).astype(np.float32)
        if not self.index:
            raise ValueError(f"no usable windows in {root} (need To={To} history + Tp={Tp} future per episode)")

    def norm_stats(self):
        return self._mean, self._std

    def _img(self, d, cam, k):
        im = cv2.imread(os.path.join(d, _SUB[cam], f"{k:06d}.jpg"))
        if im is None:
            im = np.zeros((self.img, self.img, 3), np.uint8)
        im = cv2.cvtColor(cv2.resize(im, (self.img, self.img)), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(im).float().permute(2, 0, 1) / 127.5 - 1.0   # (3,H,W) in [-1,1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, k = self.index[i]; e = self.eps[ei]; d = e["dir"]; prop = e["proprio"]
        obs = {c: torch.stack([self._img(d, c, k - self.To + 1 + j) for j in range(self.To)], 0)
               for c in self.cams}
        obs["proprio"] = torch.from_numpy(prop[k - self.To + 1:k + 1]).float()  # (To,7)
        action = torch.from_numpy(prop[k + 1:k + 1 + self.Tp]).float()          # (Tp,7)
        return obs, action

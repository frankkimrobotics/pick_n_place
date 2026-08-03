"""postprocess_picks :: retrofit "picks" records into meta.json for scenes
generated BEFORE dataset_gen recorded them natively (scene_000..099).

Derivation (per scene, from traj.npz + eef.npz):
  - a pick = one suction latch edge (0->1) paired with the next release (1->0)
  - executed grasp pose = tcp FK pose (eef.npz) at the latch frame
  - nominal grasp pose  = reconstructed: executed xy, z + PRESS_M (the plan
    pressed 15 mm past the nominal cup-contact height), top-down quat
  - frame range: the approach lasted 4.3 s (43 frames) before the latch and
    the retreat 0.8 s (8 frames) after the release (fixed segment schedule)
  - success = the object's in_bin outcome, attributed to its LAST latch
Attempts that never sealed are not recoverable here (no latch signature) --
natively recorded scenes (>= 100) include them with fail reasons.
Records carry "reconstructed": true to distinguish them.
"""
import glob
import json
import os
import sys

import numpy as np

DS = os.path.expanduser("~/pnp_dataset")
PRESS_M = 0.015
APPROACH_FRAMES = 43
RETREAT_FRAMES = 8
QUAT_DOWN = [0.0, 0.70711, 0.70711, 0.0]

for sdir in sorted(glob.glob(os.path.join(DS, "scene_*"))):
    mpath = os.path.join(sdir, "meta.json")
    if not (os.path.exists(mpath) and os.path.exists(os.path.join(sdir, "eef.npz"))):
        continue                     # scene still generating / not FK-prepped
    meta = json.load(open(mpath))
    if "picks" in meta:
        continue
    t = np.load(os.path.join(sdir, "traj.npz"))
    e = np.load(os.path.join(sdir, "eef.npz"))
    s, tgt = t["suction"].astype(int), t["target"].astype(int)
    n = len(s)
    latches = (np.where((s[1:] == 1) & (s[:-1] == 0))[0] + 1).tolist()
    releases = (np.where((s[1:] == 0) & (s[:-1] == 1))[0] + 1).tolist()
    last_latch = {}
    for L in latches:
        last_latch[int(tgt[L])] = L
    picks, prev_end = [], 0
    for L in latches:
        R = next((r for r in releases if r > L), n - 1)
        oi = int(tgt[L])
        pos = e["pos"][L]
        picks.append(dict(
            object=oi, attempt=0,
            frame_start=int(max(prev_end, L - APPROACH_FRAMES)),
            frame_latch=int(L), frame_release=int(R),
            frame_end=int(min(n, R + RETREAT_FRAMES)),
            grasp_nominal=dict(
                pos=[round(float(v), 5) for v in
                     (pos[0], pos[1], pos[2] + PRESS_M)],
                press_pos=[round(float(v), 5) for v in pos],
                quat=QUAT_DOWN),
            grasp_executed=dict(
                pos=[round(float(v), 5) for v in pos],
                quat=[round(float(v), 5) for v in e["quat"][L]]),
            success=bool(meta["objects"][oi]["in_bin"] and last_latch[oi] == L),
            fail_reason=None, reconstructed=True))
        prev_end = picks[-1]["frame_end"]
    meta["picks"] = picks
    meta["picks_reconstructed"] = True
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=1)
    print(os.path.basename(sdir), len(picks), "picks", flush=True)
print("done")

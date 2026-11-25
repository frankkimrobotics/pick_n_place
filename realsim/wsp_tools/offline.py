"""offline scoring of a weight patch: (1) twin latched states C -> keeps suction + lifts?
(2) open-loop regression on recorded demo frames (pos / suction error vs the unpatched net)."""
from wsp_lib import *
import json, copy
from realsim.wsp import apply_patch
def demo_set(scenes=range(990, 1000), every=9):
    out = []
    for s in scenes:
        sd = os.path.expanduser(f"~/pnp_dataset_mega/scene_{s:03d}")
        e = np.load(sd + "/eef.npz"); meta = json.load(open(sd + "/meta.json"))
        tk = np.load(sd + "/cradio_pool.npy").astype(np.float32); tz = np.load(sd + "/cradio_zed_pool.npy").astype(np.float32)
        pos, R, suc = e["pos"], T.quat_to_R(e["quat"]), e["suction"].astype(np.float32)
        for p in meta["picks"]:
            if not p.get("grasp_nominal"): continue
            g, gq = np.array(p["grasp_nominal"]["pos"]), np.array(p["grasp_nominal"]["quat"])
            f1 = p["frame_end"] if p["frame_end"] > 0 else len(pos)
            for t in range(p["frame_start"] + 1, min(f1, len(pos) - 20), every):
                prop = np.stack([RZ.rel_pose(pos[t], R[t], pos[t-1], R[t-1], suc[t-1]), RZ.rel_pose(pos[t], R[t], pos[t], R[t], suc[t])])
                grel = np.concatenate([R[t].T @ (g - pos[t]), T.rot6d(R[t].T @ T.quat_to_R(gq))])
                gt = np.stack([np.r_[R[t].T @ (pos[t+3+k] - pos[t]), suc[t+3+k]] for k in range(16)])
                out.append(dict(tok=np.stack([tk[t-1], tk[t], tz[t-1], tz[t]]), prop=prop, goal=grel, gt=gt, ok=p["success"]))
    return out
def c_set(d="~/pnp_rollouts_realsim/wsp/base_dev"):
    out = []
    for sc, t in traces(d).items():
        lat = t["dec_latched"]
        for j, k in enumerate(t["inf_k"]):
            di = int(np.where(t["dec_k"] == k)[0][0])
            if lat[di] and t["dec_comp"][di] > 0.003 and t["dec_obj_z"][di] < 0.01:
                out.append(dict(tok=t["inf_tok"][j].astype(np.float32).mean(1), prop=t["inf_prop"][j], goal=t["inf_goal"][j], R=t["inf_R"][j]))
    return out
def score(pol, D, Cs, ref=None):
    acts = [run(pol, x["tok"], x["prop"], x["goal"]) for x in D]
    cact = [run(pol, x["tok"], x["prop"], x["goal"]) for x in Cs]
    r = dict(pos_mm=1e3*np.mean([np.linalg.norm(a[:, :3]-x["gt"][:, :3], axis=1).mean() for a, x in zip(acts, D)]),
             suc_err=np.mean([np.abs((a[:, 9] > .5) - x["gt"][:, 3]).mean() for a, x in zip(acts, D)]),
             C_suc=np.mean([a[:8, 9].mean() for a in cact]), C_keep=np.mean([a[:8, 9].min() > 0.5 for a in cact]),
             C_dz_mm=1e3*np.mean([(x["R"] @ a[15, :3])[2] for a, x in zip(cact, Cs)]))
    if ref is not None:
        r["drift_mm"] = 1e3*np.mean([np.linalg.norm(a[:, :3]-b[:, :3], axis=1).mean() for a, b in zip(acts, ref)])
        r["suc_flip"] = np.mean([((a[:, 9] > .5) != (b[:, 9] > .5)).mean() for a, b in zip(acts, ref)])
    return r, acts
def phase_set(kind, scenes, every=1):
    """demo states of one phase: 'release' = frames [release-14, release+1] of successful picks,
    'carry' = latch+3 .. release-15"""
    out = []
    for s in scenes:
        sd = os.path.expanduser(f"~/pnp_dataset_mega/scene_{s:03d}")
        e = np.load(sd + "/eef.npz"); meta = json.load(open(sd + "/meta.json"))
        tk = np.load(sd + "/cradio_pool.npy").astype(np.float32); tz = np.load(sd + "/cradio_zed_pool.npy").astype(np.float32)
        pos, R, suc = e["pos"], T.quat_to_R(e["quat"]), e["suction"].astype(np.float32)
        for p in meta["picks"]:
            if not (p.get("grasp_nominal") and p["success"] and p.get("frame_release", -1) > 0): continue
            g, gq = np.array(p["grasp_nominal"]["pos"]), np.array(p["grasp_nominal"]["quat"])
            fr = range(p["frame_release"] - 14, p["frame_release"] + 2) if kind == "release" else range(p["frame_latch"] + 3, p["frame_release"] - 14)
            for t in list(fr)[::every]:
                if t < 1 or t + 20 >= len(pos): continue
                prop = np.stack([RZ.rel_pose(pos[t], R[t], pos[t-1], R[t-1], suc[t-1]), RZ.rel_pose(pos[t], R[t], pos[t], R[t], suc[t])])
                grel = np.concatenate([R[t].T @ (g - pos[t]), T.rot6d(R[t].T @ T.quat_to_R(gq))])
                gt = np.stack([np.r_[R[t].T @ (pos[t+3+k] - pos[t]), suc[t+3+k]] for k in range(16)])
                out.append(dict(tok=np.stack([tk[t-1], tk[t], tz[t-1], tz[t]]), prop=prop, goal=grel, gt=gt, ok=True))
    return out
def phase_score(pol, D, ref=None):
    acts = [run(pol, x["tok"], x["prop"], x["goal"]) for x in D]
    r = dict(pos_mm=1e3*np.mean([np.linalg.norm(a[:, :3]-x["gt"][:, :3], axis=1).mean() for a, x in zip(acts, D)]),
             suc_err=np.mean([np.abs((a[:, 9] > .5) - x["gt"][:, 3]).mean() for a, x in zip(acts, D)]))
    if ref is not None: r["drift_mm"] = 1e3*np.mean([np.linalg.norm(a[:, :3]-b[:, :3], axis=1).mean() for a, b in zip(acts, ref)])
    return r, acts

from wsp_lib import *
import json
np.set_printoptions(precision=3, suppress=True, linewidth=220)
pol = load(); enc = pol.obs_enc; TR = traces("~/pnp_rollouts_realsim/wsp/base_dev")
print("trace tok shape", TR[980]["inf_tok"].shape)
@torch.no_grad()
def proj(x): return enc.proj(torch.from_numpy(np.asarray(x, np.float32)).to(RZ.DEV)).cpu().numpy()
A, B, N, Bz, Az = [], [], [], [], []
for s in range(0, 120):
    sd = os.path.expanduser(f"~/pnp_dataset_mega/scene_{s:03d}")
    tk = np.load(sd + "/cradio_pool.npy", mmap_mode="r"); e = np.load(sd + "/eef.npz"); m = json.load(open(sd + "/meta.json"))
    suc = e["suction"]; z = e["pos"][:, 2]
    N.append(np.asarray(tk[::7], np.float32))
    for p in m["picks"]:
        if not p.get("grasp_nominal"): continue
        f0, f1 = p["frame_start"], (p["frame_end"] if p["frame_end"] > 0 else len(z)); gz = p["grasp_nominal"]["pos"][2]
        if p["success"] and p["frame_latch"] > 0:
            fr = [p["frame_latch"] - 1, p["frame_latch"]]
            A.append(np.asarray(tk[fr], np.float32)); Az += [z[f] - gz for f in fr]
        elif not p["success"]:
            on = np.where(suc[f0:f1] > 0.5)[0]
            if len(on) > 4:
                fr = [f0 + on[-2], f0 + on[-1]]
                B.append(np.asarray(tk[fr], np.float32)); Bz += [z[f] - gz for f in fr]
A, B, N = np.concatenate(A), np.concatenate(B), np.concatenate(N)
C = []
for sc, t in TR.items():
    lat = t["dec_latched"]
    for j, k in enumerate(t["inf_k"]):
        di = int(np.where(t["dec_k"] == k)[0][0])
        if lat[di] and t["dec_comp"][di] > 0.003 and t["dec_obj_z"][di] < 0.01:
            tok = t["inf_tok"][j].astype(np.float32); C.append(tok[1] if tok.ndim == 2 else tok[1].mean(0))
C = np.stack(C)
print("A touch-latch", A.shape, "z-goal mm %.1f" % (1e3*np.mean(Az)), "| B failed-press end", B.shape, "z-goal mm %.1f" % (1e3*np.mean(Bz)), "| C twin latched", C.shape, "| N", N.shape)
fA, fB, fC, fN = proj(A), proj(B), proj(C), proj(N)
mu = {k: v.mean(0) for k, v in dict(A=fA, B=fB, C=fC, N=fN).items()}
cs = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
print("|muB-muA| %.3f |muC-muA| %.3f |muC-muB| %.3f  typical within-A std norm %.3f" % (np.linalg.norm(mu['B']-mu['A']), np.linalg.norm(mu['C']-mu['A']), np.linalg.norm(mu['C']-mu['B']), np.linalg.norm(fA.std(0))))
print("cos(muB-muA, muC-muA) %.3f" % cs(mu['B']-mu['A'], mu['C']-mu['A']))
# LDA direction A vs B, whitened by pooled within covariance
Sw = np.cov(np.concatenate([fA-mu['A'], fB-mu['B']]).T) + 1e-3*np.eye(256)
w = np.linalg.solve(Sw, mu['B']-mu['A']); w /= np.linalg.norm(w)
for n, f in dict(A=fA, B=fB, C=fC, N=fN).items(): print(f"  LDA(A|B) projection {n}: mean {float((f@w).mean()):+.3f} sd {float((f@w).std()):.3f}")
thr = ((fA@w).mean() + (fB@w).mean())/2
print("  A below thr %.2f  B above thr %.2f  C above thr %.2f" % ((fA@w < thr).mean(), (fB@w > thr).mean(), (fC@w > thr).mean()))
np.savez("geom.npz", A=A, B=B, C=C, N=N, w=w)

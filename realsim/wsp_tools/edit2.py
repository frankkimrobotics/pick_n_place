"""E2: closed-form rank-n edit of obs_enc.head.2 (MEMIT-style, no gradient steps).
keys  k_i = Mish(head.0 x_i) at the twin's sealed-and-over-pressed states,
values v_i = cond the net itself produces when the wrist view is swapped for the pre-press (touch) view."""
from offline import *
import sys
def latched_pairs(dirs):
    out = []
    for d in dirs:
        for sc, t in traces(d).items():
            lat = t["dec_latched"]; di = {int(k): i for i, k in enumerate(t["dec_k"])}
            for j, k in enumerate(t["inf_k"]):
                i = di[int(k)]
                if lat[i] and t["dec_comp"][i] > 0.003 and t["dec_obj_z"][i] < 0.01:
                    tok = t["inf_tok"][j].astype(np.float32).mean(1)
                    cands = [t["inf_tok"][jj].astype(np.float32).mean(1) for jj in (j-1, j-2, j-3) if jj >= 0 and not lat[di[int(t["inf_k"][jj])]]]
                    out.append(dict(sc=sc, tok=tok, prop=t["inf_prop"][j], goal=t["inf_goal"][j], R=t["inf_R"][j], cands=cands))
    return out
@torch.no_grad()
def hid(enc, tok, prop, goal):
    tt = torch.from_numpy(tok)[None].to(RZ.DEV); f = enc.proj(tt)
    x = torch.cat([f.flatten(1), torch.from_numpy(np.asarray(prop, np.float32))[None].to(RZ.DEV).flatten(1), torch.from_numpy(np.asarray(goal, np.float32))[None].to(RZ.DEV)], -1)
    return enc.head[1](enc.head[0](x))[0]
def targets(pol, Cs):
    """per state: (key, value) with value from the first counterfactual that keeps suction and lifts"""
    enc = pol.obs_enc; K, V, used = [], [], []
    for x in Cs:
        for c in x["cands"]:
            tk = np.stack([c[1], c[1], x["tok"][2], x["tok"][3]])
            a = run(pol, tk, x["prop"], x["goal"])
            if a[:8, 9].min() > 0.5 and (x["R"] @ a[15, :3])[2] > 0.04:
                K.append(hid(enc, x["tok"], x["prop"], x["goal"])); V.append(enc.head[2](hid(enc, tk, x["prop"], x["goal"]))); used.append(x); break
    return torch.stack(K, 1), torch.stack(V, 1), used
def solve(enc, K, V, C0, lam, W0=None):
    W = (enc.head[2].weight.detach() if W0 is None else W0).double(); b = enc.head[2].bias.detach().double()
    Rm = V - (W @ K + b[:, None])
    return (Rm @ K.T @ torch.linalg.inv(lam * C0 + K @ K.T)).float()
def cov0(enc, scenes=range(0, 900, 6), every=5):
    """second moment of the head.2 keys over ordinary demo states (what the edit must leave alone) + 1 % ridge"""
    Dn = demo_set(scenes, every)
    Kd = torch.stack([hid(enc, x["tok"], x["prop"], x["goal"]) for x in Dn], 1).double()
    C0 = Kd @ Kd.T / Kd.shape[1]
    print("C0 from", Kd.shape[1], "demo states", flush=True)
    return C0 + 0.01 * torch.trace(C0) / C0.shape[0] * torch.eye(C0.shape[0], dtype=C0.dtype, device=C0.device)
if __name__ == "__main__":
    pol = load(); enc = pol.obs_enc; D = demo_set()
    Cs = latched_pairs(sys.argv[1:]); scs = sorted({x["sc"] for x in Cs}); print("C states", len(Cs), "scenes", scs)
    C0 = cov0(enc)
    base, ref = score(pol, D, Cs); print("base", {k: round(float(v), 3) for k, v in base.items()})
    W0 = enc.head[2].weight.detach().clone()
    K, V, used = targets(pol, Cs); print("states with a working counterfactual:", len(used), "/", len(Cs))
    tr = [i for i, x in enumerate(used) if scs.index(x["sc"]) % 2 == 0]; te_sc = [s for i, s in enumerate(scs) if i % 2 == 1]
    Cte = [x for x in Cs if x["sc"] in te_sc]
    for lam in (0.03, 0.3, 3.0, 30.0):
        dW = solve(enc, K[:, tr].double(), V[:, tr].double(), C0, lam * len(tr), W0)
        with torch.no_grad(): enc.head[2].weight.copy_(W0 + dW)
        r, _ = score(pol, D, Cte, ref)
        print(f"lam {lam}: fit on {len(tr)} states, HELD-OUT scenes {te_sc} n {len(Cte)} |dW|/|W| {float(dW.norm()/W0.norm()):.3f}", {k: round(float(v), 3) for k, v in r.items()}, flush=True)
    with torch.no_grad(): enc.head[2].weight.copy_(W0)

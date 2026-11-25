"""E3 = E2 + explicit preservation of the TWIN's own non-target states (approach, carry, release):
C0 = demo second moment + beta * twin second moment, so dW k ~ 0 on the states that already work."""
from edit2 import *
name, lams, beta = sys.argv[1], [float(x) for x in sys.argv[2].split(",")], float(sys.argv[3])
fit_dirs = [d for d in sys.argv[4:] if not d.startswith("+")]; keep_dirs = fit_dirs + [d[1:] for d in sys.argv[4:] if d.startswith("+")]
pol = load(); enc = pol.obs_enc; D = demo_set()
def twin_keep(dirs):
    out = []
    for d in dirs:
        for sc, t in traces(d).items():
            lat = t["dec_latched"]; di = {int(k): i for i, k in enumerate(t["dec_k"])}
            for j, k in enumerate(t["inf_k"]):
                i = di[int(k)]
                if lat[i] and t["dec_obj_z"][i] < 0.03: continue          # the edit's own target region
                if t["dec_comp"][i] > 0.001 or (t["dec_cmd"][i] == 1 and not lat[i]): continue   # pressing, seal unknowable: same view as the targets
                if t["dec_hold"][i] == "ik": continue                      # stuck states: nothing worth preserving
                out.append(dict(tok=t["inf_tok"][j].astype(np.float32).mean(1), prop=t["inf_prop"][j], goal=t["inf_goal"][j],
                                R=t["inf_R"][j], carry=bool(lat[i])))
    return out
Cs = latched_pairs(fit_dirs); TK = twin_keep(keep_dirs); print("target states", len(Cs), "| twin keep states", len(TK), "of which carry", sum(x["carry"] for x in TK))
C0 = cov0(enc); Kt = torch.stack([hid(enc, x["tok"], x["prop"], x["goal"]) for x in TK], 1).double()
Kc = torch.stack([hid(enc, x["tok"], x["prop"], x["goal"]) for x in TK if x["carry"]], 1).double()
C0 = C0 + beta * (Kt @ Kt.T / Kt.shape[1]) + beta * (Kc @ Kc.T / Kc.shape[1])
W0 = enc.head[2].weight.detach().clone()
base, ref = score(pol, D, Cs); print("base", {k: round(float(v), 3) for k, v in base.items()})
tkc = [x for x in TK if x["carry"]][::3]; ref_c = [run(pol, x["tok"], x["prop"], x["goal"]) for x in tkc]
K, V, used = targets(pol, Cs); print("keys", K.shape[1])
for lam in lams:
    dW = solve(enc, K.double(), V.double(), C0, lam * K.shape[1], W0)
    with torch.no_grad(): enc.head[2].weight.copy_(W0 + dW)
    r, _ = score(pol, D, Cs, ref); sv = torch.linalg.svdvals(dW)
    ac = [run(pol, x["tok"], x["prop"], x["goal"]) for x in tkc]
    cd = 1e3*np.mean([np.linalg.norm(a[:, :3]-b[:, :3], axis=1).mean() for a, b in zip(ac, ref_c)]); cf = np.mean([((a[:, 9] > .5) != (b[:, 9] > .5)).mean() for a, b in zip(ac, ref_c)])
    print(f"lam {lam} beta {beta}: |dW|/|W| {float(dW.norm()/W0.norm()):.3f} eff.rank {float(sv.sum()**2/(sv**2).sum()):.1f}", {k: round(float(v), 3) for k, v in r.items()}, f"| twin carry drift {cd:.1f} mm suc_flip {cf:.3f}", flush=True)
    torch.save(dict(ops=[dict(key="obs_enc.head.2.weight", kind="add", value=dW.cpu())],
                    note=f"E3 head.2 closed-form edit, lam {lam}, beta {beta}, {K.shape[1]} keys, twin-preserving"),
               os.path.expanduser(f"~/pnp_rollouts_realsim/wsp/patches/{name}_lam{lam:g}_b{beta:g}.pt"))

from wsp_lib import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
pol = load(); TR = traces("~/pnp_rollouts_realsim/wsp/base_dev")
def s(a, R): return f"suc mean {a[:,9].mean():.2f}  dz15 {1e3*(R@a[15,:3])[2]:+.0f}"
for sc in (980, 982, 988, 981, 989):
    t = TR[sc]; lat = t["dec_latched"]; k0 = int(np.argmax(lat))
    js = [j for j, k in enumerate(t["inf_k"]) if k >= k0 and lat[np.where(t["dec_k"] == k)[0][0]]]
    if not js: continue
    j = js[0]; jp = j - 1                      # jp: the inference before the seal (approach / touch)
    tok, prop, goal, R = t["inf_tok"][j], t["inf_prop"][j], t["inf_goal"][j].copy(), t["inf_R"][j]
    di = int(np.where(t["dec_k"] == t["inf_k"][j])[0][0]); dp = int(np.where(t["dec_k"] == t["inf_k"][jp])[0][0])
    print(f"sc {sc}: latched inf k {t['inf_k'][j]} comp {1e3*t['dec_comp'][di]:.0f} mm goal_rel_z {goal[2]:+.3f} | prev inf k {t['inf_k'][jp]} comp {1e3*t['dec_comp'][dp]:.0f} mm goal_rel_z {t['inf_goal'][jp][2]:+.3f}")
    tp = t["inf_tok"][jp]
    g0 = goal.copy(); g0[2] = 0.0
    variants = {
        "base": (tok, goal), "goal z=0": (tok, g0),
        "wrist<-prev": (np.stack([tp[1], tp[1], tok[2], tok[3]]), goal),
        "zed<-prev": (np.stack([tok[0], tok[1], tp[3], tp[3]]), goal),
        "all img<-prev": (np.stack([tp[1], tp[1], tp[3], tp[3]]), goal),
        "wrist<-prev + goal z=0": (np.stack([tp[1], tp[1], tok[2], tok[3]]), g0),
        "all img<-prev + goal z=0": (np.stack([tp[1], tp[1], tp[3], tp[3]]), g0),
    }
    for n, (tk, g) in variants.items():
        print(f"    {n:26s}", s(run(pol, tk, prop, g), R))
    # sweep goal z with prev images
    print("    goal z sweep (all img<-prev):", [(gz, round(float(run(pol, np.stack([tp[1], tp[1], tp[3], tp[3]]), prop, np.r_[goal[:2], gz, goal[3:]])[:,9].mean()),2)) for gz in (0.01,0.005,0,-0.005,-0.01,-0.02)])
    print("    goal z sweep (real img):     ", [(gz, round(float(run(pol, tok, prop, np.r_[goal[:2], gz, goal[3:]])[:,9].mean()),2)) for gz in (0.01,0.005,0,-0.005,-0.01,-0.02)])

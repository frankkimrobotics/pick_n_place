import numpy as np, sys, glob, json, os
sys.path.insert(0, os.path.dirname(__file__)); from look import load
D="/home/lisc-frank/pnp_real_runs/sysid"
for spec in sys.argv[1:]:
    j,tag=spec.split(":"); j=int(j)
    fs=sorted(glob.glob(f"{D}/*check_j{j}_{tag}_pi.csv"))
    if not fs: print(j,tag,"no log"); continue
    f=fs[-1]; L=load(f); ref=np.load(f[:-7]+"_ref.npy"); m=json.load(open(f[:-7]+"_meta.json")); tr=m["t_anchor"]+np.arange(len(ref))*0.01
    t=L["t"]; q=L["q"][:,j]-ref[0,j]; r=np.interp(t,tr,ref[:,j])-ref[0,j]; Dp=r.max()
    k=(t>tr[50])&(t<tr[50]+1.0)
    if not k.any(): print(j,tag,"log does not cover the run"); continue
    i_end=np.argmax(ref[:,j]-ref[0,j]>=Dp-1e-6); t0=tr[i_end]
    kk=np.where((t>t0)&(np.abs(q-Dp)>0.05)&(t<t0+0.9))[0]; settle=(t[kk[-1]]-t0) if len(kk) else 0
    e=q-r; mm=(t>tr[0])&(t<tr[-1])
    print(f"J{j+1} {tag}: overshoot {q[k].max()-Dp:+.3f} deg, settle(0.05 deg) {1e3*settle:.0f} ms after ramp end, rms err {np.sqrt((e[mm]**2).mean()):.3f}, max {np.abs(e[mm]).max():.3f}, rest jitter {q[(t>t0+0.6)&(t<t0+0.9)].std():.4f}")

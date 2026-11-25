import numpy as np, sys, glob, json, os
sys.path.insert(0, os.path.dirname(__file__)); from look import load
D="/home/lisc-frank/pnp_real_runs/sysid"
for tag in sys.argv[1:]:
    f=sorted(glob.glob(f"{D}/*{tag}_pi.csv"))[-1]; L=load(f); ref=np.load(f[:-7]+"_ref.npy"); m=json.load(open(f[:-7]+"_meta.json"))
    tr=m["t_anchor"]+np.arange(len(ref))*0.01; t=L["t"]; k=(t>tr[0])&(t<tr[-1])
    for j in range(6):
        if np.ptp(ref[:,j])<0.5: continue
        r=np.interp(t[k],tr,ref[:,j]); v=np.interp(t[k],tr,np.gradient(ref[:,j],0.01)); a=np.interp(t[k],tr,np.gradient(np.gradient(ref[:,j],0.01),0.01)); e=L["q"][k,j]-r
        A=np.c_[np.ones(k.sum()),v,a,np.tanh(v/0.3)]; c=np.linalg.lstsq(A,e,rcond=None)[0]
        print(f"{tag} J{j+1}: rms {np.sqrt((e**2).mean()):.3f} max {np.abs(e).max():.3f} | err = {c[0]:+.3f} {c[1]:+.4f}*v {c[2]:+.5f}*a {c[3]:+.3f}*sign(v) | residual {(e-A@c).std():.3f}")

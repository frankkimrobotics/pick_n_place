import numpy as np, glob, csv, sys
def load(f):
    r=list(csv.DictReader(open(f))); g=lambda k:np.array([float(x[k]) for x in r])
    return dict(t=g("t_epoch"), q=np.stack([g(f"q{i}") for i in range(6)],1), cv=np.stack([g(f"cmd_vel{i}") for i in range(6)],1)/17.0,
                hv=np.stack([g(f"hal_vel{i}") for i in range(6)],1), tq=np.stack([g(f"hal_torq{i}") for i in range(6)],1), ref=np.stack([g(f"target{i}") for i in range(6)],1))
if __name__=="__main__":
    f=sorted(glob.glob(f"/home/lisc-frank/pnp_real_runs/sysid/*{sys.argv[1]}_pi.csv"))[-1]; j=int(sys.argv[2]); L=load(f); t=L["t"]-L["t"][0]
    # resample to 10 ms
    tt=np.arange(0,t[-1],0.01); q=np.interp(tt,t,L["q"][:,j]); cv=np.interp(tt,t,L["cv"][:,j]); tq=np.interp(tt,t,L["tq"][:,j]); hv=np.interp(tt,t,L["hv"][:,j])
    v=np.gradient(q,0.01)
    print("others max dev", np.round(np.abs(L["q"]-L["q"][0]).max(0),3), "net", round(q[-1]-q[0],3))
    edges=np.where(np.abs(np.diff(cv))>0.3)[0]
    for e in edges:
        a,b=cv[e],cv[e+1]
        seg=slice(e+1,min(e+1+60,len(tt)))
        vv=v[seg]; dv=vv-v[max(e-3,0):e+1].mean()
        thr=0.1*abs(b-a)
        on=np.argmax(np.abs(dv)>max(thr,0.08)) if (np.abs(dv)>max(thr,0.08)).any() else -1
        t63=np.argmax(np.abs(dv)>0.63*abs(b-a)) if (np.abs(dv)>0.63*abs(b-a)).any() else -1
        # steady: last 40% of the plateau
        nxt=edges[edges>e][0] if (edges>e).any() else len(tt)-1
        ss=slice(e+1+int(0.6*(nxt-e)),nxt)
        print(f"t {tt[e]:5.2f} cmd {a:+6.2f}->{b:+6.2f}  onset {10*on:4d} ms  t63 {10*t63:4d} ms  v_ss {v[ss].mean():+7.3f} (velfb {hv[ss].mean():+7.3f})  torq_ss {tq[ss].mean():+.4f}  peak|a| {np.abs(np.gradient(vv,0.01)).max():6.0f}")

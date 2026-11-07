#!/usr/bin/env python3
"""seal_sac :: asymmetric SAC for the seal controller.
Actor sees DEPLOYABLE obs only; both critics see deployable + PRIVILEGED (plunger force,
attach state, seal-window params) — force shapes learning but never reaches the actor."""
import os, sys, time, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from seal_env import SealEnv

K_STACK = 8                       # 160 ms of history: covers proxy latency + contact memory
DEP1, PRIV, ACT = 23, 6, 3
DEP = DEP1 * K_STACK

class ObsStack:
    def __init__(self): self.buf=[]
    def reset(self, o):
        self.buf=[o]*K_STACK; return np.concatenate(self.buf)
    def push(self, o):
        self.buf.pop(0); self.buf.append(o); return np.concatenate(self.buf)

class MLP(nn.Module):
    def __init__(self, i, o, h=256):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(i,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,o))
    def forward(self,x): return self.f(x)

class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.net=MLP(DEP,ACT*2)
        with torch.no_grad():                      # low initial exploration noise:
            self.net.f[-1].bias[ACT:] -= 1.5       # std ~0.22 pre-tanh, not ~1.0
    def forward(self,x,deterministic=False):
        mu,log_s=self.net(x).chunk(2,-1)
        log_s=log_s.clamp(-5,2)
        if deterministic: a=torch.tanh(mu); lp=None
        else:
            d=torch.distributions.Normal(mu,log_s.exp())
            z=d.rsample(); a=torch.tanh(z)
            lp=(d.log_prob(z)-torch.log1p(-a.pow(2)+1e-6)).sum(-1,keepdim=True)
        return a,lp

class ActGate:
    """hysteresis on the binary channels so sampling noise cannot flicker
    suction (env detaches on every off-edge) or the lift trigger."""
    def __init__(self): self.s=0.0; self.l=0.0
    def reset(self): self.s=0.0; self.l=0.0
    def __call__(self,a):
        if a[1]> 0.2: self.s=1.0
        elif a[1]<-0.2: self.s=0.0
        if a[2]> 0.2: self.l=1.0            # lift is latching by design
        return np.array([float(a[0])*30.0, self.s, self.l])

def scale(a, gate=ActGate()):
    return gate(a)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=150000)
    ap.add_argument("--out",default=os.path.expanduser("~/pnp_rl/seal_sac1"))
    ap.add_argument("--gpu",type=int,default=1)
    a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    dev=f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    actor=Actor().to(dev)
    q1=MLP(DEP+PRIV+ACT,1).to(dev); q2=MLP(DEP+PRIV+ACT,1).to(dev)
    q1t=MLP(DEP+PRIV+ACT,1).to(dev); q2t=MLP(DEP+PRIV+ACT,1).to(dev)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    opt_a=torch.optim.Adam(actor.parameters(),3e-4)
    opt_q=torch.optim.Adam(list(q1.parameters())+list(q2.parameters()),3e-4)
    log_alpha=torch.full((1,),-3.0,device=dev,requires_grad=True)   # alpha=0.05: do not blast the BC init
    opt_al=torch.optim.Adam([log_alpha],3e-4); target_ent=-ACT
    N=200000
    buf=dict(dep=np.zeros((N,DEP),np.float32),priv=np.zeros((N,PRIV),np.float32),
             act=np.zeros((N,ACT),np.float32),rew=np.zeros((N,1),np.float32),
             dep2=np.zeros((N,DEP),np.float32),priv2=np.zeros((N,PRIV),np.float32),
             done=np.zeros((N,1),np.float32))
    ptr=0; full=False
    env=SealEnv(seed=0)
    log=open(os.path.join(a.out,"log.jsonl"),"w")
    # ---- demo seeding: noisy force-aware scripted expert (privileged F only used to ACT
    # here, which is fine: demos are a training-time device, the actor never sees F)
    n_demo=0; demo_lift=0
    stk=ObsStack()
    for epd in range(150):
        dep1,priv=env.reset(); dep=stk.reset(dep1)
        contact=False; press_end=None; suct_t=None; lift_t=None; attempt=0
        for k in range(175):
            Fc=priv[1]
            if not contact and Fc>1.0:
                contact=True; press_end=env.t+np.random.uniform(0.15,0.3)
            if not contact: ta=[-25.0+np.random.normal(0,3),0,0]
            elif env.t<press_end: ta=[-20.0+np.random.normal(0,3),0,0]
            else:
                if suct_t is None: suct_t=env.t
                dwell=0.5+0.4*attempt
                if env.t<suct_t+dwell: ta=[np.random.normal(0,2),1,0]
                elif lift_t is None or env.t<lift_t+0.45:
                    if lift_t is None: lift_t=env.t
                    ta=[np.random.normal(0,2),1,1]
                else:
                    attempt+=1; contact=False; press_end=None; suct_t=None; lift_t=None
                    ta=[-25.0,0,0]
            act_t=np.array([np.clip(ta[0]/30.0,-0.999,0.999),
                            0.98 if ta[1]>0.5 else -0.98, 0.98 if ta[2]>0.5 else -0.98])
            (dep21,priv2),r,done,info=env.step(np.array(ta,float))
            dep2=stk.push(dep21)
            buf["dep"][ptr]=dep; buf["priv"][ptr]=priv; buf["act"][ptr]=act_t
            buf["rew"][ptr]=r; buf["dep2"][ptr]=dep2; buf["priv2"][ptr]=priv2; buf["done"][ptr]=float(done)
            ptr=(ptr+1)%N; n_demo+=1
            dep,priv=dep2,priv2
            if done:
                demo_lift+=(info.get("end")=="lifted"); break
    log.write(json.dumps({"demo_transitions":n_demo,"demo_lifted":demo_lift})+"\n"); log.flush()
    n_demo_fixed=n_demo
    # class-rebalanced BC pretrain (lift frames are rare)
    wts=np.ones(n_demo_fixed,np.float32)
    wts[buf["act"][:n_demo_fixed,2]>0]=10.0
    wts[buf["act"][:n_demo_fixed,1]>0]=np.maximum(wts[buf["act"][:n_demo_fixed,1]>0],3.0)
    pw=wts/wts.sum()
    for it in range(4000):
        i=np.random.choice(n_demo_fixed,256,p=pw)
        bd=torch.from_numpy(buf["dep"][i]).to(dev); ba=torch.from_numpy(buf["act"][i]).to(dev)
        mu,_=actor.net(bd).chunk(2,-1)
        lbc=F.mse_loss(torch.tanh(mu),ba)
        opt_a.zero_grad(); lbc.backward(); opt_a.step()
    log.write(json.dumps({"bc_final_loss":round(float(lbc),5)})+"\n"); log.flush()
    gate=ActGate(); stk=ObsStack()
    def det_eval(n=10):
        det=0; gg=ActGate(); ds=ObsStack()
        for _ in range(n):
            dd1,pp=env.reset(); dd=ds.reset(dd1); gg.reset()
            for _k in range(175):
                with torch.no_grad():
                    ad,_=actor(torch.from_numpy(dd[None]).to(dev),deterministic=True)
                (dd1,pp),rr,dn,ii=env.step(gg(ad.cpu().numpy()[0]))
                dd=ds.push(dd1)
                if dn: det+=(ii.get("end")=="lifted"); break
        return det
    d0=det_eval(20)
    log.write(json.dumps({"det_post_bc":d0,"of":20})+"\n"); log.flush()
    dep1,priv=env.reset(); dep=stk.reset(dep1); gate.reset(); ep_r=0; ep_n=0; outcomes={}
    t0=time.time()
    for step in range(a.steps):
        if step<2000: act_t=np.random.uniform(-1,1,ACT)
        else:
            with torch.no_grad():
                act_t,_=actor(torch.from_numpy(dep[None]).to(dev)); act_t=act_t.cpu().numpy()[0]
        (dep21,priv2),r,done,info=env.step(gate(act_t))
        dep2=stk.push(dep21)
        buf["dep"][ptr]=dep; buf["priv"][ptr]=priv; buf["act"][ptr]=act_t
        buf["rew"][ptr]=r; buf["dep2"][ptr]=dep2; buf["priv2"][ptr]=priv2; buf["done"][ptr]=float(done)
        ptr=(ptr+1)%N; full=full or ptr==0
        dep,priv=dep2,priv2; ep_r+=r
        if done:
            e=info.get("end","?"); outcomes[e]=outcomes.get(e,0)+1; ep_n+=1
            if ep_n%50==0:
                tot=sum(outcomes.values()); sr=outcomes.get("lifted",0)/max(tot,1)
                # deterministic eval: 10 episodes with tanh(mu) + fresh gate
                det=det_eval(10)
                log.write(json.dumps({"step":step,"ep":ep_n,"seal_rate":round(sr,3),
                    "det10":det,"outcomes":outcomes,"wall":round(time.time()-t0)})+"\n"); log.flush()
                outcomes={}
                dep1,priv=env.reset(); dep=stk.reset(dep1); gate.reset()
            dep1,priv=env.reset(); dep=stk.reset(dep1); gate.reset(); ep_r=0
        if step>2000 and step%1==0:
            n=N if full else ptr
            i=np.random.randint(0,n,256)
            b={k:torch.from_numpy(v[i]).to(dev) for k,v in buf.items()}
            with torch.no_grad():
                a2,lp2=actor(b["dep2"])
                qin2=torch.cat([b["dep2"],b["priv2"],a2],-1)
                tq=torch.min(q1t(qin2),q2t(qin2))-log_alpha.exp()*lp2
                y=b["rew"]+0.98*(1-b["done"])*tq
            qin=torch.cat([b["dep"],b["priv"],b["act"]],-1)
            lq=F.mse_loss(q1(qin),y)+F.mse_loss(q2(qin),y)
            opt_q.zero_grad(); lq.backward(); opt_q.step()
            if step<6000:
                with torch.no_grad():
                    for tp_,sp_ in ((q1t,q1),(q2t,q2)):
                        for p_,ps_ in zip(tp_.parameters(),sp_.parameters()):
                            p_.mul_(0.995).add_(ps_,alpha=0.005)
                continue
            an,lpn=actor(b["dep"])
            qa=torch.min(q1(torch.cat([b["dep"],b["priv"],an],-1)),
                         q2(torch.cat([b["dep"],b["priv"],an],-1)))
            la=(log_alpha.exp().detach()*lpn-qa).mean()
            # TD3+BC-style anchor on demo samples (decays over training)
            j=np.random.choice(n_demo_fixed,128,p=pw)
            bd=torch.from_numpy(buf["dep"][j]).to(dev); ba=torch.from_numpy(buf["act"][j]).to(dev)
            mu,_=actor.net(bd).chunk(2,-1)
            w_bc=max(0.02, 1.0*np.exp(-step/30000))
            la=la+w_bc*F.mse_loss(torch.tanh(mu),ba)
            opt_a.zero_grad(); la.backward(); opt_a.step()
            lal=-(log_alpha.exp()*(lpn.detach()+target_ent)).mean()
            opt_al.zero_grad(); lal.backward(); opt_al.step()
            with torch.no_grad():
                for tp,sp in ((q1t,q1),(q2t,q2)):
                    for p,ps in zip(tp.parameters(),sp.parameters()):
                        p.mul_(0.995).add_(ps,alpha=0.005)
        if step%25000==0 and step>0:
            torch.save(dict(actor=actor.state_dict(),step=step),os.path.join(a.out,f"actor_{step//1000}k.pt"))
    torch.save(dict(actor=actor.state_dict(),step=a.steps),os.path.join(a.out,"actor_final.pt"))
    print("done")

if __name__=="__main__": main()

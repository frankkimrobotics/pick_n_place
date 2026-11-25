"""Sim eval for train_policy_zed ckpts. BARE parent (no mujoco/torch/warp imports);
each scene runs in a forked child that does ALL imports. mjwarp env."""
import argparse, os, sys, json
import multiprocessing as mp

def eval_scene(idx, a, rq):
    try:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ["MESA_SHADER_CACHE_DISABLE"]="true"
        import numpy as np, torch, cv2, mujoco, imageio
        HERE=os.path.expanduser("~/Desktop/2026/pick_and_place")
        sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,"policy"))
        sys.path.append("/home/lisc-frank/Desktop/lingbot-vision")
        import dataset_gen_div as dg
        if a["gen"]=="zed":
            import dataset_gen_zed as dg
        elif a["gen"]=="curobo":
            import dataset_gen_ycb as dg          # the curobo generator's sampler
            import json as _json
            _meta=_json.load(open(os.path.join(os.path.expanduser(a["ds"]),f"scene_{idx:03d}","meta.json")))
            BOX=_meta["box"]
            dg.BIN_XY=np.array(BOX["xy"]); dg.BIN_HALF=BOX["inner"]/2.0
            CUROBO_BOX=BOX
        import train_policy_zed as tp
        DEV="cuda:"+os.environ.get("EVAL_GPU","1")
        def rot6d_to_R(d6):
            a1,a2=d6[:3],d6[3:]
            b1=a1/(np.linalg.norm(a1)+1e-8)
            a2p=a2-(b1@a2)*b1; b2=a2p/(np.linalg.norm(a2p)+1e-8)
            return np.stack([b1,b2,np.cross(b1,b2)],1)
        def cup_contact_with(m,d,cup_gid,tip_sid,bid):
            fbuf=np.zeros(6); best=0.0
            for ci in range(d.ncon):
                c=d.contact[ci]
                if cup_gid not in (c.geom1,c.geom2): continue
                other=c.geom2 if c.geom1==cup_gid else c.geom1
                if m.geom_bodyid[other]!=bid: continue
                mujoco.mj_contactForce(m,d,ci,fbuf); best=max(best,abs(fbuf[0]))
            zax=d.site_xmat[tip_sid].reshape(3,3)[:,2]
            return best, float(np.degrees(np.arccos(np.clip(-zax[2],-1,1))))
        # ckpt + nets
        ck=torch.load(a["ckpt"],map_location=DEV,weights_only=False)
        lo,hi=ck["lo"].to(DEV),ck["hi"].to(DEV)
        unnorm=lambda x:(x+1)/2*(hi-lo)+lo
        model=ck["model"]; arch=ck.get("arch","unet"); use_goal=bool(ck.get("goal"))
        LSH=int(ck.get("lat_shift") or 0)
        if model == "act":
            fd = ck["net"]["tokp.weight"].shape[1]
        else:
            fd = ck["obs_enc"]["proj.0.weight"].shape[1]
        kw={}
        _sdw=ck["ema"] if "ema" in ck else ck["net"]
        if arch=="mlp":
            kw["d"]=_sdw["inp.weight"].shape[0]
        elif arch=="dit":
            # infer width/depth from the checkpoint. Without this the harness built the
            # DiT1D DEFAULTS (d=384, depth=6) and every mega-DiT scene died on a state_dict
            # size mismatch -- reported as 0 scenes, which reads like a model result but is
            # purely a loading bug. real_rollout_zed.py already does this; keep them in step.
            kw["d"]=int(_sdw["inp.weight"].shape[0])
            kw["depth"]=1+max(int(k.split(".")[1]) for k in _sdw if k.startswith("blocks."))
        obs_enc=None
        if model == "ode":
            net = tp.ODEField(cond_dim=512, H=ck.get("ode_H") or 16).to(DEV)
            obs_enc = tp.ObsEncoder(fd, use_goal=use_goal).to(DEV)
            net.load_state_dict({k:v for k,v in ck["ema"].items() if k in net.state_dict()}, strict=False)
            obs_enc.load_state_dict({k:v for k,v in ck["ema"].items() if k in obs_enc.state_dict()}, strict=False)
            obs_enc.eval()
        elif model == "act":
            net = tp.ACT(fd, use_goal=use_goal).to(DEV)
            net.load_state_dict(ck["net"], strict=False)
        else:
            net={"unet":tp.CondUnet1D,"dit":tp.DiT1D,"mlp":tp.MLPDenoiser}[arch](cond_dim=512,**kw).to(DEV)
            obs_enc=tp.ObsEncoder(fd,use_goal=use_goal,goal_dim=(9 if ck.get("goal_se3") else 3)).to(DEV)
            st_n={k:v for k,v in ck["ema"].items() if k in net.state_dict()}
            st_o={k:v for k,v in ck["ema"].items() if k in obs_enc.state_dict()}
            net.load_state_dict(st_n, strict=False)
            obs_enc.load_state_dict(st_o, strict=False)   # ckpts predating the graspability head
            obs_enc.eval()
        net.eval()
        dsched=None
        if model=="dp":
            from diffusers import DDIMScheduler
            dsched=DDIMScheduler(num_train_timesteps=50,beta_schedule="squaredcos_cap_v2",
                                 prediction_type="epsilon")
            dsched.set_timesteps(16)
        sys.path.append("/home/lisc-frank/Desktop/lingbot-vision")
        import extract_features as EF
        EF.DEV=DEV                      # 2-GPU machine: pin the encoder to the harness device
        _fwd,_mean,_std=EF.load_encoder(ck.get("enc","lingbot"))
        _mean,_std=_mean.to(DEV),_std.to(DEV)
        def tokens(img):
            t=torch.from_numpy(img.copy()).permute(2,0,1)[None].float().to(DEV)/255.
            t=torch.nn.functional.interpolate(t,size=(224,404),mode="bilinear",align_corners=False)
            x0=(404-224)//2; t=t[:,:,:,x0:x0+224]
            with torch.no_grad(): return _fwd(((t-_mean)/_std))[0].half()
        demo={"__file__":os.path.join(HERE,"mjwarp_pick_demo.py")}
        exec(open(demo["__file__"]).read().split("if __name__")[0],demo)
        ik=demo["ik"]; CTRL_DT=demo["CTRL_DT"]; NSUB=demo["NSUB"]
        every=int(round(1.0/(10*CTRL_DT)))
        sdir=os.path.join(os.path.expanduser(a["ds"]),f"scene_{idx:03d}")
        xml=os.path.join(sdir,"scene.xml")
        objs=dg.sample_scene(idx)
        m=mujoco.MjModel.from_xml_path(xml); d=mujoco.MjData(m)
        d.qpos[:6]=demo["Q_START"]; mujoco.mj_forward(m,d)
        ren=mujoco.Renderer(m,height=dg.H,width=dg.W)
        vopt=mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER]=0; vopt.sitegroup[:]=0; vopt.geomgroup[3]=1  # robot visuals of mega scenes
        tip_sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"tcp")
        tcp_bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"tcp")
        cup_gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"cup_tip")
        bids={o["name"]:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,o["name"]) for o in objs}
        eqs={o["name"]:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,f"suction_{o['name']}") for o in objs}
        state_step=[0]
        def render(cam):
            ren.update_scene(d,camera=cam,scene_option=vopt); img=ren.render()
            if state_step[0]==0 and img.mean()<40.0:   # only the FIRST obs must see the bright table (real GL corruption check)
                raise RuntimeError(f"CORRUPT RENDER {cam} mean {img.mean():.1f} at step 0")
            return img
        def tcp_pose():
            return d.site_xpos[tip_sid].copy(), d.site_xmat[tip_sid].reshape(3,3).copy()
        dik=mujoco.MjData(m)
        latched=None; was_latched=False; prev_tok=None; prev_prop=None; suc_state=0; frozen_goal=[None]
        LAT=int(a.get("sim_latency",0))
        LSH=0  # ckpt lat_shift, set after load
        c0_handover=[None]          # boundary state for the ab_refit decode (None -> 0)
        prev_tail=[]                # (tgt_p,tgt_R,suc) leftover actions of the previous plan
        vw=imageio.get_writer(os.path.join(a["out"],f"scene{idx:03d}.mp4"),fps=10,quality=7)
        z0={o["name"]:d.xpos[bids[o["name"]]][2] for o in objs}
        lifted=set(); touched=False
        binned={}                      # object name -> decision index when it entered the bin
        def _in_bin(o):
            p=d.xpos[bids[o["name"]]]
            if a["gen"]=="curobo":
                h=CUROBO_BOX["inner"]/2.0
                return (abs(p[0]-CUROBO_BOX["xy"][0])<h and abs(p[1]-CUROBO_BOX["xy"][1])<h
                        and p[2]<CUROBO_BOX["height"])      # inside the walled box
            return (abs(p[0]-dg.BIN_XY[0])<dg.BIN_HALF and abs(p[1]-dg.BIN_XY[1])<dg.BIN_HALF
                    and p[2]<0.12)
        for step in range(a["steps_max"]):
            state_step[0]=step
            tw=tokens(render("wrist_d405")); tz=tokens(render("zed"))
            p_t,R_t=tcp_pose()
            if prev_tok is None: prev_tok=(tw,tz); prev_prop=(p_t.copy(),R_t.copy())
            dp=R_t.T@(prev_prop[0]-p_t); dR=R_t.T@prev_prop[1]
            pr0=np.concatenate([dp,dR[:,0],dR[:,1],[float(suc_state)]])
            pr1=np.concatenate([np.zeros(3),[1,0,0],[0,1,0],[float(suc_state)]])
            prop=torch.from_numpy(np.stack([pr0,pr1]).astype(np.float32))[None].to(DEV)
            tok=torch.stack([prev_tok[0],tw,prev_tok[1],tz])[None].to(DEV)
            # latency emulation: the plan computed now starts executing LAT steps from now;
            # the previous plan's tail covers the gap. Boundary for matched decode = the pose
            # the arm will have reached by then, relative to the current obs pose.
            if LAT>0 and len(prev_tail)>=LAT:
                hp,hR=prev_tail[LAT-1][0],prev_tail[LAT-1][1]
                dp0=R_t.T@(hp-p_t); dR0=R_t.T@hR
                c0_handover[0]=np.concatenate([dp0,dR0[:,0],dR0[:,1],[float(suc_state)]])
            else:
                c0_handover[0]=None
            goal_t=None
            if use_goal:
                if latched is None: frozen_goal[0]=None
                # target: keep latched object's grasp; else nearest not-yet-binned object
                cands=[o for o in objs if not (abs(d.xpos[bids[o["name"]]][0]-dg.BIN_XY[0])<dg.BIN_HALF
                        and abs(d.xpos[bids[o["name"]]][1]-dg.BIN_XY[1])<dg.BIN_HALF)]
                tgt_o=None
                if latched is not None:
                    tgt_o=next((o for o in objs if o["name"]==latched),None)
                if tgt_o is None and cands:
                    tgt_o=min(cands,key=lambda o:np.hypot(*(d.xpos[bids[o["name"]]][:2]-p_t[:2])))
                if tgt_o is not None:
                    if latched is not None and frozen_goal[0] is not None:
                        gp=frozen_goal[0]                   # goal frozen at latch (training semantics)
                    else:
                        gp=d.xpos[bids[tgt_o["name"]]].copy(); gp[2]+=tgt_o["half_extents"][2]+demo["CUP_R"]
                        if a["goal_err"]>0:
                            gerr=np.random.default_rng(idx*31+7).normal(0,a["goal_err"],3)
                            gp=gp+gerr                      # consistent wrong goal all scene
                        if latched is not None: frozen_goal[0]=gp.copy()
                    g9=(R_t.T@(gp-p_t)).astype(np.float32)
                    if ck.get("goal_se3"):
                        Rg=np.array(demo["R_DOWN"],float)
                        g9=np.concatenate([g9,(R_t.T@Rg)[:,0],(R_t.T@Rg)[:,1]]).astype(np.float32)
                    goal_t=torch.from_numpy(g9)[None].to(DEV)
                else:
                    goal_t=torch.zeros(1,9 if ck.get("goal_se3") else 3,device=DEV)
            with torch.no_grad():
                if model=="ode":
                    cond=obs_enc(tok,prop,goal_t)
                    _id10=torch.tensor([0,0,0,1,0,0,0,1,0,0.0],device=DEV)
                    a0=((((_id10-lo)/(hi-lo))*2-1)[None,:9]).float()
                    traj=net.rollout(a0,cond)[0]
                    pose=(traj+1)/2*(hi[:9]-lo[:9])+lo[:9]
                    suc=(torch.sigmoid(net.suc_head(cond))[0]>0.5).float()
                    act=torch.cat([pose,suc[:,None]],-1).cpu().numpy()
                    if act.shape[0] < tp.TA:            # H<16: hold the last pose (replan at 8)
                        import numpy as _np
                        act=_np.concatenate([act, _np.repeat(act[-1:], tp.TA-act.shape[0], 0)], 0)
                elif model=="act":
                    pred,_,_=net(tok,prop,act=None,goal=goal_t)
                    act=unnorm(pred[0]).cpu().numpy()
                    if ck.get("action")=="bspline":
                        import sys as _s; _s.path.insert(0,os.path.join(HERE,"policy"))
                        from spline_action import ctrl_to_waypoints, ab_refit
                        act=ctrl_to_waypoints(ab_refit(act, c0=c0_handover[0]))
                elif model in("dp","cfm"):
                    cond=obs_enc(tok,prop,goal_t)
                    x=torch.randn(1,tp.ADIM,tp.TA,device=DEV)
                    if model=="cfm":
                        for k in range(16):
                            t=torch.full((1,),k/16,device=DEV)
                            x=x+net(x,t*50,cond)/16
                    else:
                        for t in dsched.timesteps:
                            eps=net(x,t[None].to(DEV).float(),cond)
                            x=dsched.step(eps,t,x).prev_sample
                    act=unnorm(x.transpose(1,2)[0]).cpu().numpy()
                    if ck.get("action")=="bspline":
                        import sys as _s; _s.path.insert(0,os.path.join(HERE,"policy"))
                        from spline_action import ctrl_to_waypoints, ab_refit
                        act=ctrl_to_waypoints(ab_refit(act, c0=c0_handover[0]))
                else:
                    pred,_,_=net(tok,prop,act=None,goal=goal_t)
                    act=unnorm(pred[0]).cpu().numpy()
                    if ck.get("action")=="bspline":
                        import sys as _s; _s.path.insert(0,os.path.join(HERE,"policy"))
                        from spline_action import ctrl_to_waypoints, ab_refit
                        act=ctrl_to_waypoints(ab_refit(act, c0=c0_handover[0]))
            # schedule this cycle's exec_h actions: prev-plan tail during "inference",
            # then the new plan from index (LAT-LSH) (vanilla skips its stale head; a
            # latency-matched ckpt (LSH=LAT) starts at 0 because its window already starts late)
            new_abs=[]
            for hh in range(act.shape[0]):
                new_abs.append((p_t+R_t@act[hh,:3], R_t@rot6d_to_R(act[hh,3:9]), act[hh,9]>0.5))
            skip=max(0,LAT-LSH)
            sched_acts=(prev_tail[:LAT] if LAT>0 else [])+new_abs[skip:skip+a["exec_h"]-min(LAT,len(prev_tail))]
            if not sched_acts: sched_acts=new_abs[:a["exec_h"]]
            prev_tail=new_abs[skip+a["exec_h"]-min(LAT,len(prev_tail)):]
            for h in range(len(sched_acts)):
                if h==len(sched_acts)-1:   # fresh prev-obs 0.1s before next inference (training t0=t-1)
                    pp,RR=tcp_pose()
                    prev_tok=(tokens(render("wrist_d405")),tokens(render("zed"))); prev_prop=(pp.copy(),RR.copy())
                tgt_p,tgt_R,sc=sched_acts[h]
                q,e=ik(m,dik,"tcp",tgt_p.tolist(),tgt_R,d.qpos[:6])
                if e>0.01: continue
                qref=np.linspace(d.qpos[:6],np.asarray(q,float),every)
                qdref,tau_ff,kp,kd=demo["_gains_and_ff"](m,d,qref)
                sched=dict(qref=qref,qdref=qdref,tau_ff=tau_ff,kp=kp,kd=kd)
                for k in range(len(qref)):
                    if sc and latched is None:
                        for o in objs:
                            f,tilt=cup_contact_with(m,d,cup_gid,tip_sid,bids[o["name"]])
                            if f>demo["SEAL_N"] and tilt<demo["SEAL_DEG"]:
                                demo["_latch_weld"](m,d,eqs[o["name"]],tcp_bid,bids[o["name"]])
                                latched=o["name"]; was_latched=True; break
                    if (not sc) and latched is not None:
                        d.eq_active[eqs[latched]]=0; latched=None
                    for _ in range(NSUB):
                        d.ctrl[:6]=demo["pd_tau"](sched,k,d.qpos[:6],d.qvel[:6])
                        mujoco.mj_step(m,d)
                suc_state=1 if latched else 0
            if not touched:
                for o in objs:
                    f,_=cup_contact_with(m,d,cup_gid,tip_sid,bids[o["name"]])
                    if f>0: touched=True; break
            for o in objs:
                if d.xpos[bids[o["name"]]][2]>z0[o["name"]]+0.06: lifted.add(o["name"])
            for o in objs:
                if o["name"] not in binned and _in_bin(o): binned[o["name"]]=step
            fa=cv2.resize(render("wrist_d405"),(432,240)); fb=cv2.resize(render("zed"),(432,240))
            vw.append_data(np.hstack([fa,fb]))
        vw.close()
        inbin=sum(1 for o in objs if _in_bin(o))
        rq.put(dict(scene=idx,touched=touched,sealed=was_latched,lifted=len(lifted),in_bin=inbin,
                    n_obj=len(objs),place_steps=sorted(binned.values())))
    except Exception as e:
        rq.put({"scene":idx,"error":str(e)})

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--scenes",default="90-99")
    ap.add_argument("--steps_max",type=int,default=90); ap.add_argument("--exec_h",type=int,default=8)
    ap.add_argument("--out",default="/tmp/zed_eval")
    ap.add_argument("--ds",default="~/pnp_dataset_zed")
    ap.add_argument("--gen",default="zed",choices=["zed","div","curobo"])
    ap.add_argument("--par",type=int,default=4,help="concurrent scene children")
    ap.add_argument("--sim_latency",type=int,default=0,help="emulate N decisions of inference latency (prev-plan tail bridges)")
    ap.add_argument("--goal_err",type=float,default=0.0,help="m: fixed per-scene goal-input offset (robustness test)")
    a=vars(ap.parse_args())
    s0,s1=[int(x) for x in a["scenes"].split("-")]
    os.makedirs(a["out"],exist_ok=True)
    results=[]
    ids=list(range(s0,s1+1)); active={}; done={}
    def launch(idx,att):
        rq=mp.Queue(); p=mp.Process(target=eval_scene,args=(idx,a,rq)); p.start()
        active[idx]=(p,rq,att)
    pend=list(ids)
    while pend or active:
        while pend and len(active)<a["par"]:
            launch(pend.pop(0),1)
        import time as _t; _t.sleep(2)
        for idx in list(active):
            p,rq,att=active[idx]
            if not rq.empty():
                r=rq.get(); p.join(); del active[idx]
                if "error" in r and "CORRUPT" in r.get("error","") and att<4:
                    print(f"[retry] scene {idx} attempt {att+1}",flush=True); launch(idx,att+1)
                else:
                    done[idx]=r; print(json.dumps(r),flush=True)
            elif not p.is_alive():
                del active[idx]
                if att<4: launch(idx,att+1)
                else: done[idx]={"scene":idx,"error":"child died"}
    results=[done[i] for i in ids if i in done]
    ok=[r for r in results if "error" not in r]
    json.dump(ok,open(os.path.join(a["out"],"results.json"),"w"),indent=1)
    print(f"SUMMARY: {len(ok)} scenes  touch {sum(r['touched'] for r in ok)}  seal {sum(r['sealed'] for r in ok)}  lift {sum(r['lifted']>0 for r in ok)}  in_bin {sum(r['in_bin']>0 for r in ok)}")

#!/usr/bin/env python3
"""realsim.eval_policy :: closed-loop eval of a ZED visual policy on the realistic Pro 630 twin.

Runs the REAL executor's per-decision logic (policy/real_rollout_zed.py: ZedPolicy, ik with the
LinuxCNC bounds, async inference + adoption-time ab_refit splice, lat_shift indexing, yawlock,
lateral/vertical step caps, z floor, tilt cap, graceful lead scaling, dq clamp, suction z-gate +
off hysteresis, spline streaming with grid anchors) against `realsim.twin.Pro630Twin`
(Pi stream law, drive dead-time / accel / saturation, fitted joint friction), delayed cameras
(capture age, frame period, JPEG q95) and delayed joint feedback.

Scenes / scoring are the old harness's (realsim/eval_ideal_harness.py == /tmp/eval_zed_clean.py):
same scene.xml, sample_scene objects, goal = nearest un-binned object top (frozen at latch),
seal = cup contact > SEAL_N with tilt < SEAL_DEG while suction is on -> weld.

  conda activate mj4
  python realsim/eval_policy.py --ckpt ~/pnp_policies/cfm_lingbot_bspline_lat2_aug/ckpt_final.pt \
      --level real --slow 2 --scenes 90-99 --out ~/pnp_rollouts_realsim/real_slow2

Parent process imports nothing heavy; each scene runs in a fresh child (OSMesa + torch safety).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def eval_scene(idx, a, rq):
    try:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        if a.get("_link"):                                 # many CPU workers behind one policy server: one thread each
            for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LP_NUM_THREADS"):
                os.environ[_k] = "1"
        os.environ["MESA_SHADER_CACHE_DISABLE"] = "true"
        import numpy as np
        import torch  # noqa: F401  (before mujoco: OSMesa/triton import order)
        import cv2
        import imageio
        import mujoco
        sys.path.insert(0, ROOT)
        sys.path.insert(0, os.path.join(ROOT, "policy"))
        sys.path.append(os.path.expanduser("~/Desktop/lingbot-vision"))
        gpus = a["gpus"].split(",")
        dev = "cuda:" + gpus[idx % len(gpus)]
        import real_rollout_zed as RZ
        import zed_mask as ZM
        RZ.DEV = dev
        import train_policy_zed as T
        from spline_action import ctrl_to_waypoints, ab_refit
        from realsim.config import make_cfg
        from realsim.twin import Pro630Twin, CameraModel
        from realsim.validate import Streamer
        import dataset_gen_zed as dg
        if a["gen"] == "div":
            import dataset_gen_div as dg
        BOX = None
        if a["gen"] == "curobo":                          # mega scenes: walled box from meta.json (old harness)
            import dataset_gen_ycb as dg
            BOX = json.load(open(os.path.join(os.path.expanduser(a["ds"]), f"scene_{idx:03d}", "meta.json")))["box"]
            dg.BIN_XY = np.array(BOX["xy"])
            dg.BIN_HALF = BOX["inner"] / 2.0
        demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
        exec(open(demo["__file__"]).read().split("if __name__")[0], demo)

        rng = np.random.default_rng(1000 + idx + 100003 * a["seed"])
        over = {}
        for kv in a["cfg"]:
            k_, v_ = kv.split("=", 1)
            over[k_] = float(v_) if k_ not in ("orient",) else v_
        dq_wrist = float(np.radians(over.pop("dq_wrist", 0.0)))  # deg per decision for joints 4-6, 0 = same as dq_max
        dq_mid = float(np.radians(over.pop("dq_mid", 0.0)))      # deg per decision for joints 2-4 (vertical moves), 0 = same
        dq_mid_z = float(over.pop("dq_mid_z", -1.0))             # dq_mid applies only above this tool height (m)
        dq_mid_up = bool(over.pop("dq_mid_up", 0.0))             # ... or on any upward move
        for k_ in ("dq_max", "lead_max"):                      # given in degrees on the CLI
            if k_ in over:
                over[k_] = float(np.radians(over[k_]))
        if a["stream"] == "tuned" and a["level"] != "ideal":   # Pi stream gains tuned on the arm 2026-11-14
            from realsim.config import STREAM_TUNED
            over = {**STREAM_TUNED, **over}
        cfg = make_cfg(a["level"], dr=a["dr"], rng=rng, **over)
        dt = 0.1 * a["slow"]

        sdir = os.path.join(os.path.expanduser(a["ds"]), f"scene_{idx:03d}")
        objs = dg.sample_scene(idx)
        if BOX is not None:                                # the generator drops (and renames past) objects under the box
            import dataset_gen_curobo as _gc
            objs = _gc.sample_objects(idx, tuple(BOX["xy"]))
        from realsim.plunger import load_with_plunger, compression
        if a["rigid_tip"]:
            _sp = mujoco.MjSpec.from_file(os.path.join(sdir, "scene.xml"))
            if a["mast"]:
                from realsim.mast import add_zed_mast
                add_zed_mast(_sp)
            m, PL = _sp.compile(), None
        else:
            pl_over = {}
            if a["dr_scene"]:
                pl_over = dict(k=300.0 * rng.uniform(0.6, 1.6), preload=2.0 * rng.uniform(0.5, 2.0),
                               seal_x=rng.uniform(0.0005, 0.003))
            m, PL = load_with_plunger(os.path.join(sdir, "scene.xml"), mast=bool(a["mast"]), **pl_over)
        d = mujoco.MjData(m)
        d.qpos[:6] = demo["Q_START"]
        goal_err = a["goal_err"]
        if a["dr_scene"]:
            from realsim.scene_dr import randomize_scene
            randomize_scene(m, d, objs, rng)
            cfg["suction_onset"] = rng.uniform(0.15, 0.50) if cfg["suction_onset"] > 0 else 0.0
            cfg["cam_noise"] = rng.uniform(0.0, 3.0)
            goal_err = max(goal_err, rng.uniform(0.0, 0.006))
        mujoco.mj_forward(m, d)
        tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        tcp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp" if PL is None else "plunger_tip")
        cup_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")
        bids = {o["name"]: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"]) for o in objs}
        eqs = {o["name"]: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{o['name']}") for o in objs}
        z0 = {o["name"]: d.xpos[bids[o["name"]]][2] for o in objs}

        # ---------------- suction physics (pump onset / valve release), checked every 10 ms tick
        S = dict(cmd=0, t_on=None, t_off=None, latched=None, ever=False, touched=False,
                 comp_max=0.0, bottom_ticks=0, seal_comp=None)

        # ---------------- safety monitor (SIM-ONLY): the arm links carry no collision geoms, so clearance is measured
        # with distance queries every 20 ms. arm = link2..6 + camera mount, cup = suction-cup body, wall = x < WALL_X.
        WALL_X, SAFE_MAX = -0.30, 0.10
        _gb = lambda g: m.body(m.geom_bodyid[g]).name
        G_ARM = [g for g in range(m.ngeom) if _gb(g) in ("link2", "link3", "link4", "link5", "link6", "camera_mount")]
        G_CUP = [g for g in range(m.ngeom) if _gb(g) == "suction_cup"]
        G_WORLD = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
        G_WALLS = [g for g in G_WORLD if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") not in ("floor", "table", "box_floor")]
        G_OBJ = {o["name"]: [g for g in range(m.ngeom) if m.geom_bodyid[g] == bids[o["name"]]] for o in objs}
        _corn = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        xy0 = {o["name"]: d.xpos[bids[o["name"]]][:2].copy() for o in objs}
        JR = m.jnt_range[:6].copy()
        SF = dict(n=0, arm_world=SAFE_MAX, arm_obj=SAFE_MAX, cup_world=SAFE_MAX, carry=SAFE_MAX, wall=9.0, jl=9.0, qd=np.zeros(6),
                  pen_arm=0, pen_carry=0, near_arm=0, checks=0, touched_ever=set(), cur=[SAFE_MAX] * 4)
        _ft = np.zeros(6)

        def _dmin(A, B):
            return min((mujoco.mj_geomDistance(m, d, ga, gb, SAFE_MAX, _ft) for ga in A for gb in B), default=SAFE_MAX)

        def _near_name():
            dm, ga, gb = min((mujoco.mj_geomDistance(m, d, ga, gb, SAFE_MAX, _ft), ga, gb) for ga in G_ARM for gb in G_WORLD)
            return f"{_gb(ga)}-{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gb) or gb}"

        def safety_tick():
            SF["n"] += 1
            SF["qd"] = np.maximum(SF["qd"], np.abs(d.qvel[:6]))
            SF["over"] = SF.get("over", 0) + int(np.abs(d.qvel[:6]).max() > np.radians(36.0))
            if SF["n"] % 2:
                return
            lat = S["latched"]
            if lat is not None:
                SF["touched_ever"].add(lat)
            others = [g for n_, gs in G_OBJ.items() if n_ != lat for g in gs]
            aw, ao, cw = _dmin(G_ARM, G_WORLD), _dmin(G_ARM, others), _dmin(G_CUP, G_WORLD)
            ca = _dmin(G_OBJ[lat], others + G_WALLS) if lat is not None else SAFE_MAX
            SF["cur"] = [aw, ao, cw, ca]
            if aw < SF["arm_world"]:
                SF["arm_world_pair"] = _near_name()
            if aw < 0.010:
                SF.setdefault("near_pairs", {})
                nm = _near_name()
                SF["near_pairs"][nm] = SF["near_pairs"].get(nm, 0) + 1
            SF["arm_world"], SF["arm_obj"], SF["cup_world"], SF["carry"] = min(SF["arm_world"], aw), min(SF["arm_obj"], ao), min(SF["cup_world"], cw), min(SF["carry"], ca)
            SF["checks"] += 1
            SF["pen_arm"] += int(min(aw, ao) < 0.0)
            SF["near_arm"] += int(min(aw, ao) < 0.010)
            SF["pen_carry"] += int(ca < -0.002)
            xm = 9.0
            for g in G_ARM:
                c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g, :3]
                xm = min(xm, float((c[0] + (_corn * m.geom_aabb[g, 3:]) @ d.geom_xmat[g].reshape(3, 3)[0]).min()))
            SF["wall"] = min(SF["wall"], xm - WALL_X)
            q = d.qpos[:6]
            SF["jl"] = min(SF["jl"], float(np.degrees(np.minimum(q - JR[:, 0], JR[:, 1] - q).min())))

        def cup_contact(bid):
            fbuf = np.zeros(6)
            best = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if cup_gid not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == cup_gid else c.geom1
                if m.geom_bodyid[other] != bid:
                    continue
                mujoco.mj_contactForce(m, d, ci, fbuf)
                best = max(best, abs(fbuf[0]))
            zax = d.site_xmat[tip_sid].reshape(3, 3)[:, 2]
            return best, float(np.degrees(np.arccos(np.clip(-zax[2], -1, 1))))

        def on_tick(tw):
            safety_tick()
            if not S["touched"]:
                S["touched"] = any(cup_contact(bids[o["name"]])[0] > 0 for o in objs)
            x = compression(m, d) if PL is not None else 0.0      # SIM-ONLY diagnostic: no real sensor
            S["comp_max"] = max(S["comp_max"], x)
            if PL is not None and x >= PL["travel"] - 0.0005:
                S["bottom_ticks"] += 1
            if S["cmd"] and S["latched"] is None and tw.t - S["t_on"] >= cfg["suction_onset"] - 1e-9:
                for o in objs:
                    f, tilt = cup_contact(bids[o["name"]])
                    # rigid tip: old force rule; plunger: lip seated = compressed >= seal_x on THIS object
                    seated = (f > demo["SEAL_N"]) if PL is None else (f > 0 and x >= PL["seal_x"])
                    if seated and tilt < demo["SEAL_DEG"]:
                        S["seal_comp"] = x
                        demo["_latch_weld"](m, d, eqs[o["name"]], tcp_bid, bids[o["name"]])
                        S["latched"] = o["name"]
                        S["ever"] = True
                        break
            if (not S["cmd"]) and S["latched"] is not None and tw.t - S["t_off"] >= cfg["suction_release"] - 1e-9:
                d.eq_active[eqs[S["latched"]]] = 0
                S["latched"] = None

        tw = Pro630Twin(m, d, cfg, rng=rng, on_tick=on_tick)
        tw.advance(0.3)                                    # settle under gravity on the drive
        if a["ticklog"]:
            tw.log = []
        streamer = Streamer(tw, dt=dt)

        ren = mujoco.Renderer(m, height=dg.H, width=dg.W)
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        vopt.sitegroup[:] = 0
        vopt.geomgroup[3] = 1      # mega (curobo/ycb) scenes keep the robot's visual meshes in group 3 and their
        #                            generator renders it; hidden here, the policy sees no arm (cos 0.87-0.94 vs 0.99)
        cams = {c: CameraModel(tw, ren, c, cfg[f"{k}_lat"], cfg[f"{k}_period"],
                               phase=rng.uniform(0, cfg[f"{k}_period"]), noise=cfg["cam_noise"], vopt=vopt, rng=rng)
                for c, k in (("wrist_d405", "wrist"), ("zed", "zed"))}

        def grab_bgr(cam):
            rgb = cams[cam].grab()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (432, 240), interpolation=cv2.INTER_AREA)   # cam servers publish 432x240
            if cfg["jpeg_q"]:
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg_q"])])
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return bgr

        dfk = mujoco.MjData(m)

        def fk(q):
            dfk.qpos[:6] = q
            mujoco.mj_kinematics(m, dfk)
            return dfk.site_xpos[tip_sid].copy(), dfk.site_xmat[tip_sid].reshape(3, 3).copy()

        dik = mujoco.MjData(m)
        if a["sim_ik_bounds"]:                     # rollout_sim_zed: widen the LinuxCNC bounds in sim
            RZ.LC_LO[:], RZ.LC_HI[:] = -10.0, 10.0

        if a.get("_link"):                                 # shared policy server (realsim/policy_server.py)
            from realsim.policy_server import RemotePolicy
            torch.set_num_threads(1)
            policy = RemotePolicy(*a["_link"])
        else:
            _tl = torch.load                               # keep the ckpt dict off the GPU (920M DiT: 9.7 -> ~5 GB)
            torch.load = lambda *x, **kw: _tl(*x, **{**kw, "map_location": "cpu"})
            policy = RZ.ZedPolicy(a["ckpt"], euler=a["euler"], frozen_noise=True)
            policy.fast = bool(a["fast_infer"])
            torch.load = _tl
            if a["patch"]:
                from realsim.wsp import apply_patch
                apply_patch(policy, a["patch"])
        TR = dict(dec=[], inf=[])
        lat_shift = policy.lat_shift if a["lat_shift"] < 0 else a["lat_shift"]
        R_DOWN = RZ.R_DOWN

        def in_bin(o):
            p = d.xpos[bids[o["name"]]]
            if BOX is not None:
                return abs(p[0] - BOX["xy"][0]) < BOX["inner"] / 2 and abs(p[1] - BOX["xy"][1]) < BOX["inner"] / 2 \
                    and p[2] < BOX["height"]
            return abs(p[0] - dg.BIN_XY[0]) < dg.BIN_HALF and abs(p[1] - dg.BIN_XY[1]) < dg.BIN_HALF and p[2] < 0.12

        vw = imageio.get_writer(os.path.join(a["out"], f"scene{idx:03d}.mp4"), fps=int(round(1 / dt)), quality=7)
        q_m0, _, _ = tw.joint_feedback()
        q_cmd = q_m0.copy()
        p_cmd, R_cmd = fk(q_cmd)
        R_lp = None
        prev_p_cmd = None
        hist, plan, pending = [], None, None
        frozen_goal = None
        goal_cur, goal_t0, goal_skip = None, 0, {}
        off_count = 0
        lifted, binned = set(), {}
        stats = dict(hold=0, lead=[], adopt=[], infer=[], cam_age=[], clips=0, dq_clip=0, ik_fail=0, plan_exhausted=0)
        FILT = None
        if a["safe"]:
            from realsim.safety import SafetyFilter
            FILT = SafetyFilter(m, margin=a["safe_margin"], v_lim=np.radians(a["safe_vlim"]), a_lim=np.radians(a["safe_alim"]),
                                wall_margin=a["safe_wall"] or None)
        RT_N, BLK_N = max(1, int(round(1.5 / dt))), max(1, int(round(a["safe_block"] / dt)))
        blocked_n, retreat_left = 0, 0
        aim_block, aim_press = 0, 0
        rt_pos, stalled, RT_C = [], False, np.array([0.30, 0.10])
        stats.update(retreats=0, filt_block=0)
        T0 = tw.t
        for k in range(a["steps"]):
            t_k = T0 + k * dt                                 # decision grid (sim clock)
            tw.advance(max(0.0, round((t_k - tw.t) / cfg["cmd_dt"]) * cfg["cmd_dt"]))
            # ---- observe (delayed joint stream + delayed, JPEG'd frames)
            q_m, _, _ = tw.joint_feedback()
            p_m, R_m = fk(q_m)
            wimg, zimg = grab_bgr("wrist_d405"), grab_bgr("zed")
            if k == 0 and (wimg.mean() < 40 or zimg.mean() < 40):
                raise RuntimeError(f"CORRUPT RENDER at step 0 ({wimg.mean():.0f}/{zimg.mean():.0f})")
            stats["cam_age"].append(cams["zed"].age)
            if a["zed_mask"]:                                  # table-only ZED view (policy/zed_mask.py)
                zimg = ZM.apply(zimg)
            hist.append(dict(p=p_m, R=R_m, suc=S["cmd"], wrist_tok=policy.encode(wimg), zed_tok=policy.encode(zimg)))
            if a["trace_mask"]:                                # masked ZED features of the same state, policy input unchanged
                hist[-1]["zed_tokm"] = policy.encode(ZM.apply(zimg)).float().mean(0).cpu().numpy().astype(np.float16)
            hist = hist[-2:]
            if a["trace"] >= 2:                                # every decision is a policy input state (DAgger)
                _o = [hist[-2] if len(hist) >= 2 else hist[-1], hist[-1]]
                tr_tok = torch.stack([_o[0]["wrist_tok"], _o[1]["wrist_tok"], _o[0]["zed_tok"], _o[1]["zed_tok"]]
                                     ).float().mean(1).cpu().numpy().astype(np.float16)
                tr_prop = np.stack([RZ.rel_pose(p_m, R_m, _o[0]["p"], _o[0]["R"], _o[0]["suc"]),
                                    RZ.rel_pose(p_m, R_m, p_m, R_m, _o[1]["suc"])])
            # ---- goal (harness semantics: nearest un-binned object; frozen while latched)
            goal_rel = None
            if policy.use_goal:
                if S["latched"] is None:
                    frozen_goal = None
                cands = [o for o in objs if not in_bin(o)]
                tgt_o = next((o for o in objs if o["name"] == S["latched"]), None)
                if a["goal_to"] > 0:                               # give up on a goal object that is not sealed in time
                    if tgt_o is not None:
                        goal_cur, goal_t0 = None, k
                    else:
                        goal_skip = {n_: e_ for n_, e_ in goal_skip.items() if e_ > k}
                        if goal_cur is not None and (k - goal_t0) * dt > a["goal_to"]:
                            goal_skip[goal_cur] = k + int(a["goal_skip"] / dt)
                            stats["goal_giveup"] = stats.get("goal_giveup", 0) + 1
                            goal_cur = None
                        c2 = [o for o in cands if o["name"] not in goal_skip]
                        if not c2 and cands:
                            goal_skip, c2 = {}, cands
                        if goal_cur is not None and any(o["name"] == goal_cur for o in c2):
                            c2 = [o for o in c2 if o["name"] == goal_cur]      # stay on it until sealed or timed out
                        cands = c2
                if tgt_o is None and cands:
                    tgt_o = min(cands, key=lambda o: np.hypot(*(d.xpos[bids[o["name"]]][:2] - p_m[:2])))
                    if a["goal_to"] > 0 and tgt_o["name"] != goal_cur:
                        goal_cur, goal_t0 = tgt_o["name"], k
                if tgt_o is not None:
                    if frozen_goal is not None:
                        gp = frozen_goal
                    else:
                        gp = d.xpos[bids[tgt_o["name"]]].copy()
                        gp[2] += tgt_o["half_extents"][2] + demo["CUP_R"]
                        if goal_err > 0:
                            gp = gp + np.random.default_rng(idx * 31 + 7 + a["seed"]).normal(0, goal_err, 3)
                        if S["latched"] is not None:
                            frozen_goal = gp.copy()
                    goal_rel = hist[-1]["R"].T @ (gp - hist[-1]["p"])
                    if policy.goal_se3:
                        dRg = hist[-1]["R"].T @ R_DOWN
                        goal_rel = np.concatenate([goal_rel, dRg[:, 0], dRg[:, 1]])
                else:
                    goal_rel = np.zeros(9 if policy.goal_se3 else 3)
            # ---- inference: start every n_exec decisions; result usable once its latency has elapsed
            t_now = t_k + cfg["host_proc"] * 0.5            # inference thread starts after the encodes
            if (plan is None or k - plan["k_obs"] >= cfg["n_exec"]) and pending is None and retreat_left == 0:
                obs = [hist[-2] if len(hist) >= 2 else hist[-1], hist[-1]]
                tok4 = torch.stack([obs[0]["wrist_tok"], obs[1]["wrist_tok"], obs[0]["zed_tok"], obs[1]["zed_tok"]])
                pT, RT = obs[1]["p"], obs[1]["R"]
                prop = np.stack([RZ.rel_pose(pT, RT, obs[0]["p"], obs[0]["R"], obs[0]["suc"]),
                                 RZ.rel_pose(pT, RT, pT, RT, obs[1]["suc"])])
                rtc = None                                    # condition the new chunk on the one being executed
                if a["rtc"] and plan is not None and "p_abs" in plan and blocked_n == 0 and plan.get("suc_obs") == S["cmd"]:
                    sft = k - plan["k_obs"]
                    Y, Wr = np.zeros((T.TA, 10), np.float32), np.zeros((T.TA, 10), np.float32)
                    for i_ in range(T.TA - sft):
                        Rr = RT.T @ plan["R_abs"][i_ + sft]
                        Y[i_] = np.concatenate([RT.T @ (plan["p_abs"][i_ + sft] - pT), Rr[:, 0], Rr[:, 1], [plan["suc"][i_ + sft]]])
                        Wr[i_, :9] = a["rtc_w"] * (1.0 if i_ < a["rtc_hard"] else max(0.0, 1.0 - (i_ - a["rtc_hard"] + 1) / (a["rtc"] + 1.0)))
                    rtc = (Y, Wr)
                    stats["rtc"] = stats.get("rtc", 0) + 1
                act = policy(tok4, prop, goal_rel, rtc=rtc) if rtc is not None else policy(tok4, prop, goal_rel)
                if a["trace"]:
                    TR["inf"].append(dict(k=k, tok=(tok4.float().cpu().numpy().astype(np.float16) if a["trace"] < 2
                                                    else np.zeros(0, np.float16)), prop=prop,
                                          goal=np.asarray(goal_rel, np.float32), act=act, p=pT.copy(), R=RT.copy()))
                lat = max(cfg["infer_min"], rng.normal(cfg["infer_mean"], cfg["infer_sd"])) if cfg["infer_mean"] > 0 else 0.0
                pending = dict(k_obs=k, act=act, p_t=pT.copy(), R_t=RT.copy(), ready=t_now + lat, suc_obs=S["cmd"])
                stats["infer"].append(lat)
            if pending is not None and t_now + 1e-9 >= pending["ready"]:
                _old = plan
                plan = pending
                pending = None
                plan["k_adopt"] = k
                plan["old"] = None if (_old is None or "p_abs" not in _old) else dict(p=_old["p_abs"], R=_old["R_abs"], k_obs=_old["k_obs"])
                stats["adopt"].append(k - plan["k_obs"])
                pT, RT = plan["p_t"], plan["R_t"]
                if policy.action_mode == "bspline":
                    c0 = np.concatenate([RT.T @ (p_cmd - pT), (RT.T @ R_cmd)[:, 0], (RT.T @ R_cmd)[:, 1], [float(S["cmd"])]])
                    v0 = np.zeros(10)
                    if prev_p_cmd is not None:
                        v0[:3] = RT.T @ (p_cmd - prev_p_cmd) / dt
                    acts = ctrl_to_waypoints(ab_refit(plan["act"], v0=v0, c0=c0)).astype(np.float32)
                else:
                    acts = plan["act"]
                plan["p_abs"] = np.stack([pT + RT @ acts[i, :3] for i in range(T.TA)])
                plan["R_abs"] = np.stack([RT @ RZ.rot6d_to_R(acts[i, 3:9]) for i in range(T.TA)])
                plan["suc"] = acts[:, 9].copy()
            # ---- target for this decision
            holding = "no plan" if plan is None else None
            tgt_p, tgt_R, suc_pred = p_cmd.copy(), R_cmd.copy(), 0.0
            if plan is not None:
                i = max(0, k - plan["k_obs"] - lat_shift)
                if i >= T.TA:
                    holding = "plan exhausted"
                    stats["plan_exhausted"] += 1
                else:
                    tgt_p, tgt_R, suc_pred = plan["p_abs"][i].copy(), plan["R_abs"][i].copy(), float(plan["suc"][i])
                    jb = k - plan.get("k_adopt", k)               # cross-fade: old chunk's remaining steps -> new chunk
                    if a["blend"] and plan.get("old") is not None and jb < a["blend"]:
                        io = k - plan["old"]["k_obs"] - lat_shift
                        if 0 <= io < T.TA:
                            wb = (jb + 1.0) / (a["blend"] + 1.0)
                            wb = wb * wb * (3.0 - 2.0 * wb)
                            tgt_p = (1.0 - wb) * plan["old"]["p"][io] + wb * tgt_p
                            Ub, _, Vb = np.linalg.svd((1.0 - wb) * plan["old"]["R"][io] + wb * tgt_R)
                            tgt_R = Ub @ Vb
                    if cfg["orient"] == "yawlock":
                        z = tgt_R[:, 2] / (np.linalg.norm(tgt_R[:, 2]) + 1e-9)
                        x = R_DOWN[:, 0] - z * (z @ R_DOWN[:, 0])
                        if np.linalg.norm(x) > 1e-3:
                            x /= np.linalg.norm(x)
                            tgt_R = np.stack([x, np.cross(z, x), z], axis=1)
                    elif cfg["orient"] == "fixed":
                        tgt_R = R_DOWN.copy()
                    if a["rot_lp"] > 0.0:                         # low-pass the orientation target (wrist chatter)
                        if R_lp is not None:
                            Ub, _, Vb = np.linalg.svd((1.0 - a["rot_lp"]) * R_lp + a["rot_lp"] * tgt_R)
                            tgt_R = Ub @ Vb
                        R_lp = tgt_R.copy()
            if plan is None:
                R_lp = None
            # ---- final-approach aim assist: close to the goal object the executor puts the lateral target on the object
            # (position from the scene here, from the ZED on hardware; --aim_err emulates its error); height and suction stay the policy's
            if a["aim"] > 0 and plan is not None and holding is None and S["latched"] is None and retreat_left == 0 \
                    and policy.use_goal and tgt_o is not None:
                _ap = d.xpos[bids[tgt_o["name"]]]
                _axy = _ap[:2] + (np.random.default_rng(idx * 977 + objs.index(tgt_o) + a["seed"]).normal(0, a["aim_err"], 2)
                                  if a["aim_err"] > 0 else 0.0)
                if np.hypot(*(p_cmd[:2] - _axy)) < a["aim"] and p_cmd[2] < _ap[2] + tgt_o["half_extents"][2] + a["aim_z"]:
                    tgt_p[:2] = _axy
                    stats["aim"] = stats.get("aim", 0) + 1
                    # descent assist: once centred the executor brings the cup down, opens the valve 5 cm above the top and
                    # presses 10 mm; after --aim_press s at the bottom without a seal it lets go of this object for 5 s
                    if a["aim_down"] > 0 and k >= aim_block and np.hypot(*(p_cmd[:2] - _axy)) < 0.010:
                        _top = _ap[2] + tgt_o["half_extents"][2]
                        _flo = _top + demo["CUP_R"] - 0.010
                        tgt_p[2] = min(tgt_p[2], max(_flo, p_cmd[2] - a["aim_down"] * dt))
                        if p_cmd[2] < _top + 0.05:
                            suc_pred = 1.0
                        aim_press = aim_press + 1 if p_cmd[2] < _flo + 0.003 else 0
                        if aim_press * dt > a["aim_press"]:
                            aim_block, aim_press = k + int(5.0 / dt), 0
                        stats["aim_down"] = stats.get("aim_down", 0) + 1
            # ---- suction: z gate + release hysteresis (real_rollout_zed)
            if plan is not None and retreat_left == 0:
                if suc_pred > 0.5 and p_m[2] >= cfg["suction_z"] and S["cmd"] == 0:
                    off_count = 1
                elif suc_pred > 0.5:
                    off_count = 0
                    if S["cmd"] == 0:
                        S["cmd"], S["t_on"] = 1, tw.t
                else:
                    off_count += 1
                    if S["cmd"] == 1 and off_count >= cfg["suction_off_n"]:
                        S["cmd"], S["t_off"] = 0, tw.t
            # ---- retreat (safety): blocked for safe_block s -> re-anchor on the arm, lift 1.5 s, re-plan from a new look
            if retreat_left > 0:
                retreat_left -= 1
                holding, tgt_R = None, R_cmd.copy()
                tgt_p = p_cmd + np.array([0.0, 0.0, cfg["step_max_z"]])
                if a["retreat_in"]:                           # also back off toward the middle of the workspace: a pure lift can be
                    _v = RT_C - p_cmd[:2]                     # blocked itself when the obstacle is beside the arm (ZED mast)
                    tgt_p[:2] += _v * min(1.0, a["retreat_in"] * cfg["step_max"] / max(1e-6, float(np.linalg.norm(_v))))
                if retreat_left == 0:
                    plan, pending = None, None
            elif stalled:                                     # blocked at the same place stall_n times: stay lifted, stop trying
                holding, tgt_p, tgt_R = "stall", p_cmd.copy(), R_cmd.copy()
            # ---- guards
            step = tgt_p - p_cmd
            lat_v = np.array([step[0], step[1], 0.0])
            vert = np.array([0.0, 0.0, step[2]])
            nl, nv = float(np.linalg.norm(lat_v)), abs(float(step[2]))
            if nl > cfg["step_max"]:
                lat_v *= cfg["step_max"] / nl
                stats["clips"] += 1
            if nv > cfg["step_max_z"]:
                vert *= cfg["step_max_z"] / nv
                stats["clips"] += 1
            tgt_p = p_cmd + lat_v + vert
            tgt_p[2] = max(tgt_p[2], cfg["z_floor"])
            if tgt_p[0] < a["ws_xmin"]:                       # workspace: the tool target never goes behind x = ws_xmin
                tgt_p[0] = min(a["ws_xmin"], p_cmd[0] + cfg["step_max"]) if p_cmd[0] < a["ws_xmin"] else a["ws_xmin"]   # outside -> come back
                stats["ws_clip"] = stats.get("ws_clip", 0) + 1
            if RZ.tilt_deg(tgt_R) > cfg["max_tilt"]:
                tgt_R = R_cmd
            lead = float(np.abs(q_cmd - q_m).max())
            stats["lead"].append(np.degrees(lead))
            lead_soft = 0.6 * cfg["lead_max"]
            lead_scale = 1.0 if lead <= lead_soft else max(0.0, (cfg["lead_max"] - lead) / (cfg["lead_max"] - lead_soft))
            if lead_scale <= 0.0 and holding is None:
                holding = "lead"
            q_tgt = q_cmd.copy()
            if holding is None:
                q_ik, pe, re = RZ.ik(m, dik, tip_sid, tgt_p, tgt_R, q_cmd)
                if pe > 0.003 or re > 0.05:
                    holding = "ik"
                    stats["ik_fail"] += 1
                else:
                    dq = q_ik - q_cmd
                    lim = np.full(6, cfg["dq_max"])
                    if dq_wrist > 0:                           # separate (lower) step limit for joints 4-6
                        lim[3:] = dq_wrist
                    if dq_mid > 0 and (p_cmd[2] > dq_mid_z or (dq_mid_up and tgt_p[2] > p_cmd[2] + 0.002)):
                        lim[1:4] = dq_mid                      # fast only well above the table (or, optionally, going up)
                    mx = float((np.abs(dq) / lim).max())       # uniform scaling keeps the direction
                    if mx > 1.0:
                        dq /= mx
                        stats["dq_clip"] += 1
                    q_tgt = q_cmd + dq * lead_scale
            if FILT is not None and holding is None:
                q_tgt, finfo = FILT(q_cmd, q_tgt, dt)
                if finfo["alpha"] == 0.0:
                    stats["filt_block"] += 1
                    blocked_n += 1
            if FILT is not None:
                if holding is not None:
                    FILT.reset()
                blocked_n = blocked_n + 1 if holding == "lead" else (blocked_n if (holding is None and finfo["alpha"] == 0.0) else 0)
                if retreat_left > 0 or stalled:
                    blocked_n = 0
                if blocked_n >= BLK_N and retreat_left == 0:
                    blocked_n, retreat_left = 0, RT_N
                    stats["retreats"] += 1
                    rt_pos.append(p_cmd.copy())
                    if a["stall_n"] and sum(1 for _p in rt_pos if np.linalg.norm(_p - p_cmd) < 0.06) >= a["stall_n"]:
                        stalled = True
                        stats["stall_step"] = stats.get("stall_step", k)
                    q_tgt = q_m.copy()                       # re-anchor the command on the measured arm
                    holding = "retreat"
            if holding is not None:
                stats["hold"] += 1
            # ---- send (after host processing) anchored on the decision grid
            tw.advance(max(0.0, round(cfg["host_proc"] / cfg["cmd_dt"]) * cfg["cmd_dt"]))
            streamer.send(q_cmd, q_tgt, tw.t, hold=holding is not None, t_anchor_k=t_k)
            q_cmd = q_tgt
            prev_p_cmd = p_cmd.copy()
            p_cmd, R_cmd = fk(q_cmd)
            if a["trace"]:
                lo_ = S["latched"]
                TR["dec"].append(dict(
                    k=k, p_m=p_m, p_cmd=prev_p_cmd, tgt=tgt_p, suc_pred=suc_pred, cmd=S["cmd"], latched=lo_ is not None,
                    comp=compression(m, d) if PL is not None else 0.0, hold=holding or "",
                    clr=np.array(SF["cur"]), qd=np.degrees(np.abs(d.qvel[:6])), q=d.qpos[:6].copy(),
                    vsat=tw.stats["vsat"], ferr=tw.stats["ferror"],        # cumulative twin counters
                    i=-1 if plan is None else k - plan["k_obs"] - lat_shift,
                    raw=np.full(3, np.nan) if (plan is None or not 0 <= max(0, k - plan["k_obs"] - lat_shift) < T.TA)
                    else plan["p_abs"][max(0, k - plan["k_obs"] - lat_shift)],
                    goal=np.full(3, np.nan) if goal_rel is None else hist[-1]["p"] + hist[-1]["R"] @ np.asarray(goal_rel)[:3],
                    obj_z=np.nan if lo_ is None else d.xpos[bids[lo_]][2] - z0[lo_]))
                if a["trace"] >= 2:
                    TR["dec"][-1].update(
                        q_m=q_m.copy(), R_m=R_m.copy(), tokp=tr_tok, prop=tr_prop, **({"tokm": np.stack([_o[0]["zed_tokm"], _o[1]["zed_tokm"]])} if a["trace_mask"] else {}), lname=lo_ or "",
                        goal9=np.zeros(9, np.float32) if goal_rel is None else np.asarray(goal_rel, np.float32),
                        opos=np.array([d.xpos[bids[o["name"]]] for o in objs]),
                        oquat=np.array([d.xquat[bids[o["name"]]] for o in objs]),
                        oz0=np.array([z0[o["name"]] for o in objs]))
            # ---- scoring
            for o in objs:
                if d.xpos[bids[o["name"]]][2] > z0[o["name"]] + 0.06:
                    lifted.add(o["name"])
                if o["name"] not in binned and in_bin(o):
                    binned[o["name"]] = k
            if a["video"]:
                vw.append_data(np.hstack([cv2.cvtColor(wimg, cv2.COLOR_BGR2RGB), cv2.cvtColor(zimg, cv2.COLOR_BGR2RGB)]))
        vw.close()
        ren.close()
        if a["trace"]:
            out = {}
            for grp in ("dec", "inf"):
                for key in (TR[grp][0] if TR[grp] else {}):
                    out[f"{grp}_{key}"] = np.array([r[key] for r in TR[grp]])
            np.savez_compressed(os.path.join(a["out"], f"trace{idx:03d}.npz"), **out)
        if a["ticklog"]:
            np.savez_compressed(os.path.join(a["out"], f"ticks{idx:03d}.npz"), **{kk: np.array([r[kk] for r in tw.log])
                                                                                  for kk in ("t", "q", "q_ref", "v_cmd")})
        rq.put(dict(scene=idx, touched=S["touched"], sealed=S["ever"], lifted=len(lifted),
                    in_bin=sum(1 for o in objs if in_bin(o)), n_obj=len(objs),
                    place_steps=sorted(binned.values()), t_first_bin=(min(binned.values()) * dt if binned else None),
                    hold_frac=stats["hold"] / a["steps"], lead_p95=float(np.percentile(stats["lead"], 95)),
                    adopt=[int(x) for x in np.bincount(stats["adopt"])] if stats["adopt"] else [],
                    cam_age_ms=1000 * float(np.mean(stats["cam_age"])), dq_clip=stats["dq_clip"], goal_giveup=stats.get("goal_giveup", 0), aim=stats.get("aim", 0), aim_down=stats.get("aim_down", 0),
                    ik_fail=stats["ik_fail"], retreats=stats["retreats"], filt_block=stats["filt_block"],
                    filt=(FILT.n if FILT is not None else None), plunger=PL is not None, comp_max_mm=1000 * S["comp_max"],
                    bottom_out_s=S["bottom_ticks"] * cfg["cmd_dt"],
                    seal_comp_mm=None if S["seal_comp"] is None else 1000 * S["seal_comp"], vsat_frac=tw.stats["vsat"] / max(1, tw.stats["ticks"]),
                    ferror_frac=tw.stats["ferror"] / max(1, tw.stats["ticks"] * tw.sub),
                    safety=dict(arm_world_mm=1000 * SF["arm_world"], arm_obj_mm=1000 * SF["arm_obj"], cup_world_mm=1000 * SF["cup_world"],
                                carry_mm=1000 * SF["carry"], wall_mm=1000 * SF["wall"], joint_margin_deg=SF["jl"],
                                qd_max_deg=np.degrees(SF["qd"]).round(2).tolist(), pen_arm_s=SF["pen_arm"] * 0.02, qd_over36_s=SF.get("over", 0) * cfg["cmd_dt"],
                                near_arm_s=SF["near_arm"] * 0.02, arm_world_pair=SF.get("arm_world_pair"),
                                near_pairs_s={k_: round(v_ * 0.02, 2) for k_, v_ in SF.get("near_pairs", {}).items()}, ws_clip=stats.get("ws_clip", 0), stall_step=stats.get("stall_step"), pen_carry_s=SF["pen_carry"] * 0.02,
                                knocked=sum(1 for o in objs if o["name"] not in SF["touched_ever"]
                                            and np.linalg.norm(d.xpos[bids[o["name"]]][:2] - xy0[o["name"]]) > 0.03)),
                    cfg=dict(level=a["level"], dr=a["dr"], dead=cfg["dead"], zed_lat=cfg["zed_lat"],
                             infer_mean=cfg["infer_mean"], slow=a["slow"])))
    except Exception as e:
        import traceback
        rq.put({"scene": idx, "error": f"{e}\n{traceback.format_exc()[-1500:]}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--level", default="real", choices=["ideal", "drive", "drive+sensor", "drive+sensor+compute", "real"])
    ap.add_argument("--dr", action="store_true", help="randomise drive/latency per scene within the measured spread")
    ap.add_argument("--ticklog", type=int, default=0, help="1 = save the 10 ms reference / command / joint log")
    ap.add_argument("--safe", type=int, default=0, help="1 = safety filter + retreat rule (realsim/safety.py)")
    ap.add_argument("--safe_margin", type=float, default=0.005)
    ap.add_argument("--safe_wall", type=float, default=0.0, help="m, filter margin to the back wall (0 = safe_margin)")
    ap.add_argument("--ws_xmin", type=float, default=-9.0, help="m, lowest tool-target x (workspace limit toward the back wall)")
    ap.add_argument("--blend", type=int, default=0, help="cross-fade a new chunk over this many decisions (0 = switch)")
    ap.add_argument("--mast", type=int, default=1, help="1 = ZED mast pillar (realsim/mast.py): invisible, but a collision / keep-out body")
    ap.add_argument("--fast_infer", type=int, default=0, help="1 = fp16 encoder + half-weight compiled denoiser (same chunks to ~0.1 mm)")
    ap.add_argument("--euler", type=int, default=16, help="flow steps of the sampler")
    ap.add_argument("--rot_lp", type=float, default=0.0, help="orientation-target low-pass: weight of the new target per decision (0 = off)")
    ap.add_argument("--rtc", type=int, default=0, help="condition each new chunk on the old one over this many soft overlap steps (0 = off)")
    ap.add_argument("--rtc_hard", type=int, default=1, help="leading overlap steps held at full weight")
    ap.add_argument("--rtc_w", type=float, default=1.0, help="overall conditioning weight")
    ap.add_argument("--safe_alim", type=float, default=0.0, help="deg/s^2 joint acceleration limit in the filter (0 = off)")
    ap.add_argument("--safe_vlim", type=float, default=30.0, help="deg/s")
    ap.add_argument("--safe_block", type=float, default=1.0, help="s blocked before a retreat")
    ap.add_argument("--retreat_in", type=float, default=0.0, help="retreat also moves toward (0.30, 0.10) by this fraction of step_max per step")
    ap.add_argument("--stall_n", type=int, default=0, help="retreats within 6 cm of each other before the run stops trying (0 = off)")
    ap.add_argument("--slow", type=float, default=2.0, help="decision period = 0.1*slow s (hardware runs used 2)")
    ap.add_argument("--lat_shift", type=int, default=-1, help="override the ckpt lat_shift (-1 = ckpt)")
    ap.add_argument("--scenes", default="90-99")
    ap.add_argument("--steps", type=int, default=150, help="decisions per scene")
    ap.add_argument("--ds", default="~/pnp_dataset_zed")
    ap.add_argument("--gen", default="zed", choices=["zed", "div", "curobo"])
    ap.add_argument("--goal_err", type=float, default=0.0)
    ap.add_argument("--goal_to", type=float, default=0.0, help="s on one goal object without a seal before moving to another (0 = off)")
    ap.add_argument("--aim", type=float, default=0.0, help="m; within this lateral distance of the goal object the lateral target is put on the object (0 = off)")
    ap.add_argument("--aim_z", type=float, default=0.15, help="m above the object top below which --aim acts")
    ap.add_argument("--aim_err", type=float, default=0.0, help="m, sd of a fixed per-object xy error on the --aim position (object localisation error)")
    ap.add_argument("--aim_down", type=float, default=0.0, help="m/s; with --aim, once centred within 1 cm the executor descends at this rate, valve on 5 cm above the top (0 = off)")
    ap.add_argument("--aim_press", type=float, default=2.0, help="s pressed at the bottom without a seal before the descent assist releases the object for 5 s")
    ap.add_argument("--goal_skip", type=float, default=30.0, help="s a given-up goal object stays excluded")
    ap.add_argument("--cfg", nargs="*", default=[], help="config overrides key=val (dq_max/lead_max in deg)")
    ap.add_argument("--rigid_tip", action="store_true", help="old rigid cup (default: 15 mm sprung plunger)")
    ap.add_argument("--sim_ik_bounds", action="store_true", help="drop the LinuxCNC IK bounds (sim-only reach)")
    ap.add_argument("--dr_scene", action="store_true",
                    help="randomise the scene per episode: lights, colours, camera extrinsics, object pose/mass/"
                         "friction, plunger spring, seal threshold, vacuum onset, goal error")
    ap.add_argument("--stream", default="deployed", choices=["deployed", "tuned"],
                    help="Pi stream law: deployed (lead 45 ms on feedforward and position: arm leads its reference) or "
                         "tuned (realsim.config.STREAM_TUNED, 2026-11-14)")
    ap.add_argument("--seed", type=int, default=0, help="offset of the per-scene rng (DR draws)")
    ap.add_argument("--patch", default="", help="weight-space patch (.pt, see realsim/wsp.py) applied on top of --ckpt")
    ap.add_argument("--zed_mask", type=int, default=0, help="1 = the policy sees the table-only ZED view (policy/zed_mask.py)")
    ap.add_argument("--trace_mask", type=int, default=0, help="1 = also log the masked ZED pooled features (dec_tokm), needs --trace 2")
    ap.add_argument("--trace", type=int, default=0, help="1: per-decision + per-inference traces (trace###.npz); 2: per-decision pooled obs + "
                         "joints + object poses instead of the raw inference tokens (DAgger collection)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--serve", type=int, default=0, help="1 = one shared policy server on the first --gpus device; workers are CPU-only, "
                    "so --par can be 30-48 instead of 3 (6.4 GB of policy per worker otherwise)")
    ap.add_argument("--serve_batch", type=int, default=96)
    ap.add_argument("--video", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = vars(ap.parse_args())
    a["out"] = os.path.expanduser(a["out"])
    os.makedirs(a["out"], exist_ok=True)
    json.dump(a, open(os.path.join(a["out"], "args.json"), "w"), indent=1)
    s0, s1 = [int(x) for x in a["scenes"].split("-")]
    ids = list(range(s0, s1 + 1))
    active, done, pend = {}, {}, list(ids)
    ctx = mp.get_context("fork")
    srv = None
    if a["serve"]:
        sys.path.insert(0, ROOT)
        from realsim.policy_server import serve
        req_q, ready = ctx.Queue(), ctx.Queue()
        resp_qs = {i: ctx.Queue() for i in ids}
        srv = ctx.Process(target=serve, args=(a, req_q, resp_qs, ready), daemon=True)
        srv.start()
        srv_meta = ready.get(timeout=1200)
        print(f"[serve] policy server up on cuda:{a['gpus'].split(',')[0]}", flush=True)

    def launch(idx, att):
        rq = ctx.Queue()
        a_w = a if srv is None else dict(a, _link=(idx, req_q, resp_qs[idx], srv_meta))
        p = ctx.Process(target=eval_scene, args=(idx, a_w, rq))
        p.start()
        active[idx] = (p, rq, att)

    while pend or active:
        while pend and len(active) < a["par"]:
            launch(pend.pop(0), 1)
        time.sleep(2)
        for idx in list(active):
            p, rq, att = active[idx]
            if not rq.empty():
                r = rq.get()
                p.join()
                del active[idx]
                if "error" in r and "CORRUPT" in r["error"] and att < 4:
                    launch(idx, att + 1)
                else:
                    done[idx] = r
                    print(json.dumps({k: v for k, v in r.items() if k != "cfg"}), flush=True)
            elif not p.is_alive():
                del active[idx]
                if att < 4:
                    launch(idx, att + 1)
                else:
                    done[idx] = {"scene": idx, "error": "child died"}
    if srv is not None:
        req_q.put((-1, "stop", None))
        try:
            st = ready.get(timeout=60)
            print(f"[serve] {st['enc']} images in {st['enc_b']} batches ({1000 * st['t_enc'] / max(1, st['enc']):.1f} ms/image), "
                  f"{st['inf']} chunks in {st['inf_b']} batches ({1000 * st['t_inf'] / max(1, st['inf']):.1f} ms/chunk)", flush=True)
        except Exception:
            pass
    res = [done[i] for i in ids if i in done]
    ok = [r for r in res if "error" not in r]
    json.dump(res, open(os.path.join(a["out"], "results.json"), "w"), indent=1)
    n_obj = sum(r["n_obj"] for r in ok)
    print(f"SUMMARY {a['level']}{' dr' if a['dr'] else ''} slow {a['slow']}: {len(ok)} scenes  "
          f"touch {sum(r['touched'] for r in ok)}  seal {sum(r['sealed'] for r in ok)}  "
          f"lift {sum(r['lifted'] > 0 for r in ok)}  in_bin {sum(r['in_bin'] for r in ok)}/{n_obj} objects "
          f"({sum(r['in_bin'] > 0 for r in ok)} scenes)", flush=True)


if __name__ == "__main__":
    main()

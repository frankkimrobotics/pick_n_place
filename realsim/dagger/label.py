#!/usr/bin/env python3
"""realsim.dagger.label :: expert labels for the states a policy visited in the twin (DAgger).

Expert = the mega demo generator's own post-latch procedure (dataset_gen_curobo.pick_object steps 3-4),
re-run FROM EACH VISITED STATE with the same cuRobo server (:9996):

    latched, below hover height  -> scripted straight lift to (rest top + 12 cm), min-jerk, from that state
    then                         -> attach object, plan_pose to above the box, plan_pose down into the mouth,
                                    release (suction 0), 0.5 s, straight retreat: planned ONCE per grasp spot
                                    (attach costs 6-24 s); states off the lift column rejoin that path at its
                                    nearest point, their offset decaying over ~0.5 s

The joint path is sampled like the demos (4 ms ticks, 10 Hz frames) and turned into the policy's
action: tool-frame poses of frames t+1+lat_shift+k, k = 0..15, suction per frame. Only LATCHED
states get a label (the privileged part: the twin knows the seal holds; the robot has no sensor).

  ~/miniconda3/envs/mj4/bin/python realsim/dagger/label.py ~/pnp_rollouts_realsim/dagger/<run> [--stride 1]
reads trace###.npz written by eval_policy.py --trace 2, writes labels.npz next to them.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "policy")]
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
import dataset_gen_curobo as C  # noqa: E402

from retime import retime_path  # noqa: E402

DS = os.path.expanduser("~/pnp_dataset_mega")
VLIM, ALIM = 0.0, 150.0                    # --vlim / --alim: expert legs slowed to the drive's limits (0 = off)


def lim(path):
    return list(retime_path(np.array(path), VLIM, ALIM, C.CTRL_DT)) if VLIM > 0 and len(path) > 3 else list(path)
LAT, TA, EVERY = 2, 16, 25                       # ckpt lat_shift, horizon, ticks per 10 Hz frame
demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
CUP_R, seg = demo["CUP_R"], demo["_seg"]
QUATS = [(demo["R_DOWN"], C.R_DOWN_QUAT), (demo["R_FLAT"], [0.0, 1.0, 0.0, 0.0])]


def rot6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])


class SceneExpert:
    def __init__(self, idx, pl):
        sd = os.path.join(DS, f"scene_{idx:03d}")
        self.box = json.load(open(os.path.join(sd, "meta.json")))["box"]["xy"]
        self.objs = C.sample_objects(idx, tuple(self.box))
        self.m = mujoco.MjModel.from_xml_path(os.path.join(sd, "scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.pl = pl
        self.n_plan = 0
        self.direct = 0.0                            # lateral radius of the direct (no lift) approach, 0 = off
        self._wkey = None

    def fk(self, q):
        self.d.qpos[:6] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, p, R, q_seed):
        q, pe, re = demo["_ik_once"](self.m, self.d, self.sid, np.asarray(p, float), R, np.asarray(q_seed, float))
        return (np.asarray(q, float), pe, re) if q is not None else (None, np.inf, np.inf)

    def dims(self, o, inflate):
        h = o["half_extents"]
        return [2 * h[0] + inflate, 2 * h[1] + inflate, 2 * h[2] + inflate]

    def world(self, oi, opos):
        cub = dict(C.base_world())
        cub.update(C.box_cuboids(tuple(self.box)))
        for j, o in enumerate(self.objs):
            if j == oi or opos[j][2] < -0.05:
                continue
            cub[o["name"]] = {"dims": self.dims(o, 0.015), "pose": [*[float(v) for v in opos[j]], 1, 0, 0, 0]}
        return [{"name": k, **v} for k, v in cub.items()]

    def plan(self, q, xyz, quat):
        self.n_plan += 1
        res = self.pl.plan_pose(q, xyz, quat=quat)
        if not res.get("success") or len(res.get("trajectory", [])) < 2:
            return None
        traj, dt = np.asarray(res["trajectory"], float), float(res["dt"])
        t_in = np.arange(len(traj)) * dt
        cs = CubicSpline(t_in, traj, axis=0, bc_type="clamped")
        return np.vstack([np.asarray(q, float)[None], cs(np.arange(0.0, t_in[-1] + 1e-9, C.CTRL_DT))])

    def event(self, q, oi, opos, oquat, oz0):
        """One carry/lower/release/retreat plan per (object, grasp spot): the generator's steps 3-4 from the
        top of the lift. attach() costs 6-24 s on the server, so it is not re-planned per state."""
        o = self.objs[oi]
        h_obj = 2 * o["half_extents"][2]
        hover_z = float(oz0[oi] + o["half_extents"][2]) + 0.12
        bx, by = self.box
        above = [bx, by, C.BOX_H + h_obj + 0.05 + CUP_R]
        low = [bx, by, C.BOX_H - 0.03 + h_obj + CUP_R]
        tip, R = self.fk(q)
        quat = min(QUATS, key=lambda rq: -float(np.trace(rq[0].T @ R)))[1]
        q_up, pe, re = self.ik([tip[0], tip[1], hover_z], R, q)
        if q_up is None or pe > 0.003 or re > 0.05:
            return None
        self.pl.set_world(self.world(oi, opos))
        self._wkey = None
        obj_p = np.asarray(opos[oi], float) + [0, 0, hover_z - tip[2]]
        self.pl.attach(q_up, self.dims(o, 0.01), [*obj_p.tolist(), *[float(v) for v in oquat[oi]]])
        carry = self.plan(q_up, above, quat)
        if carry is None:
            self.pl.detach()
            return None
        path, suc = list(carry), [1] * len(carry)
        lower = self.plan(carry[-1], low, quat)
        if lower is None:                                             # generator: scripted leg, else drop from the rim
            _, Rn = self.fk(carry[-1])
            q_low, pe, re = self.ik(low, Rn, carry[-1])
            lower = []
            if q_low is not None and pe < 0.003 and re < 0.05:
                seg(lower, carry[-1], q_low, 1.0)
        self.pl.detach()
        path += list(lower[1:])
        path = lim(path); suc = [1] * len(path)
        n_on = len(path)
        q_now = np.asarray(path[-1], float)
        path += [q_now] * int(0.5 / C.CTRL_DT); suc += [0] * int(0.5 / C.CTRL_DT)      # release, let it fall
        _, Rn = self.fk(q_now)
        q_ret, pe, re = self.ik([bx, by, C.BOX_H + 0.20], Rn, q_now)
        if q_ret is not None and pe < 0.003 and re < 0.05:
            leg = []
            seg(leg, q_now, q_ret, 1.0)
            leg = lim(leg)
            path += leg[1:]; suc += [0] * (len(leg) - 1)
        P, Rs = zip(*[self.fk(qq) for qq in path])
        return dict(anchor=tip[:2].copy(), hover_z=hover_z, P=np.array(P), R=np.array(Rs), suc=np.array(suc),
                    n_on=n_on, f_last=0)

    def expert(self, ev, q):
        """-> (P [N,3], R [N,3,3], suction [N], phase) world-frame expert tool path from joint state q, or None."""
        need = (1 + LAT + TA) * EVERY + 1
        tip, R = self.fk(q)
        if np.hypot(*(tip[:2] - ev["anchor"])) < 0.025 and tip[2] < ev["hover_z"] - 0.01:
            # lift_clear(top + 0.12, T = 1.2): straight up from wherever the tool is, then the carry
            dz = ev["hover_z"] - tip[2]
            q_up, pe, re = self.ik([tip[0], tip[1], ev["hover_z"]], R, q)
            if q_up is None or pe > 0.003 or re > 0.05:
                return None
            leg = []
            seg(leg, np.asarray(q, float), q_up, 1.2 * float(np.clip(np.sqrt(dz / 0.12), 0.35, 1.0)))
            leg = lim(leg)
            Pl, Rl = zip(*[self.fk(qq) for qq in leg[:need]])
            P, Rs, suc, phase, f = list(Pl), list(Rl), [1] * len(Pl), "lift", 0
            off = P[-1] - ev["P"][0]
        else:
            # off the lift column: rejoin the planned carry at its nearest point (never backwards) and continue
            f0 = max(0, ev["f_last"] - 50)
            dist = np.linalg.norm(ev["P"][f0:ev["n_on"]] - tip, axis=1)
            f = f0 + int(np.argmin(dist))
            if dist.min() > 0.10:
                return None
            ev["f_last"] = f
            P, Rs, suc, off = [], [], [], tip - ev["P"][f]
            phase = "carry" if f < ev["n_on"] - 25 else "release"
        n0 = len(P)
        for i in range(need - n0):                                    # carry path, start offset decaying over ~0.5 s
            j = min(f + i + (1 if n0 else 0), len(ev["P"]) - 1)
            P.append(ev["P"][j] + off * np.exp(-i / 125.0)); Rs.append(ev["R"][j]); suc.append(int(ev["suc"][j]))
        return np.array(P[:need]), np.array(Rs[:need]), np.array(suc[:need]), phase

    def target(self, q, opos):
        """index of the object the generator would go for from here: the nearest one still on the table."""
        tip, _ = self.fk(q)
        bx, by = self.box
        best, bd = None, 1e9
        for j, o in enumerate(self.objs):
            p = opos[j]
            if p[2] < -0.05 or (abs(p[0] - bx) < C.BOX_FP / 2 and abs(p[1] - by) < C.BOX_FP / 2) \
                    or not (0.15 < p[0] < 0.75 and abs(p[1]) < 0.45):
                continue
            dd = float(np.hypot(*(tip[:2] - p[:2])))
            if dd < bd:
                best, bd = j, dd
        return best

    def approach(self, q, oi, opos):
        """Pre-seal expert (generator steps 1-2) from joint state q: [lift clear ->] planned move to the hover pose
        (top + 12 cm) -> 1.6 s descent to the surface -> 0.7 s press of 10 mm, pump on 5 cm above the object.
        Within 3 cm laterally the tool just goes (on) down from where it is. -> (P, R, suction, phase) or None."""
        need = (1 + LAT + TA) * EVERY + 1
        o = self.objs[oi]
        top = float(opos[oi][2] + o["half_extents"][2])
        c = np.asarray(opos[oi][:2], float)
        tip, R = self.fk(q)
        Rd, quat = min(QUATS, key=lambda rq: -float(np.trace(rq[0].T @ R)))
        q = np.asarray(q, float)
        path, phase = [q], "press"
        d_lat = float(np.hypot(*(tip[:2] - c)))
        if d_lat > 0.03 and d_lat <= self.direct and tip[2] >= top + CUP_R + 0.015:
            # close and already clear of the top: slide straight over at this height, no lift / hover plan
            phase = "approach"
            q_ov, pe, re = self.ik([float(c[0]), float(c[1]), max(float(tip[2]), top + CUP_R + 0.03)], R, q)
            if q_ov is None or pe > 0.003 or re > 0.05:
                return None
            seg(path, q, q_ov, max(0.35, d_lat / 0.12))
        elif d_lat > 0.03:
            phase = "approach"
            if tip[2] < 0.15:                                         # lift_clear(0.15): never plan from a low / touching start
                q_up, pe, re = self.ik([tip[0], tip[1], 0.15], R, q)
                if q_up is None or pe > 0.003 or re > 0.05:
                    return None
                seg(path, q, q_up, max(0.35, 1.0 * (0.15 - tip[2]) / 0.15))
            if len(path) < need:
                key = (oi, tuple(np.round(np.asarray(opos).ravel(), 2)))
                if self._wkey != key:
                    self.pl.set_world(self.world(oi, opos))
                    self._wkey = key
                hov = self.plan(path[-1], [float(c[0]), float(c[1]), top + 0.12], quat)
                if hov is None:
                    return None
                path += list(hov[1:])
        if len(path) < need:
            tn, Rn = self.fk(path[-1])
            grasp = [float(c[0]), float(c[1]), top + CUP_R]
            if tn[2] > grasp[2] + 0.002:
                q_g, pe, re = self.ik(grasp, Rn, path[-1])
                if q_g is None or pe > 0.003 or re > 0.05:
                    return None
                seg(path, path[-1], q_g, 1.6 * float(np.clip(np.sqrt((tn[2] - grasp[2]) / 0.12), 0.25, 1.0)))
            q_p, pe, re = self.ik([grasp[0], grasp[1], grasp[2] - 0.010], Rn, path[-1])
            if q_p is None or pe > 0.003 or re > 0.05:
                return None
            seg(path, path[-1], q_p, 0.7)
        path = lim(path)
        path = path[:need] + [path[-1]] * max(0, need - len(path))
        P, Rs = zip(*[self.fk(qq) for qq in path])
        P = np.array(P)
        suc = (P[:, 2] - CUP_R - top <= 0.05) & (np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < 0.03)
        return P, np.array(Rs), suc.astype(int), phase

    @staticmethod
    def chunk(R_m, p_m, P, Rs, suc):
        """tool-frame action waypoints [16,10] relative to the MEASURED pose the policy conditions on."""
        out = np.zeros((TA, 10), np.float32)
        for k in range(TA):
            i = (1 + LAT + k) * EVERY
            out[k] = np.concatenate([R_m.T @ (P[i] - p_m), rot6d(R_m.T @ Rs[i]), [suc[i]]])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stride", type=int, default=1, help="label every n-th latched decision once the object is off the table")
    ap.add_argument("--port", type=int, default=9996)
    ap.add_argument("--pre", type=int, default=0, help="also label every n-th NOT latched decision with the approach / press expert (0 = off)")
    ap.add_argument("--out", default="labels.npz")
    ap.add_argument("--direct", type=float, default=0.0, help="within this lateral distance (m) and clear of the top: slide over, no lift")
    ap.add_argument("--reuse", default="", help="labels file in the run dir whose post-seal rows are copied instead of re-planned")
    ap.add_argument("--vlim", type=float, default=0.0, help="joint speed limit of the expert legs (deg/s); 0 = off")
    ap.add_argument("--alim", type=float, default=150.0, help="joint acceleration limit used with --vlim (deg/s^2)")
    a = ap.parse_args()
    global VLIM, ALIM
    VLIM, ALIM = a.vlim, a.alim
    run = os.path.expanduser(a.run)
    pl = C.Planner("127.0.0.1", a.port)
    rows = {k: [] for k in ("tok", "prop", "goal", "act", "scene", "k", "phase", "suc_pred", "obj_z", "comp")}
    t0, n_fail = time.time(), 0
    for f in sorted(glob.glob(os.path.join(run, "trace*.npz"))):
        idx = int(f[-7:-4])
        t = dict(np.load(f, allow_pickle=True))
        if "dec_lname" not in t:
            continue
        ex = SceneExpert(idx, pl)
        ex.direct = a.direct
        names = [o["name"] for o in ex.objs]
        n0, events = len(rows["k"]), {}
        for j in ([] if a.reuse else np.where(t["dec_lname"] != "")[0]):
            lifted = t["dec_obj_z"][j] > 0.02
            oi = names.index(str(t["dec_lname"][j]))
            q = t["dec_q_m"][j]
            ev = events.get(oi)
            if not lifted and (ev is None or (ev and np.hypot(*(ex.fk(q)[0][:2] - ev["anchor"])) > 0.015)):
                ev = events[oi] = ex.event(q, oi, t["dec_opos"][j], t["dec_oquat"][j], t["dec_oz0"][j]) or False
            if not ev or (lifted and (j % a.stride)):
                n_fail += int(not ev)
                continue
            r = ex.expert(ev, q)
            if r is None:
                n_fail += 1
                continue
            P, Rs, suc, phase = r
            rows["act"].append(ex.chunk(t["dec_R_m"][j], t["dec_p_m"][j], P, Rs, suc))
            rows["tok"].append(t["dec_tokp"][j]); rows["prop"].append(t["dec_prop"][j].astype(np.float32))
            rows["goal"].append(t["dec_goal9"][j]); rows["scene"].append(idx); rows["k"].append(int(t["dec_k"][j]))
            rows["phase"].append(phase); rows["suc_pred"].append(float(t["dec_suc_pred"][j]))
            rows["obj_z"].append(float(t["dec_obj_z"][j])); rows["comp"].append(float(t["dec_comp"][j]))
        if a.pre:                                                     # pre-seal states: approach / press labels
            n1 = len(rows["k"])
            for j in np.where(t["dec_lname"] == "")[0][::a.pre]:
                oi = ex.target(t["dec_q_m"][j], t["dec_opos"][j])
                if oi is None:
                    continue
                r = ex.approach(t["dec_q_m"][j], oi, t["dec_opos"][j])
                if r is None:
                    n_fail += 1
                    continue
                P, Rs, suc, phase = r
                rows["act"].append(ex.chunk(t["dec_R_m"][j], t["dec_p_m"][j], P, Rs, suc))
                rows["tok"].append(t["dec_tokp"][j]); rows["prop"].append(t["dec_prop"][j].astype(np.float32))
                rows["goal"].append(t["dec_goal9"][j]); rows["scene"].append(idx); rows["k"].append(int(t["dec_k"][j]))
                rows["phase"].append(phase); rows["suc_pred"].append(float(t["dec_suc_pred"][j]))
                rows["obj_z"].append(float(t["dec_obj_z"][j])); rows["comp"].append(float(t["dec_comp"][j]))
            print(f"scene {idx}: pre-seal labelled {len(rows['k']) - n1}", flush=True)
        print(f"scene {idx}: latched {int((t['dec_lname'] != '').sum())} labelled {len(rows['k']) - n0} "
              f"plans {ex.n_plan} ({time.time() - t0:.0f} s)", flush=True)
    out = {k: np.array(v) for k, v in rows.items()}
    if a.reuse:
        z = np.load(os.path.join(run, a.reuse), allow_pickle=True)
        keep = ~np.isin(z["phase"], ["approach", "press"])
        out = {k: (np.concatenate([out[k], z[k][keep]]) if len(out[k]) else z[k][keep]) for k in out}
    np.savez_compressed(os.path.join(run, a.out), **out)
    ph = out["phase"] if len(out["phase"]) else np.array([])
    print(f"LABELS {run}: {len(ph)} states, expert failed {n_fail}; phases "
          f"{ {p: int((ph == p).sum()) for p in np.unique(ph)} }; policy suction<0.5 on "
          f"{100 * float((out['suc_pred'] < 0.5).mean()) if len(ph) else 0:.0f} %", flush=True)


if __name__ == "__main__":
    main()

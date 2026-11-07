#!/usr/bin/env python3
"""dataset_gen_curobo :: expert pick-and-place demonstrations with cuRobo (v0.9 / cuRoboV2-Fast)
motion planning in the MuJoCo twin: clear a cluttered table into a 25 cm cube BOX whose position
is given at run time.

Scene (per scene id): 2-9 random objects (YCB meshes + primitives, dataset_gen_ycb.sample_scene)
on the table, kept out of the box footprint; the box = 25 cm inner cube, 1 cm walls, floor on the
table (sim_robot_mjcf box_* geoms), at --box X Y (base frame, m) or random in --box_range.
Cameras: wrist D405 (hand-eye) + fixed ZED (calibrated pose), rendered at 10 Hz like the other
generators (d405_rgb.mp4 / d435_rgb.mp4 = ZED slot).

Expert, per object (nearest to the robot first, up to --attempts tries each):
  1. plan_pose  (cuRobo RPC)  current q -> hover 12 cm above the (noisy, --obs_noise) object top,
     world = ground + ZED column + box walls + every OTHER object as an inflated cuboid
  2. scripted descent with the PUMP RUNNING (real sequence): the valve opens --suction_on_dist above
     the object top, and from then on every control tick asks whether the cup seals NOW --
     contact (the compliant lip gives 0.1-0.4 N at 5-15 mm, measured), normal tilt < SEAL_DEG,
     and rim_seal.py flatness (a 7.5 mm lip cannot conform to curvature > 2 mm). Within --snap_gap of
     a surface (or on contact) the airflow pulls it onto the lip (--snap_force) AND draws it onto the
     cup axis (--center_force within --capture_r), so a light object that the cup would otherwise shove
     away is captured, self-centres and seals -- the reason a running cup sticks on first touch. The latch ends the
     descent (no further press). Failures: "no_contact", "no_rim_seal" (curvature/edge),
     "tilted" (never met the surface face-on), "weak_contact", "pushed" (slid out from under the cup
     without ever sealing), "touched_other"
  3. attach the object to the planner's tool (its cuboid), plan_pose grasp -> above the box
     (object bottom 5 cm above the rim), world = other objects + box walls
  4. scripted lowering into the box mouth (object bottom 3 cm below the rim), release, detach,
     plan_pose straight back up to the box-top retreat pose; "box_contact" counted when the cup
     or the carried object touches a box wall during the carry/lower legs
  5. success = object ends inside the box (xy within the inner footprint, z below the rim)
Every planned trajectory (25 ms samples) is resampled to the 4 ms control tick and tracked by the
demos' computed-torque PD (mjwarp_pick_demo._gains_and_ff / pd_tau).

Output: <out>/scene_NNN/{scene.xml, d405_rgb.mp4, d435_rgb.mp4, traj.npz, eef.npz, meta.json}
meta.json: box pose, objects (init/final pose, in_box), picks[] (attempt, planned grasp, executed
grasp, fail_reason, planner times, box_contacts, moved_mm), totals.

Needs the planner server with v0.9:  (in another shell)
    cd ~/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo
    CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/curobo3/bin/python curobo_planner_server_v2.py --ground-z -0.1 --port 9996 --collision-cache 32
Run (mjwarp env):
    MUJOCO_GL=osmesa ~/miniconda3/envs/mjwarp/bin/python dataset_gen_curobo.py --count 2 --box 0.30 0.34 --out ~/pnp_dataset_curobo
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_gen_ycb as G  # noqa: E402  (sample_scene, LIGHTS, W, H, TABLE_X/Y, REC_HZ, CUROBO_PY)

BOX_FP, BOX_H, BOX_WALL = 0.25, 0.25, 0.01
ZED_CAM = ('<camera name="zed" pos="0.6014 -0.0870 0.6772" '
           'xyaxes="0.0723 0.9968 0.0349 -0.9062 0.0510 0.4197" fovy="69.1"/>')
R_DOWN_QUAT = [0.0, 0.7071, 0.7071, 0.0]        # wxyz of the demos' R_DOWN (cup down, +90 deg yaw)
CTRL_DT = 0.004


# --------------------------------------------------------------------------- planner RPC
class Planner:
    def __init__(self, host, port):
        self.addr = (host, port)
        r = self.rpc({"type": "ping"})
        assert r.get("ok"), r
        self.backend = r.get("backend")

    def rpc(self, req, timeout=120):
        s = socket.create_connection(self.addr, timeout=timeout)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
        s.close()
        return json.loads(buf.split(b"\n")[0])

    def set_world(self, cuboids):
        return self.rpc({"type": "set_world", "cuboids": cuboids})

    def plan_pose(self, q, xyz, quat=R_DOWN_QUAT, attempts=4):
        return self.rpc({"type": "plan_pose", "start_q": [float(v) for v in q],
                         "goal_pose": [float(v) for v in xyz] + list(quat), "max_attempts": attempts})

    def plan_joint(self, q, q_goal, attempts=4):
        return self.rpc({"type": "plan_joint", "start_q": [float(v) for v in q],
                         "goal_q": [float(v) for v in q_goal], "max_attempts": attempts})

    def attach(self, q, dims, pose):
        return self.rpc({"type": "attach", "grasp_q": [float(v) for v in q], "dims": [float(v) for v in dims],
                         "obj_pose": [float(v) for v in pose]})

    def detach(self):
        return self.rpc({"type": "detach"})


# --------------------------------------------------------------------------- scene
def sample_objects(idx, box_xy, margin=0.03):
    """dataset_gen_ycb clutter, minus anything whose footprint overlaps the box footprint."""
    objs = G.sample_scene(idx)
    keep = []
    for o in objs:
        ext = float(max(o["half_extents"][0], o["half_extents"][1])) * 1.42
        if (abs(o["pos"][0] - box_xy[0]) < BOX_FP / 2 + BOX_WALL + ext + margin
                and abs(o["pos"][1] - box_xy[1]) < BOX_FP / 2 + BOX_WALL + ext + margin):
            continue
        keep.append(o)
    for i, o in enumerate(keep):                      # names must stay object0..N-1 (weld names)
        o["name"] = f"object{i}"
    return keep


def build_scene_xml(objs, light, box_xy, out_xml):
    tuples = [(o["name"], o["shape"], o["size"], " ".join(f"{v:.4f}" for v in o["pos"]),
               " ".join(f"{v:g}" for v in o["rgba"]), o["quat"], o.get("gpos", "")) for o in objs]
    code = (f"import sys; sys.path.insert(0, {HERE!r});\nimport sim_robot_mjcf as g;\n"
            f"g.build(warp=True, objects={tuples!r}, light_pos={tuple(light)!r}, out_path={out_xml!r}, "
            f"box_xyz=({box_xy[0]!r}, {box_xy[1]!r}, 0.0), box_fp={BOX_FP!r}, box_h={BOX_H!r}, box_wall={BOX_WALL!r})")
    subprocess.run([G.CUROBO_PY, "-c", code], check=True, capture_output=True)
    x = open(out_xml).read()
    # Compliant, grippy suction lip. A rigid 8 mm sphere with default contact delivers a ~30 N spike on
    # first touch (measured at the latch), which launches light curved objects before the vacuum can hold
    # them; a rubber lip deforms instead and grips. priority=1 makes these params win over the object's.
    # robot VISUAL meshes (contype=0) into geom group 3 so the seal ray-casts can ignore them:
    # mj_ray is contype-blind and the suction-cup shell sits ~3.5 mm above the lip, right on the centre ray.
    x = re.sub(r'(<geom\b[^>]*?contype="0"[^>]*?)/>', r'\1 group="3"/>', x)
    x = x.replace('<geom name="cup_tip" type="sphere" size="0.008"',
                  '<geom name="cup_tip" type="sphere" size="0.008" priority="1" '
                  'solref="0.04 1.4" solimp="0.5 0.9 0.01" friction="1.8 0.05 0.002"', 1)
    open(out_xml, "w").write(x)
    if 'name="zed"' not in x:
        x = x.replace('<camera name="fixed_d435"', ZED_CAM + '\n    <camera name="fixed_d435"', 1)
        open(out_xml, "w").write(x)


def box_cuboids(box_xy, inflate=0.01):
    """The five box walls for the planner (inflated), same geometry as sim_robot_mjcf."""
    bx, by = box_xy
    h, t, H = BOX_FP / 2, BOX_WALL, BOX_H
    walls = {
        "box_floor": ([2 * h + 2 * t, 2 * h + 2 * t, t], [bx, by, -t / 2]),
        "box_xm": ([t, 2 * h, H], [bx - (h + t / 2), by, H / 2]),
        "box_xp": ([t, 2 * h, H], [bx + (h + t / 2), by, H / 2]),
        "box_ym": ([2 * h + 2 * t, t, H], [bx, by - (h + t / 2), H / 2]),
        "box_yp": ([2 * h + 2 * t, t, H], [bx, by + (h + t / 2), H / 2]),
    }
    return {k: {"dims": [d_ + inflate for d_ in dims], "pose": [*p, 1, 0, 0, 0]} for k, (dims, p) in walls.items()}


def base_world():
    """Overrides of the server's base cuboids (server ground stays at z -0.1 so the base link's spheres are
    not in collision): the table top at z 0 in front of the robot only, ZED column instead of the old D435 mount."""
    return {
        "table": {"dims": [0.60, 1.20, 0.04], "pose": [0.48, 0.0, -0.02, 1, 0, 0, 0]},
        "camera_mount_d435": {"dims": [0.12, 0.12, 0.90], "pose": [0.62, -0.09, 0.35, 1, 0, 0, 0]},
    }


# --------------------------------------------------------------------------- per-scene
def run_scene(args):
    idx, out_root, a = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import imageio
    import mujoco
    from scipy.interpolate import CubicSpline
    from rim_seal import rim_seal
    demo = {"__file__": os.path.join(HERE, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    rng = np.random.default_rng(idx * 104729 + 7)
    box_xy = tuple(a.box) if a.box else (float(rng.uniform(*a.box_range[0:2])), float(rng.uniform(*a.box_range[2:4])))
    sdir = os.path.join(out_root, f"scene_{idx:03d}")
    os.makedirs(sdir, exist_ok=True)
    objs = sample_objects(idx, box_xy)
    xml = os.path.join(sdir, "scene.xml")
    build_scene_xml(objs, G.LIGHTS[idx % 3], box_xy, xml)
    pl = Planner(a.host, a.port)

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    d.qpos[:6] = demo["Q_START"]
    mujoco.mj_forward(m, d)
    dik = mujoco.MjData(m)
    tcp_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    tcp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
    tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip_rf")
    cup_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")
    rf_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "tip_range")]
    box_gids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"box_{n}") for n in ("floor", "xm", "xp", "ym", "yp")}
    ob = []
    for o in objs:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
        ob.append(dict(name=o["name"], bid=bid, half=o["half_extents"],
                       eq=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{o['name']}"),
                       gids=set(np.where(m.geom_bodyid == bid)[0].tolist()), kind=o["kind"], shape=o["shape"]))
    ren = mujoco.Renderer(m, height=G.H, width=G.W)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0
    vopt.geomgroup[3] = 1      # the robot's visual meshes live in group 3 (masked from the seal ray-casts,
    #                            NOT from the cameras); MuJoCo renders only groups 0-2 by default, so
    #                            without this the arm is invisible in the recorded streams.
    every = int(round(1.0 / (G.REC_HZ * CTRL_DT)))
    wr405 = imageio.get_writer(os.path.join(sdir, "d405_rgb.mp4"), fps=G.REC_HZ, quality=8)
    wr435 = imageio.get_writer(os.path.join(sdir, "d435_rgb.mp4"), fps=G.REC_HZ, quality=8)
    rec = dict(t=[], q=[], qd=[], rf=[], suction=[], target=[], tcp_pos=[], tcp_quat=[])
    state = dict(tick=0, suction=0, target=-1, box_contacts=0, watch_box=False)
    picks = []

    def obj_top(o):
        return float(d.xpos[o["bid"]][2] + o["half"][2])

    def obj_dims(o, inflate=0.01):
        return [2 * o["half"][0] + inflate, 2 * o["half"][1] + inflate, 2 * o["half"][2] + inflate]

    def record():
        if state["tick"] % every:
            return
        for wr, cam in ((wr405, "wrist_d405"), (wr435, "zed")):
            ren.update_scene(d, camera=cam, scene_option=vopt)
            wr.append_data(ren.render())
        q4 = np.zeros(4)
        mujoco.mju_mat2Quat(q4, d.site_xmat[tcp_sid])
        rec["t"].append(state["tick"] * CTRL_DT)
        rec["q"].append(d.qpos[:6].copy())
        rec["qd"].append(d.qvel[:6].copy())
        rec["rf"].append(float(d.sensordata[rf_adr]))
        rec["suction"].append(state["suction"])
        rec["target"].append(state["target"])
        rec["tcp_pos"].append(d.site_xpos[tcp_sid].copy())
        rec["tcp_quat"].append(q4)

    def tick(sched, k, held=None):
        for _ in range(demo["NSUB"]):
            d.ctrl[:6] = demo["pd_tau"](sched, k, d.qpos[:6], d.qvel[:6])
            mujoco.mj_step(m, d)
        if state["watch_box"]:
            mine = {cup_gid} | (held["gids"] if held else set())
            for i in range(d.ncon):
                c = d.contact[i]
                if (c.geom1 in box_gids and c.geom2 in mine) or (c.geom2 in box_gids and c.geom1 in mine):
                    state["box_contacts"] += 1
                    break
        record()
        state["tick"] += 1

    def run_qref(qref, on_tick=None, held=None, stop=None):
        qdref, tau_ff, kp, kd = demo["_gains_and_ff"](m, d, qref)
        sched = dict(qref=qref, qdref=qdref, tau_ff=tau_ff, kp=kp, kd=kd)
        for k in range(len(qref)):
            if on_tick is not None:
                on_tick(k)
            tick(sched, k, held)
            if stop is not None and stop():
                break

    def run_plan(res, held=None):
        """Resample a cuRobo plan (dt samples) to the 4 ms tick with a cubic spline and track it."""
        traj = np.asarray(res["trajectory"], float)
        dt = float(res["dt"])
        if len(traj) < 2:
            return
        t_in = np.arange(len(traj)) * dt
        cs = CubicSpline(t_in, traj, axis=0, bc_type="clamped")
        t_out = np.arange(0.0, t_in[-1] + 1e-9, CTRL_DT)
        qref = np.vstack([d.qpos[:6].copy()[None], cs(t_out)])      # start from the measured joints
        run_qref(qref, held=held)

    def world_for(exclude=None, with_box=True):
        cub = dict(base_world())
        if with_box:
            cub.update(box_cuboids(box_xy))
        for o in ob:
            if o is exclude:
                continue
            p = d.xpos[o["bid"]]
            if p[2] < -0.05:
                continue                                              # fell off the table
            cub[o["name"]] = {"dims": obj_dims(o, 0.015), "pose": [float(p[0]), float(p[1]), float(p[2]), 1, 0, 0, 0]}
        return [{"name": k, **v} for k, v in cub.items()]

    def in_box(o):
        p = d.xpos[o["bid"]]
        return bool(abs(p[0] - box_xy[0]) < BOX_FP / 2 and abs(p[1] - box_xy[1]) < BOX_FP / 2 and p[2] < BOX_H)

    def plan_log(pk, key, res):
        pk.setdefault("plans", {})[key] = dict(success=bool(res.get("success")), solve_ms=round(1000 * float(res.get("solve_time", 0) or 0), 1),
                                              motion_s=round(float(res.get("motion_time", 0) or 0), 3), n=len(res.get("trajectory", [])))

    GEOMGROUP = np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8)      # group 3 = robot visual meshes
    RAY_BACK = 0.03
    bid2obj = {o["bid"]: o for o in ob}

    def touched_object():
        """Object body the cup tip geom is currently in contact with (None if none)."""
        for i in range(d.ncon):
            c = d.contact[i]
            if cup_gid in (c.geom1, c.geom2):
                g = c.geom2 if c.geom1 == cup_gid else c.geom1
                b = int(m.geom_bodyid[g])
                if b in bid2obj:
                    return b
        return None

    def cup_ray():
        """(gap along the cup axis to the first surface, object body id or None)."""
        tip = d.site_xpos[tip_sid].copy()
        ax = d.site_xmat[tip_sid].reshape(3, 3)[:, 2]
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(m, d, tip - ax * RAY_BACK, ax, GEOMGROUP, 1, tcp_bid, gid)
        if dist < 0 or gid[0] < 0:
            return np.inf, None
        b = int(m.geom_bodyid[int(gid[0])])
        return float(dist) - RAY_BACK, (b if b in bid2obj else None)

    ik, seg = demo["ik"], demo["_seg"]
    R_DOWN = demo["R_DOWN"]
    CUP_R, PRESS = demo["CUP_R"], demo["PRESS_M"]
    retreat_z = BOX_H + 0.20
    t_scene = time.time()

    def lift_clear(z_min, held=None, T=1.0):
        """Scripted straight-up leg to tip z >= z_min: cuRobo must never start a plan with the cup pressed
        into an object / the carried object inside the table slab."""
        tip = d.site_xpos[tcp_sid].copy()
        if tip[2] >= z_min:
            return
        R_now = d.site_xmat[tcp_sid].reshape(3, 3).copy()          # keep whatever tool yaw the grasp used
        q_up, err = ik(m, dik, "tcp", [tip[0], tip[1], z_min], R_now, d.qpos[:6].copy())
        if err > 0.005:
            return
        qref = []
        seg(qref, d.qpos[:6].copy(), q_up, T)
        run_qref(np.array(qref), held=held)

    def pick_object(o, attempt):
        lift_clear(0.15)
        p_now = d.xpos[o["bid"]].copy()
        top = obj_top(o)
        pk = dict(object=o["name"], kind=o["kind"], attempt=attempt, frame_start=len(rec["t"]), frame_latch=-1,
                  frame_release=-1, frame_end=-1, grasp_nominal=None, grasp_executed=None, success=False,
                  fail_reason=None, moved_mm=None, rim_dev_mm=None, box_contacts=0)
        picks.append(pk)
        if not (G.TABLE_X[0] - 0.06 < p_now[0] < G.TABLE_X[1] + 0.06 and G.TABLE_Y[0] - 0.08 < p_now[1] < G.TABLE_Y[1] + 0.08) or p_now[2] < -0.05:
            pk.update(frame_end=len(rec["t"]), fail_reason="off_table")
            return False
        est = p_now[:2] + rng.normal(0, a.obs_noise, 2)            # the expert's (noisy) object estimate
        hover = [float(est[0]), float(est[1]), top + 0.12]
        # 1. plan to the hover pose; the demos' +90 deg yaw first, yaw 0 as a fallback (objects next to
        #    the camera column are only reachable with the wrist camera turned away from it)
        pl.set_world(world_for(exclude=o))
        R_DOWN, quat_used = demo["R_DOWN"], R_DOWN_QUAT
        res = pl.plan_pose(d.qpos[:6], hover, quat=R_DOWN_QUAT)
        if not res.get("success"):
            res = pl.plan_pose(d.qpos[:6], hover, quat=[0.0, 1.0, 0.0, 0.0])
            if res.get("success"):
                R_DOWN, quat_used = demo["R_FLAT"], [0.0, 1.0, 0.0, 0.0]
        pk["tool_quat"] = quat_used
        plan_log(pk, "hover", res)
        if not res.get("success"):
            pk.update(frame_end=len(rec["t"]), fail_reason="plan_hover")
            return False
        state["target"] = ob.index(o)
        run_plan(res)
        # 2. descent with the PUMP ALREADY RUNNING: the valve opens --suction_on_dist above the object,
        #    so the cup grabs at the first adequate touch instead of only sealing after a full 15 mm press,
        #    and vacuum flow pulls a near object the last millimetres onto the lip (--snap_gap/--snap_force).
        #    An object can therefore be caught mid-push, which is what the real gripper does.
        q0 = d.qpos[:6].copy()
        q_grasp, e1 = ik(m, dik, "tcp", [est[0], est[1], top + CUP_R], R_DOWN, q0)
        q_press, e2 = ik(m, dik, "tcp", [est[0], est[1], top + CUP_R - a.press], R_DOWN, q_grasp)
        pk["grasp_nominal"] = dict(pos=[round(float(v), 5) for v in (est[0], est[1], top + CUP_R)], quat=quat_used)
        if max(e1, e2) > 0.003:
            pk.update(frame_end=len(rec["t"]), fail_reason="ik")
            return False
        qref = []
        seg(qref, q0, q_grasp, 1.6)
        seg(qref, q_grasp, q_press, 0.7)
        seg(qref, q_press, q_press, 0.3)
        qref = np.array(qref)
        p_before = d.xpos[o["bid"]].copy()
        latched = {"ok": False}
        st = dict(contact=False, rim_min=np.inf, rim_ok=False, tilt_min=np.inf, f_max=0.0,
                  snapped=False, captured=False, other=None)

        def on_tick(k):
            d.xfrc_applied[:, :] = 0.0
            if latched["ok"]:
                return
            tip_p = d.site_xpos[tip_sid].copy()
            cup_ax = d.site_xmat[tip_sid].reshape(3, 3)[:, 2]
            if not state["suction"] and tip_p[2] - obj_top(o) <= a.suction_on_dist:
                state["suction"] = 1                       # valve opens before contact (real sequence)
                pk["suction_cmd_frame"] = len(rec["t"])
            if not state["suction"]:
                return
            rim_ok, rim_dev, _, _ = rim_seal(m, d, tip_p, cup_ax, target_bid=o["bid"], exclude_bid=tcp_bid,
                                             geomgroup=GEOMGROUP)
            if np.isfinite(rim_dev):
                st["rim_min"] = min(st["rim_min"], float(rim_dev))
            st["rim_ok"] = st["rim_ok"] or bool(rim_ok)
            bid_touch = touched_object()
            # ---- capture: while the pump runs, whatever is under the lip is drawn onto it and
            # self-centres (the pressure difference acts over the whole lip, not just at the contact
            # point). This is why a running cup sticks on first touch instead of skidding the object away.
            gap, bid_ray = cup_ray()
            cand = bid_touch if bid_touch is not None else (bid_ray if gap <= a.snap_gap else None)
            if cand is not None and rim_ok:          # a leaking lip (rim over an edge/curve) pulls nothing
                rel = d.xpos[cand] - tip_p
                lat = rel - cup_ax * float(rel @ cup_ax)         # offset from the cup axis
                r = float(np.linalg.norm(lat))
                if r <= a.capture_r:
                    reach = 1.0 if (bid_touch is not None or gap <= 0) else (1.0 - gap / a.snap_gap)
                    fvec = -a.snap_force * reach * cup_ax                      # onto the lip
                    if r > 1e-6:
                        fvec = fvec - (a.center_force * reach * min(1.0, r / a.capture_r)) * (lat / r)
                    d.xfrc_applied[cand, :3] = fvec
                    st["snapped"] = st["snapped"] or bid_touch is None
                    st["captured"] = True
            if bid_touch is None:
                return
            st["contact"] = True
            if bid_touch != o["bid"]:
                st["other"] = bid2obj[bid_touch]["name"]
                return
            f, tilt = demo["_cup_contact"](m, d, cup_gid)
            st["tilt_min"] = min(st["tilt_min"], float(tilt))
            st["f_max"] = max(st["f_max"], float(f))
            # A RUNNING cup seals on contact when the lip conforms and meets the surface face-on; the
            # force gate is only a contact sanity check (the measured lip gives 0.1-0.4 N at 5-15 mm).
            if rim_ok and tilt < demo["SEAL_DEG"] and f > a.touch_n:
                demo["_latch_weld"](m, d, o["eq"], tcp_bid, o["bid"])
                latched["ok"] = True
                q4 = np.zeros(4)
                mujoco.mju_mat2Quat(q4, d.site_xmat[tcp_sid])
                pk["frame_latch"] = len(rec["t"])
                pk["latch_mode"] = "snap" if st["snapped"] else "touch"
                pk["latch_force_n"] = round(float(f), 2)
                pk["grasp_executed"] = dict(pos=[round(float(v), 5) for v in d.site_xpos[tcp_sid]], quat=[round(float(v), 5) for v in q4],
                                            obj_pos=[round(float(v), 5) for v in d.xpos[o["bid"]]])

        run_qref(qref, on_tick, stop=lambda: latched["ok"])
        d.xfrc_applied[:, :] = 0.0
        pk["moved_mm"] = round(1000 * float(np.linalg.norm(d.xpos[o["bid"]][:2] - p_before[:2])), 1)
        pk["rim_dev_mm"] = round(1000 * st["rim_min"], 2) if np.isfinite(st["rim_min"]) else -1
        pk["captured"] = bool(st["captured"])
        if pk.get("suction_cmd_frame") is not None and pk["frame_latch"] >= 0:
            pk["latch_delay_s"] = round((pk["frame_latch"] - pk["suction_cmd_frame"]) / G.REC_HZ, 2)
        if latched["ok"]:
            qref = []
            seg(qref, d.qpos[:6].copy(), d.qpos[:6].copy(), 0.25)      # hold while the vacuum builds
            run_qref(np.array(qref), held=o)
        else:
            state["suction"] = 0
            pk["tilt_min_deg"] = round(float(st["tilt_min"]), 1) if np.isfinite(st["tilt_min"]) else -1
            pk["contact_f_max_n"] = round(float(st["f_max"]), 3)
            pk["fail_reason"] = ("pushed" if pk["moved_mm"] > 1000 * a.push_tol else
                                 "touched_other" if st["other"] else
                                 "no_contact" if not st["contact"] else
                                 "no_rim_seal" if not st["rim_ok"] else
                                 "tilted" if st["tilt_min"] >= demo["SEAL_DEG"] else "weak_contact")
            if st["other"]:
                pk["touched_other"] = st["other"]
        if not latched["ok"]:
            # back up to the hover height so the next attempt / object starts clear of the clutter
            q_up, _ = ik(m, dik, "tcp", [est[0], est[1], top + 0.12], R_DOWN, d.qpos[:6].copy())
            qref = []
            seg(qref, d.qpos[:6].copy(), q_up, 1.0)
            run_qref(np.array(qref))
            state["target"] = -1
            pk["frame_end"] = len(rec["t"])
            return False
        # 3. carry: scripted lift to hover height (clears the table slab), attach the object to the
        #    planner tool, plan to above the box
        state["watch_box"] = True
        lift_clear(top + 0.12, held=o, T=1.2)
        if not d.eq_active[o["eq"]] or np.linalg.norm(d.xpos[o["bid"]] - d.site_xpos[tcp_sid]) > 0.15:
            pk.update(frame_end=len(rec["t"]), fail_reason="dropped")
            state["watch_box"] = False
            return False
        oq = np.zeros(4)
        mujoco.mju_mat2Quat(oq, d.xmat[o["bid"]].reshape(9))
        pl.set_world(world_for(exclude=o))
        pl.attach(d.qpos[:6], obj_dims(o, 0.01), [*d.xpos[o["bid"]].tolist(), *oq.tolist()])
        h_obj = 2 * o["half"][2]
        above = [box_xy[0], box_xy[1], BOX_H + h_obj + 0.05 + CUP_R]
        res = pl.plan_pose(d.qpos[:6], above, quat=quat_used)
        plan_log(pk, "carry", res)
        if not res.get("success"):
            # cannot plan the carry: release in place
            d.eq_active[o["eq"]] = 0
            state["suction"] = 0
            pl.detach()
            state["watch_box"] = False
            pk.update(frame_end=len(rec["t"]), fail_reason="plan_carry")
            return False
        run_plan(res, held=o)
        # 4. lower into the box mouth (object bottom 3 cm below the rim), release, detach, retreat up
        q_low, e3 = ik(m, dik, "tcp", [box_xy[0], box_xy[1], BOX_H - 0.03 + h_obj + CUP_R], R_DOWN, d.qpos[:6].copy())
        qref = []
        seg(qref, d.qpos[:6].copy(), q_low, 1.0)
        run_qref(np.array(qref), held=o)
        d.eq_active[o["eq"]] = 0
        state["suction"] = 0
        pl.detach()
        pk["frame_release"] = len(rec["t"])
        qref = []
        seg(qref, d.qpos[:6].copy(), d.qpos[:6].copy(), 0.5)      # let it fall
        run_qref(np.array(qref))
        q_ret, _ = ik(m, dik, "tcp", [box_xy[0], box_xy[1], retreat_z], R_DOWN, d.qpos[:6].copy())
        qref = []
        seg(qref, d.qpos[:6].copy(), q_ret, 1.0)
        run_qref(np.array(qref))
        state["watch_box"] = False
        state["target"] = -1
        pk["box_contacts"] = state["box_contacts"]
        state["box_contacts"] = 0
        pk["frame_end"] = len(rec["t"])
        pk["success"] = in_box(o)
        if not pk["success"]:
            pk["fail_reason"] = "not_in_box"
        return pk["success"]

    # nearest object first, each up to a.attempts tries, re-observed before every try
    order = sorted(range(len(ob)), key=lambda i: np.hypot(*ob[i]["half"][:2]) * 0 + np.linalg.norm(d.xpos[ob[i]["bid"]][:2]))
    done = 0
    for i in order:
        for attempt in range(a.attempts):
            if in_box(ob[i]):
                break
            if pick_object(ob[i], attempt):
                done += 1
                break
    # home
    pl.set_world(world_for())
    res = pl.plan_joint(d.qpos[:6], demo["Q_START"])
    if res.get("success"):
        run_plan(res)
    wr405.close()
    wr435.close()
    np.savez_compressed(os.path.join(sdir, "traj.npz"), **{k: np.asarray(v) for k, v in rec.items()})
    np.savez_compressed(os.path.join(sdir, "eef.npz"), pos=np.asarray(rec["tcp_pos"]), quat=np.asarray(rec["tcp_quat"]),
                        suction=np.asarray(rec["suction"]))
    meta = dict(scene=idx, box=dict(xy=[float(box_xy[0]), float(box_xy[1])], inner=BOX_FP, height=BOX_H, wall=BOX_WALL),
                planner=pl.backend, rec_hz=G.REC_HZ, n_objects=len(ob), cleared=done, obs_noise=a.obs_noise,
                objects=[dict(name=o["name"], kind=o["kind"], geom=o["shape"], half_extents=o["half"],
                              init_pos=[round(float(v), 4) for v in objs[k]["pos"]],
                              final_pos=[round(float(v), 4) for v in d.xpos[o["bid"]]], in_box=in_box(o)) for k, o in enumerate(ob)],
                picks=picks, frames=len(rec["t"]), wall_s=round(time.time() - t_scene, 1))
    json.dump(meta, open(os.path.join(sdir, "meta.json"), "w"), indent=1)
    return idx, len(ob), done, len(picks), len(rec["t"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_dataset_curobo"))
    ap.add_argument("--box", type=float, nargs=2, default=None, help="box centre x y (base frame, m); default random in --box_range")
    ap.add_argument("--box_range", type=float, nargs=4, default=[0.24, 0.36, 0.33, 0.42], help="xmin xmax ymin ymax for a random box")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9996)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--obs_noise", type=float, default=0.004, help="xy noise (m) on the expert's object estimate -> off-centre presses that push objects")
    ap.add_argument("--push_tol", type=float, default=0.010, help="object displacement without a latch that counts as 'pushed' (m)")
    ap.add_argument("--suction_on_dist", type=float, default=0.05, help="cup-tip height above the object top at which the valve opens (m); the cup then latches on contact")
    ap.add_argument("--touch_n", type=float, default=0.05, help="contact force that an already-running cup needs to seal (N); the old press-end check needed SEAL_N = 2 N")
    ap.add_argument("--snap_gap", type=float, default=0.004, help="airflow reach: an object within this gap of a sealable lip is pulled onto it (m); 0 disables")
    ap.add_argument("--snap_force", type=float, default=2.5, help="peak pull force onto the lip (N), falling linearly to 0 at --snap_gap")
    ap.add_argument("--press", type=float, default=0.010, help="descent depth below the expected surface (m); a running cup latches on contact, so the old 15 mm press-to-seal stroke only rams the object")
    ap.add_argument("--capture_r", type=float, default=0.012, help="lateral offset from the cup axis within which a running cup still captures an object (m)")
    ap.add_argument("--center_force", type=float, default=1.5, help="lateral self-centring force drawing a captured object onto the cup axis (N)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, a.out, a) for i in range(a.start, a.start + a.count)]
    print(f"[dataset_curobo] {a.count} scenes -> {a.out}, planner {a.host}:{a.port}, box {a.box or a.box_range}", flush=True)
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=2) as pool:
            it = pool.imap_unordered(run_scene, jobs)
            for idx, n, done, npk, nf in it:
                print(f"[dataset_curobo] scene {idx:03d}: {done}/{n} in the box, {npk} attempts, {nf} frames", flush=True)
    else:
        for j in jobs:
            idx, n, done, npk, nf = run_scene(j)
            print(f"[dataset_curobo] scene {idx:03d}: {done}/{n} in the box, {npk} attempts, {nf} frames", flush=True)
    print("[dataset_curobo] complete")


if __name__ == "__main__":
    main()

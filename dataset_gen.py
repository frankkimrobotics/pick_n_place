"""dataset_gen :: 100 random pick-and-place scenes with RGBD + state recording.

Per scene: 1-10 random objects (cylinder / cube / cuboid -- no spheres) with
random dimensions, colours, yaws, rejection-sampled non-overlapping poses on
the table; one of 3 light positions (scene % 3). The arm press-to-seal picks
every object into the bin one by one (up to 2 retries each) until the table
is clear. Recorded at 10 Hz throughout:
  - wrist D405 + fixed D435 RGB (mp4) and depth (uint16 mm, compressed npz)
  - joint angles q[6], velocities qd[6]
  - tip rangefinder value, suction state, active target index
Per-scene meta.json: every object's shape, OBB half-extents, colour, initial
pose, final pose, picked/in-bin outcome, light position, seed.

Batching note: mujoco_warp batches identical models -- every scene here is a
DIFFERENT model (object count/dims), and the wall-clock is dominated by
OSMesa rendering which is CPU. So scenes are parallelised across worker
PROCESSES instead (see --workers).

Run in the mjwarp env:
    conda activate mjwarp && python dataset_gen.py --count 100 --workers 12
The MJCF generator needs the curobo2 env, so scene XMLs are built by
shelling out to it (sim_robot_mjcf.build with objects=...).
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_OUT = "/media/lisc-frank/pnp_dataset"
FALLBACK_OUT = os.path.expanduser("~/pnp_dataset")
CUROBO_PY = "/home/lisc-frank/miniconda3/envs/curobo2/bin/python"

REC_HZ = 10
W, H = 424, 240
LIGHTS = [(0.3, 0.1, 1.5), (1.0, -0.7, 1.3), (-0.5, 0.9, 1.1)]
BIN_XY = np.array([0.10, 0.40])
BIN_HALF = 0.145
TABLE_X = (0.24, 0.52)
TABLE_Y = (-0.16, 0.16)
MAX_RETRY = 2


# --------------------------------------------------------------------------- #
# scene sampling
def sample_scene(idx):
    rng = np.random.default_rng(idx * 7919 + 13)
    n = int(rng.integers(1, 11))
    objs, placed = [], []
    for i in range(n):
        for _ in range(300):
            shape = rng.choice(["cylinder", "cube", "cuboid"])
            if shape == "cylinder":
                r, hh = rng.uniform(0.018, 0.032), rng.uniform(0.012, 0.024)
                size, half, z, ext = f"{r:.4f} {hh:.4f}", [r, r, hh], hh, r
            elif shape == "cube":
                s_ = rng.uniform(0.015, 0.030)
                size, half, z, ext = f"{s_:.4f} {s_:.4f} {s_:.4f}", [s_] * 3, s_, s_ * 1.42
            else:
                sx, sy, sz = (rng.uniform(0.015, 0.035), rng.uniform(0.015, 0.035),
                              rng.uniform(0.012, 0.030))
                size, half, z, ext = (f"{sx:.4f} {sy:.4f} {sz:.4f}", [sx, sy, sz],
                                      sz, float(np.hypot(sx, sy)))
            x = rng.uniform(*TABLE_X)
            y = rng.uniform(*TABLE_Y)
            if all(np.hypot(x - px, y - py) > ext + pe + 0.015
                   for px, py, pe in placed):
                break
        else:
            continue
        yaw = float(rng.uniform(0, np.pi)) if shape != "cylinder" else 0.0
        quat = f"{np.cos(yaw/2):.4f} 0 0 {np.sin(yaw/2):.4f}"
        rgba = rng.uniform(0.15, 0.95, 3)
        objs.append(dict(
            name=f"object{i}", shape=("box" if shape != "cylinder" else "cylinder"),
            kind=shape, size=size, half_extents=[round(v, 4) for v in half],
            pos=[round(x, 4), round(y, 4), round(z, 4)], yaw=round(yaw, 4),
            quat=quat, rgba=[round(v, 3) for v in rgba] + [1.0]))
        placed.append((x, y, ext))
    return objs


def build_scene_xml(objs, light, out_xml):
    """Build the per-scene MJCF via the generator (curobo2 env has yourdfpy)."""
    tuples = [(o["name"], o["shape"], o["size"],
               " ".join(f"{v:.4f}" for v in o["pos"]),
               " ".join(f"{v:g}" for v in o["rgba"]), o["quat"]) for o in objs]
    code = (f"import sys; sys.path.insert(0, {HERE!r});\n"
            f"import sim_robot_mjcf as g;\n"
            f"g.build(warp=True, objects={tuples!r}, light_pos={tuple(light)!r}, "
            f"out_path={out_xml!r})")
    subprocess.run([CUROBO_PY, "-c", code], check=True, capture_output=True)


# --------------------------------------------------------------------------- #
# per-scene execution (runs inside a worker process)
def run_scene(args):
    idx, out_root = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    import imageio
    demo = {"__file__": os.path.join(HERE, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)

    sdir = os.path.join(out_root, f"scene_{idx:03d}")
    os.makedirs(sdir, exist_ok=True)
    objs = sample_scene(idx)
    light = LIGHTS[idx % 3]
    xml = os.path.join(sdir, "scene.xml")
    build_scene_xml(objs, light, xml)

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    d.qpos[:6] = demo["Q_START"]
    mujoco.mj_forward(m, d)
    ren = mujoco.Renderer(m, height=H, width=W)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0
    rf_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "tip_range")]
    tcp_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
    cup_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")

    every = int(round(1.0 / (REC_HZ * demo["CTRL_DT"])))
    wr405 = imageio.get_writer(os.path.join(sdir, "d405_rgb.mp4"), fps=REC_HZ, quality=8)
    wr435 = imageio.get_writer(os.path.join(sdir, "d435_rgb.mp4"), fps=REC_HZ, quality=8)
    rec = dict(t=[], q=[], qd=[], rf=[], suction=[], target=[])
    depth = {"wrist_d405": [], "fixed_d435": []}
    state = dict(tick=0, suction=0, target=-1)

    def record(dd):
        if state["tick"] % every:
            return
        for cam, wr in (("wrist_d405", wr405), ("fixed_d435", wr435)):
            ren.update_scene(dd, camera=cam, scene_option=vopt)
            wr.append_data(ren.render())
            ren.enable_depth_rendering()
            ren.update_scene(dd, camera=cam, scene_option=vopt)
            dep = ren.render()
            ren.disable_depth_rendering()
            depth[cam].append(np.clip(dep * 1000.0, 0, 65535).astype(np.uint16))
        rec["t"].append(state["tick"] * demo["CTRL_DT"])
        rec["q"].append(dd.qpos[:6].copy())
        rec["qd"].append(dd.qvel[:6].copy())
        rec["rf"].append(float(dd.sensordata[rf_adr]))
        rec["suction"].append(state["suction"])
        rec["target"].append(state["target"])

    def run_ref(sched, on_tick=None):
        for k in range(len(sched["qref"])):
            if on_tick is not None:
                on_tick(k)
            for _ in range(demo["NSUB"]):
                d.ctrl[:6] = demo["pd_tau"](sched, k, d.qpos[:6], d.qvel[:6])
                mujoco.mj_step(m, d)
            state["tick"] += 1
            record(d)

    def pick_object(oi):
        o = objs[oi]
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
        eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{o['name']}")
        p_now = d.xpos[bid].copy()
        top = p_now[2] + o["half_extents"][2]
        if not (TABLE_X[0] - 0.06 < p_now[0] < TABLE_X[1] + 0.06
                and TABLE_Y[0] - 0.08 < p_now[1] < TABLE_Y[1] + 0.08):
            return False                                   # left the table
        rngd = np.random.default_rng(idx * 131 + oi)
        drop = BIN_XY + rngd.uniform(-0.06, 0.06, 2)
        dik = mujoco.MjData(m)
        q0 = d.qpos[:6].copy()
        ik, R_DOWN, seg = demo["ik"], demo["R_DOWN"], demo["_seg"]
        CUP_R, PRESS = demo["CUP_R"], demo["PRESS_M"]
        q_hov, e1 = ik(m, dik, "tcp", [p_now[0], p_now[1], top + 0.12], R_DOWN, q0)
        q_grasp, e2 = ik(m, dik, "tcp", [p_now[0], p_now[1], top + CUP_R], R_DOWN, q_hov)
        q_press, e3 = ik(m, dik, "tcp", [p_now[0], p_now[1], top + CUP_R - PRESS],
                         R_DOWN, q_grasp)
        q_bin, e4 = ik(m, dik, "tcp", [drop[0], drop[1], 0.42], R_DOWN, q_hov)
        q_drop, e5 = ik(m, dik, "tcp", [drop[0], drop[1], 0.22], R_DOWN, q_bin)
        if max(e1, e2, e3, e4, e5) > 0.003:
            return False
        qref = []
        seg(qref, q0, q_hov, 2.0)
        seg(qref, q_hov, q_grasp, 1.6)
        seg(qref, q_grasp, q_press, 0.7)
        i_press = len(qref)
        seg(qref, q_press, q_press, 0.3)
        seg(qref, q_press, q_hov, 1.2)
        seg(qref, q_hov, q_bin, 2.0)
        seg(qref, q_bin, q_drop, 0.8)
        i_rel = len(qref)
        seg(qref, q_drop, q_bin, 0.8)
        qref = np.array(qref)
        qdref, tau_ff, kp, kd = demo["_gains_and_ff"](m, d, qref)
        sched = dict(qref=qref, qdref=qdref, tau_ff=tau_ff, kp=kp, kd=kd)

        def on_tick(k):
            if k == i_press:
                f, tilt = demo["_cup_contact"](m, d, cup_gid)
                if f > demo["SEAL_N"] and tilt < demo["SEAL_DEG"]:
                    demo["_latch_weld"](m, d, eq_id, tcp_bid, bid)
                    state["suction"] = 1
                    state["target"] = oi
            if k == i_rel and state["suction"]:
                d.eq_active[eq_id] = 0
                state["suction"] = 0
                state["target"] = -1

        run_ref(sched, on_tick)
        state["suction"] = 0
        d.eq_active[eq_id] = 0
        p = d.xpos[bid]
        return bool(abs(p[0] - BIN_XY[0]) < BIN_HALF
                    and abs(p[1] - BIN_XY[1]) < BIN_HALF and p[2] < 0.12)

    results = []
    order = sorted(range(len(objs)),
                   key=lambda i: np.hypot(*(np.array(objs[i]["pos"][:2]))))
    for oi in order:
        ok = False
        for _ in range(1 + MAX_RETRY):
            ok = pick_object(oi)
            if ok:
                break
        results.append((oi, ok))

    wr405.close(); wr435.close()
    np.savez_compressed(
        os.path.join(sdir, "traj.npz"),
        t=np.array(rec["t"]), q=np.array(rec["q"]), qd=np.array(rec["qd"]),
        rangefinder=np.array(rec["rf"]), suction=np.array(rec["suction"]),
        target=np.array(rec["target"]))
    for cam in depth:
        np.savez_compressed(os.path.join(sdir, f"{cam.split('_')[-1]}_depth_mm.npz"),
                            depth=np.stack(depth[cam]) if depth[cam] else np.zeros((0, H, W), np.uint16))
    meta = dict(scene=idx, seed=idx * 7919 + 13, light_pos=list(light),
                rec_hz=REC_HZ, n_objects=len(objs), objects=[])
    for oi, o in enumerate(objs):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
        ok = dict(results).get(oi, False)
        meta["objects"].append(dict(
            name=o["name"], kind=o["kind"], geom=o["shape"],
            obb_half_extents=o["half_extents"], rgba=o["rgba"],
            init_pos=o["pos"], init_yaw=o["yaw"],
            final_pos=[round(float(v), 4) for v in d.xpos[bid]],
            final_quat=[round(float(v), 4) for v in d.xquat[bid]],
            in_bin=bool(ok)))
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    n_ok = sum(1 for _, ok in results if ok)
    return idx, len(objs), n_ok, len(rec["t"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out
    if out is None:
        out = DEFAULT_OUT if os.access(os.path.dirname(DEFAULT_OUT), os.W_OK) else FALLBACK_OUT
        try:
            os.makedirs(out, exist_ok=True)
        except PermissionError:
            out = FALLBACK_OUT
            os.makedirs(out, exist_ok=True)
    print(f"[dataset] writing to {out} ({a.count} scenes, {a.workers} workers)")
    jobs = [(i, out) for i in range(a.start, a.start + a.count)]
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(a.workers, maxtasksperchild=4) as pool:
        for idx, n, ok, frames in pool.imap_unordered(run_scene, jobs):
            done += 1
            print(f"[dataset] scene {idx:03d}: {ok}/{n} in bin, {frames} frames "
                  f"({done}/{a.count})", flush=True)
    print("[dataset] complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Joint multi-camera extrinsic calibration from RANDOMLY-PLACED static ChArUco boards.

No prior hand-eye and no known board positions needed:
  Stage 1  wrist hand-eye X = T_tcp_cam  via cv2.calibrateHandEye (robot moves, wrist cam
           watches static boards from many rotation-diverse poses -> AX=XB).
  Stage 2  localize each board in base: T_base_B = FK(q)*X * T_wristcam_B  (averaged over poses).
  Stage 3  each fixed cam: T_base_fixedcam = T_base_B * inv(T_fixedcam_B)  (averaged over boards).
Then compares to the existing extrinsics (config T_TCP_CAM + shift, extrinsics_d435/d405_fixed.json).

Boards detected (any subset can be present): original DICT_4X4_50 + boardA/B/C (5X5_100/6X6_100/4X4_100),
all 5x7, square 0.035m, marker 0.026m.  Captures RGB from all 3 cams; saves raw frames+poses.

  source /opt/ros/humble/setup.bash
  python3 multi_cam_calib.py            # move + collect + solve + compare
  python3 multi_cam_calib.py --process outputs/multicam_calib_XXXX   # re-solve saved data (no robot)
"""
import argparse, json, os, socket, sys, time, math, datetime
import numpy as np
import cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, make_T, quat_wxyz_to_R
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg
import pyrealsense2 as rs

PI = "10.0.0.27"
CAMS = {"wrist": "218622271300", "d435": "043422070101", "d405fix": "218622277013"}
RES = {"wrist": (640, 480), "d435": (848, 480), "d405fix": (640, 480)}
SQUARES = (5, 7); SQ = 0.032; MK = 0.032 * 26.0 / 35.0   # measured: 32mm square (printed ~0.91x); marker ~24mm
BOARD_DICTS = {"orig": "DICT_4X4_50", "boardA": "DICT_5X5_100", "boardB": "DICT_6X6_100", "boardC": "DICT_4X4_100"}

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def send_chunk(m):
    k = socket.create_connection((PI, 9994), timeout=3); k.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    k.sendall((json.dumps(m) + "\n").encode()); k.close()
def read_q():
    s = socket.create_connection((PI, 9999), timeout=3); b = b""
    while b"\n" not in b: b += s.recv(4096)
    s.close(); return list(json.loads(b.split(b"\n")[0])["joints_deg"])
def qrad(qd): return [float(v) for v in linuxcnc_deg_to_rad(qd)]
def fk_T(qd):
    r = rpc({"type": "fk", "q": qrad(qd)}); return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))
def move_to(xyz, quat, V=16.0):
    qd = read_q(); r = rpc({"type": "plan_pose", "start_q": qrad(qd), "goal_pose": list(xyz) + list(quat), "max_attempts": 18})
    if not r.get("success"): return False
    traj = np.array(r["trajectory"]); dt = r["dt"]
    peak = np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt if len(traj) > 1 else V
    sdt = dt * max(1.0, peak / V)
    send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj], "traj_dt": sdt, "t_anchor": time.time() + 0.12})
    time.sleep(sdt * (len(traj) - 1) + 1.8); return True

def down_tilt_quat(th_deg, az_deg):
    th, az = math.radians(th_deg), math.radians(az_deg)
    z = np.array([math.sin(th) * math.cos(az), math.sin(th) * math.sin(az), -math.cos(th)])
    return list(map(float, R_to_quat_wxyz(R_from_two_axes(z))))

def gen_poses():
    P = []
    for (x, y) in [(0.30, 0.0), (0.40, 0.0), (0.35, 0.14), (0.35, -0.14), (0.30, 0.12), (0.40, -0.12), (0.30, -0.12), (0.40, 0.12)]:
        P.append((x, y, 0.30, 0.0, 0.0))
    for (x, y, th, az) in [(0.33, 0.0, 22, 0), (0.37, 0.0, 22, 180), (0.35, 0.10, 22, 90), (0.35, -0.10, 22, 270),
                           (0.32, 0.08, 25, 45), (0.38, -0.08, 25, 225), (0.32, -0.08, 25, 135), (0.38, 0.08, 25, 315)]:
        P.append((x, y, 0.30, th, az))
    return P

def open_cams():
    cams = {}
    for name, ser in CAMS.items():
        w, h = RES[name]
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(ser)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
        prof = pipe.start(cfg)
        it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]])
        dist = np.array(it.coeffs[:5])
        for _ in range(10): pipe.wait_for_frames(2000)
        cams[name] = dict(pipe=pipe, K=K, dist=dist)
    return cams
def grab(cam):
    return np.asanyarray(cam["pipe"].wait_for_frames(1000).get_color_frame().get_data())

# ---- boards + detectors ----
def build_boards():
    B = {}
    for name, dname in BOARD_DICTS.items():
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dname))
        board = cv2.aruco.CharucoBoard(SQUARES, SQ, MK, d)
        B[name] = dict(board=board, det=cv2.aruco.CharucoDetector(board))
    return B
def board_pose(gray, bd, K, dist):
    cc, ci, mc, mi = bd["det"].detectBoard(gray)
    if cc is None or len(cc) < 8: return None, 0
    op, ip = bd["board"].matchImagePoints(cc, ci)
    if op is None or len(op) < 8: return None, 0
    ok, rvec, tvec = cv2.solvePnP(op, ip, K, dist)
    if not ok: return None, 0
    R, _ = cv2.Rodrigues(rvec); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
    return T, len(cc)

def avg_T(Ts):
    """robust mean of SE(3): median translation + chordal-mean rotation via SVD."""
    if not Ts: return None
    t = np.median(np.array([T[:3, 3] for T in Ts]), axis=0)
    M = np.sum([T[:3, :3] for T in Ts], axis=0); U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T
def pose_diff(Tn, To):
    dt = np.linalg.norm(Tn[:3, 3] - To[:3, 3]) * 1000.0
    Rr = Tn[:3, :3] @ To[:3, :3].T
    ang = math.degrees(math.acos(max(-1, min(1, (np.trace(Rr) - 1) / 2))))
    return dt, ang

def collect(outdir):
    os.makedirs(outdir, exist_ok=True)
    cams = open_cams()
    for name in cams: np.save(os.path.join(outdir, f"K_{name}.npy"), cams[name]["K"]); np.save(os.path.join(outdir, f"dist_{name}.npy"), cams[name]["dist"])
    poses = gen_poses(); recs = []
    print(f"[collect] {len(poses)} poses")
    for i, (x, y, z, th, az) in enumerate(poses):
        if not move_to([x, y, z], down_tilt_quat(th, az)): print(f"  pose {i}: unreachable, skip"); continue
        time.sleep(0.5)
        qd = read_q()
        for name in cams:
            img = grab(cams[name]); cv2.imwrite(os.path.join(outdir, f"p{i:02d}_{name}.png"), img)
        recs.append({"i": i, "q": qd, "T_base_tcp": fk_T(qd).tolist()})
        print(f"  pose {i}: captured (tip [{x},{y},{z}] tilt {th}/{az})")
    for c in cams.values(): c["pipe"].stop()
    json.dump({"poses": recs, "cams": list(CAMS)}, open(os.path.join(outdir, "poses.json"), "w"), indent=1)
    # return to base
    r = rpc({"type": "plan_joint", "start_q": qrad(read_q()), "goal_q": [float(v) for v in C.BASE_Q], "max_attempts": 12})
    if r.get("success"):
        traj = np.array(r["trajectory"]); dt = r["dt"]; sdt = dt * max(1.0, np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt / 20.0)
        send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj], "traj_dt": sdt, "t_anchor": time.time() + 0.12}); time.sleep(sdt * (len(traj) - 1) + 1.6)
    print(f"[collect] saved to {outdir}; arm back at base")
    return outdir

def process(outdir):
    meta = json.load(open(os.path.join(outdir, "poses.json")))
    K = {n: np.load(os.path.join(outdir, f"K_{n}.npy")) for n in meta["cams"]}
    dist = {n: np.load(os.path.join(outdir, f"dist_{n}.npy")) for n in meta["cams"]}
    boards = build_boards()
    # detect every board in every image
    wrist_obs = {b: [] for b in boards}         # board -> list of (T_base_tcp, T_wristcam_board)
    fixed_obs = {n: {b: [] for b in boards} for n in ("d435", "d405fix")}
    for rec in meta["poses"]:
        i = rec["i"]; T_bt = np.array(rec["T_base_tcp"])
        for name in meta["cams"]:
            p = os.path.join(outdir, f"p{i:02d}_{name}.png")
            if not os.path.exists(p): continue
            gray = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
            for b, bd in boards.items():
                T, n = board_pose(gray, bd, K[name], dist[name])
                if T is None: continue
                if name == "wrist": wrist_obs[b].append((T_bt, T))
                else: fixed_obs[name][b].append(T)
    for b in boards: print(f"  board {b}: wrist views={len(wrist_obs[b])}  d435={len(fixed_obs['d435'][b])}  d405fix={len(fixed_obs['d405fix'][b])}")

    # ---- Stage 1: hand-eye X = T_tcp_cam  (use the board with the most wrist views) ----
    best = max(wrist_obs, key=lambda b: len(wrist_obs[b]))
    obs = wrist_obs[best]
    if len(obs) < 6: print("NOT ENOUGH wrist views for hand-eye (need >=6)"); return
    Rg2b = [T[:3, :3] for T, _ in obs]; tg2b = [T[:3, 3] for T, _ in obs]
    Rt2c = [T[:3, :3] for _, T in obs]; tt2c = [T[:3, 3] for _, T in obs]
    Rx, tx = cv2.calibrateHandEye(Rg2b, tg2b, Rt2c, tt2c, method=cv2.CALIB_HAND_EYE_PARK)  # robust vs DANIILIDIS on noisy data
    X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = tx.ravel()      # T_tcp_cam (new hand-eye)
    print(f"\n[Stage1] hand-eye from board '{best}' ({len(obs)} views): X=T_tcp_cam solved")

    # ---- Stage 2: localize each board in base ----
    T_base_B = {}
    for b in boards:
        est = [T_bt @ X @ T_wc for (T_bt, T_wc) in wrist_obs[b]]
        if est: T_base_B[b] = avg_T(est)
    print(f"[Stage2] localized boards: {list(T_base_B)}")

    # ---- Stage 3: fixed cameras ----
    fixed_new = {}
    for name in ("d435", "d405fix"):
        ests = []
        for b in boards:
            if b in T_base_B and fixed_obs[name][b]:
                T_fc_b = avg_T(fixed_obs[name][b])
                ests.append(T_base_B[b] @ np.linalg.inv(T_fc_b))
        if ests:
            fixed_new[name] = avg_T(ests)
            sp = np.std([e[:3, 3] for e in ests], axis=0) * 1000 if len(ests) > 1 else np.zeros(3)
            print(f"[Stage3] {name}: from {len(ests)} board(s), pos spread {np.round(sp,1)} mm")

    # ---- Compare to previous ----
    print("\n===== COMPARISON WITH PREVIOUS =====")
    X_old = make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
    dt, ang = pose_diff(X, X_old); print(f"WRIST hand-eye (T_tcp_cam): dpos={dt:.1f} mm  drot={ang:.2f} deg  vs config")
    for name, jf, key in [("d435", "extrinsics_d435.json", "T_base_cam435"), ("d405fix", "extrinsics_d405_fixed.json", "T_base_cam405fixed")]:
        if name not in fixed_new: continue
        try:
            To = np.array(json.load(open(os.path.join(C.OUT_DIR, jf)))[key])
            dt, ang = pose_diff(fixed_new[name], To); print(f"{name}: dpos={dt:.1f} mm  drot={ang:.2f} deg  vs {jf}")
        except Exception as e: print(f"{name}: no previous ({e})")

    out = {"T_tcp_cam_new": X.tolist(), "board_best": best, "n_wrist_views": len(obs),
           "T_base_boards": {b: T.tolist() for b, T in T_base_B.items()},
           "T_base_d435_new": fixed_new.get("d435", np.eye(4)).tolist(),
           "T_base_d405fix_new": fixed_new.get("d405fix", np.eye(4)).tolist()}
    json.dump(out, open(os.path.join(outdir, "calib_result.json"), "w"), indent=1)
    print(f"\nsaved -> {os.path.join(outdir,'calib_result.json')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="", help="re-solve an existing capture dir (no robot)")
    args = ap.parse_args()
    if args.process:
        process(args.process)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(C.OUT_DIR, f"multicam_calib_{ts}")
        collect(d); process(d)

if __name__ == "__main__":
    main()

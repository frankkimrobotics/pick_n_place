#!/usr/bin/env python3
"""Recalibrate the ZED->base extrinsic from TWO randomly-placed ChArUco boards.

Why two: a single small board poorly constrains ROTATION, which is exactly the error we
measured on 2026-11-08 (ZED z error swinging -3 mm at x=0.30 to -30 mm at x=0.50).
Two boards separated across the workspace give a much longer baseline for the fit.

Chain (arm stays put; wrist D405 is the base-frame reference via the hand-eye):
    T_base_wristcam = FK(q) @ T_TCP_CAM
    T_base_board_i  = T_base_wristcam @ T_wristcam_board_i
    corners_base_i  = T_base_board_i  @ corners_board
    corners_zed_i   = T_zedcam_board_i @ corners_board
    T_base_zedcam   = Kabsch(corners_zed_all -> corners_base_all)       # both boards stacked

Boards share a dictionary with duplicate IDs, so markers are clustered spatially (2-means on
marker centres) and each cluster is solved as its own board.

Stage A (system python3: pyzed + pyrealsense):   --capture
Stage B (ilpolicy python: cv2 + mujoco FK):      --solve
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = "/tmp/calib_two_boards.npz"
WRIST_SER = "218622271300"


def capture():
    import cv2, socket, time
    import pyrealsense2 as rs
    import pyzed.sl as sl
    # joints first (base-frame reference)
    s = socket.create_connection(("192.168.50.2", 9999), timeout=5)
    buf = b""
    while buf.count(b"\n") < 2:
        buf += s.recv(4096)
    s.close()
    q = json.loads(buf.split(b"\n")[0])["joints_deg"]

    p = rs.pipeline(); c = rs.config(); c.enable_device(WRIST_SER)
    c.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
    prof = p.start(c)
    it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    Kw = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.0]])
    dw = np.array(it.coeffs[:5])
    for _ in range(20):
        f = p.wait_for_frames(2000)
    wrist = np.asanyarray(f.get_color_frame().get_data()).copy()
    p.stop()

    cam = sl.Camera(); ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD720; ip.camera_fps = 30
    ip.depth_mode = sl.DEPTH_MODE.NEURAL
    assert cam.open(ip) == sl.ERROR_CODE.SUCCESS, "ZED open failed"
    ci = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
    Kz = np.array([[ci.fx, 0, ci.cx], [0, ci.fy, ci.cy], [0, 0, 1.0]])
    dz = np.zeros(5)
    im = sl.Mat(); rt = sl.RuntimeParameters()
    for _ in range(25):
        cam.grab(rt)
    cam.grab(rt); cam.retrieve_image(im, sl.VIEW.LEFT)
    zed = np.ascontiguousarray(im.get_data()[:, :, :3]).copy()
    cam.close()

    np.savez(CAP, wrist=wrist, zed=zed, Kw=Kw, dw=dw, Kz=Kz, dz=dz, q=np.array(q))
    cv2.imwrite("/tmp/cap_wrist.png", wrist); cv2.imwrite("/tmp/cap_zed.png", zed)
    print(f"captured: wrist {wrist.shape} zed {zed.shape} | q={[round(v,1) for v in q]}")


# --------------------------------------------------------------------- solve
def cluster_markers(corners, ids, k=2):
    ctr = np.array([c.reshape(-1, 2).mean(0) for c in corners])
    # 2-means on marker centres (boards are laterally separated)
    a, b = ctr[ctr[:, 0].argmin()], ctr[ctr[:, 0].argmax()]
    for _ in range(25):
        da = np.linalg.norm(ctr - a, axis=1); db = np.linalg.norm(ctr - b, axis=1)
        lab = (db < da).astype(int)
        if lab.sum() == 0 or lab.sum() == len(lab): break
        a, b = ctr[lab == 0].mean(0), ctr[lab == 1].mean(0)
    return lab, ctr


def board_pose_cropped(gray, K, dist, sel_corners, cfgs):
    """Crop to ONE board (two boards share a dictionary -> duplicate IDs break whole-image
    ChArUco interpolation) and solve its pose; K's principal point is shifted by the crop."""
    import cv2
    pts = np.vstack([c.reshape(-1, 2) for c in sel_corners])
    x0, y0 = np.floor(pts.min(0) - 15).astype(int).clip(0)
    x1, y1 = np.ceil(pts.max(0) + 15).astype(int)
    x1 = min(int(x1), gray.shape[1]); y1 = min(int(y1), gray.shape[0])
    crop = gray[int(y0):y1, int(x0):x1]
    Kc = K.copy(); Kc[0, 2] -= x0; Kc[1, 2] -= y0
    best = None
    for squares, sq, ratio, dict_id in cfgs:
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard(squares, sq, sq * ratio, d)
        cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(crop)
        if cc is None or len(cc) < 6:
            continue
        op, ip = board.matchImagePoints(cc, ci)
        if op is None or len(op) < 6:
            continue
        ok, rvec, tvec = cv2.solvePnP(op, ip, Kc, dist)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(op, rvec, tvec, Kc, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1)))
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
        cand = dict(T=T, err=err, n=len(cc), cfg=(squares, sq), corners_board=op.reshape(-1, 3),
                    ids=ci.ravel().tolist())
        if best is None or cand["n"] > best["n"]:
            best = cand
    return best


def board_pose_from_cluster(gray, K, dist, sel_corners, sel_ids, cfgs):
    import cv2
    best = None
    for squares, sq, ratio, dict_id in cfgs:
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard(squares, sq, sq * ratio, d)
        try:
            cc, ci = cv2.aruco.interpolateCornersCharuco(sel_corners, sel_ids, gray, board)[1:3]
        except Exception:
            continue
        if cc is None or len(cc) < 6:
            continue
        op, ip = board.matchImagePoints(cc, ci)
        if op is None or len(op) < 6:
            continue
        ok, rvec, tvec = cv2.solvePnP(op, ip, K, dist)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(op, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1)))
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
        cand = dict(T=T, err=err, n=len(cc), cfg=(squares, sq, ratio), corners_board=op.reshape(-1, 3))
        if best is None or err < best["err"]:
            best = cand
    return best


def solve():
    import cv2
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
    import config as C
    from joint_conventions import linuxcnc_deg_to_rad
    import mujoco

    z = np.load(CAP)
    wrist, zed, Kw, dw, Kz, dz, q = (z["wrist"], z["zed"], z["Kw"], z["dw"], z["Kz"], z["dz"], z["q"])

    # FK of the wrist camera in base
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "rl/scenes/box_med.xml"))
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    d.qpos[:6] = linuxcnc_deg_to_rad(list(q)); mujoco.mj_kinematics(m, d)
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = d.site_xmat[sid].reshape(3, 3); T_base_tcp[:3, 3] = d.site_xpos[sid]
    T_base_wcam = T_base_tcp @ np.array(C.T_TCP_CAM, float)
    print("wrist cam in base:", T_base_wcam[:3, 3].round(4))

    CFGS = [((5, 7), 0.036, 26/35, cv2.aruco.DICT_4X4_50),
            ((7, 5), 0.036, 26/35, cv2.aruco.DICT_4X4_50),
            ((5, 7), 0.032, 26/35, cv2.aruco.DICT_4X4_50),
            ((7, 5), 0.032, 26/35, cv2.aruco.DICT_4X4_50),
            ((6, 6), 0.036, 26/35, cv2.aruco.DICT_4X4_50),
            ((6, 6), 0.032, 26/35, cv2.aruco.DICT_4X4_50)]
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

    poses = {}
    for name, img, K, dist in (("wrist", wrist, Kw, dw), ("zed", zed, Kz, dz)):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mc, mi, _ = det.detectMarkers(g)
        if mi is None or len(mi) < 8:
            print(f"{name}: only {0 if mi is None else len(mi)} markers -> cannot solve"); return
        lab, ctr = cluster_markers(mc, mi)
        out = []
        for cl in (0, 1):
            idx = np.where(lab == cl)[0]
            if len(idx) < 4: continue
            sc = [mc[i] for i in idx]; si = mi[idx]
            b = board_pose_cropped(g, K, dist, sc, CFGS)
            if b is None: continue
            b["centre_px"] = ctr[idx].mean(0)
            out.append(b)
            print(f"{name} board{cl}: {b['n']} corners, reproj {b['err']:.2f}px, cfg {b['cfg']}, "
                  f"pos {b['T'][:3,3].round(3)}")
        poses[name] = out
    if len(poses.get("wrist", [])) < 1 or len(poses.get("zed", [])) < 1:
        print("not enough boards solved"); return

    # match boards between cameras by their x-order in the image
    wb = sorted(poses["wrist"], key=lambda b: b["centre_px"][0])
    zb = sorted(poses["zed"], key=lambda b: b["centre_px"][0])
    # pair by shared ChArUco corner ids so only mutually-seen corners are used

    npair = min(len(wb), len(zb))
    P_base, P_zed = [], []
    for i in range(npair):
        common = sorted(set(wb[i]["ids"]) & set(zb[i]["ids"]))
        if len(common) < 4:
            print(f"  board{i}: only {len(common)} shared corners, skipped"); continue
        wi = {c: k for k, c in enumerate(wb[i]["ids"])}; zi = {c: k for k, c in enumerate(zb[i]["ids"])}
        cbw = wb[i]["corners_board"][[wi[c] for c in common]]
        cbz = zb[i]["corners_board"][[zi[c] for c in common]]
        T_base_board = T_base_wcam @ wb[i]["T"]
        P_base.append((T_base_board @ np.c_[cbw, np.ones(len(cbw))].T).T[:, :3])
        P_zed.append((zb[i]["T"] @ np.c_[cbz, np.ones(len(cbz))].T).T[:, :3])
        print(f"  board{i}: {len(common)} shared corners")
    A = np.vstack(P_zed); B = np.vstack(P_base)
    print(f"\ncorrespondences: {len(A)} corners from {npair} board(s)")
    # Kabsch
    ca, cb_ = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb_)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    t = cb_ - R @ ca
    T_new = np.eye(4); T_new[:3, :3] = R; T_new[:3, 3] = t
    resid = np.linalg.norm((R @ A.T).T + t - B, axis=1)
    print(f"fit residual: mean {1000*resid.mean():.1f} mm, max {1000*resid.max():.1f} mm")

    old = np.array(json.load(open(os.path.join(HERE, "outputs/extrinsics_zed.json")))["T_base_camzed"])
    print(f"\nold ZED pos {old[:3,3].round(4)}  ->  new {T_new[:3,3].round(4)}")
    dR = old[:3, :3].T @ R
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    print(f"translation change {1000*np.linalg.norm(T_new[:3,3]-old[:3,3]):.1f} mm | rotation change {ang:.2f} deg")

    out = os.path.join(HERE, "outputs/extrinsics_zed_twoboard.json")
    json.dump({"T_base_camzed": T_new.tolist(), "resid_mm": float(1000*resid.mean()),
               "n_corners": int(len(A)), "n_boards": int(npair)}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--solve", action="store_true")
    a = ap.parse_args()
    if a.capture: capture()
    if a.solve: solve()

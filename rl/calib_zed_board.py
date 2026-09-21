#!/usr/bin/env python3
"""calib_zed_board :: static extrinsic calibration of the fixed ZED from a ChArUco board at a KNOWN
base-frame pose -- the ZED counterpart of `calib_fixed_static.py` (which did the D435 + fixed D405).

Method (identical to calib_fixed_static.py)
-------------------------------------------
The board (5x7 squares, 35 mm square / 26 mm marker, DICT_4X4_50) lies FLAT and FACE UP on the table
with its geometric centre at base (0.4, 0, 0) and its axes aligned to the base axes (X = the 5-square
direction, Y = the 7-square direction). That fixes T_base_board up to 8 axis-aligned candidates
(4 yaws x board-Z up/down); the D435/D405 calibration already resolved the physical one to
yaw = 180 deg, zflip = True, and this script REUSES that T_base_board straight out of
`outputs/extrinsics_d435.json` by default, so the ZED lands in exactly the same base frame as the
RealSense cameras (`--search` prints all 8 candidates with sanity metrics instead).

We then detect the ChArUco in the ZED's RECTIFIED LEFT image with the SDK's rectified left intrinsics
(distortion = 0 by construction), solvePnP -> T_cam_board, and

    T_base_camzed = T_base_board @ inv(T_cam_board)

Frames: the ZED is opened with COORDINATE_SYSTEM.IMAGE and UNIT.METER, i.e. the optical convention
+X right, +Y down, +Z forward -- the same one `outputs/extrinsics_d435.json` declares. The result is
written to `outputs/extrinsics_zed.json` with the key `T_base_camzed` (base <- ZED left optical), which
is what `rl/zed_track.py` loads.

    python3 rl/calib_zed_board.py                      # board on the table at (0.4, 0, 0)
    python3 rl/calib_zed_board.py --search             # show all 8 board-orientation candidates
    python3 rl/calib_zed_board.py --provisional        # NO board: table-plane-only stopgap (see below)

--provisional
-------------
Before the board is available, a usable-but-wrong extrinsic can be bootstrapped from the geometry the
ZED can see on its own: a RANSAC plane fit of the table gives the camera's HEIGHT above the table and
its two TILT angles exactly (3 of the 6 dof). The remaining 3 (yaw about the table normal, and the xy
offset) are NOT observable without a landmark, so they are guessed: the yaw is chosen so the optical
axis' horizontal projection points along base -X (the camera looks back towards the robot, the D435
convention: image-right ~ base +Y), and the xy offset is chosen so the centroid of the objects standing
on the table lands at `--centre` (default the middle of the work area). The file is written with
"provisional": true and zed_track.py prints a warning while it is in use. Re-run WITHOUT --provisional
as soon as the board is on the table.
"""
import argparse
import json
import os

import cv2
import numpy as np
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as Rsc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs", "extrinsics_zed.json")
REF = os.path.join(ROOT, "outputs", "extrinsics_d435.json")
CENTER_BASE = np.array([0.4, 0.0, 0.0])
XR, YR = (0.18, 0.60), (-0.32, 0.32)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def avg_pose(Ts):
    tb = np.array([T[:3, 3] for T in Ts])
    qs = np.array([Rsc.from_matrix(T[:3, :3]).as_quat() for T in Ts])       # xyzw
    qs = qs * np.sign(qs[:, 3:4] + 1e-12)
    qm = qs.mean(0)
    qm /= np.linalg.norm(qm)
    T = make_T(Rsc.from_quat(qm).as_matrix(), tb.mean(0))
    return T, float(np.linalg.norm(tb - tb.mean(0), axis=1).mean() * 1000)


def open_zed(a):
    z = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = getattr(sl.RESOLUTION, a.resolution)
    ip.camera_fps = a.fps
    ip.depth_mode = getattr(sl.DEPTH_MODE, a.depth)
    ip.coordinate_units = sl.UNIT.METER
    ip.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE          # +X right, +Y down, +Z fwd (optical)
    ip.depth_minimum_distance = 0.3
    ip.depth_maximum_distance = 4.0
    ip.sdk_verbose = 0
    st = z.open(ip)
    if st != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(
            f"[calib_zed] ZED open failed: {st}\n"
            "  * the ZED needs a USB 3.0 port (lsusb -t must show the 2b03 device at 5000M, not 480M)\n"
            "  * the ZED SDK needs the V4L2 uvcvideo driver: /dev/video* must exist.\n"
            "    This box blacklists it for librealsense (/etc/modprobe.d/blacklist-uvcvideo.conf):\n"
            "      sudo modprobe uvcvideo")
    info = z.get_camera_information()
    cc = info.camera_configuration
    lc = cc.calibration_parameters.left_cam                    # RECTIFIED left intrinsics
    K = np.array([[lc.fx, 0, lc.cx], [0, lc.fy, lc.cy], [0, 0, 1]], float)
    dist = np.zeros(5)                                         # retrieve_image(VIEW.LEFT) is rectified
    print(f"[calib_zed] {info.camera_model} sn {info.serial_number}  {cc.resolution.width}x{cc.resolution.height}")
    print(f"[calib_zed] rectified left K: fx={lc.fx:.2f} fy={lc.fy:.2f} cx={lc.cx:.2f} cy={lc.cy:.2f} "
          f"(raw disto {np.round(list(lc.disto)[:5], 5).tolist()} -> using 0 on the rectified image)")
    return z, K, dist, info


def grab(z, n_frames, want_xyz=False):
    rt = sl.RuntimeParameters()
    m_img, m_xyz = sl.Mat(), sl.Mat()
    imgs, xyzs = [], []
    got = 0
    for _ in range(n_frames * 4):
        if z.grab(rt) != sl.ERROR_CODE.SUCCESS:
            continue
        z.retrieve_image(m_img, sl.VIEW.LEFT)
        imgs.append(m_img.get_data()[:, :, :3].copy())
        if want_xyz:
            z.retrieve_measure(m_xyz, sl.MEASURE.XYZ)
            xyzs.append(m_xyz.get_data()[:, :, :3].copy())
        got += 1
        if got >= n_frames:
            break
    return imgs, xyzs


# ------------------------------------------------------------------ board calibration
def calib_board(a, z, K, dist):
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict))
    board = cv2.aruco.CharucoBoard((a.squares[0], a.squares[1]), a.square, a.marker, dic)
    cdet = cv2.aruco.CharucoDetector(board)
    ch = board.getChessboardCorners()
    center_board = (ch.min(0) + ch.max(0)) / 2.0
    print(f"[calib_zed] board centre in board coords = {center_board.round(4)} "
          f"(expect ~[{2.5 * a.square:.4f}, {3.5 * a.square:.4f}, 0])")

    imgs, _ = grab(z, a.frames)
    Ts, reps, ncorn, last = [], [], 0, None
    for bgr in imgs:
        last = bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = cdet.detectBoard(gray)
        if ci is None or len(ci) < 6:
            continue
        obj, img = board.matchImagePoints(cc, ci)
        if obj is None or len(obj) < 6:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        reps.append(float(np.sqrt(((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2).sum(1)).mean()))
        R, _ = cv2.Rodrigues(rvec)
        Ts.append(make_T(R, tvec))
        ncorn = len(ci)
    if not Ts:
        if last is not None:
            cv2.imwrite(os.path.join(ROOT, "outputs", "zed_board_notfound.png"), last)
            print("[calib_zed] wrote outputs/zed_board_notfound.png")
        raise SystemExit("[calib_zed] BOARD NOT DETECTED in any frame -- is it in view, flat, face up, unoccluded?")
    T_cam_board, spread = avg_pose(Ts)
    reproj = float(np.mean(reps))
    print(f"[calib_zed] {len(Ts)}/{len(imgs)} frames detected, ~{ncorn} corners, reproj={reproj:.2f} px, "
          f"pose-spread={spread:.2f} mm, board_dist={np.linalg.norm(T_cam_board[:3, 3]):.3f} m")

    def Tbb_for(yaw_deg, zflip):
        R = Rsc.from_euler("z", yaw_deg, degrees=True).as_matrix()
        if zflip:
            R = R @ Rsc.from_euler("x", 180, degrees=True).as_matrix()
        return make_T(R, CENTER_BASE - R @ center_board)

    ref_yaw, ref_zflip, Tbb_ref = None, None, None
    if os.path.exists(REF):
        r = json.load(open(REF))
        ref_yaw, ref_zflip = r.get("selected_yaw_deg"), r.get("selected_zflip")
        Tbb_ref = np.array(r["T_base_board"], float)

    print("\n--- board-orientation candidates (camera must be ABOVE the table and LOOKING DOWN) ---")
    rows = []
    for zf in (False, True):
        for yaw in (0, 90, 180, 270):
            Tbb = Tbb_for(yaw, zf)
            Tbc = Tbb @ inv_T(T_cam_board)
            pos = Tbc[:3, 3]
            look_down = float(Tbc[:3, 2] @ np.array([0, 0, 1.0]))     # optical +Z expressed in base, z-component
            Rcb = inv_T(Tbc)[:3, :3]
            vY, vX = Rcb @ np.array([0, 1.0, 0]), Rcb @ np.array([1.0, 0, 0])
            ok = pos[2] > 0.10 and look_down < -0.2
            flag = "  <== D435 board pose" if (yaw == ref_yaw and zf == bool(ref_zflip)) else ""
            print(f"  yaw={yaw:3d} zflip={int(zf)}: cam_pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}] "
                  f"lookdown={look_down:+.2f} img_right=+Y?{vY[0]:+.2f} img_down=+X?{vX[1]:+.2f} "
                  f"plausible={ok}{flag}")
            rows.append((yaw, zf, Tbb, Tbc, ok))

    if a.yaw is not None:
        yaw, zf = a.yaw, a.zflip
        Tbb = Tbb_for(yaw, zf)
        why = "forced by --yaw/--zflip"
    elif a.search:
        cand = [r for r in rows if r[4]]
        if len(cand) != 1:
            print(f"\n[calib_zed] --search found {len(cand)} plausible candidates; pick one with --yaw/--zflip")
            raise SystemExit(1)
        yaw, zf, Tbb = cand[0][0], cand[0][1], cand[0][2]
        why = "only geometrically plausible candidate"
    else:
        if Tbb_ref is None:
            raise SystemExit(f"[calib_zed] {REF} not found; re-run with --search or --yaw/--zflip")
        yaw, zf, Tbb = ref_yaw, bool(ref_zflip), Tbb_ref
        why = f"reused from outputs/extrinsics_d435.json (yaw={ref_yaw}, zflip={ref_zflip}) so both cameras share a base frame"
        chk = [r for r in rows if r[0] == yaw and r[1] == zf]
        if chk and not chk[0][4]:
            print("\n[calib_zed] WARNING: the reused D435 board pose puts the ZED below the table or looking up. "
                  "Either the board is not placed the same way, or the ZED sees it mirrored -- check --search.")
    print(f"\n==> T_base_board: yaw={yaw} zflip={int(zf)} ({why})")

    Tbc = Tbb @ inv_T(T_cam_board)
    return dict(T_base_board=Tbb, T_base_camzed=Tbc, T_cam_board=T_cam_board, reproj=reproj,
                spread=spread, n=len(Ts), n_corners=int(ncorn), img=last, K=K, dist=dist,
                yaw=int(yaw), zflip=bool(zf),
                board=dict(square=a.square, marker=a.marker, dict=a.dict, squares=list(a.squares)))


# ------------------------------------------------------------------ provisional (no board)
def fit_plane(P, iters=400, thr=0.006, rng=None):
    """RANSAC plane through a (N,3) point cloud. Returns (normal (unit), d) with n.p + d = 0."""
    rng = rng or np.random.default_rng(0)
    best, best_in = None, -1
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False)
        p0, p1, p2 = P[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm /= ln
        d = -nrm @ p0
        k = int((np.abs(P @ nrm + d) < thr).sum())
        if k > best_in:
            best_in, best = k, (nrm, d)
    nrm, d = best
    inl = np.abs(P @ nrm + d) < thr
    Q = P[inl]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c)                     # refine on the inliers
    nrm = Vt[2] / np.linalg.norm(Vt[2])
    return nrm, float(-nrm @ c), int(inl.sum())


def calib_provisional(a, z):
    _, xyzs = grab(z, 5, want_xyz=True)
    if not xyzs:
        raise SystemExit("[calib_zed] no depth frames")
    P = np.median(np.stack(xyzs), 0)
    H, W = P.shape[:2]
    fin = np.isfinite(P).all(2) & (P[:, :, 2] > 0.3) & (P[:, :, 2] < 4.0)
    # only the central half of the image: the table should dominate it
    cen = np.zeros((H, W), bool)
    cen[H // 5:H * 4 // 5, W // 5:W * 4 // 5] = True
    pts = P[fin & cen]
    if len(pts) < 5000:
        raise SystemExit(f"[calib_zed] only {len(pts)} valid depth points -- is the ZED pointed at the table?")
    sub = pts[np.random.default_rng(0).choice(len(pts), min(60000, len(pts)), replace=False)]
    nrm, d, ninl = fit_plane(sub)
    if nrm[2] > 0:                                       # make the normal point back TOWARDS the camera
        nrm, d = -nrm, -d
    height = abs(d)                                      # camera distance to the plane along the normal
    print(f"[calib_zed] table plane: normal_cam={np.round(nrm, 4).tolist()} d={d:+.4f} "
          f"({ninl}/{len(sub)} inliers), camera height above the table = {height:.3f} m")

    # base axes expressed in the CAMERA frame: base +Z = the plane normal pointing at the camera.
    bz = nrm / np.linalg.norm(nrm)
    fwd = np.array([0, 0, 1.0])                          # optical axis in the camera frame
    bx = -(fwd - (fwd @ bz) * bz)                        # default yaw: optical axis points along base -X
    if np.linalg.norm(bx) < 1e-6:
        bx = np.array([1.0, 0, 0]) - (np.array([1.0, 0, 0]) @ bz) * bz
    bx /= np.linalg.norm(bx)
    if a.yaw_deg:
        Rz = Rsc.from_rotvec(np.deg2rad(a.yaw_deg) * bz).as_matrix()
        bx = Rz @ bx
    by = np.cross(bz, bx)
    R_cam_base = np.stack([bx, by, bz], 1)               # columns = base axes in camera coords
    R_base_cam = R_cam_base.T
    # translation: put the table at base z = 0, then shift xy so the objects' centroid sits at --centre
    t = np.array([0.0, 0.0, height])
    Tbc = make_T(R_base_cam, t)
    hgt = (P.reshape(-1, 3) @ R_base_cam[2, :]) + Tbc[2, 3]
    hgt = hgt.reshape(H, W)
    up = fin & cen & (hgt > 0.012) & (hgt < 0.15)
    if up.sum() > 500:
        B = P[up] @ R_base_cam.T + t
        ctr = np.median(B[:, :2], 0)
        t[:2] += np.array(a.centre) - ctr
        print(f"[calib_zed] {int(up.sum())} above-table px, their median xy = {np.round(ctr, 3).tolist()} "
              f"-> shifted to {a.centre}")
    else:
        t[0] += a.centre[0]
        t[1] += a.centre[1]
        print(f"[calib_zed] no objects found above the table; xy offset set to {a.centre} blindly")
    Tbc = make_T(R_base_cam, t)
    return dict(T_base_camzed=Tbc, provisional=True, plane_normal_cam=bz.tolist(),
                plane_height_m=float(height), plane_inliers=int(ninl))


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--square", type=float, default=0.035, help="ChArUco square size (m)")
    ap.add_argument("--marker", type=float, default=0.026, help="ChArUco marker size (m)")
    ap.add_argument("--squares", type=int, nargs=2, default=[5, 7], help="squaresX squaresY")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--resolution", default="HD1080", choices=["HD720", "HD1080", "HD1200", "HD2K"],
                    help="higher = better corner localisation; the tracker may run at a different one")
    ap.add_argument("--depth", default="NEURAL", choices=["NEURAL", "NEURAL_PLUS", "ULTRA", "PERFORMANCE"])
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--search", action="store_true", help="pick the board orientation from geometry instead of reusing the D435's")
    ap.add_argument("--yaw", type=int, default=None, choices=[0, 90, 180, 270], help="force the board yaw")
    ap.add_argument("--zflip", action="store_true", help="with --yaw: board Z down")
    ap.add_argument("--provisional", action="store_true", help="no board: table-plane-only stopgap (see the module docstring)")
    ap.add_argument("--yaw_deg", type=float, default=0.0, help="--provisional: extra yaw (deg) about the table normal")
    ap.add_argument("--centre", type=float, nargs=2, default=[0.39, 0.0], help="--provisional: base xy the objects' centroid is pinned to")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    z, K, dist, info = open_zed(a)
    try:
        r = calib_provisional(a, z) if a.provisional else calib_board(a, z, K, dist)
    finally:
        pass
    Tbc = r["T_base_camzed"]
    pos = Tbc[:3, 3]
    rpy = Rsc.from_matrix(Tbc[:3, :3]).as_euler("xyz", degrees=True)
    look = Tbc[:3, 2]                                    # optical +Z (view direction) in base coords

    res = {
        "frame_convention": "optical +X right +Y down +Z fwd; T_base_cam = base<-cam "
                            "(ZED opened with COORDINATE_SYSTEM.IMAGE, UNIT.METER; left RECTIFIED frame)",
        "method": "table_plane_provisional" if a.provisional else "static_known_board",
        "camera": {"model": str(info.camera_model), "serial": int(info.serial_number),
                   "resolution": f"{info.camera_configuration.resolution.width}x{info.camera_configuration.resolution.height}",
                   "K_left_rectified": K.tolist()},
        "T_base_camzed": Tbc.tolist(),
        "cam_pos_base": pos.tolist(),
        "rpy_xyz_deg": rpy.tolist(),
        "view_dir_base": look.tolist(),
    }
    if a.provisional:
        res.update(provisional=True,
                   provisional_note="table plane only: height + tilt are measured, yaw and xy are GUESSED "
                                    f"(optical axis assumed to point along base -X, object centroid pinned to {a.centre}). "
                                    "Re-run rl/calib_zed_board.py with the ChArUco board at (0.4, 0, 0).",
                   plane_normal_cam=r["plane_normal_cam"], plane_height_m=r["plane_height_m"],
                   plane_inliers=r["plane_inliers"])
    else:
        res.update(board=r["board"], board_center_base=CENTER_BASE.tolist(),
                   board_orientation="axis-aligned to base: X=5-sq dir, Y=7-sq dir, Z up "
                                     "(T_base_board reused from the D435 calibration unless --search/--yaw)",
                   selected_yaw_deg=r["yaw"], selected_zflip=r["zflip"],
                   T_base_board=r["T_base_board"].tolist(),
                   reproj_px=r["reproj"], pose_spread_mm=r["spread"], n_frames=r["n"],
                   n_corners=r["n_corners"],
                   note="static known-board calib of the fixed ZED; same board and same base frame as "
                        "outputs/extrinsics_d435.json. Loaded by rl/zed_track.py.")

    print("\n===== ZED EXTRINSICS =====")
    print(f"cam_pos_base = [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")
    print(f"rpy_xyz      = [{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}] deg")
    print(f"view dir (optical +Z) in base = [{look[0]:+.3f}, {look[1]:+.3f}, {look[2]:+.3f}]  "
          f"({'looking DOWN' if look[2] < -0.2 else 'NOT looking down -- suspicious'})")
    if not a.provisional:
        print(f"reproj = {r['reproj']:.2f} px over {r['n']} frames (~{r['n_corners']} corners), "
              f"pose spread {r['spread']:.2f} mm")
    ok = pos[2] > 0.15 and 0.15 < np.linalg.norm(pos[:2] - CENTER_BASE[:2]) < 1.5 and look[2] < -0.2
    print(f"sanity (above the table, a few tens of cm away, looking down): {'OK' if ok else 'FAILED -- do not trust this'}")

    # overlay: base axes at the board centre, drawn on the last image
    if not a.provisional and r.get("img") is not None:
        img = r["img"].copy()
        Tcam_base = inv_T(Tbc)
        L = 0.08
        basepts = np.array([CENTER_BASE, CENTER_BASE + [L, 0, 0], CENTER_BASE + [0, L, 0], CENTER_BASE + [0, 0, L]], float)
        cb = (Tcam_base[:3, :3] @ basepts.T).T + Tcam_base[:3, 3]
        pp, _ = cv2.projectPoints(cb, np.zeros(3), np.zeros(3), K, dist)
        pp = pp.reshape(-1, 2).astype(int)
        o = tuple(pp[0])
        cv2.line(img, o, tuple(pp[1]), (0, 0, 255), 3)
        cv2.line(img, o, tuple(pp[2]), (0, 255, 0), 3)
        cv2.line(img, o, tuple(pp[3]), (255, 0, 0), 3)
        cv2.putText(img, "baseX", tuple(pp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "baseY", tuple(pp[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ov = os.path.join(ROOT, "outputs", "overlay_zed.png")
        cv2.imwrite(ov, img)
        print(f"wrote {ov}")

    z.close()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Static extrinsic calibration of the two FIXED cameras (D435 + fixed D405)
from a ChArUco board at a KNOWN base-frame pose.

Given (from the user):
  * board center (id-8 / geometric center) at base [0.4, 0, 0]
  * board flat, face up  -> board +Z = base +Z
  * board axes aligned to base axes: X=5-square dir, Y=7-square dir
  * sign cues:  D435 view -> base +Y points image-RIGHT
                D405 view -> base +Y points TOWARD the camera (near)
Solve:  T_base_cam = T_base_board @ inv(T_cam_board)
"""
import json, time, numpy as np, cv2, pyrealsense2 as rs
from scipy.spatial.transform import Rotation as Rsc

SQUARE, MARKER = 0.035, 0.026
SQUARES = (5, 7)                       # squaresX=5 (board X), squaresY=7 (board Y)
DICT = cv2.aruco.DICT_4X4_50
CENTER_BASE = np.array([0.4, 0.0, 0.0])
CAMS = [("d435", "043422070101"), ("d405fixed", "218622277013")]
W, H, FPS, NFRAMES = 1280, 720, 15, 25

dic = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard(SQUARES, SQUARE, MARKER, dic)
cdet = cv2.aruco.CharucoDetector(board)

# board geometric center in board coords (midpoint of interior-corner bbox)
ch = board.getChessboardCorners()          # (N,3) interior chessboard corners
center_board = (ch.min(0) + ch.max(0)) / 2.0
print(f"board center in board coords = {center_board.round(4)}  (expect ~[0.0875,0.1225,0])")

def make_T(R, t):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = np.asarray(t).reshape(3); return T
def inv_T(T):
    R, t = T[:3,:3], T[:3,3]; Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti
def avg_pose(Ts):
    tb = np.array([T[:3,3] for T in Ts])
    qs = np.array([Rsc.from_matrix(T[:3,:3]).as_quat() for T in Ts])  # xyzw
    qs = qs * np.sign(qs[:,3:4]+1e-12)
    qm = qs.mean(0); qm /= np.linalg.norm(qm)
    T = make_T(Rsc.from_quat(qm).as_matrix(), tb.mean(0))
    spread_mm = float(np.linalg.norm(tb - tb.mean(0), axis=1).mean()*1000)
    return T, spread_mm

def open_cam(serial):
    p = rs.pipeline(); c = rs.config(); c.enable_device(serial)
    c.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    pr = p.start(c)
    intr = pr.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]], float)
    dist = np.array(intr.coeffs, float)
    for _ in range(12):
        try: p.wait_for_frames(2000)
        except RuntimeError: pass
    return p, K, dist

def grab_board(p, K, dist, n):
    Ts, reps, last = [], [], None
    for _ in range(n):
        try: fr = p.wait_for_frames(2000)
        except RuntimeError: continue
        cfr = fr.get_color_frame()
        if not cfr: continue
        bgr = np.asanyarray(cfr.get_data()).copy(); last = bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = cdet.detectBoard(gray)
        if ci is None or len(ci) < 6: continue
        obj, img = board.matchImagePoints(cc, ci)
        if obj is None or len(obj) < 6: continue
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue
        proj,_ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        reps.append(float(np.sqrt(((proj.reshape(-1,2)-img.reshape(-1,2))**2).sum(1)).mean()))
        R,_ = cv2.Rodrigues(rvec); Ts.append(make_T(R, tvec)); n_corners=len(ci)
    if not Ts: return None
    T, sp = avg_pose(Ts)
    return dict(T_cam_board=T, spread_mm=sp, n=len(Ts), reproj=float(np.mean(reps)),
               n_corners=int(len(ci)), img=last, K=K, dist=dist)

# ---- capture both cameras ----
det = {}
for name, ser in CAMS:
    p, K, dist = open_cam(ser)
    r = grab_board(p, K, dist, NFRAMES); p.stop()
    if r is None:
        print(f"{name}: BOARD NOT DETECTED"); continue
    det[name] = r
    print(f"{name} ({ser}): {r['n']}/{NFRAMES} det, corners~{r['n_corners']}, "
          f"reproj={r['reproj']:.2f}px, pose-spread={r['spread_mm']:.2f}mm, "
          f"board_dist={np.linalg.norm(r['T_cam_board'][:3,3]):.3f}m")

# ---- candidate T_base_board: 8 axis-aligned orientations (4 yaws x board-Z up/down),
#      anchored so board center -> CENTER_BASE. The physical truth is one of these.
def Tbb_for(yaw_deg, zflip):
    R = Rsc.from_euler('z', yaw_deg, degrees=True).as_matrix()
    if zflip:
        R = R @ Rsc.from_euler('x', 180, degrees=True).as_matrix()
    t = CENTER_BASE - R @ center_board
    return make_T(R, t)

cands = {(y, z): Tbb_for(y, z) for z in (False, True) for y in (0, 90, 180, 270)}

def cam_metrics(Tbb):
    out = {}
    for name in det:
        Tbc = Tbb @ inv_T(det[name]["T_cam_board"])      # T_base_cam
        Rcb = inv_T(Tbc)[:3,:3]                            # base->cam rotation
        vY = Rcb @ np.array([0,1,0.0])                     # base +Y in cam frame (x=right,y=down,z=fwd)
        vX = Rcb @ np.array([1,0,0.0])                     # base +X in cam frame
        out[name] = dict(T_base_cam=Tbc, z=float(Tbc[2,3]), vY=vY, vX=vX)
    return out

# Constraints: both fixed cams mounted ABOVE the board (base z>0). Cues pin yaw/sign:
#   D435 image:  right = base +Y (vY[0]>0)  AND  down = base +X (vX[1]>0)
#   D405 image:  base +Y points TOWARD the camera (vY[2]<0)
print("\n--- candidate search (cams above z>0; D435 right=+Y & down=+X; D405 +Y toward cam) ---")
scored = []
for (y, zf), Tbb in cands.items():
    m = cam_metrics(Tbb)
    above = all(m[n]["z"] > 0 for n in m)
    d = m.get("d435"); p = m.get("d405fixed")
    d435_ok = (d is not None) and (d["vY"][0] > 0) and (d["vX"][1] > 0)
    d405_ok = (p is not None) and (p["vY"][2] < 0)
    # cue strength: how strongly +Y aligns right & +X aligns down (D435) and +Y toward cam (D405)
    cue_strength = ((d["vY"][0] + d["vX"][1]) if d else 0) + ((-p["vY"][2]) if p else 0)
    s = (1 if above else 0)*100 + int(d435_ok) + int(d405_ok)
    info = {n:f"z={m[n]['z']:.2f}" for n in m}
    print(f"  yaw={y:3d} zflip={int(zf)}: {info} above={above} | "
          f"d435(Y_right={d['vY'][0]:+.2f},X_down={d['vX'][1]:+.2f})ok={d435_ok} | "
          f"d405(Y_fwd={p['vY'][2]:+.2f})ok={d405_ok} | str={cue_strength:.2f}")
    scored.append((s, cue_strength, (y, zf), Tbb, m))
scored.sort(key=lambda x:(-x[0], -x[1]))
best_score, best_str, (best_yaw, best_zf), Tbb, M = scored[0]
print(f"\n==> selected yaw={best_yaw} zflip={int(best_zf)} (above+cues), cue_strength={best_str:.2f}")

# ---- results ----
res = {"method":"static_known_board",
       "frame_convention":"optical +X right +Y down +Z fwd; T_base_cam = base<-cam",
       "board":{"square":SQUARE,"marker":MARKER,"dict":"DICT_4X4_50","squares":list(SQUARES)},
       "board_center_base":CENTER_BASE.tolist(),
       "selected_yaw_deg":best_yaw, "selected_zflip":bool(best_zf),
       "T_base_board":Tbb.tolist()}
print("\n===== EXTRINSICS =====")
for name in det:
    Tbc = M[name]["T_base_cam"]
    pos = Tbc[:3,3]; rpy = Rsc.from_matrix(Tbc[:3,:3]).as_euler('xyz', degrees=True)
    res[name] = {"serial":dict(CAMS)[name], "reproj_px":det[name]["reproj"],
                 "pose_spread_mm":det[name]["spread_mm"], "n_frames":det[name]["n"],
                 "T_base_cam":Tbc.tolist(),
                 "cam_pos_base":pos.tolist(), "rpy_xyz_deg":rpy.tolist()}
    print(f"{name:10s}: pos_base=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]  "
          f"rpy=[{rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}]deg  reproj={det[name]['reproj']:.2f}px")

# ---- overlays: draw board frame + base axes anchored at center, for visual check ----
for name in det:
    img = det[name]["img"].copy(); K=det[name]["K"]; dist=det[name]["dist"]
    Tcb = det[name]["T_cam_board"]
    rvec,_ = cv2.Rodrigues(Tcb[:3,:3]); tvec = Tcb[:3,3]
    cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.05, 2)      # board origin axes
    # base axes at base point [0.4,0,0]: transform base pts -> cam -> project
    Tbc = M[name]["T_base_cam"]; Tcam_base = inv_T(Tbc)
    L=0.08
    basepts = np.array([[0.4,0,0],[0.4+L,0,0],[0.4,L,0],[0.4,0,L]],float)
    cb = (Tcam_base[:3,:3]@basepts.T).T + Tcam_base[:3,3]
    pp,_ = cv2.projectPoints(cb, np.zeros(3), np.zeros(3), K, dist); pp=pp.reshape(-1,2).astype(int)
    o=tuple(pp[0])
    cv2.line(img,o,tuple(pp[1]),(0,0,255),3)   # base X red
    cv2.line(img,o,tuple(pp[2]),(0,255,0),3)   # base Y green
    cv2.line(img,o,tuple(pp[3]),(255,0,0),3)   # base Z blue
    cv2.putText(img,"baseY",tuple(pp[2]),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(img,"baseX",tuple(pp[1]),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imwrite(f"overlay_{name}.png", img)
    print(f"wrote overlay_{name}.png")

json.dump(res, open("extrinsics_fixed_static.json","w"), indent=2)
print("\nwrote extrinsics_fixed_static.json")

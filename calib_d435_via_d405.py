#!/usr/bin/env python3
"""Recalibrate the fixed D435 extrinsic from a RANDOMLY-placed ChArUco board, using the wrist D405
at the base pose as the base-frame reference (no arm motion):

  T_base_d405cam = FK(q_base) @ T_TCP_CAM         (wrist cam pose in base, arm at base)
  T_base_board   = T_base_d405cam @ T_d405_board  (D405 sees the board)
  T_base_d435    = T_base_board  @ inv(T_d435_board)   (D435 sees the same board)

Accuracy is hand-eye-limited (~1-2 cm). Board = 5x7 ChArUco, 32 mm square, DICT_4X4_50.
Both the wrist D405 AND the fixed D435 must see the whole board (arm stays at base).

  python3 calib_d435_via_d405.py            # writes fresh extrinsics_d435_static.json (backs up old)
"""
import json
import os
import shutil
import socket
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from joint_conventions import linuxcnc_deg_to_rad

PI = "10.0.0.27"; WRIST = "218622271300"; D435 = "043422070101"
SQUARES = (5, 7); SQ = 0.032; MK = 0.032 * 26.0 / 35.0        # printed 32 mm square, ~24 mm marker
BOARD_DICTS = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
               "DICT_6X6_100": cv2.aruco.DICT_6X6_100, "DICT_4X4_100": cv2.aruco.DICT_4X4_100}


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def read_q():
    s = socket.create_connection((PI, 9999), timeout=3); b = b""
    while b"\n" not in b: b += s.recv(4096)
    s.close(); return list(json.loads(b.split(b"\n")[0])["joints_deg"])
def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).ravel(); return T
def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])
def fk_T(qd):
    r = rpc({"type": "fk", "q": [float(v) for v in linuxcnc_deg_to_rad(qd)]})
    return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))


def capture(serial, w=848, h=480):
    import pyrealsense2 as rs
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]]); dist = np.array(it.coeffs[:5])
    for _ in range(12): pipe.wait_for_frames(2000)
    img = np.asanyarray(pipe.wait_for_frames(2000).get_color_frame().get_data())
    pipe.stop()
    return img, K, dist


def board_pose(img, K, dist, dict_id):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(SQUARES, SQ, MK, d); det = cv2.aruco.CharucoDetector(board)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cc, ci, mc, mi = det.detectBoard(gray)
    if cc is None or len(cc) < 8:
        return None, 0, None
    op, ip = board.matchImagePoints(cc, ci)
    ok, rvec, tvec = cv2.solvePnP(op, ip, K, dist)
    if not ok:
        return None, 0, None
    R, _ = cv2.Rodrigues(rvec); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
    proj, _ = cv2.projectPoints(op, rvec, tvec, K, dist)
    reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1)))
    return T, len(cc), reproj


def main():
    from joint_conventions import rad_to_linuxcnc_deg
    T_tcp_cam = np.array(C.T_TCP_CAM, float)
    try:
        q = read_q()                                              # actual joints (best)
    except Exception:
        q = [float(v) for v in rad_to_linuxcnc_deg(C.BASE_Q)]     # :9999 wedged -> nominal BASE_Q
        print("  (:9999 feedback unavailable; using nominal BASE_Q -- ensure the arm is AT base)")
    T_base_tcp = fk_T(q); T_base_d405 = T_base_tcp @ T_tcp_cam
    print(f"arm q(deg)={[round(v,1) for v in q]}")
    print(f"T_base_d405cam pos={T_base_d405[:3,3].round(3)}")

    print("capturing wrist D405 + fixed D435 (finding a board BOTH see)...")
    img405, K405, d405 = capture(WRIST, 640, 480)
    img435, K435, d435d = capture(D435, 848, 480)
    for tag, img in [("d405_board_view", img405), ("d435_board_view", img435)]:
        cv2.imwrite(os.path.join(C.OUT_DIR, tag + ".png"), img)
    Tw = Td = None; rw = rd = None; chosen = None
    for name, did in BOARD_DICTS.items():
        tw, nw, rwi = board_pose(img405, K405, d405, did)
        td, nd, rdi = board_pose(img435, K435, d435d, did)
        print(f"  {name:14} D405={nw:2d}c  D435={nd:2d}c")
        if tw is not None and td is not None and chosen is None:
            Tw, Td, rw, rd, chosen = tw, td, rwi, rdi, name
    if chosen is None:
        print("  NO common board seen by BOTH cams -- move ONE board into both views (D405 wrist at base + D435). "
              "Views saved to outputs/*_board_view.png")
        return
    print(f"  -> using {chosen} (D405 reproj {rw:.2f}px, D435 reproj {rd:.2f}px)")

    T_base_board = T_base_d405 @ Tw
    T_base_d435 = T_base_board @ np.linalg.inv(Td)
    print(f"T_base_board pos={T_base_board[:3,3].round(3)}  (board on table -> z~0)")
    print(f"NEW T_base_d435 pos={T_base_d435[:3,3].round(3)}")
    op = os.path.join(C.OUT_DIR, "extrinsics_d435_static.json")
    try:
        old = np.array(json.load(open(op))["T_base_cam435"], float)
        print(f"OLD T_base_d435 pos={old[:3,3].round(3)}  shift={np.linalg.norm(T_base_d435[:3,3]-old[:3,3])*1000:.0f} mm")
        shutil.copy(op, op + ".bak_prev")
    except Exception:
        pass
    out = {"T_base_cam435": T_base_d435.tolist(), "serial": D435, "method": "d405_at_base_bridge",
           "square": SQ, "d405_reproj_px": rw, "d435_reproj_px": rd, "board_z_base": float(T_base_board[2, 3])}
    json.dump(out, open(op, "w"), indent=2)
    print(f"saved -> {op}  (board_z={T_base_board[2,3]*1000:.0f}mm; expect ~0 if flat on table)")


if __name__ == "__main__":
    main()

"""Render a single episode to an mp4: wrist + fixed RGB | info panel (6 joint-angle sparklines with
a moving marker, the current phase colour-coded, suction ON/OFF). No deps beyond numpy+cv2.

  python3 il/viz_episode.py outputs/il_pnp_test/sim_ep0002_XXXX out.mp4
"""
import argparse
import glob
import os
import numpy as np
import cv2

PHASE_COL = {  # BGR
    "start": (160, 160, 160), "reach": (60, 180, 210), "descend": (60, 200, 60),
    "grasp": (60, 220, 220), "lift": (60, 140, 220), "carry": (200, 90, 200),
    "place": (80, 200, 240), "release": (240, 120, 80), "home": (150, 150, 150), "idle": (110, 110, 110),
}
JN = ["J1", "J2", "J3", "J4", "J5", "J6"]


def draw_panel(pw, ph, joints_deg, k, phase, suction):
    p = np.full((ph, pw, 3), 28, np.uint8)
    col = PHASE_COL.get(phase, (200, 200, 200))
    cv2.putText(p, f"phase: {phase.upper()}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    # suction indicator
    on = bool(suction[k])
    cv2.circle(p, (pw - 60, 24), 12, (60, 220, 60) if on else (70, 70, 70), -1)
    cv2.putText(p, "SUCTION " + ("ON" if on else "off"), (pw - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60) if on else (140, 140, 140), 2)
    # 6 joint sparklines
    y = 52; sh = (ph - y - 10) // 6; N = len(joints_deg)
    for j in range(6):
        s = joints_deg[:, j]; lo, hi = s.min(), s.max(); rng = max(hi - lo, 1e-3)
        y0 = y + j * sh; gy0 = y0 + 14; gh = sh - 18; gx0 = 60; gw = pw - 80
        cv2.putText(p, f"{JN[j]} {s[k]:+6.1f}", (10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
        pts = [(int(gx0 + gw * i / max(N - 1, 1)), int(gy0 + gh * (1 - (s[i] - lo) / rng))) for i in range(N)]
        cv2.polylines(p, [np.array(pts, np.int32)], False, (90, 140, 200), 1)
        cx, cy = pts[k]
        cv2.circle(p, (cx, cy), 4, (60, 220, 220), -1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode"); ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--height", type=int, default=300)
    a = ap.parse_args()
    arr = np.load(os.path.join(a.episode, "arrays.npz"), allow_pickle=True)
    joints_deg = np.degrees(arr["joints"].astype(np.float32)); suction = arr["suction"]
    phase = arr["phase"]; N = len(joints_deg)
    wr = sorted(glob.glob(os.path.join(a.episode, "wrist", "*.jpg")))
    fx = sorted(glob.glob(os.path.join(a.episode, "fixed", "*.jpg")))
    H = a.height; pw = 460; vw = None
    for k in range(min(N, len(wr), len(fx))):
        w = cv2.imread(wr[k]); f = cv2.imread(fx[k])
        w = cv2.resize(w, (int(H * w.shape[1] / w.shape[0]), H))
        f = cv2.resize(f, (int(H * f.shape[1] / f.shape[0]), H))
        cv2.putText(w, "wrist D405", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(f, "fixed D435", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panel = draw_panel(pw, H, joints_deg, k, str(phase[k]), suction)
        frame = np.concatenate([w, f, panel], axis=1)
        if vw is None:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()
    print(f"wrote {a.out}  ({min(N, len(wr))} frames @ {a.fps} fps)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""blue_dot_mask :: segment the BLUE marker dot on the metal just below the suction cup's black
mushroom dome. The dot is a fixed cup feature (rigid to the camera at rest) that translates
up/down when the cup compresses on contact -> a clean contact signal.

Detection: (1) find the dark mushroom dome dynamically (low-V blob inside the cup mask);
(2) restrict to a band just BELOW the dome bottom, centred on the dome; (3) threshold the
DARKER/saturated blue (rejecting the bright specular glare on the dome that also reads blue);
(4) keep the largest blob.

Saves outputs/blue_dot_mask.npz {blue, centroid, bbox, roi, dome_bottom, base_y} and
outputs/blue_dot.png (RGB+overlay | refined-blue | result). Tunable via args.
"""
import argparse, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C


def segment_blue_dot(bgr, cupm, a):
    """Return (mask uint8, info dict). bgr: HxWx3, cupm: cup bool mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); V = hsv[:, :, 2]
    # 1) dark mushroom dome inside the cup
    dark = cv2.morphologyEx(((V < a.dome_v) & cupm).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 1:
        return np.zeros(V.shape, np.uint8), {}
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); dy, dx = np.where(lab == k)
    bx, by = int(ce[k][0]), int(dy.max())                      # dome centre-x, bottom-y
    # 2) ROI band just below the dome bottom, central
    H, W = V.shape
    x0, x1 = max(0, bx - a.roi_w), min(W, bx + a.roi_w)
    y0, y1 = max(0, by - a.roi_up), min(H, by + a.roi_down)
    roi = np.zeros((H, W), np.uint8); roi[y0:y1, x0:x1] = 255
    # 3) DARKER saturated blue (reject bright dome glare via the V upper bound)
    blue = cv2.inRange(hsv, (a.hlo, a.slo, a.vlo), (a.hhi, 255, a.vhi))
    blue = cv2.bitwise_and(blue, roi)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n2, lab2, st2, ce2 = cv2.connectedComponentsWithStats(blue, 8)
    if n2 <= 1:
        return np.zeros((H, W), np.uint8), {"dome_bottom": by, "roi": [x0, y0, x1, y1]}
    j = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA])); keep = (lab2 == j).astype(np.uint8) * 255
    info = {"centroid": [float(ce2[j][0]), float(ce2[j][1])],
            "bbox": [int(st2[j, cv2.CC_STAT_LEFT]), int(st2[j, cv2.CC_STAT_TOP]),
                     int(st2[j, cv2.CC_STAT_WIDTH]), int(st2[j, cv2.CC_STAT_HEIGHT])],
            "area": int(st2[j, cv2.CC_STAT_AREA]), "dome_bottom": by, "roi": [x0, y0, x1, y1]}
    return keep, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--cup", default=os.path.join(C.OUT_DIR, "cup_mask.npz"))
    ap.add_argument("--from-file", default="", help="use a saved .npy BGR frame instead of the camera")
    ap.add_argument("--dome-v", type=int, default=70, help="V below this = the dark dome")
    ap.add_argument("--hlo", type=int, default=86); ap.add_argument("--hhi", type=int, default=116)   # blue hue
    ap.add_argument("--slo", type=int, default=90)
    ap.add_argument("--vlo", type=int, default=40); ap.add_argument("--vhi", type=int, default=170)    # vhi rejects glare
    ap.add_argument("--roi-w", type=int, default=50); ap.add_argument("--roi-up", type=int, default=10)
    ap.add_argument("--roi-down", type=int, default=55)
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()

    cup = np.load(args.cup); cupm = cup["mask"].astype(bool); H, W = cupm.shape
    if args.from_file:
        bgr = np.load(args.from_file)
    else:
        import pyrealsense2 as rs, time
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(args.serial)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
        for _ in range(5):
            try:
                pipe.start(cfg); break
            except Exception:
                time.sleep(1.5)
        for _ in range(args.frames):
            fr = pipe.wait_for_frames(2000)
        bgr = np.asanyarray(fr.get_color_frame().get_data()); pipe.stop()
        np.save(os.path.join(C.OUT_DIR, "cup_frame_raw.npy"), bgr)

    keep, info = segment_blue_dot(bgr, cupm, args)
    if "centroid" in info:
        print(f"blue dot: area={info['area']}px  centroid=({info['centroid'][0]:.1f},{info['centroid'][1]:.1f})  "
              f"bbox={info['bbox']}  dome_bottom_y={info['dome_bottom']}")
    else:
        print("NO blue dot found — tune --hlo/--hhi/--slo/--vlo/--vhi or --roi-down")

    out = os.path.join(C.OUT_DIR, "blue_dot_mask.npz")
    np.savez(out, blue=keep.astype(bool),
             centroid=np.array(info.get("centroid", [np.nan, np.nan]), float),
             bbox=np.array(info.get("bbox", [0, 0, 0, 0])),
             roi=np.array(info.get("roi", [0, 0, 0, 0])),
             dome_bottom=int(info.get("dome_bottom", 0)),
             base_y=float(info.get("centroid", [np.nan, np.nan])[1]))   # rest-position y for tracking
    # viz
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue_raw = cv2.inRange(hsv, (args.hlo, args.slo, args.vlo), (args.hhi, 255, args.vhi))
    ov = rgb.copy(); ov[keep > 0] = (255, 0, 0)
    if info.get("roi"):
        x0, y0, x1, y1 = info["roi"]; cv2.rectangle(ov, (x0, y0), (x1, y1), (255, 255, 0), 1)
    if "centroid" in info:
        cv2.circle(ov, (int(info["centroid"][0]), int(info["centroid"][1])), 5, (0, 255, 0), 1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(rgb); ax[0].set_title("RGB")
        ax[1].imshow(blue_raw, cmap="gray"); ax[1].set_title(f"blue H[{args.hlo},{args.hhi}] S>{args.slo} V[{args.vlo},{args.vhi}]")
        ax[2].imshow(ov); ax[2].set_title(f"blue dot mask  area={info.get('area', 0)}px")
        for a in ax:
            a.axis("off")
        png = os.path.join(C.OUT_DIR, "blue_dot.png"); fig.savefig(png, dpi=110, bbox_inches="tight")
        print("wrote", out, "and", png)
    except Exception as e:
        print("saved", out, "(plot failed:", e, ")")


if __name__ == "__main__":
    main()

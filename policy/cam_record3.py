#!/usr/bin/env python3
"""cam_record3 :: record all three rig cameras during a real rollout.

  wrist D405 (218622271300) + ZED 2i   -- owned by policy/zed_cam_server.py, read from /dev/shm
  fixed D405 (218622277013)            -- owned HERE, opened directly (free during ZED rollouts)

Runs under SYSTEM python3 (the only interpreter whose librealsense build sees the D405s).
Start zed_cam_server.py first so the wrist/zed shm jpegs exist; this process captures the
fixed D405 directly and snapshots the two shm streams, tagging every frame with time.time()
so the three streams can be aligned afterwards.

Output (one dir per run):
    <out>/<tag>/wrist/<t>.jpg   <out>/<tag>/zed/<t>.jpg   <out>/<tag>/fixed/<t>.jpg
    <out>/<tag>/frames.csv      rows: t_wall, cam, rel_path
    <out>/<tag>/meta.json       serials, fps, start time

    PYTHONPATH=~/librealsense/build/release /usr/bin/python3 policy/cam_record3.py \
        --out ~/pnp_recordings --tag cfm_goaldiv_run1
Ctrl-C to stop; it then muxes each camera's jpgs into <cam>.mp4 if ffmpeg is available.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

SHM = "/dev/shm"
FIXED_SER = "218622277013"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_recordings"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--no_fixed", action="store_true", help="skip the fixed D405 (shm snapshots only)")
    args = ap.parse_args()

    base = os.path.join(args.out, args.tag)
    for c in ("wrist", "zed", "fixed"):
        os.makedirs(os.path.join(base, c), exist_ok=True)
    csv_f = open(os.path.join(base, "frames.csv"), "w", newline="")
    wr = csv.writer(csv_f)
    wr.writerow(["t_wall", "cam", "rel_path"])

    fixed = None
    if not args.no_fixed:
        try:
            fixed = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(FIXED_SER)
            cfg.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, args.fps)
            fixed.start(cfg)
            print(f"[rec] fixed D405 {FIXED_SER} streaming", flush=True)
        except Exception as e:
            print(f"[rec] fixed D405 unavailable ({e}); recording wrist+zed only", flush=True)
            fixed = None

    json.dump(dict(fixed_ser=FIXED_SER, fps=args.fps, t_start=time.time(),
                   have_fixed=fixed is not None),
              open(os.path.join(base, "meta.json"), "w"))

    last = {"wrist": -1.0, "zed": -1.0}  # mtime of last-saved shm jpg per cam
    n = {"wrist": 0, "zed": 0, "fixed": 0}
    t_prev = 0.0
    period = 1.0 / args.fps
    print(f"[rec] -> {base}   (Ctrl-C to stop)", flush=True)
    try:
        while True:
            now = time.time()
            # fixed D405 direct
            if fixed is not None:
                try:
                    fr = fixed.wait_for_frames(1000)
                    col = np.asanyarray(fr.get_color_frame().get_data())
                    t = time.time()
                    p = f"fixed/{t:.3f}.jpg"
                    cv2.imwrite(os.path.join(base, p), col, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    wr.writerow([f"{t:.3f}", "fixed", p]); n["fixed"] += 1
                except Exception:
                    pass
            # shm snapshots: save only when the published frame changed (mtime)
            for cam, shm in (("wrist", f"{SHM}/pnp_wrist.jpg"), ("zed", f"{SHM}/pnp_zed.jpg")):
                try:
                    mt = os.path.getmtime(shm)
                    if mt > last[cam]:
                        with open(shm, "rb") as f:
                            data = f.read()
                        t = time.time()
                        p = f"{cam}/{t:.3f}.jpg"
                        with open(os.path.join(base, p), "wb") as f:
                            f.write(data)
                        wr.writerow([f"{t:.3f}", cam, p]); n[cam] += 1
                        last[cam] = mt
                except FileNotFoundError:
                    pass
            if now - t_prev > 5.0:
                csv_f.flush()
                print(f"[rec] wrist {n['wrist']}  zed {n['zed']}  fixed {n['fixed']}", flush=True)
                t_prev = now
            dt = period - (time.time() - now)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[rec] stopping", flush=True)
    finally:
        csv_f.close()
        if fixed is not None:
            try:
                fixed.stop()
            except Exception:
                pass
        _mux(base, n)


def _mux(base, n):
    """best-effort mp4 per camera from the jpgs, in wall-clock order."""
    if not _has_ffmpeg():
        print("[rec] ffmpeg not found; leaving jpg frames", flush=True)
        print(f"[rec] frames: {n}", flush=True)
        return
    for cam in ("wrist", "zed", "fixed"):
        files = sorted(glob.glob(os.path.join(base, cam, "*.jpg")))
        if len(files) < 2:
            continue
        lst = os.path.join(base, f"{cam}_list.txt")
        with open(lst, "w") as f:
            for fp in files:
                f.write(f"file '{fp}'\n")
                f.write("duration 0.066\n")
            f.write(f"file '{files[-1]}'\n")
        out = os.path.join(base, f"{cam}.mp4")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-vf", "fps=15,format=yuv420p", "-loglevel", "error", out])
        print(f"[rec] wrote {out} ({len(files)} frames)", flush=True)


def _has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""cam_bench :: frame-rate / drop / age bench for one camera, one process (no GIL sharing).

Measures, per frame: sensor capture time (ZED TIME_REFERENCE.IMAGE, D405 global-time frame
timestamp), host time when the frame is in hand, frame number (drops), and the cost of the
current publish path (resize + JPEG q95 + atomic file write to a scratch name in /dev/shm).

    PYTHONPATH=~/librealsense/build/release /usr/bin/python3 policy/cam_bench.py --cam zed --fps 60 --secs 30
    ... --cam wrist --fps 60       (D405 218622271300)   --cam fixed (D405 218622277013)
    ... --cam zed --fps 60 --depth_every 12 --depth_mode NEURAL   (cloud at 5 Hz as in the server)

Writes a JSON summary to --out (default: stdout only). Run several at once for the "all cameras" case.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

SER = {"wrist": "218622271300", "fixed": "218622277013"}


def publish_cost(img, size, name):
    t = time.perf_counter()
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA) if img.shape[1] != size[0] else img
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])
    p = f"/dev/shm/cambench_{name}.jpg"
    with open(p + ".tmp", "wb") as f:
        f.write(buf.tobytes())
    os.replace(p + ".tmp", p)
    return (time.perf_counter() - t) * 1e3


def run_zed(a):
    import pyzed.sl as sl
    z = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD720
    ip.camera_fps = int(a.fps)
    ip.depth_mode = getattr(sl.DEPTH_MODE, a.depth_mode)
    ip.coordinate_units = sl.UNIT.METER
    ip.sdk_verbose = 0
    st = z.open(ip)
    if st != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"ZED open failed: {st}")
    info = z.get_camera_information()
    cfg = info.camera_configuration
    meta = {"model": str(info.camera_model), "res": [cfg.resolution.width, cfg.resolution.height], "fps_set": cfg.fps}
    rt_img = sl.RuntimeParameters(); rt_img.enable_depth = False
    rt_dep = sl.RuntimeParameters(); rt_dep.enable_depth = a.depth_mode != "NONE"
    img, xyz = sl.Mat(), sl.Mat()
    rows = []
    t_end = time.time() + a.secs + 2.0
    n = 0
    drop0 = None
    while time.time() < t_end:
        use_depth = a.depth_every > 0 and n % a.depth_every == 0
        t0 = time.time()
        if z.grab(rt_dep if use_depth else rt_img) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.002)
            continue
        t_grab = time.time()
        t_cap = z.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() * 1e-9
        z.retrieve_image(img, sl.VIEW.LEFT)
        a3 = img.get_data()[:, :, :3]
        t_ret = time.time()
        if use_depth:
            z.retrieve_measure(xyz, sl.MEASURE.XYZ)
        t_dep = time.time()
        pc = publish_cost(a3, (432, 240), "zed")
        dropped = z.get_frame_dropped_count()
        if drop0 is None:
            drop0 = dropped
        rows.append((t_cap, t0, t_grab, t_ret, t_dep, pc, dropped - drop0, float(a3[::8, ::8].mean()), use_depth))
        n += 1
    try:
        meta["exposure"] = z.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)[1]
        meta["gain"] = z.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)[1]
    except Exception as e:
        meta["exposure"] = str(e)
    z.close()
    return meta, rows


def run_rs(a):
    import pyrealsense2 as rs
    p = rs.pipeline()
    c = rs.config()
    c.enable_device(SER[a.cam])
    c.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, int(a.fps))
    if not a.no_rs_depth:
        c.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, int(a.fps))
    prof = p.start(c)
    dev = prof.get_device()
    for s in dev.query_sensors():
        if s.supports(rs.option.global_time_enabled):
            s.set_option(rs.option.global_time_enabled, 1)
    meta = {"model": dev.get_info(rs.camera_info.name), "serial": SER[a.cam], "fps_set": a.fps}
    rows = []
    t_end = time.time() + a.secs + 2.0
    md_ok = {}
    while time.time() < t_end:
        fs = p.wait_for_frames(3000)
        t_host = time.time()
        col = fs.get_color_frame()
        t_cap = col.get_timestamp() * 1e-3
        dom = str(col.get_frame_timestamp_domain()).split(".")[-1]
        toa = None
        for k in ("time_of_arrival", "sensor_timestamp", "actual_exposure", "frame_timestamp", "backend_timestamp"):
            mv = getattr(rs.frame_metadata_value, k)
            ok = col.supports_frame_metadata(mv)
            md_ok[k] = ok
            if ok and k == "time_of_arrival":
                toa = col.get_frame_metadata(mv) * 1e-3
        exp = col.get_frame_metadata(rs.frame_metadata_value.actual_exposure) if md_ok.get("actual_exposure") else -1
        arr = np.asanyarray(col.get_data())
        pc = publish_cost(arr, (424, 240), a.cam)
        rows.append((t_cap, t_host, toa if toa else np.nan, col.get_frame_number(), pc, float(arr[::8, ::8].mean()), exp))
    meta["ts_domain"] = dom
    meta["metadata"] = md_ok
    p.stop()
    return meta, rows


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarise(a, meta, rows):
    R = np.array(rows, dtype=float)
    R = R[R[:, 1 if a.cam != "zed" else 2] > R[0, 1 if a.cam != "zed" else 2] + 2.0]   # drop 2 s warm-up
    out = {"cam": a.cam, "fps_set": a.fps, "secs": a.secs, **meta}
    if a.cam == "zed":
        t_cap, t_grab, t_ret, t_dep, pc, drops, bright, dflag = R[:, 0], R[:, 2], R[:, 3], R[:, 4], R[:, 5], R[:, 6], R[:, 7], R[:, 8]
        loop = np.diff(R[:, 1]) * 1e3
        out["depth_mode"] = a.depth_mode
        out["depth_every"] = a.depth_every
        out["sdk_dropped"] = int(drops[-1] - drops[0])
        out["retrieve_depth_ms_median"] = float(np.median((t_dep - t_ret)[dflag > 0]) * 1e3) if dflag.any() else 0.0
    else:
        t_cap, t_grab, toa, fn, pc, bright, exp = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4], R[:, 5], R[:, 6]
        gaps = np.diff(fn)
        out["frame_number_gaps"] = int((gaps > 1).sum())
        out["frames_missing"] = int((gaps - 1).clip(0).sum())
        out["exposure_us_median"] = float(np.median(exp))
        if np.isfinite(toa).any():
            out["arrival_minus_capture_ms_median"] = float(np.nanmedian(toa - t_cap) * 1e3)
    d = np.diff(t_cap) * 1e3
    age = (t_grab - t_cap) * 1e3
    out.update({
        "n": int(len(R)),
        "fps_measured": float((len(t_cap) - 1) / (t_cap[-1] - t_cap[0])),
        "cap_dt_ms": {"median": float(np.median(d)), "p95": pct(d, 95), "p99": pct(d, 99), "max": float(d.max())},
        "gaps_gt_1p5x": int((d > 1.5 * 1000.0 / a.fps).sum()),
        "age_capture_to_host_ms": {"median": float(np.median(age)), "p95": pct(age, 95), "max": float(age.max())},
        "publish_cost_ms": {"median": float(np.median(pc)), "p95": pct(pc, 95)},
        "brightness_mean": float(np.mean(bright)),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", choices=["zed", "wrist", "fixed"], required=True)
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--secs", type=float, default=30)
    ap.add_argument("--depth_every", type=int, default=0, help="ZED: enable depth on every k-th grab (0 = never)")
    ap.add_argument("--depth_mode", default="NEURAL", help="ZED DEPTH_MODE name, or NONE")
    ap.add_argument("--no_rs_depth", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    meta, rows = run_zed(a) if a.cam == "zed" else run_rs(a)
    s = summarise(a, meta, rows)
    print(json.dumps(s, indent=1), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=1)


if __name__ == "__main__":
    main()

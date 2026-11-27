#!/usr/bin/env python3
"""cam_server60 :: low-latency replacement for policy/zed_cam_server.py (same /dev/shm outputs).

Differences from zed_cam_server.py (bench 2026-11-27, policy/cam_bench.py):
  * one PROCESS per camera -- the old threads shared the GIL, so the wrist D405 arrived in bursts
    (median gap 7 ms, p95 92 ms) every time the ZED thread grabbed/encoded;
  * every frame is published (default 60 fps) -- the old `t_next = now + period` throttle aliased
    15 fps on a 30 fps grab to 10.7 Hz;
  * ZED depth only on every k-th grab (RuntimeParameters.enable_depth), cloud still at --zed_depth_hz;
  * ZED resize on the contiguous BGRA frame (2.5 ms) instead of the [:, :, :3] view (12 ms);
  * sensor CAPTURE time and a sequence number per frame (ZED TIME_REFERENCE.IMAGE, D405 global time);
  * raw-frame ring buffers /dev/shm/pnp_ring_<cam> (64 slots) so a recorder can keep every frame and a
    consumer can skip the JPEG decode (see RingReader);
  * RealSense cameras are started BEFORE the ZED: starting a D405 (libusb backend re-enumerates it)
    while the ZED streams coincided with uvcvideo -71 errors on the ZED and "CAMERA NOT DETECTED";
  * a supervisor exits non-zero if any camera stops publishing for --stale_s (the old server kept
    running with a ZED thread stuck for 6.8 h).

Files kept compatible with policy/real_rollout_zed.py:
    /dev/shm/pnp_wrist.jpg, pnp_zed.jpg, pnp_zed_xyz.npy, pnp_range.txt, pnp_fixed.jpg (--fixed)
    /dev/shm/pnp_cam_stamp.json  {"wrist": t_pub, "zed": t_pub, "n": ..., plus
                                  "<cam>_cap": capture time, "<cam>_seq": frame seq, "<cam>_fps": rate;
                                  "zedxyz*": the same for pnp_zed_xyz.npy -- zedxyz_seq == the zed_seq of
                                  the colour frame grabbed together with that cloud}
    /dev/shm/pnp_ring_<cam>      raw rings (RingReader): wrist / fixed / zed uint8 colour, 64 slots;
                                  zedxyz float16 180x320x3 cloud (camera frame, m), 16 slots, same seq as zed

    PYTHONPATH=~/librealsense/build/release /usr/bin/python3 policy/cam_server60.py [--fps 60] [--fixed]
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SHM = "/dev/shm"
SER = {"wrist": "218622271300", "fixed": "218622277013"}
CAMS = ("wrist", "zed", "fixed")
SLOTS = 64
HDR = 64            # bytes: int64 write count, int64 h, int64 w, int64 c, padding
SLOT_HDR = 32       # bytes per slot: int64 seq, f64 t_cap, f64 t_pub, int64 seq_check


# ----------------------------------------------------------------- shared memory helpers
def put(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


DTYPES = [np.uint8, np.float16, np.float32]              # header code -> element type


class RingWriter:
    """Seqlock ring of raw frames: slot = [seq, t_cap, t_pub, seq_check] + h*w*c elements of `dtype`.
    Header int64s: [write count, h, w, c, dtype code]."""

    def __init__(self, cam, h, w, c=3, dtype=np.uint8, slots=SLOTS):
        self.path = f"{SHM}/pnp_ring_{cam}"
        self.dtype = np.dtype(dtype)
        self.fs = h * w * c * self.dtype.itemsize                # bytes per frame
        self.slots = slots
        size = HDR + slots * (SLOT_HDR + self.fs)
        with open(self.path + ".tmp", "wb") as f:
            f.truncate(size)
        os.replace(self.path + ".tmp", self.path)
        self.m = np.memmap(self.path, np.uint8, "r+", shape=(size,))
        self.hdr = self.m[:HDR].view(np.int64)
        self.hdr[1:6] = (h, w, c, [np.dtype(d) for d in DTYPES].index(self.dtype), slots)
        self.hdr[0] = 0
        self.shape = (h, w, c)

    def write(self, seq, t_cap, t_pub, img):
        k = int(self.hdr[0])
        off = HDR + (k % self.slots) * (SLOT_HDR + self.fs)
        sh = self.m[off:off + SLOT_HDR]
        sh[24:32].view(np.int64)[0] = -1                      # mark slot as being written
        self.m[off + SLOT_HDR:off + SLOT_HDR + self.fs] = np.ascontiguousarray(img, self.dtype).reshape(-1).view(np.uint8)
        sh[0:8].view(np.int64)[0] = seq
        sh[8:16].view(np.float64)[0] = t_cap
        sh[16:24].view(np.float64)[0] = t_pub
        sh[24:32].view(np.int64)[0] = seq
        self.hdr[0] = k + 1


class RingReader:
    """Reader side: latest() -> (seq, t_cap, t_pub, img) or None; since(count) for lossless tailing."""

    def __init__(self, cam):
        self.path = f"{SHM}/pnp_ring_{cam}"
        self.m = np.memmap(self.path, np.uint8, "r")
        hdr = self.m[:HDR].view(np.int64)
        self.hdr = hdr
        self.shape = tuple(int(x) for x in hdr[1:4])
        self.dtype = np.dtype(DTYPES[int(hdr[4])])
        self.slots = int(hdr[5]) or SLOTS
        self.fs = int(np.prod(self.shape)) * self.dtype.itemsize

    def count(self):
        return int(self.hdr[0])

    def read(self, k):
        off = HDR + (k % self.slots) * (SLOT_HDR + self.fs)
        for _ in range(3):
            sh = np.array(self.m[off:off + SLOT_HDR])
            seq = int(sh[0:8].view(np.int64)[0])
            img = np.array(self.m[off + SLOT_HDR:off + SLOT_HDR + self.fs]).view(self.dtype).reshape(self.shape)
            chk = int(self.m[off + 24:off + 32].view(np.int64)[0])
            if chk == seq and seq >= 0:
                return seq, float(sh[8:16].view(np.float64)[0]), float(sh[16:24].view(np.float64)[0]), img
        return None

    def latest(self):
        n = self.count()
        return self.read(n - 1) if n else None


class Stamps:
    """Merged pnp_cam_stamp.json written by several processes (shared array + lock)."""
    KEYS = [f"{c}{s}" for c in CAMS + ("zedxyz",) for s in ("", "_cap", "_seq", "_fps")]

    def __init__(self):
        self.lock = mp.Lock()
        self.a = mp.Array("d", len(self.KEYS) + 1, lock=False)

    def set(self, cam, t_pub, t_cap, seq, fps):
        with self.lock:
            i = self.KEYS.index(cam)
            self.a[i:i + 4] = [t_pub, t_cap, float(seq), fps]
            self.a[-1] += 1
            d = {k: self.a[j] for j, k in enumerate(self.KEYS) if self.a[j] or k in ("wrist", "zed")}
            d["n"] = int(self.a[-1])
            put(f"{SHM}/pnp_cam_stamp.json", json.dumps(d).encode())

    def last_pub(self, cam):
        return self.a[self.KEYS.index(cam)]


# ----------------------------------------------------------------- camera processes
def _jpg(img):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


class Rate:
    def __init__(self):
        self.t, self.v = None, 0.0

    def tick(self, t):
        if self.t is not None and t > self.t:
            self.v = 0.95 * self.v + 0.05 * (1.0 / (t - self.t)) if self.v else 1.0 / (t - self.t)
        self.t = t
        return self.v


def rs_proc(cam, fps, stamps, ready):
    import pyrealsense2 as rs
    p, c = rs.pipeline(), rs.config()
    c.enable_device(SER[cam])
    c.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, fps)
    c.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, fps)
    prof = p.start(c)
    for s in prof.get_device().query_sensors():
        if s.supports(rs.option.global_time_enabled):
            s.set_option(rs.option.global_time_enabled, 1)
    ring = RingWriter(cam, 240, 424)
    out = f"{SHM}/pnp_{cam}.jpg"
    print(f"[cam] {cam} D405 {SER[cam]} streaming at {fps} fps", flush=True)
    ready.set()
    rate, n = Rate(), 0
    while True:
        f = p.wait_for_frames(3000)
        col = f.get_color_frame()
        t_cap = col.get_timestamp() * 1e-3
        img = np.asanyarray(col.get_data())
        if cam == "wrist":
            dep = f.get_depth_frame()
            put(f"{SHM}/pnp_range.txt", f"{dep.get_distance(212, 120) if dep else 0.0:.4f}".encode())
        put(out, _jpg(img))
        t_pub = time.time()
        ring.write(col.get_frame_number(), t_cap, t_pub, img)
        stamps.set(cam, t_pub, t_cap, col.get_frame_number(), rate.tick(t_cap))
        n += 1
        if n % (fps * 10) == 0:
            print(f"[cam] {cam} {n} frames, {rate.v:.1f} fps, capture->published {1e3 * (t_pub - t_cap):.1f} ms", flush=True)


def zed_proc(fps, depth_hz, depth_mode, stamps, ready):
    import cv2
    import pyzed.sl as sl
    z = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD720
    ip.camera_fps = fps
    ip.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    ip.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    ip.coordinate_units = sl.UNIT.METER
    ip.sdk_verbose = 0
    st = z.open(ip)
    if st != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"[cam] ZED open failed: {st} (if 'CAMERA NOT DETECTED' with the ZED on lsusb: "
                         f"replug it or `sudo usbreset 2b03:f880`)")
    info = z.get_camera_information()
    cc = info.camera_configuration
    print(f"[cam] {info.camera_model} sn {info.serial_number} {cc.resolution.width}x{cc.resolution.height} "
          f"@ {cc.fps} fps, v_fov {cc.calibration_parameters.left_cam.v_fov:.1f} deg (sim fovy 69.1)", flush=True)
    rt_img = sl.RuntimeParameters(); rt_img.enable_depth = False
    rt_dep = sl.RuntimeParameters(); rt_dep.enable_depth = True
    every = max(1, int(round(fps / depth_hz))) if depth_hz > 0 else 0
    img, xyz = sl.Mat(), sl.Mat()
    ring = RingWriter("zed", 240, 432)
    ring_xyz = None                                          # sized on the first depth frame (HD720 / 4 = 180 x 320)
    ready.set()
    rate, rate_xyz, n, fails = Rate(), Rate(), 0, 0
    while True:
        use_depth = every and n % every == 0
        if z.grab(rt_dep if use_depth else rt_img) != sl.ERROR_CODE.SUCCESS:
            fails += 1
            if fails > 200:
                raise SystemExit("[cam] ZED grab failing for 200 tries -> exit")
            time.sleep(0.002)
            continue
        fails = 0
        t_cap = z.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() * 1e-9
        z.retrieve_image(img, sl.VIEW.LEFT)
        small = cv2.resize(img.get_data(), (432, 240), interpolation=cv2.INTER_AREA)[:, :, :3]
        small = np.ascontiguousarray(small)
        put(f"{SHM}/pnp_zed.jpg", _jpg(small))
        t_pub = time.time()
        ring.write(n, t_cap, t_pub, small)
        stamps.set("zed", t_pub, t_cap, n, rate.tick(t_cap))
        if use_depth:
            # the cloud belongs to THIS grab: same capture time and sequence number as the colour frame n
            z.retrieve_measure(xyz, sl.MEASURE.XYZ)
            P = np.ascontiguousarray(xyz.get_data()[::4, ::4, :3]).astype(np.float16)
            buf = io.BytesIO(); np.save(buf, P)
            put(f"{SHM}/pnp_zed_xyz.npy", buf.getvalue())
            t_pub_xyz = time.time()
            if ring_xyz is None:
                ring_xyz = RingWriter("zedxyz", P.shape[0], P.shape[1], 3, np.float16, slots=16)
            ring_xyz.write(n, t_cap, t_pub_xyz, P)
            stamps.set("zedxyz", t_pub_xyz, t_cap, n, rate_xyz.tick(t_cap))
        n += 1
        if n % (fps * 10) == 0:
            print(f"[cam] zed {n} frames, {rate.v:.1f} fps, capture->published {1e3 * (t_pub - t_cap):.1f} ms, "
                  f"SDK dropped {z.get_frame_dropped_count()}", flush=True)


# ----------------------------------------------------------------- supervisor
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=60, choices=[15, 30, 60], help="all cameras (D405 also 90)")
    ap.add_argument("--wrist_fps", type=int, help="override for the D405s (5/15/30/60/90)")
    ap.add_argument("--zed_depth_hz", type=float, default=5.0)
    ap.add_argument("--zed_depth", default=os.environ.get("ZED_DEPTH", "NEURAL"))
    ap.add_argument("--no_wrist", action="store_true")
    ap.add_argument("--no_zed", action="store_true")
    ap.add_argument("--fixed", action="store_true", help="also publish the fixed D405 as pnp_fixed.jpg")
    ap.add_argument("--stale_s", type=float, default=2.0)
    a = ap.parse_args()
    rs_fps = a.wrist_fps or a.fps
    stamps = Stamps()
    procs = {}
    ctx = mp.get_context("fork")
    rs_cams = ([] if a.no_wrist else ["wrist"]) + (["fixed"] if a.fixed else [])
    for cam in rs_cams:                                   # RealSense first, then the ZED
        ev = ctx.Event()
        procs[cam] = ctx.Process(target=rs_proc, args=(cam, rs_fps, stamps, ev), daemon=True)
        procs[cam].start()
        if not ev.wait(15):
            raise SystemExit(f"[cam] {cam} did not start")
    if not a.no_zed:
        time.sleep(1.0)
        ev = ctx.Event()
        procs["zed"] = ctx.Process(target=zed_proc, args=(a.fps, a.zed_depth_hz, a.zed_depth, stamps, ev), daemon=True)
        procs["zed"].start()
        if not ev.wait(30):
            raise SystemExit("[cam] zed did not start")
    print(f"[cam] running: {', '.join(procs)}", flush=True)
    t0 = time.time()
    while True:
        time.sleep(0.5)
        now = time.time()
        for cam, p in procs.items():
            if not p.is_alive():
                print(f"[cam] {cam} process died (exit {p.exitcode}) -> exit", flush=True)
                sys.exit(1)
            if now - t0 > 5 and now - stamps.last_pub(cam) > a.stale_s:
                print(f"[cam] {cam} stale for {now - stamps.last_pub(cam):.1f} s -> exit", flush=True)
                for q in procs.values():
                    q.terminate()
                sys.exit(2)


if __name__ == "__main__":
    main()

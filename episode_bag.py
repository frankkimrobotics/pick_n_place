"""episode_bag :: one rosbag2 per pick-and-place episode (todo_plan S5).

Records, per episode (default outputs/episodes/obj_NN/), a self-contained rosbag2:

  /camera/color/image_raw   sensor_msgs/CompressedImage (jpeg)  D405 RGB   (direct grab)
  /camera/depth/image_raw   sensor_msgs/Image  16UC1 (mm)       D405 depth (direct grab)
  /camera/color/camera_info sensor_msgs/CameraInfo              intrinsics (for reuse)
  /joint_states             sensor_msgs/JointState (position)   from /mycobot/drive_feedback (deg->rad)
  /joint_vel                sensor_msgs/JointState (velocity)   from /mycobot/drive_feedback (deg/s->rad/s)
  /joint_cmd                std_msgs/String                     tapped /mycobot/cmd/move (commanded traj)
  /phase                    std_msgs/String                     pipeline events: pregrasp/contact/lift/move/release

Every message carries header.stamp (capture time); the bag receive-timestamp is the
same wall clock. Joints come straight off /mycobot/drive_feedback (pos + vel, no
differentiation) at whatever rate the desktop stream runs (50 Hz now; set the Pi's
STREAM_RATE_HZ=100 to log at the 100 Hz servo rate). Camera frames are pulled from a
caller-supplied getter (the GapMonitor's latest_rgbd) in a ~30 Hz thread.

Storage: tries mcap (smaller bags for CompressedImage); falls back to sqlite3 (the
default plugin) with a printed note if the mcap storage plugin isn't installed.

The bag taps ROS topics itself, so the controller only calls start()/write_phase()/stop().
"""
import json
import math
import os
import shutil
import threading
import time

import numpy as np
import cv2

from rclpy.serialization import serialize_message
from sensor_msgs.msg import CompressedImage, Image, JointState, CameraInfo
from std_msgs.msg import String
from builtin_interfaces.msg import Time as TimeMsg
import array
import rosbag2_py

try:                                    # native .mcap writer (no rosbag2 mcap plugin needed)
    from mcap_ros2.writer import Writer as _McapWriter
    from rosbags.typesys import Stores, get_typestore
    _HAVE_MCAP = True
except Exception:                       # noqa: BLE001
    _HAVE_MCAP = False
_TS = None                              # lazy rosbags typestore (message definitions)


class _N:
    """duck-typed message: the mcap_ros2 encoder wants list fields, rclpy gives ndarrays."""


def _conv(m):
    if hasattr(m, "get_fields_and_field_types"):
        o = _N()
        for fn in m.get_fields_and_field_types():
            setattr(o, fn, _conv(getattr(m, fn)))
        return o
    if isinstance(m, np.ndarray):
        return m.tolist()
    if isinstance(m, array.array):
        return bytes(m) if m.typecode in ("b", "B") else m.tolist()
    if isinstance(m, (bytes, bytearray)):
        return bytes(m)
    if isinstance(m, (list, tuple)):
        return [_conv(x) for x in m]
    return m


class _McapBackend:
    """Writes one .mcap FILE via mcap_ros2 + rosbags typestore (schemas)."""
    used = "mcap"

    def __init__(self, out_path, topics):
        global _TS
        if _TS is None:
            _TS = get_typestore(Stores.ROS2_HUMBLE)
        self.f = open(out_path, "wb")
        self.w = _McapWriter(self.f)
        self.schemas = {}
        for name, typ in topics:
            msgdef, _ = _TS.generate_msgdef(typ, ros_version=2)
            self.schemas[name] = self.w.register_msgdef(typ, msgdef)

    def write(self, topic, msg, ns):
        self.w.write_message(topic=topic, schema=self.schemas[topic],
                             message=_conv(msg), log_time=ns, publish_time=ns)

    def finish(self):
        self.w.finish(); self.f.close()


class _Rosbag2Backend:
    """Writes a rosbag2 directory (sqlite3) -- the fallback."""
    def __init__(self, uri, topics, storage):
        last = None
        for sid in (([storage] if storage == "sqlite3" else [storage, "sqlite3"])):
            try:
                if os.path.isdir(uri):
                    shutil.rmtree(uri)
                w = rosbag2_py.SequentialWriter()
                w.open(rosbag2_py.StorageOptions(uri=uri, storage_id=sid),
                       rosbag2_py.ConverterOptions("", ""))
                for name, typ in topics:
                    w.create_topic(rosbag2_py.TopicMetadata(
                        name=name, type=typ, serialization_format="cdr"))
                self.w = w; self.used = sid
                return
            except Exception as e:  # noqa: BLE001
                last = e
        raise last

    def write(self, topic, msg, ns):
        self.w.write(topic, serialize_message(msg), ns)

    def finish(self):
        self.w = None               # rosbag2 flushes on drop

_TOPICS = [
    ("/camera/color/image_raw", "sensor_msgs/msg/CompressedImage"),
    ("/camera/depth/image_raw", "sensor_msgs/msg/Image"),
    ("/camera/color/camera_info", "sensor_msgs/msg/CameraInfo"),
    ("/joint_states", "sensor_msgs/msg/JointState"),
    ("/joint_vel", "sensor_msgs/msg/JointState"),
    ("/joint_cmd", "std_msgs/msg/String"),
    ("/phase", "std_msgs/msg/String"),
]


def _stamp(t):
    s = int(t)
    return TimeMsg(sec=s, nanosec=int((t - s) * 1e9))


class EpisodeBag:
    """One writer reused across episodes; start()/stop() bracket each one.

    node        : an rclpy node (subscriptions are created on it).
    joint_names : URDF joint order for the JointState messages.
    """

    def __init__(self, node, joint_names, fb_topic="/mycobot/drive_feedback",
                 cmd_topic="/mycobot/cmd/move", cam_hz=30.0, storage="mcap"):
        self.node = node
        self.joint_names = list(joint_names)
        self.nj = len(self.joint_names)
        self.cam_dt = 1.0 / cam_hz
        self.storage = storage
        self._w = None
        self._lock = threading.Lock()
        self._active = False
        self._used_storage = None
        self._frame_getter = None
        self._cam_thread = None
        self._stop = threading.Event()
        self._counts = {}
        # tap the live robot topics once; callbacks no-op until an episode is active
        node.create_subscription(JointState, fb_topic, self._on_fb, 100)
        node.create_subscription(String, cmd_topic, self._on_cmd, 20)

    # ---- lifecycle ----
    def _open(self, uri):
        if self.storage == "mcap" and _HAVE_MCAP:        # native .mcap file
            out = uri if uri.endswith(".mcap") else uri + ".mcap"
            try:
                b = _McapBackend(out, _TOPICS)
                self._used_storage = "mcap"; self._uri = out
                return b
            except Exception as e:  # noqa: BLE001
                print(f"  [bag] native mcap writer failed ({e}); falling back to sqlite3")
        b = _Rosbag2Backend(uri, _TOPICS, "sqlite3")     # fallback
        self._used_storage = b.used; self._uri = uri
        if b.used != self.storage:
            print(f"  [bag] '{self.storage}' unavailable; using '{b.used}'")
        return b

    def start(self, uri, frame_getter=None):
        """Open a bag at `uri` and begin recording. mcap -> writes `uri`.mcap (one file);
        sqlite3 -> a rosbag2 dir. `frame_getter` is a callable -> (rgb, depth, K) or None."""
        os.makedirs(os.path.dirname(uri) or ".", exist_ok=True)
        with self._lock:
            self._w = self._open(uri)
            self._active = True
            self._counts = {n: 0 for n, _ in _TOPICS}
        self._frame_getter = frame_getter
        self._stop.clear()
        if frame_getter is not None:
            self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
            self._cam_thread.start()
        self.write_phase("start")
        print(f"  [bag] recording -> {uri} ({self._used_storage})")

    def stop(self):
        """Flush + close the current episode bag; returns a per-topic count dict."""
        if not self._active:
            return {}
        self.write_phase("end")
        self._stop.set()
        if self._cam_thread is not None:
            self._cam_thread.join(timeout=1.5)
            self._cam_thread = None
        with self._lock:
            counts = dict(self._counts)
            self._active = False
            w = self._w; self._w = None
        if w is not None:
            try:
                w.finish()          # flush + close (mcap: finalize index; rosbag2: drop)
            except Exception as e:  # noqa: BLE001
                print(f"  [bag] finish error: {e}")
        print("  [bag] closed: " + ", ".join(
            f"{'/'.join(k.strip('/').split('/')[-2:])}={v}"
            for k, v in counts.items() if v))
        return counts

    # ---- writers ----
    def _write(self, topic, msg, t=None):
        t = time.time() if t is None else t
        with self._lock:
            if self._active and self._w is not None:
                self._w.write(topic, msg, int(t * 1e9))      # backend serializes
                self._counts[topic] = self._counts.get(topic, 0) + 1

    def _on_fb(self, msg):
        if not self._active:
            return
        t = time.time()
        st = _stamp(t)
        n = min(self.nj, len(msg.position))
        js = JointState()
        js.header.stamp = st
        js.name = self.joint_names
        js.position = [math.radians(float(p)) for p in msg.position[:n]]
        self._write("/joint_states", js, t)
        if len(msg.velocity) >= n:
            jv = JointState()
            jv.header.stamp = st
            jv.name = self.joint_names
            jv.velocity = [math.radians(float(v)) for v in msg.velocity[:n]]
            self._write("/joint_vel", jv, t)

    def _on_cmd(self, msg):
        if not self._active:
            return
        self._write("/joint_cmd", msg)

    def write_phase(self, label):
        m = String()
        m.data = str(label)
        self._write("/phase", m)

    def _cam_loop(self):
        K_sent = False
        while not self._stop.is_set():
            t0 = time.time()
            try:
                r = self._frame_getter()
                if r is not None:
                    rgb, depth, K = r
                    self._write_camera(rgb, depth)
                    if not K_sent and K is not None:
                        self._write_caminfo(K, depth.shape[:2])
                        K_sent = True
            except Exception:  # noqa: BLE001
                pass
            dt = self.cam_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    def _write_camera(self, rgb, depth):
        t = time.time()
        st = _stamp(t)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            ci = CompressedImage()
            ci.header.stamp = st
            ci.format = "jpeg"
            ci.data = jpg.tobytes()
            self._write("/camera/color/image_raw", ci, t)
        d = depth
        if d.dtype.kind == "f":             # meters float -> uint16 millimeters
            d = (np.nan_to_num(d) * 1000.0).astype(np.uint16)
        elif d.dtype != np.uint16:
            d = d.astype(np.uint16)
        im = Image()
        im.header.stamp = st
        im.height, im.width = int(d.shape[0]), int(d.shape[1])
        im.encoding = "16UC1"
        im.is_bigendian = 0
        im.step = im.width * 2
        im.data = d.tobytes()
        self._write("/camera/depth/image_raw", im, t)

    def _write_caminfo(self, K, hw):
        ci = CameraInfo()
        ci.header.stamp = _stamp(time.time())
        ci.height, ci.width = int(hw[0]), int(hw[1])
        ci.distortion_model = "plumb_bob"
        ci.k = [float(x) for x in np.asarray(K, float).reshape(-1)[:9]]
        self._write("/camera/color/camera_info", ci)

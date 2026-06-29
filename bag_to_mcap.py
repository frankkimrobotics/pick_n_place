#!/usr/bin/env python3
"""bag_to_mcap :: convert a rosbag2 sqlite3 (.db3) bag to .mcap using the python
mcap libraries (no rosbag2 mcap storage plugin / no sudo needed).

Reads each message via rosbag2_py (the sqlite3 reader IS installed), deserializes
it, and re-writes it to a single .mcap with the mcap_ros2 writer (chunk-compressed).

  source /opt/ros/humble/setup.bash
  python3 bag_to_mcap.py outputs/episodes/run_XXXX            # -> run_XXXX.mcap
"""
import argparse
import os
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from mcap_ros2.writer import Writer
from rosbags.typesys import Stores, get_typestore

import array
import numpy as np

_TS = get_typestore(Stores.ROS2_HUMBLE)        # message definitions (Humble sqlite bags omit them)


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


def convert(in_uri, out_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=in_uri, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    cls = {n: get_message(t) for n, t in types.items()}
    counts = {}
    with open(out_path, "wb") as f:
        w = Writer(f)
        schemas = {}
        for n, t in types.items():
            msgdef, _ = _TS.generate_msgdef(t, ros_version=2)
            schemas[n] = w.register_msgdef(t, msgdef)
        while reader.has_next():
            topic, data, ts = reader.read_next()
            msg = _conv(deserialize_message(data, cls[topic]))
            w.write_message(topic=topic, schema=schemas[topic], message=msg,
                            log_time=ts, publish_time=ts)
            counts[topic] = counts.get(topic, 0) + 1
        w.finish()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="rosbag2 dir (contains *_0.db3 + metadata.yaml)")
    ap.add_argument("--out", default=None, help="output .mcap (default: <bag>.mcap)")
    a = ap.parse_args()
    uri = a.bag.rstrip("/")
    out = a.out or (uri + ".mcap")
    print(f"converting {uri} -> {out}")
    counts = convert(uri, out)
    sz = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({sz:.1f} MB)")
    for t, n in counts.items():
        print(f"   {t}: {n}")


if __name__ == "__main__":
    main()

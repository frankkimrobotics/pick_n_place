#!/usr/bin/env python3
"""chunk_to_pi :: forward weld chunks from a ROS topic (/planner/weld_chunks, what the
online planner and servo_touch --stream publish) to the Pi's online_servo chunk port
(TCP). This is the desktop->Pi link for the streaming servo controller:

  online_planner / servo_touch --stream --/planner/weld_chunks--> [chunk_to_pi]
        --TCP JSON lines--> pi:9994 (online_servo welds + servos)

  source /opt/ros/humble/setup.bash
  python3 chunk_to_pi.py --pi 10.0.0.27 --port 9994
"""
import argparse
import socket
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ChunkToPi(Node):
    def __init__(self, host, port, topic):
        super().__init__("chunk_to_pi")
        self.host, self.port = host, port
        self.sock = None
        self.lock = threading.Lock()
        self.n = 0
        self.create_subscription(String, topic, self._on, 20)
        self._connect()
        self.get_logger().info(f"forwarding {topic} -> {host}:{port}")

    def _connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=3)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.get_logger().info(f"connected to {self.host}:{self.port}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"connect failed: {e}")
            self.sock = None

    def _on(self, msg):
        with self.lock:
            if self.sock is None:
                self._connect()
            if self.sock is None:
                return
            try:
                self.sock.sendall((msg.data.strip() + "\n").encode())
                self.n += 1
                if self.n % 20 == 1:
                    self.get_logger().info(f"forwarded {self.n} chunks")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"send failed ({e}); will reconnect")
                self.sock = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default="10.0.0.27")
    ap.add_argument("--port", type=int, default=9994)
    ap.add_argument("--topic", default="/planner/weld_chunks")
    a = ap.parse_args()
    rclpy.init()
    node = ChunkToPi(a.pi, a.port, a.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

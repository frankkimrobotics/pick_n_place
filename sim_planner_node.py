#!/usr/bin/env python3
"""sim_planner_node :: wraps the cuRobo socket planner (:9997) as a ROS2 node, so
the planner participates in the ROS2 graph and the orchestrator talks to it over
ROS2 (no custom .srv build needed — request/result over std_msgs/String JSON).

  subscribes /mycobot/curobo/plan_request  std_msgs/String (JSON)
     {"id": "...", "type": "plan_pose"|"plan_joint"|"fk"|"attach"|"detach"|
                           "set_world"|"clear_world"|"ping", ...passthrough...}
  publishes  /mycobot/curobo/plan_result   std_msgs/String (JSON)
     {"id": "...", "success": bool, "trajectory": [[6 rad]], "dt": float, ...}

Trajectories are URDF radians (as cuRobo returns). The cuRobo planner itself runs
in the curobo2 conda env behind the socket; this node is the rclpy front-end.

Run: source /opt/ros/humble/setup.bash; python3 sim_planner_node.py
"""
import argparse
import json
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

_PASSTHROUGH = {"attach", "detach", "set_world", "clear_world", "ping"}


class PlannerNode(Node):
    def __init__(self, host, port):
        super().__init__("sim_curobo_planner")
        self.host, self.port = host, port
        self.pub = self.create_publisher(String, "/mycobot/curobo/plan_result", 10)
        self.create_subscription(String, "/mycobot/curobo/plan_request", self._on_req, 10)
        b = self._rpc({"type": "ping"})
        self.get_logger().info(
            f"cuRobo planner bridge up -> {host}:{port} backend={b.get('backend')}")

    def _rpc(self, req, timeout=40):
        s = socket.create_connection((self.host, self.port), timeout=timeout)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            ch = s.recv(65536)
            if not ch:
                break
            buf += ch
        s.close()
        return json.loads(buf)

    def _on_req(self, msg):
        try:
            req = json.loads(msg.data)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"bad plan_request: {e}")
            return
        rid = req.get("id", "")
        kind = req.get("type")
        try:
            if kind == "plan_pose":
                res = self._rpc({"type": "plan_pose", "start_q": req["start_q"],
                                 "goal_pose": req["goal_pose"],
                                 "max_attempts": int(req.get("max_attempts", 5))})
            elif kind == "plan_joint":
                res = self._rpc({"type": "plan_joint", "start_q": req["start_q"],
                                 "goal_q": req["goal_q"],
                                 "max_attempts": int(req.get("max_attempts", 5))})
            elif kind == "fk":
                res = self._rpc({"type": "fk", "q": req["q"]})
            elif kind in _PASSTHROUGH:
                res = self._rpc(req)
            else:
                res = {"success": False, "status": f"unknown type {kind}"}
        except Exception as e:  # noqa: BLE001
            res = {"success": False, "status": f"planner error: {e}"}
        res["id"] = rid
        out = String()
        out.data = json.dumps(res)
        self.pub.publish(out)
        if kind in ("plan_pose", "plan_joint"):
            self.get_logger().info(
                f"{kind} id={rid} -> success={res.get('success')} "
                f"n={len(res.get('trajectory') or [])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9997)
    args = ap.parse_args()
    rclpy.init()
    node = PlannerNode(args.host, args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose the cup-mask gap signal: move the cup to known heights above the object and
measure gap = annulus_depth - rim_depth at several annulus OFFSETS from the cup rim.
The right offset is the one whose gap tracks the true cup-tip-to-surface distance
(so we clear the cup edge-bleed zone but stay on the object top)."""
import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, quat_wxyz_to_R, make_T
from real_multi import detect_objects


def main():
    ser = "218622271300"
    import rclpy, pyrealsense2 as rs, cv2
    from std_msgs.msg import String
    from object_pointclouds import deproject_mask
    from capture_and_plot import segment
    from multiview_fuse import pick_res
    from perturb_loop import PlannerClient, RobotState, execute

    class A: pass
    args = A(); args.max_h=0.13; args.max_foot=0.16; args.xmin=0.15; args.xmax=0.55; args.ymin=-0.28; args.ymax=0.30

    cup = np.load(os.path.join(C.OUT_DIR, "cup_mask.npz")); mask = cup["mask"].astype(np.uint8)
    k = np.ones((3,3), np.uint8)
    rim = (mask & ~cv2.erode(mask, k, iterations=1)).astype(bool)
    offs = [3, 8, 14, 20]                                   # annulus start offsets (px), each 3px wide
    rings = {o: (cv2.dilate(mask,k,iterations=o+3) & ~cv2.dilate(mask,k,iterations=o)).astype(bool) for o in offs}

    pc = PlannerClient(); rclpy.init(); node = rclpy.create_node("diag")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time":0.15,"pos_gain":1.0,"vff_scale":1.0}
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0,0,-1.0]))))
    CW, CH = pick_res(ser)

    def fk(q): r=pc.rpc({"type":"fk","q":list(map(float,q))}); return make_T(quat_wxyz_to_R(np.array(r["quat"][0])),np.array(r["pos"][0]))
    def fixj6(t): t=np.array(t,float); t[:,5]=t[0,5]; return t
    def goto(xyz,lbl,v):
        q=state.get_q(); r=pc.plan_pose(list(map(float,q)),list(map(float,xyz))+down,max_attempts=14)
        if not r.get("success"): print(f"[{lbl}] plan fail"); return False
        execute(state,pub,fixj6(r["trajectory"]),r["dt"],"pid",v,2.0,lbl,track=track); return True
    def to_base():
        r=pc.plan_joint(list(map(float,state.get_q())),list(map(float,C.BASE_Q)))
        if r.get("success"): execute(state,pub,fixj6(r["trajectory"]),r["dt"],"pid",22.0,3.0,"to-base",track=track)

    pipe=rs.pipeline(); cfg=rs.config(); cfg.enable_device(ser)
    cfg.enable_stream(rs.stream.color,CW,CH,rs.format.bgr8,30); cfg.enable_stream(rs.stream.depth,CW,CH,rs.format.z16,30)
    prof=pipe.start(cfg); scale=prof.get_device().first_depth_sensor().get_depth_scale(); align=rs.align(rs.stream.color)
    intr=prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1.]])
    for _ in range(15): pipe.wait_for_frames(2000)
    def grab():
        for _ in range(5):
            try: f=align.process(pipe.wait_for_frames(1500))
            except RuntimeError: continue
            c=f.get_color_frame(); d=f.get_depth_frame()
            if c and d: return np.asanyarray(c.get_data())[:,:,::-1].copy(), np.asanyarray(d.get_data()).astype(np.float32)*scale
        return None,None
    def med(depth,m,lo,hi):
        v=depth[m]; v=v[(v>lo)&(v<hi)]; return float(np.median(v)) if len(v)>=8 else float("nan")

    to_base(); time.sleep(0.3)
    Tbc=fk(state.get_q())@make_T(np.eye(3),[0,0,C.CAM_TCP_Z_SHIFT])@C.T_TCP_CAM
    rgb,depth=grab()
    objs=detect_objects(rgb,depth,K,Tbc,segment,deproject_mask,args)
    if not objs: print("no object"); pipe.stop(); rclpy.shutdown(); return
    o=max(objs,key=lambda d:d["n"]); P=o["pts"]; cxy=o["centroid"][:2].copy()
    col=P[np.linalg.norm(P[:,:2]-cxy,axis=1)<0.02]; surf=float(np.median(col[:,2])) if len(col)>20 else float(np.percentile(P[:,2],80))
    face=P[np.abs(P[:,2]-surf)<0.01]
    if len(face)>20: cxy=face[:,:2].mean(0)
    print(f"object centre [{cxy[0]:.3f},{cxy[1]:.3f}] surf_z={surf:.3f}")

    for h in (0.040, 0.025, 0.012):
        if not goto([cxy[0],cxy[1],surf+h],f"h={h}",8.0): continue
        time.sleep(0.4)
        tcpz=float(fk(state.get_q())[2,3]); true_dist=tcpz-surf
        rgb,depth=grab()
        rd=med(depth,rim,0.04,0.30)
        print(f"\n  tcp_z={tcpz:.3f}  TRUE cup-tip->surface dist = {true_dist*1000:.0f} mm   rim_depth={rd:.3f}")
        for off in offs:
            ad=med(depth,rings[off],0.04,0.60); gap=ad-rd
            print(f"    annulus@{off:>2}px: depth={ad:.3f}  gap(=ann-rim)={gap*1000:6.1f} mm")
    to_base(); pipe.stop()
    print("\n== look for the offset whose gap ~ TRUE dist and SHRINKS as height drops ==")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

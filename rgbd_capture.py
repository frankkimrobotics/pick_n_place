"""rgbd_capture :: 10 fps RGB+depth capture from the wrist D405 and the fixed
D435 (far side of the table, 0.6 m) during the press-to-seal pick cycle.

Renders a 2x2 video: [D405 RGB | D405 depth] / [D435 RGB | D435 depth].
Depth is inverse-normalised per camera range (near = bright); black = no
return. Run in the mjwarp env (OSMesa offscreen).
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys

import numpy as np
import mujoco
import imageio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
demo = {"__file__": os.path.join(HERE, "mjwarp_pick_demo.py")}
exec(open(demo["__file__"]).read().split("if __name__")[0], demo)

OUT = os.path.join(HERE, "outputs", "mujoco_sim", "rgbd_pick.mp4")
FPS = 10
W, H = 424, 240                      # half D405 res; same aspect for both views
RANGES = {"wrist_d405": (0.07, 0.80), "fixed_d435": (0.25, 1.50)}   # metres


def depth_vis(depth, near, far):
    d = np.clip(depth, near, far)
    v = 1.0 - (d - near) / (far - near)          # near = bright
    v[depth >= far * 0.999] = 0.0                # no-return -> black
    g = (v * 255).astype(np.uint8)
    # simple blue->yellow ramp so depth reads as colour, not just grey
    img = np.stack([g, g, 255 - g], axis=-1)
    return img


def main():
    m = mujoco.MjModel.from_xml_path(demo["XML"])
    sched = demo["build_schedule"](m)
    ren = mujoco.Renderer(m, height=H, width=W)
    # a real camera sees geometry only: kill visualization decor (the yellow
    # rangefinder ray and the tcp/tip site markers render into BOTH the RGB
    # and depth passes otherwise)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    vopt.sitegroup[:] = 0
    writer = imageio.get_writer(OUT, fps=FPS, quality=8)
    every = int(round(1.0 / (FPS * demo["CTRL_DT"])))

    def grab(d, cam):
        ren.update_scene(d, camera=cam, scene_option=vopt)
        rgb = ren.render().copy()
        ren.enable_depth_rendering()
        ren.update_scene(d, camera=cam, scene_option=vopt)
        depth = ren.render().copy()
        ren.disable_depth_rendering()
        return rgb, depth_vis(depth, *RANGES[cam])

    def frame_cb(d, k, weld_tick):
        if k % every:
            return
        r405, d405 = grab(d, "wrist_d405")
        r435, d435 = grab(d, "fixed_d435")
        top = np.hstack([r405, d405])
        bot = np.hstack([r435, d435])
        writer.append_data(np.vstack([top, bot]))

    res = demo["run_cpu"](m, sched, frame_cb=frame_cb)
    writer.close()
    print(f"wrote {OUT}  ({FPS} fps, in_bin={res['in_bin']})")


if __name__ == "__main__":
    main()

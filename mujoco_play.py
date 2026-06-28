"""mujoco_play :: render the exported pick-and-place replay in MuJoCo (headless).

Run in a MuJoCo env (mujoco 3.x). Headless: set MUJOCO_GL=osmesa.

  MUJOCO_GL=osmesa /home/lisc-frank/miniconda3/bin/python pick_and_place/mujoco_play.py
      [--camera iso|front] [--fps N] [--slowdown K]

Loads outputs/mujoco/{scene.xml,traj.npz}, drives the robot-link + object mocap
bodies frame-by-frame (pure kinematic replay of the planner's solution), and
writes outputs/mujoco/mujoco_demo.mp4 (+ a hero PNG).
"""
import os
import argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MJ_DIR = os.path.join(HERE, "outputs", "mujoco")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="iso")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--slowdown", type=int, default=1,
                    help="repeat each frame K times to slow the video down")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    scene = os.path.join(MJ_DIR, "scene.xml")
    data = np.load(os.path.join(MJ_DIR, "traj.npz"), allow_pickle=True)
    pos, quat = data["pos"], data["quat"]
    bodies = [str(b) for b in data["bodies"]]
    fps = args.fps or int(data["fps"])

    model = mujoco.MjModel.from_xml_path(scene)
    mjd = mujoco.MjData(model)

    # map each trajectory body -> its mocap index
    mocap_idx = []
    for b in bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if bid < 0:
            raise RuntimeError(f"body {b} not in model")
        mocap_idx.append(model.body_mocapid[bid])
    mocap_idx = np.array(mocap_idx)

    renderer = mujoco.Renderer(model, args.height, args.width)
    nF = pos.shape[0]
    frames = []
    for i in range(nF):
        mjd.mocap_pos[mocap_idx] = pos[i]
        mjd.mocap_quat[mocap_idx] = quat[i]
        mujoco.mj_forward(model, mjd)
        renderer.update_scene(mjd, camera=args.camera)
        img = renderer.render()
        for _ in range(args.slowdown):
            frames.append(img)
    print(f"[mj] rendered {nF} frames ({args.camera}) at {args.width}x{args.height}")

    import imageio
    os.makedirs(MJ_DIR, exist_ok=True)
    mp4 = os.path.join(MJ_DIR, "mujoco_demo.mp4")
    try:
        imageio.mimsave(mp4, frames, fps=fps, quality=8)
        out = mp4
    except Exception as e:
        out = os.path.join(MJ_DIR, "mujoco_demo.gif")
        imageio.mimsave(out, frames[::2], duration=2.0 / fps)
        print(f"[mj] mp4 failed ({e}); wrote gif")
    # hero still ~75% through (placing)
    imageio.imwrite(os.path.join(MJ_DIR, "mujoco_hero.png"), frames[int(nF*0.75)])
    print(f"[mj] video -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""wrench_showcase :: run policy_final_12of12 on FRESH randomized scenes with
randomized VISUALS (lighting, colors, block palette) and render one video.
The policy is state-based, so appearance changes nothing about behaviour --
this demonstrates the geometry randomization (sizes, gaps, array position,
heights) with visual variety per scene.

    python3 wrench_showcase.py --scenes 6 --seed0 200
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import warp as wp                                          # noqa: E402

OUT = os.path.expanduser("~/pnp_rl/wrench_extract")


def randomize_visuals(xml, rng):
    """Random lights + color palette, injected into a generated scene XML."""
    def col(lo=0.15, hi=0.9, sat=None):
        c = rng.uniform(lo, hi, 3)
        return " ".join(f"{v:.2f}" for v in c)
    # key light + colored fill light, random headlight level
    lx, ly = rng.uniform(-0.5, 1.2), rng.uniform(-0.9, 0.9)
    key = (f'<light pos="{lx:.2f} {ly:.2f} {rng.uniform(1.0, 2.0):.2f}" '
           f'dir="{0.3 - lx:.2f} {-ly:.2f} -1.4" '
           f'diffuse="{col(0.55, 1.0)}" specular="0.3 0.3 0.3" castshadow="true"/>')
    fill = (f'<light pos="{rng.uniform(-0.8, 1.4):.2f} {rng.uniform(-1, 1):.2f} '
            f'{rng.uniform(0.8, 1.6):.2f}" directional="false" '
            f'diffuse="{col(0.1, 0.45)}" castshadow="false"/>')
    amb = rng.uniform(0.05, 0.35)
    vis = (f'<visual><headlight ambient="{amb:.2f} {amb:.2f} {amb:.2f}" '
           f'diffuse="{rng.uniform(0.1, 0.5):.2f} {rng.uniform(0.1, 0.5):.2f} '
           f'{rng.uniform(0.1, 0.5):.2f}"/></visual>')
    xml = xml.replace("<worldbody>", f"{vis}\n  <worldbody>\n    {key}\n    {fill}", 1)
    # floor / table / box palette
    xml = re.sub(r'(name="floor"[^/]*rgba=")[^"]+',
                 lambda m: m.group(1) + col(0.15, 0.6) + " 1", xml)
    xml = re.sub(r'(name="table"[^/]*rgba=")[^"]+',
                 lambda m: m.group(1) + col(0.2, 0.8) + " 1", xml)
    bx = col(0.1, 0.8)
    xml = re.sub(r'(name="pbx_floor"[^/]*rgba=")[^"]+',
                 lambda m: m.group(1) + bx + " 1", xml)
    xml = re.sub(r'(name="pbx_x[mp]"[^/]*rgba=")[^"]+',
                 lambda m: m.group(1) + bx + " 0.5", xml)
    xml = re.sub(r'(name="pbx_y[mp]"[^/]*rgba=")[^"]+',
                 lambda m: m.group(1) + bx + " 0.5", xml)
    # target block: random SATURATED hue; neighbours: random muted colors
    hue = rng.uniform(0, 1)
    import colorsys
    tr, tg, tb = colorsys.hsv_to_rgb(hue, 0.95, 0.95)
    xml = re.sub(r'(name="g_object0"[^/]*rgba=")[^"]+',
                 lambda m: m.group(1) + f"{tr:.2f} {tg:.2f} {tb:.2f} 1", xml)
    for n in range(1, 9):
        xml = re.sub(rf'(name="g_object{n}"[^/]*rgba=")[^"]+',
                     lambda m: m.group(1) + col(0.2, 0.85) + " 1", xml)
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=200)
    ap.add_argument("--ckpt", default="policy_final_12of12.pt")
    ap.add_argument("--h", type=int, default=512)
    a = ap.parse_args()

    import cv2
    import mujoco
    import torch
    wp.init()
    import wrench_pick
    import wrench_extract as W
    from wrench_extract import AC, ExtractEnv, EP

    dev = "cuda"
    ac = AC(h=a.h, std_max=-0.7).to(dev)
    ac.load_state_dict(torch.load(os.path.join(OUT, a.ckpt),
                                  map_location=dev, weights_only=True))
    ac.eval()

    rows, clips = [], []
    for s in range(a.scenes):
        vseed = a.seed0 + s
        rng = np.random.default_rng(vseed + 991)
        xml_path = os.path.join(OUT, f"scene_{vseed}.xml")
        # fresh geometry (same distribution as training/eval)
        g = np.random.default_rng(vseed + 7777)
        half = float(g.uniform(0.015, 0.026))
        wrench_pick.BLOCK_HALF_XY = half
        wrench_pick.PITCH = 2 * half + float(g.uniform(0.002, 0.005))
        wrench_pick.ARRAY_C = (float(g.uniform(0.30, 0.40)),
                               float(g.uniform(-0.06, 0.06)))
        meta = wrench_pick.make_scene(np.random.default_rng(vseed), xml_path)
        json.dump(meta, open(xml_path + ".meta.json", "w"))
        xml = randomize_visuals(open(xml_path).read(), rng)
        open(xml_path, "w").write(xml)

        env = ExtractEnv(xml_path, meta, nworld=1, seed=vseed,
                         device=dev, dr=False)
        env.auto_reset = False
        for _ in range(64):                      # full task from home
            if int(env.start_mode[0]) == 3:
                break
            env.reset(torch.ones(1, dtype=torch.bool, device=dev))
        obs = env.observe()
        rec, info = [], {}
        done = torch.zeros(1, dtype=torch.bool)
        for t in range(EP + 12):
            with torch.no_grad():
                act = torch.tanh(ac.pi(obs))
            obs, r, done, info = env.step(act)
            rec.append((env.qpos[0].cpu().numpy().copy(),
                        bool(env.sealed[0]), float(env.wr_margin[0])))
            if done.any():
                break
        rows.append(dict(seed=vseed, placed=bool(info["placed"][0]),
                         broke=bool(info["broke"][0]), steps=len(rec)))
        print(f"[showcase {vseed}] placed={rows[-1]['placed']} "
              f"broke={rows[-1]['broke']} steps={len(rec)}", flush=True)
        clips.append((xml_path, rec))
        del env

    outp = os.path.join(OUT, "video_showcase.mp4")
    vw = None
    for xml_path, rec in clips:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        ren = mujoco.Renderer(m, height=480, width=640)
        for qp, sealed, margin in rec:
            d.qpos[:] = qp
            mujoco.mj_forward(m, d)
            ren.update_scene(d, camera="iso")
            im = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
            c = (60, 220, 60) if sealed else (40, 40, 230)
            cv2.putText(im, f"sealed={int(sealed)} margin={margin:.2f}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            if vw is None:
                vw = cv2.VideoWriter(outp.replace(".mp4", "_raw.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                     (im.shape[1], im.shape[0]))
            vw.write(im)
        ren.close()
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {outp.replace('.mp4', '_raw.mp4')} "
              f"-c:v libx264 -pix_fmt yuv420p {outp}")
    os.remove(outp.replace(".mp4", "_raw.mp4"))
    n = len(rows)
    print(f"[showcase] placed {sum(r['placed'] for r in rows)}/{n}  "
          f"broke {sum(r['broke'] for r in rows)}/{n}  video -> {outp}")


if __name__ == "__main__":
    main()

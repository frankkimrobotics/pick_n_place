"""Convert episode rosbags (mcap/sqlite written by episode_bag.py) → canonical episodes, so REAL
data lands in the same format as SIM. Needs `pip install rosbags`.

episode_bag topics: /camera/color/image_raw (CompressedImage jpg), /joint_states (pos rad),
/phase (String), and — once added per the plan — /mycobot/suction (Bool) + /d435/... (fixed).
D435 not recorded by default → fixed_rgb = zeros unless --fixed-topic is given. Frames resample to
FPS by the color-frame timeline; episodes split on /phase start..end (else the whole bag = 1 episode).

  python3 il/data/bags_to_dataset.py --bags outputs/episodes/*.mcap --out outputs/il_episodes
"""
import argparse
import glob
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode_writer import EpisodeWriter
from schema import FPS

COLOR = "/camera/color/image_raw"; JS = "/joint_states"; SUC = "/mycobot/suction"; PH = "/phase"


def _read(path, fixed_topic):
    from pathlib import Path
    from rosbags.highlevel import AnyReader
    want = {COLOR, JS, SUC, PH} | ({fixed_topic} if fixed_topic else set())
    out = {t: [] for t in want}
    with AnyReader([Path(path)]) as reader:
        for con, ts, raw in reader.messages():
            if con.topic in want:
                out[con.topic].append((ts * 1e-9, reader.deserialize(raw, con.msgtype)))
    return out


def _jpg(msg):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _nearest(seq, t):
    return min(seq, key=lambda kv: abs(kv[0] - t))[1] if seq else None


def convert_bag(path, out_root, source="real", fixed_topic="", img=None):
    d = _read(path, fixed_topic)
    color = sorted(d[COLOR]); js = sorted(d[JS]); suc = sorted(d.get(SUC, [])); ph = sorted(d[PH])
    fixed = sorted(d.get(fixed_topic, [])) if fixed_topic else []
    if not color or not js:
        print(f"  skip {path}: missing color/joints"); return 0
    t0, t1 = color[0][0], color[-1][0]
    stamps = np.arange(t0, t1, 1.0 / FPS)
    # split into episodes by /phase start..end markers (fallback: whole bag)
    spans, cur = [], None
    for t, m in ph:
        s = m.data.strip()
        if s == "start":
            cur = [t, None]
        elif s == "end" and cur:
            cur[1] = t; spans.append(cur); cur = None
    if not spans:
        spans = [[t0, t1]]
    n = 0
    for ei, (a, b) in enumerate(spans):
        w = EpisodeWriter(out_root, f"{os.path.basename(path)}_{ei}", source=source, fps=FPS)
        for t in stamps:
            if not (a <= t <= b):
                continue
            cimg = _jpg(_nearest(color, t))
            fimg = _jpg(_nearest(fixed, t)) if fixed else np.zeros_like(cimg)
            q = np.array(_nearest(js, t).position[:6], float)
            s_on = int(_nearest(suc, t).data) if suc else 0
            ptxt = (_nearest(ph, t).data if ph else "")
            w.add(t - a, q, s_on, cimg, fimg, phase=ptxt)
        w.close(success=True); n += 1
    print(f"  {path} -> {n} episode(s)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True, help="mcap/sqlite bag paths or globs")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "il_episodes"))
    ap.add_argument("--fixed-topic", default="", help="D435 topic if recorded, e.g. /d435/color/image_raw")
    args = ap.parse_args()
    paths = [p for g in args.bags for p in (glob.glob(g) or [g])]
    total = sum(convert_bag(p, args.out, fixed_topic=args.fixed_topic) for p in paths)
    print(f"done: {total} episodes -> {args.out}")


if __name__ == "__main__":
    main()

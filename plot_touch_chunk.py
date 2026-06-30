#!/usr/bin/env python3
"""Plot a touch_chunk run: tcp-z trajectory with phase bands + contact marker, the D405
contact-detection signal (dome shift + cup depth), and D405 + D435 RGB thumbnails at key
moments.

  python3 plot_touch_chunk.py outputs/episodes/touch_YYYYmmdd_HHMMSS
"""
import json
import os
import socket
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


PCOLOR = {"approach_pregrasp": "#cfe8ff", "descend": "#ffe6cc", "contact_dwell": "#ffd6d6",
          "return": "#d6f5d6", "done": "#eeeeee", "init": "#ffffff"}


def main():
    d = sys.argv[1]
    jz = np.load(os.path.join(d, "joints.npz")); jt = jz["t"]; jq = jz["q"]
    events = json.load(open(os.path.join(d, "events.json")))
    signals = json.load(open(os.path.join(d, "signals.json")))
    meta = json.load(open(os.path.join(d, "meta.json")))
    t0 = float(jt[0])

    idx = np.arange(0, len(jt), max(1, len(jt) // 250))
    zt = jt[idx] - t0
    zz = np.array([rpc({"type": "fk", "q": [float(x) for x in jq[i]]})["pos"][0][2] for i in idx])

    phases = [(e[0] - t0, e[2]) for e in events if e[1] == "phase"]
    contacts = [(e[0] - t0, e[2]) for e in events if e[1] == "contact"]
    frames = [(e[0] - t0, e[2]) for e in events if e[1] == "frame"]
    tend = float(jt[-1]) - t0

    st = np.array([s[0] - t0 for s in signals])
    shift = np.array([(s[2] - s[1]) if (s[1] is not None and s[2] is not None) else np.nan for s in signals])
    cupd = np.array([s[3] if s[3] is not None else np.nan for s in signals])

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(4, 4, height_ratios=[2.2, 1.6, 1.0, 1.0], hspace=0.4, wspace=0.08)
    ax0 = fig.add_subplot(gs[0, :]); ax1 = fig.add_subplot(gs[1, :], sharex=ax0)

    def bands(ax):
        for i, (pt, pn) in enumerate(phases):
            pe = phases[i + 1][0] if i + 1 < len(phases) else tend
            ax.axvspan(pt, pe, color=PCOLOR.get(pn, "#eee"), alpha=0.6, lw=0)
        for ct, cd in contacts:
            ax.axvline(ct, color="red", ls="--", lw=1.5)

    bands(ax0)
    ax0.plot(zt, zz, "k-", lw=1.5)
    if "xyz" in meta:
        ax0.axhline(meta["xyz"][2], color="orange", ls=":", lw=1, label=f"object top {meta['xyz'][2]:.3f}m")
        ax0.legend(loc="upper right", fontsize=8)
    ax0.set_ylabel("tcp z (m)")
    ax0.set_title(f"touch_chunk {os.path.basename(d)} — trajectory + phases + contact")
    for pt, pn in phases:
        ax0.text(pt + 0.05, ax0.get_ylim()[1], pn, fontsize=7, va="top", color="#333")

    bands(ax1)
    ax1.plot(st, shift, "b-", lw=1.3, label="dome shift (px)")
    ax1.axhline(12, color="purple", ls=":", lw=1, label="contact threshold (12px)")
    ax1.set_ylabel("dome shift (px)", color="b"); ax1.set_xlabel("time (s)")
    ax1b = ax1.twinx(); ax1b.plot(st, cupd, "g-", lw=1, alpha=0.6)
    ax1b.set_ylabel("cup depth (mm)", color="g")
    ax1.legend(loc="upper left", fontsize=8)
    for ct, cd in contacts:
        ax1.annotate(cd, (ct, ax1.get_ylim()[1]), fontsize=7, color="red", va="top")

    def nearest_frame(tt):
        return None if not frames else min(frames, key=lambda f: abs(f[0] - tt))[1]

    keys = []
    for nm in ("approach_pregrasp", "descend", "return"):
        for pt, pn in phases:
            if pn == nm:
                keys.append((nm, pt + 0.5)); break
    if contacts:
        keys.insert(2, ("contact", contacts[0][0]))
    keys = keys[:4]
    while len(keys) < 4:
        keys.append(("", tend * (len(keys) + 1) / 5.0))

    for col, (lab, tt) in enumerate(keys):
        fi = nearest_frame(tt)
        for row, cam in enumerate(("d405", "d435")):
            ax = fig.add_subplot(gs[2 + row, col]); ax.axis("off")
            p = os.path.join(d, cam, f"{fi:04d}.jpg") if fi is not None else None
            if p and os.path.exists(p):
                ax.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
            ax.set_title(f"{cam} · {lab}\nt={tt:.1f}s", fontsize=7)

    out = os.path.join(d, "plot.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

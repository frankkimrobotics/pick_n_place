#!/usr/bin/env python3
"""Plot the cup-mask regions (cup / rim / ring / dome) + the rim-gapped OFFSET ring used by
multi_touch, overlaid on a live D405 frame. Saves outputs/cup_masks.png."""
import os, sys, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import config as C
import pyrealsense2 as rs

W, H = 640, 480
cm = np.load(os.path.join(C.OUT_DIR, "cup_mask.npz"))
cup, rim, ring, dome = cm["mask"], cm["rim"], cm["ring"], cm["dome"]
def dil(m, r): return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8))
oring = (dil(cup, 6 + 12) > 0) & (dil(cup, 6) == 0)          # gap=6, width=12 (matches multi_touch)

# grab one D405 frame for the backdrop
try:
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device("218622271300")
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    for _ in range(12): pipe.wait_for_frames(2000)
    rgb = np.asanyarray(pipe.wait_for_frames(1000).get_color_frame().get_data())[:, :, ::-1].copy()
    pipe.stop()
except Exception as e:
    print("no camera, using grey backdrop:", e); rgb = np.full((H, W, 3), 90, np.uint8)

layers = [("cup (mask)", cup, (200, 200, 200)), ("dome (mushroom)", dome, (60, 120, 255)),
          ("rim", rim, (255, 60, 60)), ("ring (orig 12px)", ring, (255, 220, 0)),
          ("OFFSET ring (rim-gapped)", oring, (40, 220, 90))]

over = rgb.copy()
for _, m, col in layers:
    ov = over.copy(); ov[m] = col; over = cv2.addWeighted(ov, 0.45, over, 0.55, 0)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, col, 1)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].imshow(rgb); ax[0].set_title("D405 view (wrist cam, cup in frame)"); ax[0].axis("off")
ax[1].imshow(over); ax[1].set_title("cup-mask regions + offset ring"); ax[1].axis("off")
ax[1].legend(handles=[Patch(facecolor=np.array(c) / 255, label=f"{n}  ({int(m.sum())}px)")
                      for n, m, c in layers], loc="upper right", fontsize=9, framealpha=0.9)
fig.suptitle("Suction-cup depth masks: rim/ring sit on the shadowed edge; the OFFSET ring is gapped clear of it",
             fontsize=12)
fig.tight_layout()
out = os.path.join(C.OUT_DIR, "cup_masks.png")
fig.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out)
for n, m, _ in layers: print(f"  {n:26s} {int(m.sum()):5d} px")

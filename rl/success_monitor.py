#!/usr/bin/env python3
"""success_monitor :: independent pick-and-place SUCCESS MONITOR using the two D405s that are idle
during a run (fixed D405 218622277013, wrist D405 218622271300). It never touches the D435 that
rl/rgb_track.py holds, and it never commands the robot.

Why a second monitor: rl/real_policy_ctrl.py's own "[result]" verdict trusts the D435 tracker's
*largest* blob after the run, which is wrong as soon as a second object (e.g. a second can) is on
the table -- the "goal" blob might just be the other object sitting where it always was. This
monitor instead diffs full BEFORE/AFTER frames from two extra, unused cameras and only calls
something a change if a patch of pixels actually flipped between "table colour" and "not table
colour", then locates that patch in 3-D (fixed cam, calibrated) or in pixel space (wrist cam,
homography-calibrated on the fly).

Geometry
--------
Fixed D405 (serial 218622277013): mounted off to the front-right corner, looking obliquely across
the table -- a good side view of lift height and of the goal region. `outputs/extrinsics_d405_fixed.json`
gives T_base_cam405fixed, i.e. base <- camera OPTICAL frame, but it was solved on the DEPTH optical
frame while we align depth to colour and backproject colour pixels. Exactly like rgb_track.py:
    Tbc(raw) = base <- depth-optical
    Tdc      = colour-optical <- depth-optical (from the RealSense stream extrinsics, inverted)
    Tbc      = Tbc(raw) @ Tdc     # now base <- colour-optical, which matches aligned-to-colour pixels
Backprojecting an aligned colour pixel (u, v) with depth z through the colour intrinsics (fx, fy, cx0,
cy0) gives a point in the colour-optical frame; Tbc @ that point is the base-frame 3-D position.

Wrist D405 (serial 218622271300): rigidly mounted on the arm, so its base-frame pose changes with
every joint move -- the old hand-eye calibration is for a different task and is NOT used here. At
the controller's HOME pose (where BEFORE/AFTER frames are always taken) it looks straight down at
the whole workspace, so a *fixed*, home-pose-only pixel->base-frame mapping exists: we fit it as a
planar homography, sampled live by pairing the wrist camera's own blob pixels against the D435
tracker's base-frame blobs (broadcast on udp 9701) every time a `verdict` is requested, and persist
it to ~/pnp_rl/wrist_homography.json once enough pairs with good spread have accumulated. Until then
the wrist camera is used as a pixel-space-only second witness (does it see anything appear/vanish
near where the fixed camera says it does -- checked in pixel space via the same homography once it
exists, or just "did it see a component of the right kind at all" before that).

Change detection ("diff" segmentation, same idea as rgb_track.py --mode diff)
-------------------------------------------------------------------------
1. Convert BEFORE and AFTER (temporal-median) colour frames to Lab.
2. change_mask = (||Lab(after) - Lab(before)|| > dthr) AND inside the camera's valid region
   (fixed cam: projected work-area polygon; wrist cam: full frame minus a border margin).
3. morphological open (kills salt noise) then close (fills small holes) with a 5x5 ellipse.
4. connected components >= min_area px, not touching the image border (the arm entering/leaving
   the frame produces a huge border-touching blob -- rejected, same rule as rgb_track).
5. Classify each component APPEARED vs VANISHED by comparing, within the component's own pixels,
   its Lab distance from the *before* frame's table colour vs from the *after* frame's table colour
   (each a per-frame median Lab over the valid region, so slow lighting drift cancels out):
     d_before = median || Lab_before(px) - table_colour_before ||   (over the component's pixels)
     d_after  = median || Lab_after(px)  - table_colour_after  ||
   d_after > d_before  ->  the patch looks like "table" in the BEFORE frame and "not table" in the
   AFTER frame -> an object is there now that wasn't -> APPEARED.
   d_before > d_after  ->  the reverse -> VANISHED (something was there and is gone).

3-D localisation (fixed cam only -- the wrist cam has no metric extrinsics)
-------------------------------------------------------------------------
APPEARED components are placed using the AFTER depth (the object is there now); VANISHED components
use the BEFORE depth (the object was there then). Valid depths are 0.15..1.5 m. With >= 30 valid
points: top = 90th-percentile base-frame z, (cx, cy) = mean xy of the points within 1.5 cm of top
(a flat "cap" estimate, exactly like rgb_track's depth branch). With < 30 valid points (thin/glossy
object, sparse D405 depth): intersect the camera ray through the component's pixel centroid with the
horizontal plane z = top/2, where `top` is the object height passed in the verdict command (same
fallback rgb_track uses with its --top argument). Components whose estimated height exceeds 0.15 m
are rejected as the arm/gripper being back in frame, not the object.

Verdict rule (see `classify_verdict` for the exact thresholds/order)
-------------------------------------------------------------------------
"picked"  := a VANISHED (fixed-cam) component within 6 cm of `obj`, MINUS a "pushed" override: if
             there is *also* an APPEARED component within 8 cm of `obj`, the object never really left
             the pick point (it was shoved a few cm, not lifted) -> "pushed, not picked" wins.
"placed"  := an APPEARED (fixed-cam) component within 10 cm of `goal`; d_goal is that component's
             distance to goal (the closest APPEARED component to goal, whatever the distance, so a
             "dropped" placement still gets a distance reported).
  PICK AND PLACE OK                         picked and d_goal < 0.06
  carried, placed N cm off                  picked and d_goal < 0.15
  picked, not seen near the goal (dropped?) picked, otherwise
  pushed, not picked                        vanished-near-obj AND appeared-near-obj
  seal FAILED (object did not move)         not picked, but *some* change was seen elsewhere
  no change detected                        not picked, and no change anywhere at all

Usage
-----
  PYTHONPATH=~/librealsense/build/release python3 rl/success_monitor.py &
  # ... snap before a run, verdict after ...
  PYTHONPATH=~/librealsense/build/release python3 rl/success_monitor.py --once \
      --goal 0.40 0.00 --obj 0.45 0.15 --top 0.05
"""
import argparse
import json
import os
import socket
import select
import time
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CMD_PORT = 9702
TRACK_PORT = 9701                      # rl/rgb_track.py's D435 broadcast, used only to learn the wrist homography
FIXED_SERIAL = "218622277013"
WRIST_SERIAL = "218622271300"
XR, YR = (0.18, 0.60), (-0.32, 0.32)   # work area, base frame, metres
ZMAX_ARM = 0.15                        # component height above this = the arm/gripper, not the object
ZTOP_ROI = 0.15                        # generous object-height ceiling used when projecting the fixed-cam ROI

PNP_RL = os.path.expanduser("~/pnp_rl")
DEBUG_DIR_DEFAULT = os.path.join(PNP_RL, "monitor")
HOMOGRAPHY_FILE = os.path.join(PNP_RL, "wrist_homography.json")
PAIRS_FILE = os.path.join(PNP_RL, "wrist_homography_pairs.json")

W, H = 640, 480


# ---------------------------------------------------------------------------------------------- camera plumbing
class Cam:
    """One opened, colour+depth-aligned RealSense stream, plus (fixed cam only) its base-frame pose."""

    def __init__(self, name, serial, base_extrinsics=None):
        import pyrealsense2 as rs
        self.name = name
        self.serial = serial
        self.rs = rs
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
        prof = pipe.start(cfg)
        self.pipe = pipe
        self.align = rs.align(rs.stream.color)
        self.ds = prof.get_device().first_depth_sensor().get_depth_scale()
        cprof = prof.get_stream(rs.stream.color).as_video_stream_profile()
        dprof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = cprof.get_intrinsics()
        self.fx, self.fy, self.cx0, self.cy0 = intr.fx, intr.fy, intr.ppx, intr.ppy
        self.Tbc = None
        if base_extrinsics is not None:
            Tbc = np.array(base_extrinsics, dtype=np.float64)
            e = cprof.get_extrinsics_to(dprof)                 # colour-frame point -> depth-frame point
            Tdc = np.eye(4)
            Tdc[:3, :3] = np.array(e.rotation).reshape(3, 3).T
            Tdc[:3, 3] = e.translation
            self.Tbc = Tbc @ Tdc                                # base <- colour-optical (matches aligned pixels)
            self.Tcb = np.linalg.inv(self.Tbc)
        for _ in range(10):
            pipe.wait_for_frames()                              # let auto-exposure settle

    def grab(self):
        fs = self.align.process(self.pipe.wait_for_frames())
        col = np.asanyarray(fs.get_color_frame().get_data()).copy()
        dep = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.ds
        return col, dep

    def grab_median(self, n=5):
        """Temporal median over n frames: knocks down single-frame D405 depth holes and colour noise."""
        cols, deps = [], []
        for _ in range(n):
            c, d = self.grab()
            cols.append(c); deps.append(d)
        col = np.median(np.stack(cols, 0), axis=0).astype(np.uint8)
        dep = np.median(np.stack(deps, 0), axis=0).astype(np.float32)
        return col, dep

    def roi_mask(self, top=ZTOP_ROI):
        """Fixed cam only: project the 8 work-area corners (z=0 and z=top) into the image, convex hull."""
        assert self.Tcb is not None
        corners = [(XR[0], YR[0]), (XR[1], YR[0]), (XR[1], YR[1]), (XR[0], YR[1])]
        pts = []
        for zc in (0.0, top):
            for (x, y) in corners:
                pc = self.Tcb @ np.array([x, y, zc, 1.0])
                pts.append((int(self.fx * pc[0] / pc[2] + self.cx0), int(self.fy * pc[1] / pc[2] + self.cy0)))
        hull = cv2.convexHull(np.array(pts, np.int32))
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask, hull

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


def border_mask(margin=6, top=100):
    """Valid region for the wrist cam: full frame minus a border margin, and minus a taller strip at
    the TOP of the image. At the controller's home pose the wrist D405 looks straight down but its
    field of view still catches the mount pole / desk background above the table's near horizon (and
    anyone walking past it) -- observed directly in testing (2026-09-20): an unchanged table produced
    two spurious "changed" components up there from a hand passing near the camera, while the table
    region itself was perfectly quiet. `top` crops that dead zone out; tune per rig if the wrist cam
    is ever remounted."""
    m = np.zeros((H, W), np.uint8)
    m[max(margin, top):H - margin, margin:W - margin] = 255
    return m


# ---------------------------------------------------------------------------------------------- change segmentation
KERN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERN_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def segment_change(before_bgr, after_bgr, valid_mask, dthr=18.0, min_area=300):
    """Lab-space BEFORE/AFTER diff -> classified connected components. See module docstring for the
    APPEARED/VANISHED rule. Returns a list of dicts: kind, mask(bool HxW), bbox(x,y,w,h), centroid(u,v),
    area, d_before, d_after."""
    lab_b = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.linalg.norm(lab_a - lab_b, axis=2)
    change = ((diff > dthr) & (valid_mask > 0)).astype(np.uint8) * 255
    change = cv2.morphologyEx(change, cv2.MORPH_OPEN, KERN)
    change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, KERN_CLOSE)
    table_b = np.median(lab_b[valid_mask > 0].reshape(-1, 3), axis=0)
    table_a = np.median(lab_a[valid_mask > 0].reshape(-1, 3), axis=0)
    ncc, comp_lab, stats, cents = cv2.connectedComponentsWithStats(change)
    out = []
    for i in range(1, ncc):
        x, y, w, h, area = stats[i]
        if area < min_area or x <= 0 or y <= 0 or x + w >= W or y + h >= H:
            continue
        m = comp_lab == i
        d_before = float(np.median(np.linalg.norm(lab_b[m] - table_b, axis=1)))
        d_after = float(np.median(np.linalg.norm(lab_a[m] - table_a, axis=1)))
        kind = "appeared" if d_after > d_before else "vanished"
        out.append(dict(kind=kind, mask=m, bbox=(int(x), int(y), int(w), int(h)),
                         centroid=(float(cents[i][0]), float(cents[i][1])), area=int(area),
                         d_before=d_before, d_after=d_after))
    return out, change, (table_b, table_a)


# ---------------------------------------------------------------------------------------------- 3-D localisation
def localize_fixed(cam, comp, before_dep, after_dep, top_fallback):
    """Base-frame (x, y, top) of a fixed-cam component; None if rejected (arm height, no valid data)."""
    dep = after_dep if comp["kind"] == "appeared" else before_dep
    m = comp["mask"]
    zs = dep[m]
    valid = (zs > 0.15) & (zs < 1.5)
    if int(valid.sum()) >= 30:
        vs, us = np.nonzero(m)
        z = dep[vs, us]
        sel = (z > 0.15) & (z < 1.5)
        us, vs, z = us[sel], vs[sel], z[sel]
        P = np.stack([(us - cam.cx0) / cam.fx * z, (vs - cam.cy0) / cam.fy * z, z, np.ones_like(z)], 1) @ cam.Tbc.T
        # keep only points at table level up to arm height: a vanished can's BEFORE depth is holey on the
        # D405 and the holes look through to the far background (s3_01: a can at x 0.50 localised at 0.15)
        P = P[(P[:, 2] > -0.02) & (P[:, 2] < ZMAX_ARM + 0.05)]
        if len(P) < 30:
            return _localize_plane(cam, comp, top_fallback)
        top = float(np.percentile(P[:, 2], 90))
        T = P[P[:, 2] > top - 0.015]
        cx, cy = float(T[:, 0].mean()), float(T[:, 1].mean())
        src = "depth"
        if top < 0.005:                                           # depth says "table": the blob is a shadow/reflection
            return _localize_plane(cam, comp, top_fallback)
    else:                                                         # plane fallback through the centroid ray
        return _localize_plane(cam, comp, top_fallback)
    if top > ZMAX_ARM or not (XR[0] < cx < XR[1] and YR[0] < cy < YR[1]):
        return None
    return dict(cx=cx, cy=cy, top=top, n=comp["area"], src=src, kind=comp["kind"])


def _localize_plane(cam, comp, top_fallback):
    """Centroid ray intersected with the plane z = top/2; None when outside the work area (background
    people/objects behind the table land at x < 0 with this camera, s3_01)."""
    u0, v0 = comp["centroid"]
    r = np.array([(u0 - cam.cx0) / cam.fx, (v0 - cam.cy0) / cam.fy, 1.0])
    o = cam.Tbc[:3, 3]; d = cam.Tbc[:3, :3] @ r
    if abs(d[2]) < 1e-6:
        return None
    s = (top_fallback / 2 - o[2]) / d[2]
    p = o + s * d
    if s <= 0 or not (XR[0] < p[0] < XR[1] and YR[0] < p[1] < YR[1]):
        return None
    return dict(cx=float(p[0]), cy=float(p[1]), top=float(top_fallback), n=comp["area"], src="plane", kind=comp["kind"])


def localize_wrist(comp, homography):
    """Pixel centroid, plus base-frame (x, y) via the learned homography if one exists yet."""
    u, v = comp["centroid"]
    out = dict(u=u, v=v, n=comp["area"], kind=comp["kind"])
    if homography is not None:
        pt = cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), homography)[0, 0]
        out["cx"], out["cy"] = float(pt[0]), float(pt[1])
    return out


# ---------------------------------------------------------------------------------------------- wrist homography
def wrist_object_blobs(after_bgr, dthr=18.0, min_area=300):
    """Current-frame (not before/after diff) object blobs on the wrist cam, for homography pairing:
    Lab distance from the frame's own median colour (the table, since it's the majority of pixels)."""
    valid = border_mask(10)
    lab = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    table = np.median(lab[valid > 0].reshape(-1, 3), axis=0)
    sel = (np.linalg.norm(lab - table, axis=2) > dthr) & (valid > 0)
    mask = cv2.morphologyEx(sel.astype(np.uint8) * 255, cv2.MORPH_OPEN, KERN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERN_CLOSE)
    ncc, comp_lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, ncc):
        x, y, w, h, area = stats[i]
        if area < min_area or x <= 0 or y <= 0 or x + w >= W or y + h >= H:
            continue
        out.append((float(cents[i][0]), float(cents[i][1])))
    return out


def load_pairs():
    if os.path.exists(PAIRS_FILE):
        try:
            return json.load(open(PAIRS_FILE))
        except Exception:
            return []
    return []


def save_pairs(pairs):
    os.makedirs(os.path.dirname(PAIRS_FILE), exist_ok=True)
    json.dump(pairs, open(PAIRS_FILE, "w"))


def load_homography():
    if os.path.exists(HOMOGRAPHY_FILE):
        try:
            d = json.load(open(HOMOGRAPHY_FILE))
            return np.array(d["H"], dtype=np.float64)
        except Exception:
            return None
    return None


def fit_homography(pairs):
    """>= 6 pairs, need spread in both pixel- and base-space or RANSAC has nothing to reject."""
    if len(pairs) < 6:
        return None
    px = np.array([[p[0], p[1]] for p in pairs], np.float32)
    bx = np.array([[p[2], p[3]] for p in pairs], np.float32)
    if px[:, 0].std() < 15 and px[:, 1].std() < 15:
        return None
    if bx[:, 0].std() < 0.03 and bx[:, 1].std() < 0.03:
        return None
    Hmat, _ = cv2.findHomography(px, bx, cv2.RANSAC, 5.0)
    if Hmat is None:
        return None
    os.makedirs(os.path.dirname(HOMOGRAPHY_FILE), exist_ok=True)
    json.dump(dict(H=Hmat.tolist(), n_pairs=len(pairs), t=time.time()), open(HOMOGRAPHY_FILE, "w"))
    return Hmat


def poll_d435_cands(timeout=0.5):
    """Listen briefly on the D435 tracker's broadcast port for base-frame candidate blobs. Sharing
    port 9701 with real_policy_ctrl.py's own listener is inherently racy (only one UDP socket gets
    any given datagram) -- that's fine, we only need an occasional sample to grow the pair list."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    cands = None
    try:
        s.bind(("127.0.0.1", TRACK_PORT))
        t0 = time.time()
        while time.time() - t0 < timeout:
            r, _, _ = select.select([s], [], [], max(0.0, timeout - (time.time() - t0)))
            if not r:
                break
            try:
                m = json.loads(s.recv(4096))
            except (json.JSONDecodeError, OSError):
                continue
            c = m.get("cands")
            if c:
                cands = c
    except OSError:
        pass                                                      # port busy (real_policy_ctrl.py has it) -- fine
    finally:
        s.close()
    return cands


def pair_bootstrap(wrist_blobs, d435_cands):
    """Simple, defensive pairing with no prior mapping: 1<->1 direct; 2<->2 by rank along whichever
    axis has the larger spread in each space (assumed to be the same physical ordering)."""
    if not wrist_blobs or not d435_cands:
        return []
    bx = [(c["cx"], c["cy"]) for c in d435_cands]
    if len(wrist_blobs) == 1 and len(bx) == 1:
        return [(wrist_blobs[0][0], wrist_blobs[0][1], bx[0][0], bx[0][1])]
    if len(wrist_blobs) == 2 and len(bx) == 2:
        wu = [p[0] for p in wrist_blobs]; wv = [p[1] for p in wrist_blobs]
        axis = 0 if (max(wu) - min(wu)) >= (max(wv) - min(wv)) else 1
        w_sorted = sorted(wrist_blobs, key=lambda p: p[axis])
        b_axis = 0 if (max(x for x, _ in bx) - min(x for x, _ in bx)) >= (max(y for _, y in bx) - min(y for _, y in bx)) else 1
        b_sorted = sorted(bx, key=lambda p: p[b_axis])
        return [(w_sorted[i][0], w_sorted[i][1], b_sorted[i][0], b_sorted[i][1]) for i in range(2)]
    return []                                                      # >2 blobs: ambiguous, skip this round


# ---------------------------------------------------------------------------------------------- verdict logic
def classify_verdict(fixed_appeared, fixed_vanished, obj, goal, wrist_vanished_n=0):
    """Pure function of the fixed-cam 3-D component lists (see module docstring for the rule table).
    Returns (verdict:str, d_goal:float|None, picked:bool, placed:bool, near_goal:dict|None)."""
    def dist(c, p):
        return float(np.hypot(c["cx"] - p[0], c["cy"] - p[1]))

    vanished_near_obj = [c for c in fixed_vanished if dist(c, obj) < 0.06]
    appeared_near_obj = [c for c in fixed_appeared if dist(c, obj) < 0.08]
    near_goal = min(fixed_appeared, key=lambda c: dist(c, goal)) if fixed_appeared else None
    d_goal = dist(near_goal, goal) if near_goal is not None else None

    if vanished_near_obj and appeared_near_obj:
        return "pushed, not picked", d_goal, False, False, near_goal
    # picked evidence: the start position emptied (fixed cam), or the wrist cam saw something vanish while the
    # fixed cam saw the object appear at the goal (s3_01: the fixed cam missed the vanished can, the wrist did not)
    picked_ev = bool(vanished_near_obj) or (not appeared_near_obj and near_goal is not None and d_goal < 0.06
                                            and (wrist_vanished_n > 0 or not fixed_vanished))
    if picked_ev:
        placed = d_goal is not None and d_goal < 0.10
        if d_goal is not None and d_goal < 0.06:
            return "PICK AND PLACE OK", d_goal, True, True, near_goal
        if d_goal is not None and d_goal < 0.15:
            return "carried, placed %.0f cm off" % (100 * d_goal), d_goal, True, placed, near_goal
        return "picked, not seen near the goal (dropped?)", d_goal, True, placed, near_goal
    if fixed_appeared or fixed_vanished:
        return "seal FAILED (object did not move)", d_goal, False, False, near_goal
    return "no change detected", None, False, False, None


# ---------------------------------------------------------------------------------------------- debug images
def draw_panel(before, after, change_mask, comps, label):
    dbg = np.zeros((H, W * 3, 3), np.uint8)
    dbg[:, 0:W] = before
    dbg[:, W:2 * W] = after
    cm = cv2.cvtColor(change_mask, cv2.COLOR_GRAY2BGR)
    dbg[:, 2 * W:3 * W] = cm
    for c in comps:
        x, y, w, h = c["bbox"]
        colour = (0, 255, 0) if c["kind"] == "appeared" else (0, 0, 255)
        for off in (0, W, 2 * W):
            cv2.rectangle(dbg, (x + off, y), (x + w + off, y + h), colour, 2)
            tag = c["kind"][:3].upper()
            extra = ""
            if "cx" in c:
                extra = f" ({c['cx']:.3f},{c['cy']:.3f})"
            cv2.putText(dbg, tag + extra, (x + off, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    cv2.putText(dbg, f"{label}: before | after | change", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return dbg


# ---------------------------------------------------------------------------------------------- the monitor
class Monitor:
    def __init__(self, use_wrist=True, debug_dir=DEBUG_DIR_DEFAULT, dthr=18.0, min_area=300):
        self.debug_dir = debug_dir
        self.dthr = dthr
        self.min_area = min_area
        os.makedirs(self.debug_dir, exist_ok=True)
        ext = json.load(open(os.path.join(ROOT, "outputs", "extrinsics_d405_fixed.json")))
        self.cams = {}
        try:
            self.cams["fixed"] = Cam("fixed", FIXED_SERIAL, ext["T_base_cam405fixed"])
        except Exception as e:
            print(f"[monitor] fixed D405 failed to open: {e}", flush=True)
        if use_wrist:
            try:
                self.cams["wrist"] = Cam("wrist", WRIST_SERIAL, None)
            except Exception as e:
                print(f"[monitor] wrist D405 failed to open: {e}", flush=True)
        if not self.cams:
            raise RuntimeError("no cameras opened")
        print(f"[monitor] cams up: {list(self.cams)}", flush=True)
        self.before = {}
        self.pairs = load_pairs()
        self.homography = load_homography()

    # -- commands -------------------------------------------------------------------------------
    def snap(self):
        self.before = {}
        for name, cam in self.cams.items():
            col, dep = cam.grab_median(5)
            self.before[name] = dict(col=col, dep=dep)
        return dict(ok=True, cams=list(self.cams), t=time.time())

    def objects(self, top=0.10):
        """Current object list from the FIXED cam alone (no before/after): Lab distance from the table median
        inside the work-area ROI, depth-localised like an 'appeared' component. Used as a calibrated
        cross-check of the D435 tracker's centres."""
        if "fixed" not in self.cams:
            return dict(ok=False, error="no fixed cam")
        cam = self.cams["fixed"]
        col, dep = cam.grab_median(5)
        valid, _ = cam.roi_mask(top=max(top, ZTOP_ROI))
        # depth-first (lighting independent): every valid depth pixel in the ROI -> base frame; points standing
        # 1-15 cm above the table plane (table offset = median height over the ROI) -> 2 cm xy grid clusters
        zv = (dep > 0.15) & (dep < 1.5) & (valid > 0)
        vs, us = np.nonzero(zv); z = dep[vs, us]
        P = np.stack([(us - cam.cx0) / cam.fx * z, (vs - cam.cy0) / cam.fy * z, z, np.ones_like(z)], 1) @ cam.Tbc.T
        h_table = float(np.median(P[:, 2])) if len(P) else 0.0
        P[:, 2] -= h_table
        P = P[(P[:, 2] > 0.010) & (P[:, 2] < ZMAX_ARM) & (P[:, 0] > XR[0]) & (P[:, 0] < XR[1]) & (P[:, 1] > YR[0]) & (P[:, 1] < YR[1])]
        keys = np.floor(P[:, :2] / 0.02).astype(np.int64)
        cells = {}
        for idx, k in enumerate(map(tuple, keys)):
            cells.setdefault(k, []).append(idx)
        occ = {k for k, v in cells.items() if len(v) >= 4}
        seen, out = set(), []
        for k in occ:
            if k in seen:
                continue
            st, comp = [k], []
            while st:
                c = st.pop()
                if c in seen or c not in occ:
                    continue
                seen.add(c); comp.append(c)
                st += [(c[0] + i, c[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            C = P[np.concatenate([cells[c] for c in comp])]
            if len(C) < 100:
                continue
            tp = float(np.percentile(C[:, 2], 90)); T = C[C[:, 2] > tp - 0.015]
            out.append(dict(cx=float(T[:, 0].mean()), cy=float(T[:, 1].mean()), top=tp, n=int(len(C)), src="depth"))
        out.sort(key=lambda c: -c["n"])
        return dict(ok=True, objects=out, h_table=h_table)

    def verdict(self, goal, obj, top):
        if not self.before:
            return dict(ok=False, error="no snap() taken yet")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = dict(ok=True, cams=list(self.cams))
        fixed_appeared, fixed_vanished = [], []
        wrist_appeared, wrist_vanished = [], []
        after = {}
        for name, cam in self.cams.items():
            col, dep = cam.grab_median(5)
            after[name] = dict(col=col, dep=dep)
            if name == "fixed":
                valid, _ = cam.roi_mask(top=max(top, ZTOP_ROI))
            else:
                valid = border_mask(6)
            comps, change_mask, _ = segment_change(self.before[name]["col"], col, valid, self.dthr, self.min_area)
            used, located = [], []                                # kept in lockstep: used[i] <-> located[i]
            for c in comps:
                if name == "fixed":
                    loc = localize_fixed(cam, c, self.before[name]["dep"], dep, top)
                    if loc is None:
                        continue                                  # arm height / no valid depth -> not drawn either
                    (fixed_appeared if loc["kind"] == "appeared" else fixed_vanished).append(loc)
                else:
                    loc = localize_wrist(c, self.homography)
                    (wrist_appeared if loc["kind"] == "appeared" else wrist_vanished).append(loc)
                used.append(c); located.append(loc)
            dbg_comps = [dict(bbox=c["bbox"], kind=c["kind"], **({"cx": l["cx"], "cy": l["cy"]} if "cx" in l else {}))
                         for c, l in zip(used, located)]
            panel = draw_panel(self.before[name]["col"], col, change_mask, dbg_comps, name)
            path = os.path.join(self.debug_dir, f"{ts}_{name}.png")
            cv2.imwrite(path, panel)
            result.setdefault("debug_images", {})[name] = path

        verdict, d_goal, picked, placed, near_goal = classify_verdict(fixed_appeared, fixed_vanished, obj, goal,
                                                                       wrist_vanished_n=len(wrist_vanished))
        # raw frames for offline replay of the verdict logic
        np.savez_compressed(os.path.join(self.debug_dir, f"{ts}_frames.npz"),
                            **{f"{n}_{k}_{w}": v[k] for n in after for k in ("col", "dep") for w, v in (("before", self.before[n]), ("after", after[n]))},
                            goal=np.array(goal), obj=np.array(obj), top=float(top))

        # -- wrist agreement (pixel-only if no homography yet, else distance-checked) --------------
        if "wrist" in self.cams:
            if self.homography is not None:
                wrist_picked = any(np.hypot(w["cx"] - obj[0], w["cy"] - obj[1]) < 0.08 for w in wrist_vanished if "cx" in w)
                wrist_placed = any(np.hypot(w["cx"] - goal[0], w["cy"] - goal[1]) < 0.12 for w in wrist_appeared if "cx" in w)
            else:
                wrist_picked = len(wrist_vanished) > 0
                wrist_placed = len(wrist_appeared) > 0
            agree = (wrist_picked == picked) and (wrist_placed == placed)
        else:
            agree = None

        # -- wrist homography: opportunistic pairing against the D435 tracker, accumulate + refit --
        if "wrist" in self.cams:
            try:
                wblobs = wrist_object_blobs(after["wrist"]["col"], self.dthr, self.min_area)
                cands = poll_d435_cands(0.5)
                new_pairs = pair_bootstrap(wblobs, cands) if cands else []
                if new_pairs:
                    self.pairs.extend(new_pairs)
                    save_pairs(self.pairs)
                    fitted = fit_homography(self.pairs)
                    if fitted is not None:
                        self.homography = fitted
                        print(f"[monitor] wrist homography fitted from {len(self.pairs)} pairs", flush=True)
            except Exception as e:
                print(f"[monitor] homography update skipped: {e}", flush=True)

        result.update(
            verdict=verdict, d_goal=d_goal, moved=bool(picked), picked=picked, placed=placed, agree=agree,
            fixed=dict(appeared=[strip(c) for c in fixed_appeared], vanished=[strip(c) for c in fixed_vanished]),
            wrist=dict(appeared=[strip(c) for c in wrist_appeared], vanished=[strip(c) for c in wrist_vanished],
                       homography=self.homography is not None, n_pairs=len(self.pairs)),
        )
        self.before = after
        return result

    def close(self):
        for cam in self.cams.values():
            cam.close()


def strip(d):
    return {k: v for k, v in d.items() if k != "mask"}


# ---------------------------------------------------------------------------------------------- UDP server
def serve(mon):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", CMD_PORT))
    print(f"[monitor] serving udp://127.0.0.1:{CMD_PORT}", flush=True)
    while True:
        data, addr = sock.recvfrom(65536)
        try:
            cmd = json.loads(data)
            if cmd.get("cmd") == "snap":
                reply = mon.snap()
            elif cmd.get("cmd") == "objects":
                reply = mon.objects(float(cmd.get("top", 0.10)))
            elif cmd.get("cmd") == "verdict":
                reply = mon.verdict(tuple(cmd["goal"]), tuple(cmd["obj"]), float(cmd.get("top", 0.05)))
                print(f"[monitor] verdict -> {reply['verdict']} (d_goal {reply['d_goal']})", flush=True)
            else:
                reply = dict(ok=False, error=f"unknown cmd {cmd.get('cmd')!r}")
        except Exception as e:
            reply = dict(ok=False, error=str(e))
        sock.sendto(json.dumps(reply).encode(), addr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no_wrist", action="store_true")
    ap.add_argument("--debug_dir", default=DEBUG_DIR_DEFAULT)
    ap.add_argument("--dthr", type=float, default=18.0)
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--goal", type=float, nargs=2, default=None)
    ap.add_argument("--obj", type=float, nargs=2, default=None)
    ap.add_argument("--top", type=float, default=0.05)
    ap.add_argument("--calib_pairs", type=float, nargs="+", default=None,
                     help="add pairs by hand: groups of 'u v x y' (pixel, base xy), e.g. --calib_pairs 120 340 0.35 -0.10 400 200 0.42 0.05")
    a = ap.parse_args()

    if a.calib_pairs is not None:
        if len(a.calib_pairs) % 4 != 0:
            raise SystemExit("--calib_pairs needs groups of 4 numbers: u v x y")
        pairs = load_pairs()
        for i in range(0, len(a.calib_pairs), 4):
            pairs.append(list(a.calib_pairs[i:i + 4]))
        save_pairs(pairs)
        fitted = fit_homography(pairs)
        print(f"[monitor] {len(pairs)} pairs saved" + (", homography fitted" if fitted is not None else ", not enough spread yet"))
        return

    mon = Monitor(use_wrist=not a.no_wrist, debug_dir=a.debug_dir, dthr=a.dthr, min_area=a.min_area)
    if a.once:
        print(json.dumps(mon.snap()))
        input("[monitor] BEFORE captured -- press Enter once the AFTER scene is ready... ")
        if a.goal is None or a.obj is None:
            raise SystemExit("--once needs --goal and --obj")
        print(json.dumps(mon.verdict(tuple(a.goal), tuple(a.obj), a.top), default=str))
        mon.close()
        return
    try:
        serve(mon)
    finally:
        mon.close()


if __name__ == "__main__":
    main()

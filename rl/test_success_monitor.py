#!/usr/bin/env python3
"""test_success_monitor :: offline unit tests for the change-classification core of success_monitor.py.

No camera is opened. The BEFORE/AFTER colour crops come from a real debug panel the monitor already
saved during a live (unchanged-scene) test against the fixed D405
(~/pnp_rl/monitor/20260920_160837_fixed.png, left third = before, middle third = after -- see
draw_panel() in success_monitor.py for the panel layout). Synthetic "object" discs are then painted
onto copies of that real table image at pixels chosen by *forward-projecting* known base-frame points
through the same camera model success_monitor.py uses, so the expected base-frame answer is known
in closed form and the test also exercises the inverse (ray/plane) path.

The camera intrinsics/extrinsics below were read once from the live fixed D405
(serial 218622277013) on 2026-09-20, the same rig success_monitor.py calibrates against
(outputs/extrinsics_d405_fixed.json, composed with the colour<-depth extrinsics fix -- see
success_monitor.Cam.__init__). They are hardcoded here purely so this file needs no camera.
"""
import os
import types

import cv2
import numpy as np

import success_monitor as sm

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_PNG = os.path.expanduser("~/pnp_rl/monitor/20260920_160837_fixed.png")

FX, FY, CX0, CY0 = 391.09783935546875, 390.51995849609375, 325.2290954589844, 244.70899963378906
TBC_FIXED = np.array([
    [-0.8774651119223005, 0.3011723118849026, -0.37329639392586683, 0.6881314555599884],
    [0.47943113016916866, 0.5277410862223397, -0.7011669889061543, 0.43429743494260825],
    [-0.014168238169692074, -0.7942194901198304, -0.6074657853402758, 0.4285018245320597],
    [0.0, 0.0, 0.0, 1.0],
])


def fake_cam():
    """A duck-typed stand-in for success_monitor.Cam: has exactly the attributes localize_fixed()
    and Cam.roi_mask() read (fx, fy, cx0, cy0, Tbc, Tcb), no RealSense device behind it."""
    cam = types.SimpleNamespace()
    cam.fx, cam.fy, cam.cx0, cam.cy0 = FX, FY, CX0, CY0
    cam.Tbc = TBC_FIXED
    cam.Tcb = np.linalg.inv(TBC_FIXED)
    return cam


def project(cam, x, y, z):
    """Base-frame (x, y, z) -> fixed-cam pixel (u, v), the forward direction of localize_fixed's ray."""
    pc = cam.Tcb @ np.array([x, y, z, 1.0])
    return cam.fx * pc[0] / pc[2] + cam.cx0, cam.fy * pc[1] / pc[2] + cam.cy0


def load_before_after():
    assert os.path.exists(PANEL_PNG), (
        f"{PANEL_PNG} missing -- run success_monitor.py --once (or snap+verdict on the live rig) once "
        "to produce a debug panel before running this test"
    )
    panel = cv2.imread(PANEL_PNG)
    w = sm.W
    before = panel[:, 0:w].copy()
    after = panel[:, w:2 * w].copy()
    assert before.shape == (sm.H, sm.W, 3) and after.shape == (sm.H, sm.W, 3)
    return before, after


def roi_for(cam, top):
    return sm.Cam.roi_mask(cam, top=top)[0]


# --------------------------------------------------------------------------------------------- tests
def test_no_change_on_real_unchanged_pair():
    """Regression check: the actual BEFORE/AFTER pair captured seconds apart on a static table (two
    cans, nothing touched) must yield zero components -- this is what deliverable 3 step 2 verified
    live; here it's re-checked offline against the saved frames."""
    cam = fake_cam()
    before, after = load_before_after()
    roi = roi_for(cam, top=sm.ZTOP_ROI)
    comps, _, _ = sm.segment_change(before, after, roi, dthr=18.0, min_area=300)
    assert comps == [], f"static scene produced spurious components: {[(c['kind'], c['bbox']) for c in comps]}"


def test_appeared_near_goal():
    """Paint a dark disc onto a copy of the AFTER frame at the pixel where the goal (0.40, 0.00) at
    half the object height projects; segment_change must find exactly one APPEARED component there,
    and localize_fixed (ray/plane fallback -- no real depth is available for a synthetic frame) must
    place it within a couple of cm of the goal."""
    cam = fake_cam()
    before, after = load_before_after()
    top = 0.05
    gx, gy = 0.40, 0.00
    u, v = project(cam, gx, gy, top / 2)
    assert 0 <= u < sm.W and 0 <= v < sm.H, "goal pixel should land inside the frame"

    synth_after = after.copy()
    cv2.circle(synth_after, (int(u), int(v)), 16, (25, 25, 25), -1)   # dark object, well off the table's white

    roi = roi_for(cam, top=max(top, sm.ZTOP_ROI))
    comps, _, _ = sm.segment_change(before, synth_after, roi, dthr=18.0, min_area=300)
    appeared = [c for c in comps if c["kind"] == "appeared"]
    assert len(appeared) >= 1, f"expected an APPEARED component near the goal, got {comps}"

    zeros = np.zeros((sm.H, sm.W), np.float32)                        # no real depth for a synthetic frame
    locs = [sm.localize_fixed(cam, c, zeros, zeros, top) for c in appeared]
    locs = [l for l in locs if l is not None]
    assert locs, "APPEARED component was rejected by localize_fixed (unexpected height?)"
    best = min(locs, key=lambda l: np.hypot(l["cx"] - gx, l["cy"] - gy))
    d = float(np.hypot(best["cx"] - gx, best["cy"] - gy))
    assert d < 0.03, f"localized appeared point {d * 100:.1f} cm from the goal it was drawn at"
    assert best["src"] == "plane"


def test_full_pick_and_place_ok():
    """End-to-end: synthesize a VANISHED component at `obj` (object removed) and an APPEARED
    component at `goal` (object placed there) in one AFTER frame, run them through segment_change +
    localize_fixed + classify_verdict, and expect the "PICK AND PLACE OK" string."""
    cam = fake_cam()
    before, after = load_before_after()
    top = 0.05
    ox, oy = 0.45, 0.15
    gx, gy = 0.40, 0.00
    uo, vo = project(cam, ox, oy, top / 2)
    ug, vg = project(cam, gx, gy, top / 2)

    roi = roi_for(cam, top=max(top, sm.ZTOP_ROI))
    table_lab = np.median(cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)[roi > 0].reshape(-1, 3), axis=0)
    table_bgr = tuple(int(x) for x in cv2.cvtColor(np.uint8([[table_lab]]), cv2.COLOR_LAB2BGR)[0, 0])

    synth_before = before.copy()
    cv2.circle(synth_before, (int(uo), int(vo)), 16, (25, 25, 25), -1)   # object sitting at obj, before the run

    synth_after = after.copy()
    cv2.circle(synth_after, (int(uo), int(vo)), 18, table_bgr, -1)        # object gone from obj (a touch bigger: clean cover)
    cv2.circle(synth_after, (int(ug), int(vg)), 16, (25, 25, 25), -1)     # object now sitting at goal

    comps, _, _ = sm.segment_change(synth_before, synth_after, roi, dthr=18.0, min_area=300)
    zeros = np.zeros((sm.H, sm.W), np.float32)
    appeared, vanished = [], []
    for c in comps:
        loc = sm.localize_fixed(cam, c, zeros, zeros, top)
        if loc is None:
            continue
        (appeared if loc["kind"] == "appeared" else vanished).append(loc)
    assert appeared and vanished, f"expected both an appeared and a vanished component, got {comps}"

    verdict, d_goal, picked, placed, _ = sm.classify_verdict(appeared, vanished, (ox, oy), (gx, gy))
    assert picked, "vanished-near-obj component should have set picked=True"
    assert placed, "appeared-near-goal component should have set placed=True"
    assert verdict == "PICK AND PLACE OK", f"got {verdict!r} (d_goal={d_goal})"
    assert d_goal is not None and d_goal < 0.06


def test_classify_verdict_rules():
    """Pure state-machine coverage of every result string in classify_verdict, with synthetic
    base-frame components (no images / cameras involved at all)."""
    obj, goal = (0.45, 0.15), (0.40, 0.00)

    def c(kind, xy):
        return dict(cx=xy[0], cy=xy[1], top=0.05, n=500, src="depth", kind=kind)

    # PICK AND PLACE OK: vanished at obj, appeared within 6 cm of goal
    v, d, picked, placed, _ = sm.classify_verdict([c("appeared", (0.402, 0.01))], [c("vanished", obj)], obj, goal)
    assert v == "PICK AND PLACE OK" and picked and placed

    # carried, placed N cm off: vanished at obj, appeared 10 cm from goal (and far from obj, so it
    # isn't also read as "appeared near obj" / pushed)
    v, d, picked, placed, _ = sm.classify_verdict([c("appeared", (0.30, 0.00))], [c("vanished", obj)], obj, goal)
    assert v.startswith("carried, placed") and picked and d is not None and 0.06 <= d < 0.15

    # picked, not seen near the goal (dropped?): vanished at obj, appeared far from goal
    v, d, picked, placed, _ = sm.classify_verdict([c("appeared", (0.20, 0.30))], [c("vanished", obj)], obj, goal)
    assert v == "picked, not seen near the goal (dropped?)" and picked and not placed

    # pushed, not picked: vanished AND appeared both near obj (object just slid)
    v, d, picked, placed, _ = sm.classify_verdict([c("appeared", (0.47, 0.16))], [c("vanished", obj)], obj, goal)
    assert v == "pushed, not picked" and not picked

    # seal FAILED (object did not move): some change elsewhere, nothing near obj
    v, d, picked, placed, _ = sm.classify_verdict([c("appeared", (0.20, -0.25))], [], obj, goal)
    assert v == "seal FAILED (object did not move)" and not picked

    # no change detected: nothing at all
    v, d, picked, placed, _ = sm.classify_verdict([], [], obj, goal)
    assert v == "no change detected" and d is None and not picked and not placed


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

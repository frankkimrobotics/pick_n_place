"""Regression tests for pick verification, using synthetic depth maps.

Every case here is a failure mode that actually happened on the robot. They are worth
pinning offline because each one cost a physical run to find, and none of them need the
camera: the module reads a point-cloud file, so a hand-built cloud exercises the exact
code path the executor runs.

Run: python policy/test_pick_verify.py
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pick_verify as pv

T_EYE = np.eye(4)          # identity extrinsic -> camera frame == base frame
TABLE_Z = 0.0
SITE = (0.40, 0.00)


def write_cloud(shm, pts):
    """Publish a synthetic cloud where the module expects the shm map."""
    p = np.asarray(pts, np.float32).reshape(-1, 3)
    np.save(os.path.join(shm, "pnp_zed_xyz.npy"), p)


def table_patch(cx, cy, r=0.25, n=4000, z=0.0, jitter=0.002, seed=0):
    """A disc of table-height returns, with the ZED's few-mm depth noise."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n); rr = r * np.sqrt(rng.uniform(0, 1, n))
    return np.c_[cx + rr * np.cos(th), cy + rr * np.sin(th),
                 z + rng.normal(0, jitter, n)]


def block(cx, cy, top, r=0.03, n=90, seed=1):
    """An object: returns on its top face."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n); rr = r * np.sqrt(rng.uniform(0, 1, n))
    return np.c_[cx + rr * np.cos(th), cy + rr * np.sin(th),
                 top + rng.normal(0, 0.002, n)]


def run():
    shm = tempfile.mkdtemp(prefix="pv_test_")
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
        if not ok:
            fails.append(name)

    print("site_state:")
    # bare table -> cleared
    write_cloud(shm, table_patch(*SITE))
    check("bare table", pv.site_state(SITE, TABLE_Z, T_EYE, shm=shm)[0], pv.CLEARED)

    # object sitting on the site -> occupied
    write_cloud(shm, np.vstack([table_patch(*SITE), block(SITE[0], SITE[1], 0.05)]))
    check("object present", pv.site_state(SITE, TABLE_Z, T_EYE, shm=shm)[0], pv.OCCUPIED)

    # THE false pass: the arm occludes the site, leaving too few returns. The old boolean
    # returned "cleared" here and carried air to the box.
    write_cloud(shm, table_patch(0.70, 0.30, r=0.05))      # all returns far from SITE
    check("occluded site", pv.site_state(SITE, TABLE_Z, T_EYE, shm=shm)[0], pv.UNKNOWN)

    # a stale map must not be trusted either
    write_cloud(shm, table_patch(*SITE))
    f = os.path.join(shm, "pnp_zed_xyz.npy")
    os.utime(f, (time.time() - 60, time.time() - 60))
    check("stale map", pv.site_state(SITE, TABLE_Z, T_EYE, shm=shm)[0], pv.UNKNOWN)

    # an object only just taller than the threshold still counts
    write_cloud(shm, np.vstack([table_patch(*SITE), block(SITE[0], SITE[1], 0.025)]))
    check("low object (25 mm)", pv.site_state(SITE, TABLE_Z, T_EYE, shm=shm)[0], pv.OCCUPIED)

    print("\nlifted vs dragged:")
    # before: the target on its site, nothing else nearby
    write_cloud(shm, np.vstack([table_patch(*SITE), block(SITE[0], SITE[1], 0.05)]))
    before = pv.occupancy_cells(SITE, TABLE_Z, T_EYE, shm=shm)
    obj = pv.occupancy_cells(SITE, TABLE_Z, T_EYE, shm=shm, R=0.04)

    # genuine lift: neighbourhood empty afterwards
    write_cloud(shm, table_patch(*SITE))
    after = pv.occupancy_cells(SITE, TABLE_Z, T_EYE, shm=shm)
    check("lifted -> not dragged", pv.drag_verdict(before, after, obj)[0], False)

    # drag: site is clear but the object re-appears 8 cm away
    write_cloud(shm, np.vstack([table_patch(*SITE),
                                block(SITE[0] + 0.08, SITE[1], 0.05, seed=5)]))
    after = pv.occupancy_cells(SITE, TABLE_Z, T_EYE, shm=shm)
    check("dragged 8 cm -> dragged", pv.drag_verdict(before, after, obj)[0], True)
    check("  and the site itself reads cleared",
          pv.site_state(SITE, TABLE_Z, T_EYE, shm=shm)[0], pv.CLEARED)

    # clutter that was already there must not read as a drag
    write_cloud(shm, np.vstack([table_patch(*SITE), block(SITE[0], SITE[1], 0.05),
                                block(SITE[0] + 0.09, SITE[1], 0.05, seed=7)]))
    before_c = pv.occupancy_cells(SITE, TABLE_Z, T_EYE, shm=shm)
    obj_c = pv.occupancy_cells(SITE, TABLE_Z, T_EYE, shm=shm, R=0.04)
    write_cloud(shm, np.vstack([table_patch(*SITE),
                                block(SITE[0] + 0.09, SITE[1], 0.05, seed=7)]))
    after_c = pv.occupancy_cells(SITE, TABLE_Z, T_EYE, shm=shm)
    check("pre-existing clutter -> not dragged",
          pv.drag_verdict(before_c, after_c, obj_c)[0], False)

    print("\ncoverage gate (occlusion must not read as 'nothing appeared'):")
    write_cloud(shm, table_patch(*SITE))
    cov_ref = pv.table_coverage(SITE, TABLE_Z, T_EYE, shm=shm)
    write_cloud(shm, table_patch(*SITE)[:400])                # 90 % of returns gone
    cov_now = pv.table_coverage(SITE, TABLE_Z, T_EYE, shm=shm)
    frac = cov_now / max(cov_ref, 1)
    print(f"  coverage {cov_ref} -> {cov_now} ({100 * frac:.0f}%)")
    check("heavy occlusion trips the gate", frac < 0.6, True)
    write_cloud(shm, table_patch(*SITE, seed=3))
    frac_ok = pv.table_coverage(SITE, TABLE_Z, T_EYE, shm=shm) / max(cov_ref, 1)
    check("clear view passes the gate", frac_ok >= 0.6, True)

    print("\ninspect geometry:")
    ix, iy = pv.inspect_xy((0.33, -0.04), (0.42, 0.26))
    d = float(np.hypot(ix - 0.33, iy + 0.04))
    print(f"  inspect pose is {100 * d:.1f} cm off the pick site")
    check("inspect pose clears the 3 cm site radius", d > 0.08, True)

    print("\nplace arrival:")
    # an object released at the box, landing 12 cm away after a bounce
    place = (0.42, 0.26)
    write_cloud(shm, table_patch(*place, r=0.2))
    ref = pv.occupancy_cells(place, TABLE_Z, T_EYE, shm=shm, R=0.15)
    write_cloud(shm, np.vstack([table_patch(*place, r=0.2),
                                block(place[0] - 0.12, place[1], 0.05, seed=9)]))
    now = pv.occupancy_cells(place, TABLE_Z, T_EYE, shm=shm, R=0.15)
    check("bounced object is detected at r=0.15", len(now - ref) >= 3, True)
    now6 = pv.occupancy_cells(place, TABLE_Z, T_EYE, shm=shm, R=0.06)
    ref6 = ref  # same reference cells, tighter query radius
    check("  and MISSED at the old r=0.06", len(now6 - ref6) >= 3, False)

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())

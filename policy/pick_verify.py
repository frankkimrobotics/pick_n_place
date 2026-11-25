"""Pick verification from the fixed ZED depth map.

Why this is a separate module: the verification was wrong twice in ways that only show
up on the real robot, so it needs to be testable on its own (see /tmp/verify_test.py)
rather than only reachable from inside the rollout loop.

The hard-won constraint behind the design:

  * There is NO vacuum/pressure sensor on this cell -- every motion.digital-in pin is
    unlinked, so the pump gives no seal feedback.
  * Payload torque cannot substitute for one. Measured on the arm (/tmp/calib_load.json):
    returning to an IDENTICAL joint configuration (0.01 deg match) the load metric
    J2 + 0.5*J3 drifts by 0.0477 pin, while a 0.3 kg object at the cup would produce only
    0.0212 pin. The signal sits 2-14x BELOW the drive's stiction noise.

So vision is the only channel, and the only thing that makes it trustworthy is a clear
line of sight. Hence three states -- never a boolean, because "I cannot see the site" is
a real and common answer that must not be silently rounded to either verdict.
"""
import io
import os
import time

import numpy as np

CLEARED = "cleared"
OCCUPIED = "occupied"
UNKNOWN = "unknown"


def site_state(xy, table_z, T_zed, shm="/dev/shm", r=0.03, arm_dz=0.12,
               max_age=2.0, min_pts=8, obj_dz=0.015):
    """Occupancy of base-frame `xy`: CLEARED, OCCUPIED or UNKNOWN (+ an explanation).

    `arm_dz` drops points at robot height so a hovering arm is not mistaken for the
    object; `min_pts` is what separates a real "empty table" reading from the arm's
    occlusion shadow, which also returns nothing.
    """
    try:
        f = os.path.join(shm, "pnp_zed_xyz.npy")
        age = time.time() - os.path.getmtime(f)
        if age > max_age:
            return UNKNOWN, f"map stale by {age:.1f} s"
        with open(f, "rb") as fh:
            P = np.load(io.BytesIO(fh.read())).reshape(-1, 3).astype(np.float32)
        P = P[np.isfinite(P).all(1)]
        Pb = (T_zed[:3, :3] @ P.T).T + T_zed[:3, 3]
        col = np.hypot(Pb[:, 0] - xy[0], Pb[:, 1] - xy[1]) < r
        near = Pb[col & (Pb[:, 2] < table_z + arm_dz)]
        if len(near) < min_pts:
            return UNKNOWN, f"only {len(near)} pts at the site (occluded or dropout)"
        z_hi = float(np.median(np.sort(near[:, 2])[-min_pts:]))
        mm = 1000.0 * (z_hi - table_z)
        if z_hi < table_z + obj_dz:
            return CLEARED, f"top {mm:+.0f} mm over table, {len(near)} pts"
        return OCCUPIED, f"surface {mm:+.0f} mm over table, {len(near)} pts"
    except Exception as e:                                    # noqa: BLE001
        return UNKNOWN, f"read failed ({e})"


def poll_site_state(xy, table_z, T_zed, tries=6, dt=0.35, **kw):
    """First definite verdict across several reads -- the map publishes at 3 Hz, so a
    single read can land mid-publish or on a momentary dropout."""
    st, info = UNKNOWN, "no reads"
    for _ in range(max(1, tries)):
        time.sleep(dt)
        st, info = site_state(xy, table_z, T_zed, **kw)
        if st != UNKNOWN:
            return st, info
    return st, info


def inspect_xy(pick_xy, place_xy, frac=0.35):
    """A pose laterally off the pick site, on the way to the box.

    Judging the site from the lift pose is impossible: the lift ends directly above the
    pick xy, where the arm either reads AS the object or hides it. This offset is motion
    the carry had to make anyway, so the clear view is free.
    """
    return (pick_xy[0] + frac * (place_xy[0] - pick_xy[0]),
            pick_xy[1] + frac * (place_xy[1] - pick_xy[1]))


# ---------------------------------------------------------------------------
# Lifted vs dragged.
#
# Site-local clearance answers "is the object off this spot", which a DRAG satisfies
# just as well as a lift: a cup that pushes the object 9 cm sideways empties a 3 cm
# radius and reads as a clean pick. Distinguishing the two needs the NEIGHBOURHOOD --
# a lifted object leaves it empty, a dragged one is still sitting in it.
#
# Clutter makes an absolute count useless (other objects are in the radius too), so
# this compares occupancy CELLS against a reference captured with a clear view before
# the grasp. What matters is not how much is there, but whether an object-sized blob
# appeared somewhere that was empty before.

def occupancy_cells(xy, table_z, T_zed, shm="/dev/shm", R=0.12, cell=0.012,
                    obj_dz=0.015, arm_dz=0.12):
    """Occupied cells at object height within R of xy, as a set of integer indices."""
    try:
        f = os.path.join(shm, "pnp_zed_xyz.npy")
        with open(f, "rb") as fh:
            P = np.load(io.BytesIO(fh.read())).reshape(-1, 3).astype(np.float32)
        P = P[np.isfinite(P).all(1)]
        Pb = (T_zed[:3, :3] @ P.T).T + T_zed[:3, 3]
        sel = ((np.hypot(Pb[:, 0] - xy[0], Pb[:, 1] - xy[1]) < R)
               & (Pb[:, 2] > table_z + obj_dz) & (Pb[:, 2] < table_z + arm_dz))
        pts = Pb[sel]
        ij = np.floor((pts[:, :2] - np.asarray(xy)) / cell).astype(int)
        # a cell counts as occupied only with a few returns, so depth speckle does not
        # register as a newly-appeared object
        uniq, cnt = np.unique(ij, axis=0, return_counts=True)
        return {tuple(u) for u, c in zip(uniq, cnt) if c >= 3}
    except Exception:                                         # noqa: BLE001
        return set()


def drag_verdict(cells_before, cells_after, obj_cells_before, min_new=3, frac=0.35):
    """Did an object-sized blob appear where the pre-grasp map was empty?

    Returns (dragged, n_new, need). `obj_cells_before` is the target's own footprint,
    which sets the scale: a displaced object re-appears with a comparable footprint,
    while a stray cell or two is noise.
    """
    new = cells_after - cells_before
    need = max(int(min_new), int(round(frac * max(len(obj_cells_before), 1))))
    return len(new) >= need, len(new), need


def table_coverage(xy, table_z, T_zed, shm="/dev/shm", R=0.12, obj_dz=0.015):
    """How much of the neighbourhood's TABLE surface is currently visible.

    The drag test only sees occupancy that APPEARED, so occlusion can never fake a drag
    -- but it can hide a real one, which would report a dragged object as a clean pick.
    The table layer is the honest coverage gauge: it is always physically there, so a
    drop in its point count is occlusion and nothing else. Below a fraction of the
    reference the drag test has not actually looked at the neighbourhood and must say
    "inconclusive" instead of "confirmed".
    """
    try:
        with open(os.path.join(shm, "pnp_zed_xyz.npy"), "rb") as fh:
            P = np.load(io.BytesIO(fh.read())).reshape(-1, 3).astype(np.float32)
        P = P[np.isfinite(P).all(1)]
        Pb = (T_zed[:3, :3] @ P.T).T + T_zed[:3, 3]
        sel = ((np.hypot(Pb[:, 0] - xy[0], Pb[:, 1] - xy[1]) < R)
               & (Pb[:, 2] < table_z + obj_dz))
        return int(sel.sum())
    except Exception:                                         # noqa: BLE001
        return 0

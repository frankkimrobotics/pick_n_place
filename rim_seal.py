"""Rim-based suction seal model + validation against real-world outcomes.
Seals iff the surface under the 7.5mm cup rim is flat within the cup's compliance."""
import numpy as np, mujoco

_VISUAL_BODIES = {}


def _visual_only_bodies(m):
    """Bodies whose every geom is non-colliding, i.e. pure VISUAL meshes. mj_ray is contype-blind, so
    without this the gripper's own suction-cup shell (a visual mesh ~3.5 mm above the lip) is hit by the
    centre ray and the seal test ends up comparing the object against the gripper."""
    key = id(m)
    if key not in _VISUAL_BODIES:
        collid, seen = set(), set()
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            seen.add(b)
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                collid.add(b)
        _VISUAL_BODIES[key] = seen - collid
    return _VISUAL_BODIES[key]


def _ray(m, d, p, a, geomgroup, exclude_bid, skip_bids, max_hits=4):
    """mj_ray that walks past hits on bodies we must ignore. Returns (distance, body id)."""
    total, origin = 0.0, np.asarray(p, float).copy()
    for _ in range(max_hits):
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(m, d, origin, a, geomgroup, 1, exclude_bid, gid)
        if dist < 0 or gid[0] < 0:
            return -1.0, -1
        b = int(m.geom_bodyid[gid[0]])
        if b not in skip_bids:
            return total + float(dist), b
        step = float(dist) + 1e-4
        origin = origin + a * step
        total += step
    return -1.0, -1

RIM_R   = 0.0075     # measured cup lip radius (m)
N_RIM   = 8          # rim sample points
MAX_DEV = 0.0020     # max surface deviation under the rim the cup can conform to (m)
BACK    = 0.03       # ray start distance back along the cup axis

def rim_seal(m, d, tip_pos, axis, target_bid=None, rim_r=RIM_R, n=N_RIM,
             max_dev=MAX_DEV, exclude_bid=-1, debug=False, geomgroup=None):
    """Cast n rays along `axis` at the rim circle. Returns (sealed, deviation, n_hit, bids)."""
    a = axis / np.linalg.norm(axis)
    u = np.cross(a, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6: u = np.cross(a, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u); v = np.cross(a, u)
    dists, bids = [], []
    # mj_ray ignores contype, so a robot VISUAL mesh (e.g. the suction-cup shell, which sits a few mm
    # above the tip) is hit by the centre ray and the seal test then compares the object against the
    # gripper. Callers that put the robot visuals in their own geom group mask it out here.
    if geomgroup is None:
        geomgroup = np.array([1,1,1,1,1,1], dtype=np.uint8)
    geomgroup = np.asarray(geomgroup, dtype=np.uint8)
    skip = _visual_only_bodies(m)
    for k in range(n):
        th = 2*np.pi*k/n
        p = tip_pos + rim_r*(np.cos(th)*u + np.sin(th)*v) - a*BACK
        dist, b = _ray(m, d, p, a, geomgroup, exclude_bid, skip)
        if dist < 0:
            if debug: print(f"   ray{k}: MISS")
            return False, float("inf"), k, bids          # rim overhangs an edge -> leak
        dists.append(dist - BACK)                        # depth relative to the tip plane
        bids.append(b)
    # centre ray: a rotationally symmetric dome has ZERO rim spread but is still curved,
    # so compare the rim against the centre (sagitta) as well as against itself (tilt/edge).
    dc, bc = _ray(m, d, tip_pos - a*BACK, a, geomgroup, exclude_bid, skip)
    if dc < 0:
        return False, float("inf"), n, bids
    centre = dc - BACK
    bids.append(bc)
    spread  = float(max(dists) - min(dists))          # tilt / edge overhang
    sagitta = float(abs(np.mean(dists) - centre))     # dome curvature
    dev = max(spread, sagitta)
    same = (len(set(bids)) == 1) and (target_bid is None or bids[0] == target_bid)
    if debug:
        print(f"   rim(mm) {[round(x*1000,2) for x in dists]} centre {centre*1000:.2f} "
              f"spread {spread*1000:.2f} sagitta {sagitta*1000:.2f} same {same}")
    return bool(dev <= max_dev and same), dev, n, bids

"""realsim.mast :: the ZED camera's mast as a keep-out pillar.

2026-11-15: on the real cell the wrist camera mount hit the mast that carries the fixed ZED and broke. The mast was in no
model (scenes, safety filter, guard). It is a vertical pillar under the camera: from the ground up to the camera, radius 1 cm.
It is invisible in every render (alpha 0, geom group 5): it stands under/behind the ZED, outside its view, and the policy was
trained without it. It is a world geom, so realsim.safety.SafetyFilter and the twin's clearance monitor both see it.
"""
import mujoco

MAST_R = 0.010
MAST_Z0 = -0.12          # scene ground plane


def add_zed_mast(spec, cam_pos=None, radius=MAST_R, z0=MAST_Z0):
    """cam_pos None -> the scene's camera named 'zed' (twin); pass the calibrated base-frame position on the robot."""
    if cam_pos is None:
        cam = next((c for c in spec.worldbody.cameras if c.name == "zed"), None)
        if cam is None:
            return None
        cam_pos = list(cam.pos)
    x, y, z = (float(v) for v in cam_pos)
    g = spec.worldbody.add_geom(name="zed_mast", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, 0, 0],
                                fromto=[x, y, z0, x, y, z], rgba=[0.2, 0.2, 0.2, 0.0], group=5)
    return g

"""realsim.plunger :: sprung suction-cup plunger for already-generated scenes (MuJoCo 4 MjSpec).

The real cup sits on a spring plunger that compresses ~15 mm. There is NO sensor on it (no
switch, no vacuum gauge, no force cell), so nothing here is observable by the policy or the
executor -- the plunger only changes the PHYSICS:

  * presses up to the stroke are absorbed by the spring instead of stalling the drive
    (the demos press PRESS_M = 15 mm past the grasp pose -- exactly the stroke);
  * past the stroke it bottoms out and the arm is rigid against the object again;
  * the seal needs the lip seated (compression >= seal_x) with vacuum on;
  * a sealed object hangs from the plunger tip, not from the rigid flange.

`load_with_plunger(xml)` rewrites the rigid `cup_tip` geom on the `tcp` body into
body `plunger_tip` + slide joint `cup_plunger` (axis 0 0 -1 in the tcp frame = compression
positive; same convention as sim_robot_mjcf.build(plunger=True)) and re-targets the
`suction_*` welds to the tip. The tcp SITE stays on the flange = the cup tip at full extension,
so policy targets / FK are unchanged. The plunger dof lands at qpos[6], right after the arm.
"""
import mujoco

PLUNGER = dict(
    travel=0.015,      # m, user: "compresses around 15 mm"
    k=300.0,           # N/m       ASSUMED (level-compensator class spring)
    preload=2.0,       # N at full extension, ASSUMED -> 2.0 N .. 6.5 N over the stroke
    damping=8.0,       # N s/m     (sim_robot_mjcf plunger)
    mass=0.012,        # kg moving tip
    seal_x=0.001,      # m compression for the lip to be seated, ASSUMED
    contact_tc=0.004,  # s solref time constant of the cup contact AND the end stop. The scenes' default
                       # 0.02 is far too soft for a 12 g tip: a 25 mm over-press put the RIGID cup 20 mm
                       # inside the object, and the plunger tip 7.6 mm inside at 8 N (0.004 -> 0.7 mm)
)


def load_with_plunger(xml_path, mast=True, **over):
    p = dict(PLUNGER, **over)
    spec = mujoco.MjSpec.from_file(xml_path)
    if mast:
        from realsim.mast import add_zed_mast
        add_zed_mast(spec)
    tcp = spec.body("tcp")
    old = next(g for g in tcp.geoms if g.name == "cup_tip")
    size, rgba = list(old.size), list(old.rgba)
    spec.delete(old)
    tip = tcp.add_body(name="plunger_tip", pos=[0, 0, 0])
    tip.mass = p["mass"]
    tip.inertia = [2e-7, 2e-7, 2e-7]
    tip.explicitinertial = True
    tip.add_joint(name="cup_plunger", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, -1],
                  range=[0.0, p["travel"]], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                  stiffness=p["k"], springref=-p["preload"] / p["k"], damping=p["damping"],
                  solref_limit=[p["contact_tc"], 1.0])
    tip.add_geom(name="cup_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=size, rgba=rgba,
                 solref=[p["contact_tc"], 1.0], priority=5)     # priority: the cup's solref wins the pair
    for e in spec.equalities:
        if e.name.startswith("suction_") and e.name1 == "tcp":
            e.name1 = "plunger_tip"
    m = spec.compile()
    return m, p


def compression(m, d):
    """plunger compression (m), >= 0 -- SIM-ONLY, no real sensor exists."""
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cup_plunger")
    return float(d.qpos[m.jnt_qposadr[j]])

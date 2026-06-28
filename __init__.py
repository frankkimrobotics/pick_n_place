"""pick_and_place :: self-contained pick-and-place SIMULATION for the MyCobot Pro 630.

Eye-in-hand RGBD (synthetic) -> SAM-style mask -> 1 cm circular suction grasp point
-> collision-free / segmented motion -> suction transport -> place in box -> evaluation.

Runs in the `curobo2` conda env. Reuses (without editing) the existing repo's
cuRobo V2 Planner, URDF, hand-eye calibration and joint conventions.
"""

# Pick-and-Place Simulation Pipeline — Plan & Self-Review

MyCobot Pro 630 (6-DOF) + eye-in-hand RealSense D405 (RGBD) + single suction cup.
**The real robot is OFF** — everything here runs as a self-contained kinematic/geometric
**simulation** driven by the robot URDF. No ROS, no hardware, no sockets.

> Hard constraint: **do not edit the existing codebase**. We only *import* and *reuse*
> existing assets/code. All new code lives under `pick_and_place/`.

---

## 1. What the pipeline does (end-to-end)

```
home q ──▶ move to OBSERVE pose ──▶ render eye-in-hand RGBD ──▶ segment object mask
        ──▶ deproject mask → object point cloud (base frame, + surface normals)
        ──▶ detect 1 cm-diameter circular suction patch  → grasp point p + normal n
        ──▶ compute object 3D OBB (collision volume)
        ──▶ PICK:  plan→pre-grasp ▸ approach(descend) ▸ ACTIVATE SUCTION ▸ lift
        ──▶ PLACE (one of two modes, both implemented):
              A) segmented:  approach-above-box ▸ move ▸ lower ▸ release
              B) planned:    attach OBB to robot as collision volume,
                             add box as world obstacle, collision-free plan to box ▸ lower ▸ release
        ──▶ RELEASE SUCTION (object drops into box)
        ──▶ EVALUATE  → PASS/FAIL + metrics + report.json + rendered video
```

## 2. Environment (verified)

Runs **entirely in the `curobo2` conda env** (Python 3.10):
- `curobo` 0.8.0 — `MotionPlanner` V2 API, CUDA available. Provides FK, IK-via-pose,
  collision-free planning, `update_world(SceneCfg)`, `attachment_manager` (attach/detach
  carried object), `Cuboid` (= an OBB: pose + dims).
- `trimesh` 4.12 + **`ray_pyembree`** (fast ray casting) → synthetic RGBD rendering.
- `numpy`, `scipy` (rotations), `matplotlib`, `imageio` (video), `yourdfpy`, `PIL`.

> The other `curobo` env (0.7.7) only has the legacy `MotionGen` API and is NOT used.

## 3. Reused existing assets (imported, never edited)

| Asset | Path | Use |
|---|---|---|
| URDF (canonical) | `frankkimrobotics/ros2_mycobot/src/mycobot_description/urdf/mycobot_pro_630.urdf` | kinematics + visual/collision meshes |
| cuRobo robot cfg | `.../curobo/mycobot_pro_630.yml` (`ee_link: tcp`) | planner robot model |
| **`Planner` class** | `.../curobo/curobo_planner_server_v2.py` | imported directly; gives `plan_pose`, `plan_joint`, `fk`, and `.mp` (the `MotionPlanner` for `update_world`/`attachment_manager`) |
| Hand-eye extrinsic | `mycobot_mpc/captures/calib_20260622_215855/handeye_result.json` | `T_tcp_cam` (camera pose in tcp frame) |
| D405 intrinsics | `.../calib_20260622_215855/d405_intrinsics.json` | fx,fy,cx,cy, 848×480 |
| Joint conventions | `mycobot_mpc/joint_conventions.py` | home q, joint order, limits |
| Deproject math | `ros2node/perception/object_pointclouds.py` (`deproject_mask`) | re-implemented identically (that file imports `pyrealsense2`, so we mirror the 5-line pinhole formula and cite it) |

Key frames / conventions (all verified):
- `tcp` link = **real suction-cup tip** (0.145 m below flange; tcp **+Z = approach axis, points
  down**). cuRobo `ee_link = tcp`, so a planned tcp pose *is* the suction-tip pose.
- Quaternions are **wxyz** everywhere in cuRobo. Poses = `[x,y,z, qw,qx,qy,qz]`, metres.
- `T_base_cam = FK_tcp(q) @ T_tcp_cam`. Camera optical frame: +X right, +Y down, +Z fwd.
- Joints: `[joint1..joint6]`, URDF radians. Home (URDF) = `[-π/2,0,0,0,0,0]`.

## 4. Module layout (`pick_and_place/`)

| File | Responsibility |
|---|---|
| `config.py` | All constants: paths, `T_tcp_cam`, intrinsics, image size, depth params, home q, suction params (cup Ø=10 mm → r=5 mm; seal pos-tol 5 mm, angle-tol 15°), standoff/lift distances, table z, default object & box poses. |
| `geometry.py` | Pose/quaternion helpers (wxyz↔matrix via scipy, look-at, pose compose), all numpy. |
| `sim_planner.py` | `SimPlanner`: thin wrapper that constructs the reused `Planner` (custom world yaml w/ table+box), exposes `fk`, `plan_pose`, `plan_joint`, plus `set_world`, `attach_obb`, `detach` via `Planner.mp`. Graceful fallback if attach API mismatches. |
| `scene.py` | Builds the simulated world (trimesh): table slab, **pick object** (a low cylinder "puck" with a flat top — a clean suction target — at a known reachable pose), **place box** (open-top bin at the box pose). Ground-truth object pose & true OBB. |
| `perception.py` | `render_rgbd(cam_pose, K, WxH, meshes)` via embree ray cast → `(rgb, depth_m, mask, normals)`. `segment()` → ground-truth object mask (default) or optional SAM3 hook. `deproject(mask, depth, K)` → cam points; `to_base()` transform. |
| `grasp_detection.py` | Detect the **1 cm circular suction patch**: per surface point, fit a plane over a 5 mm-radius neighborhood; score by planarity (low residual), patch completeness (cup fully supported), and approach feasibility (normal near-vertical preferred). Return best grasp point `p`, normal `n`, full tcp grasp pose, pre-grasp (standoff along +n), lift pose. |
| `obb.py` | 3D OBB of the object cloud via `trimesh.bounds.oriented_bounds` → `(center pose wxyz, dims)`; used as cuRobo `Cuboid` collision volume + attachment. |
| `simulator.py` | Kinematic world-state sim: steps a joint trajectory via FK, tracks attached-object pose (`T_base_obj = FK_tcp(q) @ T_tcp_obj`), suction model (seal check), trimesh-based carried-OBB collision sweep vs world, captures render frames. |
| `pipeline.py` | Orchestrates the whole sequence; selectable `--place-mode {segmented,planned}`. |
| `evaluate.py` | Success criteria + metrics → `report.json` + console summary. |
| `run_demo.py` | CLI entry: `python -m pick_and_place.run_demo --place-mode planned --box-pose X Y Z [qw qx qy qz]`. |
| `outputs/` | RGBD/mask PNGs, grasp viz, trajectory plot, frames + `demo.mp4/gif`, `report.json`. |

## 5. Grasp-point detection (the "1 cm circle")

The suction cup seals on a flat circular footprint of Ø ≈ 10 mm. Detection on the
segmented object cloud (base frame, with per-point normals from the render):
1. Candidate = each object surface point with an upward-ish normal (dot with +Z_base > cos 60°).
2. For each candidate, take neighbors within r = 5 mm (the cup radius):
   - **Planarity**: RMS distance to the local best-fit plane < `flat_tol` (≈1 mm).
   - **Support/completeness**: neighborhood must cover the full disk (point count ≥
     density·π r², and no large angular gap) so the cup isn't half off an edge.
   - **Normal consistency**: neighbor normals agree (cone < ~10°).
3. Score = w₁·(−planarity_rms) + w₂·(verticality) + w₃·(−dist to object centroid).
   Pick best → grasp point `p`, sealing normal `n`.
4. Grasp tcp pose: position `p`; orient so **tcp +Z = −n** (cup pushes into surface);
   yaw chosen to keep wrist within limits. Pre-grasp = `p + standoff·n`; lift = pre-grasp + up.

## 6. Suction model & object transport (sim)

- **Activate**: succeeds iff tcp tip is within `seal_pos_tol` (5 mm) of the object surface
  **and** approach axis within `seal_ang_tol` (15°) of the local surface normal. On success,
  freeze `T_tcp_obj = inv(FK_tcp(q)) @ T_base_obj`; object now rides the tcp.
- **Transport**: while attached, `T_base_obj = FK_tcp(q) @ T_tcp_obj` every step.
- **Planned mode**: also `attach_obb(...)` so cuRobo plans the carried OBB collision-free;
  `set_world(box)` adds the bin as an obstacle.
- **Release**: detach; object settles onto the box bottom (snap to rest in sim).

## 7. Evaluation (PASS requires all)

1. **Detection** — a valid 1 cm suction patch found (planarity/support thresholds met).
2. **Reachability** — every planned segment returns success; IK/plan found, within joint limits.
3. **Seal** — grasp seal check passes (pos & angle tolerances).
4. **Collision-free** — planner reports self/world collision-free; carried-OBB sweep clears world.
5. **Place** — final object pose: XY inside box footprint, resting inside the bin, upright
   tilt < tol.
Outputs aggregate `PASS/FAIL`, per-criterion booleans, numeric margins, timings → `report.json`,
plus the rendered demo video and key figures.

## 8. Self-review (risks & mitigations)

- **cuRobo V2 attach/world signatures** are not fully documented → `sim_planner` discovers them
  defensively and **falls back** to (Mode A segmented + our own trimesh OBB collision sweep) if the
  attach API differs, so the demo always completes. *(Mode B is best-effort true-collision; Mode A
  is always available — both satisfy the "back to box" requirement.)*
- **Planner warmup** (~30–60 s, GPU) — one-time at startup; acceptable.
- **No real camera/SAM3** → synthetic RGBD with **ground-truth** mask (deterministic). The grasp
  detector and deproject run exactly as they would on real masks; SAM3 client kept as optional hook.
- **Normals**: taken from ray-hit triangles (exact) → robust planarity/seal checks.
- **Quaternion order**: cuRobo wxyz vs scipy xyzw — centralized in `geometry.py` to avoid mix-ups.
- **Reach**: object & box poses chosen inside the Pro 630 workspace (~0.25–0.40 m radial, on table).
- **Box pose "given later"**: parametrized with a sane default and a CLI flag — drop-in when provided.

## 9. Run

```bash
conda activate curobo2
cd /home/lisc-frank/Desktop/2026
python -m pick_and_place.run_demo --place-mode planned        # or: segmented
#   optional: --box-pose 0.0 0.32 -0.06   (xyz, metres, base frame)
# Artifacts → pick_and_place/outputs/
```

"""viz :: 3D scene rendering for the demo video + still figures (matplotlib).

A third-person animation of the pick-and-place: the arm skeleton (link origins via
yourdfpy FK on the SAME URDF cuRobo uses), the table, the object at its current
pose, the place-box, the carried OBB, and the tcp approach axis. Also helpers to
save the eye-in-hand RGBD and the grasp-detection still.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import config as C
from geometry import inv_T


class ArmFK:
    """Lightweight URDF FK for drawing the arm skeleton (link origins)."""
    LINKS = ["base_link", "base", "link1", "link2", "link3", "link4",
             "link5", "link6", "suction_cup", "tcp"]

    def __init__(self):
        import yourdfpy
        self.urdf = yourdfpy.URDF.load(C.URDF_PATH, build_scene_graph=True,
                                       load_meshes=False)

    def link_points(self, q):
        cfg = {n: float(v) for n, v in zip(C.JOINT_NAMES, q)}
        self.urdf.update_cfg(cfg)
        pts = []
        for lk in self.LINKS:
            try:
                T = self.urdf.get_transform(lk, "base_link")
                pts.append(T[:3, 3])
            except Exception:
                pass
        return np.array(pts)

    def link_transforms(self, q, links):
        """Return {link_name: 4x4 world transform} for the given links at config q."""
        cfg = {n: float(v) for n, v in zip(C.JOINT_NAMES, q)}
        self.urdf.update_cfg(cfg)
        return {lk: np.asarray(self.urdf.get_transform(lk, "base_link")) for lk in links}


def _box_faces(center_T, dims):
    """8 corners + 6 quad faces of a box given its 4x4 pose and extents."""
    hx, hy, hz = np.asarray(dims) / 2.0
    c = np.array([[sx*hx, sy*hy, sz*hz] for sx in (-1, 1)
                  for sy in (-1, 1) for sz in (-1, 1)])
    c = (center_T[:3, :3] @ c.T).T + center_T[:3, 3]
    idx = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    return [c[i] for i in idx]


class SceneRenderer:
    def __init__(self, scene, box_info, elev=22, azim=-60):
        self.scene = scene
        self.box_info = box_info
        self.arm = ArmFK()
        self.elev, self.azim = elev, azim

    def __call__(self, q, sim):
        fig = plt.figure(figsize=(6.4, 5.2), dpi=90)
        ax = fig.add_subplot(111, projection="3d")
        self._draw(ax, q, sim)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)
        return img

    def _draw(self, ax, q, sim):
        # table top
        tw, td, _ = C.TABLE_DIMS
        tx, ty = 0.30, 0.10
        z = C.TABLE_Z
        corners = np.array([[tx-tw/2, ty-td/2, z], [tx+tw/2, ty-td/2, z],
                            [tx+tw/2, ty+td/2, z], [tx-tw/2, ty+td/2, z]])
        ax.add_collection3d(Poly3DCollection([corners], color=(0.6, 0.6, 0.62),
                                             alpha=0.25))
        # arm skeleton
        pts = self.arm.link_points(q)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="0.2", lw=3, ms=3)
        # suction tip (eef) + approach axis
        from geometry import quat_wxyz_to_R
        pos, quat = sim.P.fk_tip(q)
        ap = quat_wxyz_to_R(quat)[:, 2]
        ax.plot([pos[0]], [pos[1]], [pos[2]], "o",
                color="red" if sim.attached else "orange", ms=7)
        ax.add_collection3d(Line3DCollection(
            [[pos, pos + 0.05 * ap]], colors="orange", lw=2))
        # object at current pose
        objm = sim.object_mesh_at(sim.object_T)
        ax.add_collection3d(Poly3DCollection(
            objm.vertices[objm.faces], color=(0.8, 0.35, 0.24), alpha=0.95))
        # carried OBB
        if sim.attached and sim.T_tcp_obb is not None:
            T = sim.P.fk_T(q) @ sim.T_tcp_obb
            ax.add_collection3d(Poly3DCollection(
                _box_faces(T, sim.obb_dims), facecolor=(0, 0.7, 0.2),
                alpha=0.12, edgecolor=(0, 0.5, 0.1)))
        # box
        bm = self.scene["box"]
        ax.add_collection3d(Poly3DCollection(bm.vertices[bm.faces],
                            color=(0.27, 0.43, 0.78), alpha=0.5))
        self._format(ax)

    def _format(self, ax):
        ax.set_xlim(0.0, 0.5); ax.set_ylim(-0.1, 0.45); ax.set_zlim(-0.12, 0.45)
        ax.set_box_aspect((0.5, 0.55, 0.57))
        ax.view_init(self.elev, self.azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("MyCobot Pro 630 — pick & place (sim)")


# ---- still figures -------------------------------------------------------- #
def save_eye_in_hand(frame, mask, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), dpi=100)
    axes[0].imshow(frame["rgb"]); axes[0].set_title("eye-in-hand RGB")
    d = frame["depth"].copy(); d[d == 0] = np.nan
    im = axes[1].imshow(d, cmap="viridis"); axes[1].set_title("depth (m)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title("object mask")
    for a in axes:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def save_grasp_still(dp, grasp, obb, path):
    fig = plt.figure(figsize=(6.5, 5.5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    pb = dp["pts_base"]
    ax.scatter(pb[:, 0], pb[:, 1], pb[:, 2], s=2, c=pb[:, 2], cmap="viridis")
    p, n = grasp["point"], grasp["normal"]
    ax.scatter([p[0]], [p[1]], [p[2]], c="red", s=80, marker="*",
               label="1 cm suction point")
    ax.add_collection3d(Line3DCollection([[p, p + 0.04 * n]], colors="red", lw=2))
    ax.add_collection3d(Poly3DCollection(_box_faces(obb["T_base_obb"], obb["dims"]),
                        facecolor=(0, 0.7, 0.2), alpha=0.1,
                        edgecolor=(0, 0.5, 0.1)))
    ax.set_title("Detected suction grasp + object OBB")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def save_video(frames, path, fps=None):
    import imageio
    fps = C.VIDEO_FPS if fps is None else fps
    if not frames:
        return None
    # pad frames to a common size
    h = max(f.shape[0] for f in frames); w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        if f.shape[0] != h or f.shape[1] != w:
            g = np.full((h, w, 3), 255, np.uint8)
            g[:f.shape[0], :f.shape[1]] = f
            f = g
        padded.append(f)
    try:
        imageio.mimsave(path, padded, fps=fps)
        return path
    except Exception:
        gif = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(gif, padded, duration=1.0 / fps)
        return gif

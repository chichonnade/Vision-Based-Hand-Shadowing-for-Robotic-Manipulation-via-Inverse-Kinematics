"""Renderer for WiLor hand pose estimation results."""

from typing import Any, Optional
import numpy as np
import torch
import trimesh
import pyrender


def _create_raymond_lights() -> list[pyrender.Node]:
    """Return raymond light nodes for the scene."""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def _get_light_poses(n_lights: int=5, elevation: float=np.pi / 3,
                     dist: float=12) -> list[np.ndarray]:
    """Get lights in a circle around origin at elevation."""
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = _make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = _make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def _make_translation(t: torch.Tensor) -> torch.Tensor:
    """Create a 4x4 translation matrix."""
    return _make_4x4_pose(torch.eye(3), t)


def _make_rotation(rx: float=0, ry: float=0, rz: float=0, order: str="xyz") -> torch.Tensor:
    """Create a 4x4 rotation matrix from euler angles."""
    Rx = _rotx(rx)
    Ry = _roty(ry)
    Rz = _rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    else:
        R = Rz @ Ry @ Rx  # Default to xyz
    return _make_4x4_pose(R, torch.zeros(3))


def _make_4x4_pose(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Create a 4x4 pose matrix from rotation and translation.

    Args:
        R: Rotation matrix (*, 3, 3)
        t: Translation vector (*, 3)

    Returns:
        Pose matrix (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def _rotx(theta: float) -> torch.Tensor:
    """Rotation matrix around x-axis."""
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def _roty(theta: float) -> torch.Tensor:
    """Rotation matrix around y-axis."""
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def _rotz(theta: float) -> torch.Tensor:
    """Rotation matrix around z-axis."""
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


class WilorRenderer:
    """Renderer for MANO hand mesh predictions.

    This class provides rendering capabilities for WiLor hand pose estimation results.
    It renders 3D hand meshes onto 2D images with proper lighting and camera projection.
    """

    def __init__(self, faces: np.ndarray):
        """Initialize the renderer.

        Args:
            faces: MANO mesh faces array of shape (F, 3)
        """

        # Add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        self._faces = faces
        self._faces_left = self._faces[:, [0, 2, 1]]

    def _vertices_to_trimesh(
        self,
        vertices: np.ndarray,
        camera_translation: np.ndarray,
        mesh_base_color: tuple=(1.0, 1.0, 0.9),
        rot_axis: Optional[list]=None,
        rot_angle: float=0,
        is_right: int=1) -> trimesh.Trimesh:
        """Convert vertices to a trimesh object.

        Args:
            vertices: Mesh vertices array of shape (N, 3)
            camera_translation: Camera translation vector
            mesh_base_color: RGB color tuple for the mesh
            rot_axis: Rotation axis
            rot_angle: Rotation angle in degrees
            is_right: 1 for right hand, 0 for left hand

        Returns:
            trimesh.Trimesh object
        """
        if rot_axis is None:
            rot_axis = [1, 0, 0]
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(
                vertices.copy() + camera_translation,
                self._faces.copy(),
                vertex_colors=vertex_colors
            )
        else:
            mesh = trimesh.Trimesh(
                vertices.copy() + camera_translation,
                self._faces_left.copy(),
                vertex_colors=vertex_colors
            )

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def _render_rgba(
            self,
            vertices: np.ndarray,
            cam_t: Optional[np.ndarray]=None,
            rot_axis: Optional[list]=None,
            rot_angle: float=0,
            camera_z: float=3,
            mesh_base_color: tuple=(1.0, 1.0, 0.9),
            scene_bg_color: tuple=(0, 0, 0),
            render_res: Optional[list]=None,
            focal_length: Optional[float]=None,
            is_right: int=1) -> np.ndarray:
        """Render mesh vertices to an RGBA image.

        Args:
            vertices: Mesh vertices array of shape (N, 3)
            cam_t: Camera translation vector
            rot_axis: Rotation axis
            rot_angle: Rotation angle in degrees
            camera_z: Camera z-position if cam_t is None
            mesh_base_color: RGB color tuple for the mesh
            scene_bg_color: RGB color tuple for the background
            render_res: [width, height] of the rendered image
            focal_length: Camera focal length
            is_right: 1 for right hand, 0 for left hand

        Returns:
            RGBA image as numpy array of shape (H, W, 4) with values in [0, 1]
        """
        if rot_axis is None:
            rot_axis = [1, 0, 0]
        if render_res is None:
            render_res = [256, 256]

        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0],
            viewport_height=render_res[1],
            point_size=1.0
        )

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])

        if is_right:
            mesh_base_color = mesh_base_color[::-1]

        mesh = self._vertices_to_trimesh(
            vertices,
            np.array([0, 0, 0]),
            mesh_base_color,
            rot_axis,
            rot_angle,
            is_right=is_right
        )
        mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12
        )

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self._add_point_lighting(scene, camera_node)
        self._add_lighting(scene, camera_node)

        light_nodes = _create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, _rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def _add_lighting(
            self,
            scene: pyrender.Scene,
            cam_node: pyrender.Node,
            color: np.ndarray = np.ones(3),
            intensity: float = 1.0):
        """Add directional lighting to the scene."""
        light_poses = _get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def _add_point_lighting(
            self,
            scene: pyrender.Scene,
            cam_node: pyrender.Node,
            color: np.ndarray=np.ones(3),
            intensity: float=1.0):
        """Add point lighting to the scene."""
        light_poses = _get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def render(
            self,
            results: list[dict[str, Any]],
            img_rgb: np.ndarray,
            mesh_color: tuple = (0.25098039, 0.274117647, 0.65882353)) -> np.ndarray:
        """
        Render hand mesh predictions onto an image.

        This method modifies the input image in-place by overlaying rendered 3D hand meshes.

        Args:
            results: list of prediction dictionaries from WiLor pipeline. Each dict should contain:
                - 'wilor_preds': dict with 'pred_vertices', 'pred_cam_t_full', 'scaled_focal_length'
                - 'is_right': 1 for right hand, 0 for left hand
            img_rgb: Input RGB image as numpy array (will be modified in-place)
            mesh_color: RGB color tuple for the mesh (default is light purple)

        Returns:
            The modified image (same object as input img)
        """
        # Convert to RGB and normalize.
        render_image = img_rgb.astype(np.float32) / 255.0

        # Render each hand.
        for out in results:
            verts = out["wilor_preds"]['pred_vertices'][0]
            is_right = out['is_right']
            cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
            scaled_focal_length = out["wilor_preds"]['scaled_focal_length']

            cam_view = self._render_rgba(
                verts,
                cam_t=cam_t,
                render_res=[img_rgb.shape[1], img_rgb.shape[0]],
                is_right=is_right,
                mesh_base_color=mesh_color,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )

            # Overlay mesh onto image using alpha compositing.
            render_image = (
                render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) +
                cam_view[:, :, :3] * cam_view[:, :, 3:])

        # Convert back to uint8.
        render_image = (255 * render_image).astype(np.uint8)

        # Copy result back to input image (modify in-place).
        img_rgb[:, :, :] = render_image

        return img_rgb

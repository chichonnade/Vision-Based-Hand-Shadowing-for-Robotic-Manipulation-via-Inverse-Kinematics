"""
3D Camera Space Conversion Pipeline Step
========================================

This module handles conversion of 2D hand landmarks to 3D camera space coordinates
using depth information and camera intrinsics.

The CameraSpaceJointsFromImageSpace transformation takes ImageSpaceJoints (2D pixel
coordinates) and DepthMap data, then uses camera intrinsics to deproject the 2D
points into 3D camera space coordinates.

Key Features:
- Depth-based 3D reconstruction from 2D landmarks
- Camera intrinsics-based deprojection (pinhole camera model)
- Robust handling of invalid depth values
- Coordinate bounds checking and validation
- Landmark visualization on depth frames
- Consistent with transformation pipeline architecture

Usage:
    depth_converter = CameraSpaceJointsFromImageSpace(display_landmarks=True)
    camera_joints = depth_converter((image_joints, depth_map))  # Returns CameraSpaceJoints
"""

import logging
from typing import Optional
import os

import numpy as np
import cv2

from vbhs.pipeline import transformations
from vbhs.pipeline import types
from vbhs.config import config
from vbhs.pipeline.hands import deprojection

_logger = logging.getLogger(__name__)

# TODO (isaac): this class should not do the visualization. That should be done in a separate class.
class CameraSpaceHandsFromImageSpace(
        transformations.Transformation[
            tuple[types.HandLandmarksImageSpace, types.CameraFrame],
            types.HandLandmarksCameraSpace]):
    """
    3D camera space conversion transformation using depth information.

    This transformation takes 2D hand landmarks and CameraFrame data, then uses camera
    intrinsics to deproject the 2D points into 3D camera space coordinates.
    """

    def __init__(self,
                 min_depth_m: float = 0.1,
                 max_depth_m: float = 5.0,
                 depth_scale: float = 1000.0,
                 display_landmarks: bool = False,
                 fps: int = 30,
                 mp4_output_dir: str | None = None):
        """
        Initialize the 3D conversion pipeline.

        Args:
            min_depth_m: Minimum valid depth in meters (default: 0.1m)
            max_depth_m: Maximum valid depth in meters (default: 5.0m)
            depth_scale: Scale factor to convert depth values to meters (default: 1000.0 for mm→m)
            display_landmarks: Whether to display landmarks overlayed on depth frame
            fps: Frame rate at which to record depth MP4 (default: 30)
            mp4_output_dir: Directory to save depth MP4 recording (default: None)

        Note:
            - min_depth_m/max_depth_m filter out unrealistic depth values
            - depth_scale handles unit conversion (RealSense typically uses millimeters)
            - Camera intrinsics are extracted from the CameraFrame object
            - display_landmarks shows visual feedback of landmark detection and 3D positions
            - mp4_output_dir creates human_depth.mp4 file with timestamped name if provided
        """
        super().__init__()

        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.depth_scale = depth_scale
        self.display_landmarks = display_landmarks
        self.mp4_output_dir = mp4_output_dir
        self.fps = fps

        # Video recording setup
        self._depth_video_writer: Optional[cv2.VideoWriter] = None  # pylint: disable=no-member
        self._depth_output_filename: Optional[str] = None
        self._setup_depth_video_writer()

        # Window setup for landmark display
        self._window_name = "Hand Landmarks - RGB & Depth"
        self._window_initialized = False
        self._display_width = 1708  # Fixed display width for side-by-side view (854 * 2)
        self._display_height = 480  # Fixed display height for consistency

        if self.display_landmarks:
            self._setup_display_window()

        _logger.info('3D conversion initialized')
        _logger.debug('Config: depth_range=[%.1f, %.1f]m, scale=%s', min_depth_m, max_depth_m, depth_scale)
        _logger.debug('Display landmarks: %s', "enabled" if display_landmarks else "disabled")
        if mp4_output_dir:
            _logger.debug('Depth recording: enabled, output to %s', mp4_output_dir)

    def _setup_display_window(self):
        """Setup the display window for landmark visualization."""
        # pylint: disable=no-member
        try:
            # Create the named window
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

            # Set window to always stay on top
            cv2.setWindowProperty(self._window_name, cv2.WND_PROP_TOPMOST, 1)

            # Set window to a reasonable size
            cv2.resizeWindow(self._window_name, self._display_width, self._display_height)

            self._window_initialized = True
            _logger.debug('Display window "%s" initialized (%dx%d)',
                        self._window_name, self._display_width, self._display_height)

        except Exception as e:
            _logger.error('Error setting up display window: %s', e)
            self._window_initialized = False

    def _transform(
            self, joints_and_camera: tuple[types.HandLandmarksImageSpace, types.CameraFrame], /
            ) -> types.HandLandmarksCameraSpace:
        """
        Convert 2D hand landmarks to 3D camera space coordinates.

        Args:
            joints_and_camera: Tuple of (ImageSpaceJoints, CameraFrame) containing 2D landmarks and camera data

        Returns:
            CameraSpaceJoints: 3D hand landmark positions in camera coordinates
        """
        image_space_joints, camera_frame = joints_and_camera

        try:
            # Extract depth image, dimensions, and camera intrinsics
            depth_image = camera_frame.depth.values
            intrinsics = camera_frame.intrinsics

            # Convert left hand landmarks if detected
            left_hand_landmarks = None
            if image_space_joints.left_hand_landmarks:
                left_hand_landmarks = deprojection.deproject_hand_landmarks(
                    image_space_joints.left_hand_landmarks, depth_image, intrinsics,
                    self.depth_scale, self.min_depth_m, self.max_depth_m
                )

            # Convert right hand landmarks if detected
            right_hand_landmarks = None
            if image_space_joints.right_hand_landmarks:
                right_hand_landmarks = deprojection.deproject_hand_landmarks(
                    image_space_joints.right_hand_landmarks, depth_image, intrinsics,
                    self.depth_scale, self.min_depth_m, self.max_depth_m
                )

            # Process landmarks for display and/or recording
            # Always call this method if mp4_output_dir is set (for depth recording)
            # or if display_landmarks is enabled (for visualization)
            if self.display_landmarks or self.mp4_output_dir:
                self._display_landmarks_on_depth(
                    camera_frame.rgb,
                    depth_image,
                    image_space_joints,
                    left_hand_landmarks,
                    right_hand_landmarks
                )

            return types.HandLandmarksCameraSpace(
                left_hand_landmarks=left_hand_landmarks,
                right_hand_landmarks=right_hand_landmarks
            )

        except Exception as e:
            _logger.error('Error converting landmarks to camera space: %s', e)
            return types.HandLandmarksCameraSpace(
                left_hand_landmarks=None,
                right_hand_landmarks=None
            )

    def _display_landmarks_on_depth(
            self,
            rgb_image: np.ndarray,
            depth_image: np.ndarray,
            image_space_joints: types.HandLandmarksImageSpace,
            left_hand_3d: Optional[types.HandPose3D],
            right_hand_3d: Optional[types.HandPose3D]):
        """
        Display landmarks overlayed on both RGB and depth frames side by side.

        Args:
            rgb_image: RGB image array
            depth_image: Depth image array
            image_space_joints: 2D hand landmarks in image space
            left_hand_3d: 3D landmarks for left hand or None
            right_hand_3d: 3D landmarks for right hand or None
        """
        # pylint: disable=no-member
        rgb_display = rgb_image.copy()

        # Convert depth image to 8-bit for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET
        )

        # Ensure both images have the same height for side-by-side display
        if rgb_display.shape[0] != depth_colormap.shape[0]:
            # Resize to match the depth image height
            target_height = depth_colormap.shape[0]
            aspect_ratio = rgb_display.shape[1] / rgb_display.shape[0]
            target_width = int(target_height * aspect_ratio)
            rgb_display = cv2.resize(rgb_display, (target_width, target_height))

        # Process left hand - draw on both images
        if image_space_joints.left_hand_landmarks and left_hand_3d:
            self._draw_hand_landmarks(
                rgb_display,
                image_space_joints.left_hand_landmarks,
                left_hand_3d,
                (0, 255, 0),  # Green for left hand
                "LEFT"
            )
            self._draw_hand_landmarks(
                depth_colormap,
                image_space_joints.left_hand_landmarks,
                left_hand_3d,
                (0, 255, 0),  # Green for left hand
                "LEFT"
            )

        # Process right hand - draw on both images
        if image_space_joints.right_hand_landmarks and right_hand_3d:
            self._draw_hand_landmarks(
                rgb_display,
                image_space_joints.right_hand_landmarks,
                right_hand_3d,
                (0, 0, 255),  # Red for right hand
                "RIGHT"
            )
            self._draw_hand_landmarks(
                depth_colormap,
                image_space_joints.right_hand_landmarks,
                right_hand_3d,
                (0, 0, 255),  # Red for right hand
                "RIGHT"
            )

        # Concatenate RGB and depth images side by side
        combined_image = np.hstack([rgb_display, depth_colormap])

        # Initialize depth video writer if needed (using combined frame dimensions)
        if self.mp4_output_dir and self._depth_video_writer is None:
            self._initialize_depth_video_writer(combined_image)

        # Write combined frame to video if recording (regardless of display_landmarks value)
        if self._depth_video_writer is not None:
            self._depth_video_writer.write(combined_image)

        # Display the image only if display_landmarks is enabled
        if self.display_landmarks and self._window_initialized:
            # Resize the image to fit the display window to prevent cropping
            display_image = cv2.resize(combined_image, (self._display_width, self._display_height))
            cv2.imshow(self._window_name, display_image)
            cv2.waitKey(1)  # Non-blocking wait

    def _draw_hand_landmarks(
            self,
            image: np.ndarray,
            landmarks_2d: types.HandPose2D,
            landmarks_3d: types.HandPose3D,
            color: tuple[int, int, int],
            hand_label: str) -> None:
        """
        Draw hand landmarks on the image.

        Args:
            image: Image to draw on
            landmarks_2d: 2D landmark positions in image coordinates
            landmarks_3d: 3D landmark positions in camera space
            color: Color tuple (B, G, R) for drawing
            hand_label: Label for the hand ("LEFT" or "RIGHT")
        """
        for key, landmark_2d in landmarks_2d.items():
            x, y = int(landmark_2d[0]), int(landmark_2d[1])

            # Draw landmark point
            cv2.circle(image, (x, y), 3, color, -1)

            # Highlight thumb base and index base
            if key == 'THUMB_MCP':
                cv2.circle(image, (x, y), 8, color, 2)
                cv2.putText(image, f'{hand_label} THUMB', (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            elif key == 'INDEX_FINGER_MCP':
                cv2.circle(image, (x, y), 8, color, 2)
                cv2.putText(image, f'{hand_label} INDEX', (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _setup_depth_video_writer(self):
        """Setup depth video writer if mp4_output_dir is provided."""
        if self.mp4_output_dir is None:
            return

        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.mp4_output_dir, exist_ok=True)
            self._depth_output_filename = os.path.join(self.mp4_output_dir, "human_depth.mp4")

        except Exception as e:
            _logger.error("Failed to setup depth video writer: %s", e)
            self._depth_video_writer = None

    def _initialize_depth_video_writer(self, depth_colormap: np.ndarray):
        """Initialize depth video writer with frame dimensions from first frame."""
        # pylint: disable=no-member
        if self.mp4_output_dir is None or self._depth_video_writer is not None:
            return

        try:
            height, width = depth_colormap.shape[:2]

            # Setup video writer with MP4 codec (default 15 fps)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._depth_video_writer = cv2.VideoWriter(
                self._depth_output_filename,
                fourcc,
                self.fps,
                (width, height)
            )

            if not self._depth_video_writer.isOpened():
                _logger.warning("Failed to open depth video writer")
                self._depth_video_writer = None
                return

        except Exception as e:
            _logger.error("Failed to initialize depth video writer: %s", e)
            self._depth_video_writer = None

    def cleanup(self):
        """Cleanup video recording and display resources."""
        # pylint: disable=no-member
        if self._depth_video_writer is not None:
            try:
                self._depth_video_writer.release()
            except Exception as e:
                _logger.error('Error releasing depth video writer: %s', e)
            finally:
                self._depth_video_writer = None

        # Cleanup display window
        if self.display_landmarks and self._window_initialized:
            cv2.destroyWindow(self._window_name)
            self._window_initialized = False


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw angles to a 3×3 rotation matrix."""
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    R_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return R_z @ R_y @ R_x


class RobotSpaceHandsFromCameraSpace(
        transformations.Transformation[
            types.HandLandmarksCameraSpace, types.HandLandmarksRobotSpace]):
    """Convert 3-D camera-space joints to robot base coordinates."""

    def __init__(
        self,
        cam_to_robot_config: Optional[dict] = None,
        urdf_offset_config: Optional[dict] = None):
        super().__init__()
        self.cam_to_robot_config = cam_to_robot_config or config.CAM_TO_ROBOT_CONFIG
        self.urdf_offset_config = urdf_offset_config or config.URDF_OFFSET_CONFIG
        self._setup_transformation_matrices()

        _logger.info("Robot space transformation initialized")
        _logger.debug('Camera translation: %s', self.cam_to_robot_config["translation"])
        _logger.debug('Camera rotation: %s degrees', self.cam_to_robot_config["rotation_angle"])

    def _setup_transformation_matrices(self) -> None:
        angle_rad_cam = np.radians(self.cam_to_robot_config["rotation_angle"])
        sin_angle_cam = np.sin(angle_rad_cam)
        cos_angle_cam = np.cos(angle_rad_cam)

        self.R_cam = np.array(
            [
                [-1, 0, 0],
                [0, sin_angle_cam, -cos_angle_cam],
                [0, -cos_angle_cam, -sin_angle_cam],
            ]
        )

        self.T_cam = np.array(
            [
                self.cam_to_robot_config["translation"]["x"],
                self.cam_to_robot_config["translation"]["y"],
                self.cam_to_robot_config["translation"]["z"],
            ]
        )

        r, p, y = self.urdf_offset_config["rpy"].values()
        self.R_urdf = rpy_to_matrix(r, p, y)
        self.T_urdf = np.array(
            [
                self.urdf_offset_config["translation"]["x"],
                self.urdf_offset_config["translation"]["y"],
                self.urdf_offset_config["translation"]["z"],
            ]
        )

        self.R_final = self.R_urdf @ self.R_cam
        self.T_final = self.R_urdf @ self.T_cam + self.T_urdf

        _logger.debug("Final rotation matrix:\n%s", self.R_final)
        _logger.debug("Final translation vector: %s", self.T_final)

    def _transform(self, camera_joints: types.HandLandmarksCameraSpace, /) -> types.HandLandmarksRobotSpace:
        left = (
            self._transform_landmarks_to_robot(camera_joints.left_hand_landmarks)
            if camera_joints.left_hand_landmarks
            else None
        )
        right = (
            self._transform_landmarks_to_robot(camera_joints.right_hand_landmarks)
            if camera_joints.right_hand_landmarks
            else None
        )
        return types.HandLandmarksRobotSpace(left_hand_landmarks=left, right_hand_landmarks=right)

    def _transform_landmarks_to_robot(
            self, camera_landmarks: types.HandPose3D
            ) -> types.HandPose3D:
        """Transform a hand pose from camera space to robot space."""
        return {key: self._transform_point_to_robot(pt)
                if pt is not None else None
                for key, pt in camera_landmarks.items()}

    def _transform_point_to_robot(
            self, camera_point: tuple[float, float, float]
            ) -> tuple[float, float, float]:
        """Transform a point from camera space to robot space."""
        point = np.array(camera_point)
        return tuple(self.R_final @ point + self.T_final)


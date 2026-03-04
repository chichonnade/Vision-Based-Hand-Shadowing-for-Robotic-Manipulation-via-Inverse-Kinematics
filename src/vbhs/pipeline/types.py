"""Types for the pipeline."""
from typing import Optional
import dataclasses

import numpy as np
from numpy import typing as npt

HandLandmark2D = tuple[float, float]
HandLandmark3D = tuple[float, float, float]
HandPose2D = dict[str, HandLandmark2D]
HandPose3D = dict[str, Optional[HandLandmark3D]]


@dataclasses.dataclass
class CameraIntrinsics:
    """Camera intrinsics with support for RealSense parameters."""
    fov: float
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    ppx: float = 0.0
    ppy: float = 0.0


@dataclasses.dataclass
class DepthMap:
    """Depth map captured by the sensor."""
    values: npt.NDArray[np.float32]


@dataclasses.dataclass
class CameraFrame:
    """RGB‑D frame captured by the sensor."""
    rgb: npt.NDArray[np.uint8]  # RGB frame of shape H×W×3
    depth: DepthMap  # depth of shape H×W in meters
    intrinsics: CameraIntrinsics


@dataclasses.dataclass
class RobotArmCommands:
    """Robot joint angle commands for both arms joint 1-5."""
    left_arm_angles: Optional[list[float]]  # Joint angles in radians
    right_arm_angles: Optional[list[float]]  # Joint angles in radians


@dataclasses.dataclass
class RobotGripperCommands:
    """Robot gripper commands for both hands joint 6"""
    left_gripper_angle: Optional[float]  # Gripper angle in radians
    right_gripper_angle: Optional[float]  # Gripper angle in radians


@dataclasses.dataclass
class RobotControlCommands:
    """Complete robot control data. All joint 1-6 for each arm."""
    robot_commands: tuple[Optional[list[float]], Optional[list[float]]] # Robot joint commands for each arm joint 1 to 6


@dataclasses.dataclass
class HandLandmarksImageSpace:
    """2D hand landmark positions in image-space coordinates (u, v)."""
    left_hand_landmarks: Optional[HandPose2D]  # 21 landmarks (u, v) or None
    right_hand_landmarks: Optional[HandPose2D]  # 21 landmarks (u, v) or None


@dataclasses.dataclass
class HandLandmarksCameraSpace:
    """3D hand landmark positions in camera-space coordinates (x, y, z)."""
    left_hand_landmarks: Optional[HandPose3D]  # 21 landmarks (x, y, z) or None
    right_hand_landmarks: Optional[HandPose3D]  # 21 landmarks (x, y, z) or None


@dataclasses.dataclass
class HandLandmarksRobotSpace:
    """Hand landmarks in robot space coordinates (not actual robot joint angles)."""
    left_hand_landmarks: Optional[HandPose3D]  # 21 landmarks in robot frame
    right_hand_landmarks: Optional[HandPose3D]  # 21 landmarks in robot frame

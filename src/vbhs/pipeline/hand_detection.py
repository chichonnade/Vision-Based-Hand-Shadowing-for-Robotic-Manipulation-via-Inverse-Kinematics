"""
Hand Landmark Detection Pipeline Step
====================================

This module handles hand landmark detection from RGB camera frames using WiLoR.

The ImageSpaceJointsFromCameraFrame transformation takes CameraFrame objects
and detects hand landmarks, returning them as ImageSpaceJoints with 2D coordinates
in image space (u, v pixels).

Key Features:
- WiLoR hand detection for both left and right hands
- Robust error handling and confidence thresholding
- Memory-efficient processing with configurable detection parameters
- Automatic hand landmark extraction and coordinate conversion
- Consistent with transformation pipeline architecture

Usage:
    hand_detector = ImageSpaceJointsFromCameraFrame(confidence_threshold=0.5)
    image_joints = hand_detector(camera_frame)  # Returns ImageSpaceJoints
"""
import logging
from typing import Optional

import cv2

from vbhs.pipeline import transformations
from vbhs.pipeline import types
from vbhs.pipeline.hands import wilor_hand_detector

_logger = logging.getLogger(__name__)

class HandLandmarksFromCameraFrame(
        transformations.Transformation[
            types.CameraFrame, types.HandLandmarksImageSpace]):
    """
    Hand landmark detection transformation using WiLoR.

    This transformation takes CameraFrame objects and outputs ImageSpaceJoints
    containing 2D hand landmark positions in image coordinates for both hands.
    """

    def __init__(self,
                 camera_intrinsics: types.CameraIntrinsics,
                 enable_smoothing: bool = False,
                 smoothing_alpha: float = 0.7,
                 enable_mesh_visualization: bool = False):
        """
        Initialize WiLoR hand detection.

        Args:
            camera_intrinsics: Camera intrinsics for hand detection.
            enable_smoothing: If True, applies EMA smoothing to landmarks.
            smoothing_alpha: EMA smoothing factor (0.0-1.0). Higher values are more responsive.
            enable_mesh_visualization: If True, displays rendered hand meshes in a cv2 window.
        """
        super().__init__()

        self._enable_smoothing = enable_smoothing
        self._smoothing_alpha = smoothing_alpha
        self._smoothed_left_hand_landmarks: Optional[types.HandPose2D] = None
        self._smoothed_right_hand_landmarks: Optional[types.HandPose2D] = None
        self._hand_detector = wilor_hand_detector.WilorHandDetector(
            camera_intrinsics,
            enable_visualization=enable_mesh_visualization
        )

    def _transform(self, camera_frame: types.CameraFrame, /) -> types.HandLandmarksImageSpace:
        """
        Detect hand landmarks from camera frame.

        Args:
            camera_frame: Input CameraFrame with RGB image and metadata

        Returns:
            ImageSpaceJoints: 2D hand landmark positions in image coordinates
        """
        # Convert BGR to RGB for MediaPipe (RealSense provides BGR format).
        rgb_image = cv2.cvtColor(camera_frame.rgb, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

        # Extract hand landmarks.
        left_hand_landmarks, right_hand_landmarks = self._hand_detector.detect(rgb_image)

        if self._enable_smoothing:
            # Apply EMA smoothing
            left_hand_landmarks = self._smooth_landmarks(
                left_hand_landmarks, self._smoothed_left_hand_landmarks)
            right_hand_landmarks = self._smooth_landmarks(
                right_hand_landmarks, self._smoothed_right_hand_landmarks)

            # Update the stored smoothed landmarks for the next frame
            self._smoothed_left_hand_landmarks = left_hand_landmarks
            self._smoothed_right_hand_landmarks = right_hand_landmarks

        return types.HandLandmarksImageSpace(
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks)

    def _smooth_landmarks(
            self,
            current_landmarks: Optional[types.HandPose2D],
            previous_smoothed_landmarks: Optional[types.HandPose2D]
            ) -> Optional[types.HandPose2D]:
        """Apply exponential moving average to hand landmarks."""
        if current_landmarks is None:
            # Hand not detected, so smoothing is not possible.
            return None

        if previous_smoothed_landmarks is None:
            # First detection, no previous data to smooth with.
            return current_landmarks

        smoothed_landmarks: types.HandPose2D = {}
        for key, (current_u, current_v) in current_landmarks.items():
            prev_u, prev_v = previous_smoothed_landmarks[key]
            smoothed_u = self._smoothing_alpha * current_u + (1 - self._smoothing_alpha) * prev_u
            smoothed_v = self._smoothing_alpha * current_v + (1 - self._smoothing_alpha) * prev_v
            smoothed_landmarks[key] = (smoothed_u, smoothed_v)

        return smoothed_landmarks

    def cleanup(self):
        """Cleanup MediaPipe resources."""
        self._hand_detector.cleanup()

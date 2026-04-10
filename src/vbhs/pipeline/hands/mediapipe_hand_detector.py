"""MediaPipe-based hand detector implementation.

Uses the MediaPipe Tasks API (HandLandmarker) introduced in mediapipe >=0.10.18.
"""
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp  # type: ignore
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from vbhs.config import config
from vbhs.pipeline import types
from vbhs.pipeline.hands import hand_detector

_MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")


class MediaPipeHandDetector(hand_detector.HandDetector):
    """MediaPipe-based hand detector using the Tasks API."""

    def __init__(self,
                camera_intrinsics: types.CameraIntrinsics,
                 detection_confidence: float=0.5,
                 tracking_confidence: float=0.5,
                 max_num_hands: int=2,
                 model_complexity: int=1):
        """Initialize MediaPipe hand detection.

        Args:
            camera_intrinsics: Camera intrinsics for hand detection.
            detection_confidence: Minimum confidence for hand detection (0.0-1.0)
            tracking_confidence: Minimum confidence for hand tracking (0.0-1.0)
            max_num_hands: Maximum number of hands to detect (1 or 2)
            model_complexity: Model complexity (ignored, kept for API compat)
        """
        super().__init__(camera_intrinsics)
        self._detection_confidence = detection_confidence
        self._frame_counter = 0

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def detect(
            self, rgb_image: cv2.typing.MatLike
            ) -> tuple[Optional[types.HandPose2D], Optional[types.HandPose2D]]:
        """Detect hand landmarks from an RGB image."""
        image_height, image_width = rgb_image.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self._frame_counter += 1
        results = self._landmarker.detect_for_video(
            mp_image, self._frame_counter)

        left_hand_landmarks: Optional[types.HandPose2D] = None
        right_hand_landmarks: Optional[types.HandPose2D] = None

        if not results.hand_landmarks or not results.handedness:
            return left_hand_landmarks, right_hand_landmarks

        for hand_lms, handedness_list in zip(results.hand_landmarks,
                                              results.handedness):
            hand_label = handedness_list[0].category_name  # "Left" or "Right"
            confidence = handedness_list[0].score

            if confidence < self._detection_confidence:
                continue

            if len(hand_lms) != len(config.MEDIAPIPE_HAND_LANDMARKS):
                raise ValueError(
                    f'Expected {len(config.MEDIAPIPE_HAND_LANDMARKS)} '
                    f'landmarks, got {len(hand_lms)}')

            landmarks_uv: types.HandPose2D = {}
            for key, index in config.MEDIAPIPE_HAND_LANDMARKS.items():
                landmark = hand_lms[index]
                u = landmark.x * image_width
                v = landmark.y * image_height
                landmarks_uv[key] = (u, v)

            # NOTE: MediaPipe labels are from the perspective of the person
            # in the image, swapped from FPV perspective.
            if hand_label == 'Right':
                left_hand_landmarks = landmarks_uv
            elif hand_label == 'Left':
                right_hand_landmarks = landmarks_uv
            else:
                raise ValueError(f'Hand label {hand_label} is not valid')

        return left_hand_landmarks, right_hand_landmarks

    def cleanup(self):
        """Cleanup MediaPipe resources."""
        self._landmarker.close()

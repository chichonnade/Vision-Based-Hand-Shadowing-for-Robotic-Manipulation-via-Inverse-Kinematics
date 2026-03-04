"""MediaPipe-based hand detector implementation."""
from typing import Optional

import cv2
import mediapipe as mp  # type: ignore

from vbhs.config import config
from vbhs.pipeline import types
from vbhs.pipeline.hands import hand_detector


class MediaPipeHandDetector(hand_detector.HandDetector):
    """MediaPipe-based hand detector."""

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
            model_complexity: Model complexity (0=lite, 1=full)
        """
        super().__init__(camera_intrinsics)
        self._detection_confidence = detection_confidence
        self._tracking_confidence = tracking_confidence
        self._max_num_hands = max_num_hands
        self._model_complexity = model_complexity

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self._max_num_hands,
            model_complexity=self._model_complexity,
            min_detection_confidence=self._detection_confidence,
            min_tracking_confidence=self._tracking_confidence
        )

    def detect(
            self, rgb_image: cv2.typing.MatLike
            ) -> tuple[Optional[types.HandPose2D], Optional[types.HandPose2D]]:
        """Detect hand landmarks from an RGB image."""
        # Extract image dimensions from the input image
        image_height, image_width = rgb_image.shape[:2]

        # Extract hand landmarks (adapted from _extract_hand_landmarks method)
        results = self._hands.process(rgb_image)
        left_hand_landmarks: Optional[types.HandPose2D] = None
        right_hand_landmarks: Optional[types.HandPose2D] = None

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return left_hand_landmarks, right_hand_landmarks

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            confidence = handedness.classification[0].score

            if confidence < self._detection_confidence:
                continue

            if not len(hand_landmarks.landmark) == len(config.MEDIAPIPE_HAND_LANDMARKS):
                raise ValueError(
                    f'Expected {len(config.MEDIAPIPE_HAND_LANDMARKS)} '
                    f'landmarks, got {len(hand_landmarks.landmark)}')

            landmarks_uv: types.HandPose2D = {}
            for key, index in config.MEDIAPIPE_HAND_LANDMARKS.items():
                landmark = hand_landmarks.landmark[index]
                u = landmark.x * image_width
                v = landmark.y * image_height
                landmarks_uv[key] = (u, v)

            # TODO (isaac): this note seems to say the opposite of what it should.
            # NOTE: MediaPipe labels are from the perspective of the person in the image,
            # swapped from FPV perspective
            if hand_label == 'Right':
                left_hand_landmarks = landmarks_uv
            elif hand_label == 'Left':
                right_hand_landmarks = landmarks_uv
            else:
                raise ValueError(f'Hand label {hand_label} is not valid')

        return left_hand_landmarks, right_hand_landmarks

    def cleanup(self):
        """Cleanup MediaPipe resources."""
        self._hands.close()

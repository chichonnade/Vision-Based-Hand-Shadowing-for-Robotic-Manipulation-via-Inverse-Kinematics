"""
Hand Landmarks Correction Pipeline Stage
========================================

This module provides a pipeline stage for applying heuristics to correct raw
3D hand landmark data. It is designed to handle issues like handedness
swapping and unrealistic jumps in landmark positions.

The CorrectHandLandmarks transformation takes HandLandmarksRobotSpace data,
applies correction algorithms, and outputs a cleaned-up version with the
same type.
"""
import logging
from typing import Optional, Tuple
import numpy as np

from vbhs.pipeline import transformations
from vbhs.pipeline import types
from vbhs.pipeline.hands import target_pose

_logger = logging.getLogger(__name__)


class CorrectHandLandmarks(
        transformations.Transformation[
            types.HandLandmarksRobotSpace, types.HandLandmarksRobotSpace]):
    """Applies heuristics to correct for issues in hand landmark detection."""

    def __init__(self, swap_threshold_meters: float, confidence_alpha: float ,
                 confidence_threshold: float, min_confidence: float,
                 force_hand: str | None = None):
        """
        Initializes the landmark correction stage.

        Args:
            swap_threshold_meters: The distance threshold (in meters) to confirm a
                                   handedness swap.
            confidence_alpha: The smoothing factor for the confidence average.
            confidence_threshold: The required difference in confidence to
                                  trigger a swap check.
            min_confidence: The minimum confidence required to consider a hand
                            detected.
            force_hand: If set to 'left' or 'right', forces a single detected
                        hand to be classified as that hand.
        """
        super().__init__()
        # NOTE: the previous landmarks are only updated if the hand is detected.
        self._previous_left_hand_landmarks: Optional[types.HandPose3D] = None
        self._previous_right_hand_landmarks: Optional[types.HandPose3D] = None
        self._left_confidence: float = 0.0
        self._right_confidence: float = 0.0
        self._alpha = confidence_alpha
        self._confidence_threshold = confidence_threshold
        self._swap_threshold_meters = swap_threshold_meters
        self._min_confidence = min_confidence
        self._force_hand = force_hand
        _logger.info('Hand landmark corrector initialized')

    def _transform(self, landmarks: types.HandLandmarksRobotSpace, /) -> types.HandLandmarksRobotSpace:
        """
        Applies correction heuristics to the provided hand landmarks.

        Args:
            landmarks: The raw hand landmarks in robot space.

        Returns:
            The corrected hand landmarks in robot space.
        """
        raw_left_hand_landmarks = landmarks.left_hand_landmarks
        raw_right_hand_landmarks = landmarks.right_hand_landmarks

        # Apply heuristics to correct for handedness swaps or outliers.
        corrected_left_landmarks, corrected_right_landmarks = self._apply_heuristics(
            raw_left_hand_landmarks, raw_right_hand_landmarks)

        # Update confidence scores based on the raw detector output. This allows
        # the confidence to heal even if the correction logic makes a mistake.
        self._left_confidence = (self._alpha * float(raw_left_hand_landmarks is not None) +
                                 (1 - self._alpha) * self._left_confidence)
        self._right_confidence = (self._alpha * float(raw_right_hand_landmarks is not None) +
                                  (1 - self._alpha) * self._right_confidence)

        # If the confidence is below the minimum confidence, return None.
        if self._left_confidence < self._min_confidence:
            corrected_left_landmarks = None
        if self._right_confidence < self._min_confidence:
            corrected_right_landmarks = None

        # Store the current landmarks for the next frame's heuristics. This uses
        # the corrected landmarks to ensure the proximity check uses the most
        # accurate last known position.
        if corrected_left_landmarks is not None:
            self._previous_left_hand_landmarks = corrected_left_landmarks
        if corrected_right_landmarks is not None:
            self._previous_right_hand_landmarks = corrected_right_landmarks

        return types.HandLandmarksRobotSpace(
            left_hand_landmarks=corrected_left_landmarks,
            right_hand_landmarks=corrected_right_landmarks
        )

    def _detect_handedness_swap(
            self,
            current_left: Optional[types.HandPose3D],
            current_right: Optional[types.HandPose3D],
            ) -> Tuple[Optional[types.HandPose3D], Optional[types.HandPose3D]]:
        """
        Detects and corrects if a single detected hand is mislabeled based on
        historical confidence and proximity.

        Args:
            current_left: The current left hand landmarks.
            current_right: The current right hand landmarks.

        Returns:
            A tuple of (corrected_left_hand_landmarks, corrected_right_hand_landmarks).
        """
        only_one_hand_detected = (current_left is None) != (current_right is None)
        if not only_one_hand_detected:
            return current_left, current_right

        is_left_detected = current_left is not None
        is_right_detected = current_right is not None

        if is_left_detected:
            # If the historical confidence in right is significantly higher,
            # swap the left and the right hands.
            should_check_swap = (self._right_confidence >
                                 self._left_confidence + self._confidence_threshold)
            if should_check_swap and self._previous_right_hand_landmarks is not None:
                distance = target_pose.target_position_distance(
                    current_left, self._previous_right_hand_landmarks, target_pose.Hand.LEFT)
                if distance is not None and distance < self._swap_threshold_meters:
                    return None, current_left  # Swap the left and the right hands.
        elif is_right_detected:
            # If the historical confidence in left is significantly higher,
            # swap the left and the right hands.
            should_check_swap = (self._left_confidence >
                                 self._right_confidence + self._confidence_threshold)
            if should_check_swap and self._previous_left_hand_landmarks is not None:
                distance = target_pose.target_position_distance(
                    current_right, self._previous_left_hand_landmarks, target_pose.Hand.RIGHT)
                if distance is not None and distance < self._swap_threshold_meters:
                    return current_right, None  # Swap the left and the right hands.

        # If none of the swap conditions are met, return the original data.
        return current_left, current_right

    def _apply_heuristics(
          self,
          current_left_hand_landmarks: Optional[types.HandPose3D],
          current_right_hand_landmarks: Optional[types.HandPose3D],
          ) -> Tuple[Optional[types.HandPose3D], Optional[types.HandPose3D]]:
        """
        Apply heuristics to correct for hand swapping issues.

        This function applies a series of checks to fix common detection errors.

        Args:
            current_left_hand_landmarks: Detected landmarks for the left hand.
            current_right_hand_landmarks: Detected landmarks for the right hand.

        Returns:
            A tuple of (corrected_left_hand_landmarks, corrected_right_hand_landmarks).
        """
        # If force_hand is enabled and only one hand is detected, override handedness
        if self._force_hand and ((current_left_hand_landmarks is None) !=
                                 (current_right_hand_landmarks is None)):
            if self._force_hand == 'left':
                # If a right hand is detected, move it to the left.
                if current_right_hand_landmarks is not None:
                    return current_right_hand_landmarks, None
                # If a left hand is detected, it's already correct.
                return current_left_hand_landmarks, None

            if self._force_hand == 'right':
                # If a left hand is detected, move it to the right.
                if current_left_hand_landmarks is not None:
                    return None, current_left_hand_landmarks
                # If a right hand is detected, it's already correct.
                return None, current_right_hand_landmarks

        corrected_left, corrected_right = self._detect_handedness_swap(
            current_left_hand_landmarks, current_right_hand_landmarks
        )

        # TODO (isaac): Implement more robust heuristics to prevent hand swapping in 3D space.
        # * If two hands are detected, but their positions are swapped compared
        #   to the previous frame, un-swap them.
        # * Add outlier detection to reject detections that are physically impossible
        #   (e.g., hand jumping a large distance in a single frame).

        return corrected_left, corrected_right

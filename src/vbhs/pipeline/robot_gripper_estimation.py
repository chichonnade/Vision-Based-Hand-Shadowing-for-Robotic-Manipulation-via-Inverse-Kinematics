"""
Robot Gripper Command Generation Pipeline Step
==============================================

This module handles generation of gripper commands from robot space hand landmarks.
It calculates gripper angles based on the distance between thumb and index finger tips.

The RobotGripperCommandsFromHandLandmarks transformation takes a HandLandmarksRobotSpace
instance (containing hand landmark positions in robot frame) and generates appropriate
gripper commands for both left and right hands.

Key Features:
- Thumb-to-index finger distance based gripper control
- Configurable finger distance to gripper angle mapping
- Exponential smoothing for stable gripper control
- Independent control for left and right grippers
- Robust handling of missing or incomplete landmark data

Usage:
    gripper_controller = RobotGripperCommandsFromHandLandmarks()
    gripper_commands = gripper_controller(robot_joints)  # Returns RobotGripperCommands
"""
import logging
from typing import Optional

import numpy as np

from vbhs.pipeline import transformations
from vbhs.pipeline import types
from vbhs.pipeline.hands import target_pose
from vbhs.config import config

_logger = logging.getLogger(__name__)


class RobotGripperCommandsFromHandLandmarks(
        transformations.Transformation[
            types.HandLandmarksRobotSpace, types.RobotGripperCommands]):
    """
    Gripper command generation from robot space hand landmarks.

    This transformation analyzes hand pose (specifically thumb-index finger distance)
    to generate appropriate gripper commands for teleoperation control.
    """

    def __init__(self, offset: float):
        """Initialize the gripper command generator.

        Args:
            offset: Offset to add to the gripper angle.
        """
        super().__init__()

        # Extract configuration parameters from gripper spec.
        # TODO (isaac): understand the spec.
        self._min_angle = config.GRIPPER_SPEC['joint_angles']['min_angle']
        self._max_angle = config.GRIPPER_SPEC['joint_angles']['max_angle']
        self._offset = offset
        
        # Store last valid gripper angles for fallback
        self._last_valid_left_angle: Optional[float] = None
        self._last_valid_right_angle: Optional[float] = None

        _logger.info('Gripper command generator initialized')
        _logger.debug('Gripper angle range: [%.3f, %.3f] radians', self._min_angle, self._max_angle)

    def _gripper_angle_from_landmarks(self, hand_landmarks: Optional[types.HandPose3D],
                                      hand: target_pose.Hand) -> Optional[float]:
        """Calculate gripper angle for a specific hand."""
        if hand_landmarks is None:
            return None
        # TODO (isaac): stop recomputing this everywhere.
        hand_position = target_pose.calculate_target_position(hand_landmarks, hand)
        angle = self._calculate_gripper_angle(hand_position, hand_landmarks, hand)
        
        # Store last valid angle for future fallback
        if angle is not None:
            if hand == target_pose.Hand.LEFT:
                self._last_valid_left_angle = angle
            else:
                self._last_valid_right_angle = angle
        
        return angle


    def _transform(self, hand_landmarks: types.HandLandmarksRobotSpace, /) -> types.RobotGripperCommands:
        """
        Generate gripper commands from robot space joint positions.

        Args:
            robot_joints: Robot space joint positions and angles

        Returns:
            RobotGripperCommands: Gripper angle commands for both hands
        """
        left_gripper_angle = self._gripper_angle_from_landmarks(
            hand_landmarks.left_hand_landmarks, target_pose.Hand.LEFT)
        right_gripper_angle = self._gripper_angle_from_landmarks(
            hand_landmarks.right_hand_landmarks, target_pose.Hand.RIGHT)

        return types.RobotGripperCommands(
            left_gripper_angle=left_gripper_angle,
            right_gripper_angle=right_gripper_angle
        )

    def _calculate_gripper_angle(
            self,
            target_position: list[float],
            hand_landmarks: types.HandPose3D,
            hand: target_pose.Hand) -> Optional[float]:
        """
        Calculate gripper angle for a specific hand using thumb-index finger distance.
        
        Implements fallback hierarchy:
        1. Try tips (THUMB_TIP, INDEX_FINGER_TIP)
        2. Try knuckles (THUMB_IP, INDEX_FINGER_DIP)
        3. Use last valid angle for this hand
        4. Use mid-open default

        Args:
            target_position: Target gripper position
            hand_landmarks: Hand landmarks in 3D
            hand: Which hand (LEFT or RIGHT)

        Returns:
            Gripper angle in radians
        """
        # Check if target position is valid
        if target_position is None:
            return self._get_ultimate_fallback(hand)
        
        # Priority 1: Try tips (THUMB_TIP, INDEX_FINGER_TIP)
        thumb_pos = hand_landmarks.get('THUMB_TIP')
        index_pos = hand_landmarks.get('INDEX_FINGER_TIP')
        
        if thumb_pos is not None and index_pos is not None:
            angle = self._compute_angle_from_landmarks(target_position, thumb_pos, index_pos)
            if angle is not None:
                return angle
        
        # Priority 2: Try knuckles (THUMB_IP for thumb, INDEX_FINGER_DIP for index)
        _logger.debug('Gripper tips missing, trying knuckle landmarks')
        thumb_pos = hand_landmarks.get('THUMB_IP')
        index_pos = hand_landmarks.get('INDEX_FINGER_DIP')
        
        if thumb_pos is not None and index_pos is not None:
            angle = self._compute_angle_from_landmarks(target_position, thumb_pos, index_pos)
            if angle is not None:
                _logger.debug('Using knuckle landmarks for gripper angle')
                return angle
        
        # Priority 3: Use last valid angle
        last_valid = (self._last_valid_left_angle if hand == target_pose.Hand.LEFT 
                     else self._last_valid_right_angle)
        if last_valid is not None:
            _logger.debug('Using last valid gripper angle: %.3f radians', last_valid)
            return last_valid
        
        # Priority 4: Use mid-open default
        return self._get_ultimate_fallback(hand)
    
    def _compute_angle_from_landmarks(
            self,
            target_position: np.ndarray,
            thumb_pos: tuple[float, float, float],
            index_pos: tuple[float, float, float]) -> Optional[float]:
        """
        Compute gripper angle from two landmark positions.
        
        Args:
            target_position: Base position (gripper center)
            thumb_pos: Thumb landmark position
            index_pos: Index finger landmark position
            
        Returns:
            Gripper angle in radians or None if computation fails
        """
        try:
            thumb_array = np.array(thumb_pos)
            index_array = np.array(index_pos)
            
            # Calculate angle between vectors from gripper base position to landmarks
            base_to_thumb = thumb_array - target_position
            base_to_index = index_array - target_position
            
            # Calculate acute angle between thumb and index finger
            dot_product = np.dot(base_to_thumb, base_to_index)
            norm = np.linalg.norm(base_to_thumb) * np.linalg.norm(base_to_index)
            
            if norm == 0:
                _logger.warning('Zero length finger vector detected, cannot compute angle')
                return None
                
            angle = np.arccos(np.clip(dot_product / norm, -1.0, 1.0))
            
            # Acute angle is required for the gripper
            if angle > np.pi / 2:
                angle = np.pi - angle
            
            return np.clip(angle + self._offset, self._min_angle, self._max_angle)
        except (ValueError, RuntimeError) as e:
            _logger.warning('Failed to compute gripper angle: %s', e)
            return None
    
    def _get_ultimate_fallback(self, hand: target_pose.Hand) -> float:
        """Get the ultimate fallback gripper angle (mid-open position)."""
        fallback_angle = (self._min_angle + self._max_angle) / 2.0
        _logger.debug('%s gripper using mid-open fallback: %.3f radians', 
                     hand.value, fallback_angle)
        return fallback_angle

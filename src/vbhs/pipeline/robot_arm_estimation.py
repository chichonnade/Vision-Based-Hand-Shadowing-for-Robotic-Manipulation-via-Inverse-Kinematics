"""
Robot Arm Command Generation Pipeline Step
=========================================

This module handles generation of robot arm joint commands from robot space hand landmarks.
It calculates target positions based on hand pose and uses inverse kinematics to determine
joint angles for joints 1-5 of each arm.

The RobotArmCommandsFromHandLandmarks transformation takes RobotSpaceJoints
(containing hand landmarks in robot frame) and generates appropriate arm joint
commands for teleoperation control.

Key Features:
- Target position calculation from thumb-index finger base landmarks
- Joint 5 orientation based on hand pose vector
- Inverse kinematics for joint angle calculation
- Independent control for left and right arms
- Smoothing for stable joint movements
- Robust handling of missing landmark data

Usage:
    arm_controller = RobotArmCommandsFromHandLandmarks(robot_id)
    arm_commands = arm_controller(robot_joints)  # Returns RobotArmCommands
"""
# pylint: disable=c-extension-no-member

import logging
from typing import Any, Optional

from vbhs.pipeline import transformations
from vbhs.pipeline import types
from vbhs.pipeline import inverse_kinematics
from vbhs.pipeline.hands import target_pose
from vbhs.utils import debug_visualization

_logger = logging.getLogger(__name__)


class RobotArmCommandsFromHandLandmarks(
        transformations.Transformation[
            types.HandLandmarksRobotSpace, types.RobotArmCommands]):
    """
    Robot arm command generation from robot space hand landmarks.

    This transformation analyzes hand pose and position to generate appropriate
    arm joint commands using inverse kinematics for teleoperation control.
    """

    def __init__(self,
                 robot_id: int,
                 left_arm_joints: list[int],
                 right_arm_joints: list[int],
                 left_end_effector: int,
                 right_end_effector: int,
                 debug_visualizer: debug_visualization.DebugVisualizer,
                 enable_smoothing: bool = False,
                 smoothing_alpha: float = 0.3,
                 min_z_height: float = 0.05):
        """
        Initialize the robot arm command generator.

        Args:
            robot_id: PyBullet robot body ID (None for placeholder mode)
            left_arm_joints: list of joint indices for left arm (joints 1-5)
            right_arm_joints: list of joint indices for right arm (joints 1-5)
            left_end_effector: Link index for left arm end effector
            right_end_effector: Link index for right arm end effector
            debug_visualizer: Debug visualizer instance
            enable_smoothing: Whether to apply smoothing to joint angles
            smoothing_alpha: Smoothing factor (0-1, lower = more smoothing)
            min_z_height: Minimum Z height (in meters) for target positions. Targets
                below this height are rejected to prevent collision with ground plane.
                Default: 0.05m (5cm)

        Note:
            - If robot_id is None, runs in placeholder mode (returns zero angles)
            - Joint indices should correspond to actual robot joint configuration
            - End effector indices should point to the gripper attachment point
            - Targets below min_z_height are filtered out for safety
        """
        super().__init__()

        self.robot_id = robot_id
        self.left_arm_joints = left_arm_joints
        self.right_arm_joints = right_arm_joints
        self.left_end_effector = left_end_effector
        self.right_end_effector = right_end_effector
        self.enable_smoothing = enable_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.min_z_height = min_z_height
        self._debug_visualizer = debug_visualizer

        # Previous joint angles for smoothing
        self._prev_left_angles: Optional[list[float]] = None
        self._prev_right_angles: Optional[list[float]] = None

        self._left_ik_solver = inverse_kinematics.IKSolver(
            robot_id=self.robot_id,
            end_effector_idx=self.left_end_effector,
            joint_indices=self.left_arm_joints
        )
        self._right_ik_solver = inverse_kinematics.IKSolver(
            robot_id=self.robot_id,
            end_effector_idx=self.right_end_effector,
            joint_indices=self.right_arm_joints
        )

        _logger.info('Robot arm command generator initialized')
        _logger.debug('Robot ID: %s', self.robot_id)
        _logger.debug('Left arm joints: %s', self.left_arm_joints)
        _logger.debug('Right arm joints: %s', self.right_arm_joints)
        _logger.debug('Smoothing: %s (alpha=%s)', "enabled" if enable_smoothing else "disabled", smoothing_alpha)
        _logger.debug('Minimum Z height safety: %.3fm', self.min_z_height)

    def _transform(self, hand_landmarks: types.HandLandmarksRobotSpace, /) -> types.RobotArmCommands:
        """
        Generate robot arm commands from robot space hand landmarks.

        Args:
            hand_landmarks: Robot space hand landmark positions

        Returns:
            RobotArmCommands: Joint angle commands for both arms
        """
        # Calculate joint angles for each arm
        left_arm_angles = self._calculate_arm_angles('left', hand_landmarks)
        right_arm_angles = self._calculate_arm_angles('right', hand_landmarks)

        if self.enable_smoothing:
            if left_arm_angles is not None:
                left_arm_angles = self._apply_smoothing(left_arm_angles, self._prev_left_angles)
                self._prev_left_angles = left_arm_angles[:]
            else:
                # Clear previous angles when target is rejected to avoid stale smoothing
                self._prev_left_angles = None
            if right_arm_angles is not None:
                right_arm_angles = self._apply_smoothing(right_arm_angles, self._prev_right_angles)
                self._prev_right_angles = right_arm_angles[:]
            else:
                # Clear previous angles when target is rejected to avoid stale smoothing
                self._prev_right_angles = None

        return types.RobotArmCommands(
            left_arm_angles=left_arm_angles,
            right_arm_angles=right_arm_angles
        )

    def _calculate_arm_angles(self, arm: str, hand_landmarks: types.HandLandmarksRobotSpace) -> Optional[list[float]]:
        """
        Calculate joint angles for a specific arm using inverse kinematics.

        Args:
            arm: 'left' or 'right'
            hand_landmarks: Robot space hand landmark data

        Returns:
            list of 5 joint angles in radians
        """
        if arm == 'left':
            target_hand_pose = hand_landmarks.left_hand_landmarks
            arm_joints = self.left_arm_joints
            end_effector = self.left_end_effector
            ik_solver = self._left_ik_solver
            hand = target_pose.Hand.LEFT
        elif arm == 'right':
            target_hand_pose = hand_landmarks.right_hand_landmarks
            arm_joints = self.right_arm_joints
            end_effector = self.right_end_effector
            ik_solver = self._right_ik_solver
            hand = target_pose.Hand.RIGHT
        else:
            raise ValueError(f'Invalid arm: {arm}')

        if target_hand_pose is None:
            return None

        target_pos = target_pose.calculate_target_position(target_hand_pose, hand)
        orientation_quat = target_pose.calculate_target_orientation(target_hand_pose, hand)

        # Safety check: reject targets below minimum Z height to prevent ground collision
        if target_pos is not None and target_pos[2] < self.min_z_height:
            _logger.warning(
                '%s arm target position below minimum Z height (%.3fm < %.3fm). Rejecting target.',
                arm.capitalize(), target_pos[2], self.min_z_height)
            target_pos = None

        # Use fallback orientation when full orientation unavailable but position is valid
        if target_pos is not None and orientation_quat is None:
            orientation_quat = target_pose.calculate_fallback_orientation(target_hand_pose, hand)
            if orientation_quat is not None:
                _logger.debug(
                    '%s arm: Using fallback orientation (tip landmarks missing, using default forward direction).',
                    arm.capitalize())
            else:
                _logger.debug(
                    '%s arm: Position valid but both full and fallback orientation unavailable. '
                    'Using position-only IK.',
                    arm.capitalize())

        self._debug_visualizer.visualize(
            arm, target_pos, orientation_quat, end_effector, arm_joints)

        if target_pos is None:
            return None

        joint_angles = ik_solver.solve(target_pos, orientation_quat)
        
        # Log when IK returns None despite having valid position
        if joint_angles is None:
            _logger.warning(
                '%s arm: IK solver returned None despite valid target position (%.3f, %.3f, %.3f). '
                'This may indicate unreachable target or IK constraints issue.',
                arm.capitalize(), target_pos[0], target_pos[1], target_pos[2])
        else:
            _logger.debug(
                '%s arm: IK successful - joint angles: [%.3f, %.3f, %.3f, %.3f, %.3f]',
                arm.capitalize(), joint_angles[0], joint_angles[1], joint_angles[2], 
                joint_angles[3], joint_angles[4])
        
        return joint_angles

    def _apply_smoothing(self, current_angles: list[float], previous_angles: Optional[list[float]]) -> list[float]:
        """
        Apply exponential smoothing to joint angles.

        Args:
            current_angles: Current calculated joint angles
            previous_angles: Previous smoothed joint angles (None for first frame)

        Returns:
            list of smoothed joint angles
        """
        if previous_angles is None or len(previous_angles) != len(current_angles):
            return current_angles[:]

        # Exponential smoothing: new_value = alpha * current + (1-alpha) * previous
        smoothed_angles = []
        for current_angle, previous_angle in zip(current_angles, previous_angles):
            smoothed_angle = (
                self.smoothing_alpha * current_angle +
                (1 - self.smoothing_alpha) * previous_angle)
            smoothed_angles.append(smoothed_angle)

        return smoothed_angles

    def reset_smoothing(self):
        """Reset smoothing state (useful when starting new sequences)."""
        self._left_ik_solver.reset()
        self._right_ik_solver.reset()
        self._prev_left_angles = None
        self._prev_right_angles = None
        _logger.debug('Arm joint smoothing state reset')


    def get_config_info(self) -> dict[str, Any]:
        """
        Get current robot arm configuration information.

        Returns:
            Dictionary containing current configuration parameters
        """
        return {
            'robot_id': self.robot_id,
            'left_arm_joints': self.left_arm_joints,
            'right_arm_joints': self.right_arm_joints,
            'left_end_effector': self.left_end_effector,
            'right_end_effector': self.right_end_effector,
            'enable_smoothing': self.enable_smoothing,
            'smoothing_alpha': self.smoothing_alpha,
            'previous_angles': {
                'left': self._prev_left_angles,
                'right': self._prev_right_angles
            }
        }

"""
Complete Robot Control Command Generation Pipeline Step
======================================================

This module handles generation of complete robot control commands from robot space hand landmarks.
It combines arm joint commands (joints 1-5) and gripper commands (joint 6) into unified
robot control data for teleoperation.

The RobotControlCommandsFromRobotSpaceJoints transformation takes RobotSpaceJoints
(containing hand landmarks in robot frame) and generates complete robot control
commands as a tuple of two 6-joint lists (left arm, right arm).

Key Features:
- Unified control combining arm IK and gripper control
- Target position calculation and tracking
- Independent control for left and right arms
- Comprehensive robot state management
- Error handling and safe fallback positions
- Integration with existing arm and gripper controllers

Usage:
    robot_controller = RobotControlCommandsFromRobotSpaceJoints(robot_id)
    control_commands = robot_controller(robot_joints)  # Returns RobotControlCommands
"""
import logging

from vbhs.utils import debug_visualization
from vbhs.pipeline import types
from vbhs.pipeline import transformations
from vbhs.pipeline import robot_arm_estimation
from vbhs.pipeline import robot_gripper_estimation

_logger = logging.getLogger(__name__)


class RobotControlCommandsFromHandLandmarks(
        transformations.Transformation[
            types.HandLandmarksRobotSpace, types.RobotControlCommands]):
    """
    Complete robot control command generation from robot space hand landmarks.

    This transformation combines arm joint control (joints 1-5) and gripper control (joint 6)
    to provide unified robot control commands for teleoperation.
    """

    def __init__(self,
                 robot_id: int,
                 left_arm_joints: list[int],
                 right_arm_joints: list[int],
                 left_end_effector: int,
                 right_end_effector: int,
                 debug_visualizer: debug_visualization.DebugVisualizer,
                 enable_smoothing: bool = False,
                 arm_smoothing_alpha: float = 0.3,
                 min_z_height: float = 0.05):
        """
        Initialize the complete robot control command generator.

        Args:
            robot_id: PyBullet robot body ID (None for placeholder mode)
            left_arm_joints: List of joint indices for left arm (joints 1-5)
            right_arm_joints: List of joint indices for right arm (joints 1-5)
            left_end_effector: Link index for left arm end effector
            right_end_effector: Link index for right arm end effector
            debug_visualizer: Debug visualizer instance
            enable_smoothing: Whether to apply smoothing to commands
            arm_smoothing_alpha: Smoothing factor for arm joints (0-1)
            min_z_height: Minimum Z height for target positions (meters, default: 0.05m)

        Note:
            - Coordinates both arm and gripper control systems
            - Provides unified interface for complete robot control
            - Manages target positions for visualization and monitoring
            - Targets below min_z_height are rejected to prevent ground collision
        """
        super().__init__()

        self.robot_id = robot_id
        self.enable_smoothing = enable_smoothing

        self._arm_controller = robot_arm_estimation.RobotArmCommandsFromHandLandmarks(
            robot_id, left_arm_joints, right_arm_joints,
            left_end_effector, right_end_effector,
            debug_visualizer,
            enable_smoothing=enable_smoothing,
            smoothing_alpha=arm_smoothing_alpha,
            min_z_height=min_z_height)

        # Tighten grip by approx 10 degrees.
        self._gripper_controller = robot_gripper_estimation.RobotGripperCommandsFromHandLandmarks(
            offset=-0.175)

        _logger.info('Complete robot control command generator initialized')
        _logger.debug('Robot ID: %s', self.robot_id)
        _logger.debug('Smoothing: %s', "enabled" if enable_smoothing else "disabled")
        _logger.debug('Arm smoothing alpha: %s', arm_smoothing_alpha)

    def _transform(self, robot_joints: types.HandLandmarksRobotSpace, /) -> types.RobotControlCommands:
        """
        Generate complete robot control commands from robot space hand landmarks.

        Args:
            robot_joints: Robot space hand landmark positions

        Returns:
            RobotControlCommands: Tuple of (left_arm_joints, right_arm_joints) where each is [j1,j2,j3,j4,j5,j6]
        """
        # Generate arm commands (joints 1-5)
        arm_commands = self._arm_controller(robot_joints)

        # Generate gripper commands (joint 6)
        gripper_commands = self._gripper_controller(robot_joints)

        # Combine arm and gripper commands into 6-joint lists
        if arm_commands.left_arm_angles is None or gripper_commands.left_gripper_angle is None:
            left_joints = None
        else:
            left_joints = arm_commands.left_arm_angles + [gripper_commands.left_gripper_angle]

        if arm_commands.right_arm_angles is None or gripper_commands.right_gripper_angle is None:
            right_joints = None
        else:
            right_joints = arm_commands.right_arm_angles + [gripper_commands.right_gripper_angle]

        return types.RobotControlCommands(
            robot_commands=(left_joints, right_joints))


    def reset_all_smoothing(self):
        """Reset smoothing state for both arm and gripper controllers."""
        self._arm_controller.reset_smoothing()
        _logger.debug('All robot control smoothing state reset')

    def get_config_info(self) -> dict:
        """
        Get comprehensive configuration information for all controllers.

        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'robot_id': self.robot_id,
            'enable_smoothing': self.enable_smoothing,
            'arm_controller_config': self._arm_controller.get_config_info(),
    }

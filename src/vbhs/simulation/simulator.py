"""
Dual Arm Teleoperation Simulation
=================================

This module provides a complete PyBullet simulation environment for dual-arm teleoperation
using hand tracking. It integrates the full vision pipeline with robot control.

The DualArmTeleopSimulation class handles:
- PyBullet simulation setup and environment
- Dual-arm robot loading and configuration
- Integration with hand tracking pipeline
- Real-time robot control from hand landmarks
- Visualization and monitoring

Key Features:
- Complete simulation environment setup
- Dual-arm robot control with IK
- Hand tracking integration
- Virtual camera systems
- Real-time teleoperation
- Smooth joint control with safety limits

Usage:
    sim = DualArmTeleopSimulation(urdf_path="robot.urdf")
    sim.run_teleoperation()
"""
# pylint: disable=c-extension-no-member

import logging
from typing import Dict, List, Optional
import os
import time

import pybullet as p
import pybullet_data

from vbhs.utils import debug_visualization

_logger = logging.getLogger(__name__)

# Remove pipeline imports - handled in main teleoperation script


class DualArmTeleopSimulation:
    """
    Complete dual-arm teleoperation simulation using PyBullet and hand tracking.

    This class provides a complete simulation environment that integrates hand tracking
    with dual-arm robot control in PyBullet.

    Joint Structure (Dual_S101_Assembly.urdf):
    - left_joint_0: FIXED (index 0)
    - left_joint_1: REVOLUTE (index 1) - Base rotation
    - left_joint_2: REVOLUTE (index 2) - Shoulder
    - left_joint_3: REVOLUTE (index 3) - Elbow
    - left_joint_4: REVOLUTE (index 4) - Wrist pitch
    - left_joint_5: REVOLUTE (index 5) - Wrist roll
    - left_joint_6: REVOLUTE (index 6) - Gripper
    - right_joint_0: FIXED (index 7)
    - right_joint_1: REVOLUTE (index 8) - Base rotation
    - right_joint_2: REVOLUTE (index 9) - Shoulder
    - right_joint_3: REVOLUTE (index 10) - Elbow
    - right_joint_4: REVOLUTE (index 11) - Wrist pitch
    - right_joint_5: REVOLUTE (index 12) - Wrist roll
    - right_joint_6: REVOLUTE (index 13) - Gripper
    """

    def __init__(self,
                 urdf_path: str,
                 left_arm_joints: Optional[Dict[str, int]] = None,
                 right_arm_joints: Optional[Dict[str, int]] = None,
                 left_gripper_joint: Optional[int] = None,
                 right_gripper_joint: Optional[int] = None,
                 left_end_effector: Optional[int] = None,
                 right_end_effector: Optional[int] = None,
                 enable_virtual_cameras: bool = True,
                 use_gui: bool = True):
        """
        Initialize dual-arm teleoperation simulation.

        Args:
            urdf_path: Path to dual-arm robot URDF file
            left_arm_joints: Dictionary mapping joint names to indices for left arm
                (left_joint_1 to left_joint_5)
            right_arm_joints: Dictionary mapping joint names to indices for right arm
                (right_joint_1 to right_joint_5)
            left_gripper_joint: Joint index for left gripper (left_joint_6)
            right_gripper_joint: Joint index for right gripper (right_joint_6)
            left_end_effector: Link index for left end effector
            right_end_effector: Link index for right end effector
            enable_virtual_cameras: Whether to enable virtual camera views
            use_gui: Whether to run with PyBullet GUI
        """
        self.urdf_path = urdf_path
        self.use_gui = use_gui

        # Robot configuration
        self.left_arm_joints = left_arm_joints or {
            'left_joint_1': 1, 'left_joint_2': 2, 'left_joint_3': 3,
            'left_joint_4': 4, 'left_joint_5': 5
        }
        self.right_arm_joints = right_arm_joints or {
            'right_joint_1': 8, 'right_joint_2': 9, 'right_joint_3': 10,
            'right_joint_4': 11, 'right_joint_5': 12
        }
        self.left_gripper_joint = left_gripper_joint or 6
        self.right_gripper_joint = right_gripper_joint or 13
        self.left_end_effector = left_end_effector or 5
        self.right_end_effector = right_end_effector or 12

        # Simulation state
        self.physics_client = None
        self.sim_cameras = enable_virtual_cameras

        # Visual markers for target positions
        self.left_target_marker = None
        self.right_target_marker = None
        self.left_target_label = None
        self.right_target_label = None

        _logger.info('Dual-arm teleoperation simulation initialized')
        _logger.debug('URDF: %s', urdf_path)
        _logger.debug('Virtual cameras: %s', enable_virtual_cameras)

        # Initialize simulation
        self._setup_simulation()
        self.robot_id = self._load_robot()
        self._setup_initial_positions()

        # Debug visualization
        self.debug_visualizer = debug_visualization.DebugVisualizer(robot_id=self.robot_id)

    def _setup_simulation(self):
        """Setup PyBullet simulation environment."""
        _logger.info("Setting up dual arm teleop simulation")

        # Connect with GUI
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)


        # Configure visualizer
        if self.use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=0,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )

        # Setup environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)

        # Load plane
        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.8, 0.8, 0.8, 1.0])

        _logger.info("Simulation environment ready")

    def _load_robot(self) -> int:
        """Load dual arm robot URDF."""
        _logger.info("Loading dual arm robot: %s", self.urdf_path)

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        # Load robot at origin
        robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        if robot_id is None:
            raise RuntimeError("Failed to load dual arm robot URDF")

        _logger.info("Dual arm robot loaded successfully")

        # Set virtual camera view if available
        if self.sim_cameras and self.use_gui:
            self.set_virtual_camera_view("front")
            _logger.debug("Virtual camera view initialized")

        return robot_id

    def get_joint_positions(self) -> tuple[List[float], List[float]]:
        """Get joint positions."""
        robot_state = self.get_robot_state()
        joint_positions = robot_state['joint_positions']

        left_arm_positions = [joint_positions[i] for i in self.left_arm_joints.values()] + [joint_positions[self.left_gripper_joint]]
        right_arm_positions = [joint_positions[i] for i in self.right_arm_joints.values()] + [joint_positions[self.right_gripper_joint]]

        return (left_arm_positions, right_arm_positions)

    def _setup_initial_positions(self):
        """Set initial joint positions."""
        _logger.debug("Setting initial robot positions")

        # Set initial positions for left arm
        for joint_idx in self.left_arm_joints.values():
            p.resetJointState(self.robot_id, joint_idx, 0.0)

        # Set initial positions for right arm
        for joint_idx in self.right_arm_joints.values():
            p.resetJointState(self.robot_id, joint_idx, 0.0)

        # Set gripper positions
        if self.left_gripper_joint:
            p.resetJointState(self.robot_id, self.left_gripper_joint, 0.0)
            self.prev_left_gripper = 0.0
        if self.right_gripper_joint:
            p.resetJointState(self.robot_id, self.right_gripper_joint, 0.0)
            self.prev_right_gripper = 0.0

        _logger.debug("Initial positions set")

    def control_arms(
            self,
            left_control_data: Optional[List[float]] = None,
            right_control_data: Optional[List[float]] = None):
        """
        Control one or both arms using 6-joint control data.

        If both left and right control data are provided, they are applied
        simultaneously in a single command.

        Args:
            left_control_data: Optional list of 6 joint angles for the left arm
                [j1, j2, j3, j4, j5, j6_gripper].
            right_control_data: Optional list of 6 joint angles for the right arm
                [j1, j2, j3, j4, j5, j6_gripper].
        """
        joint_indices: List[int] = []
        target_positions: List[float] = []

        # Process left arm control data
        if left_control_data:
            if len(left_control_data) != 6:
                raise ValueError('Expected 6 joint values for left arm, got '
                                 f'{len(left_control_data)}')
            arm_joints = list(self.left_arm_joints.values())
            gripper_joint = self.left_gripper_joint
            arm_angles = left_control_data[:5]
            gripper_angle = left_control_data[5]

            joint_indices.extend(arm_joints + [gripper_joint])
            target_positions.extend(arm_angles + [gripper_angle])

        # Process right arm control data
        if right_control_data:
            if len(right_control_data) != 6:
                raise ValueError('Expected 6 joint values for right arm, got '
                                 f'{len(right_control_data)}')
            arm_joints = list(self.right_arm_joints.values())
            gripper_joint = self.right_gripper_joint
            arm_angles = right_control_data[:5]
            gripper_angle = right_control_data[5]

            joint_indices.extend(arm_joints + [gripper_joint])
            target_positions.extend(arm_angles + [gripper_angle])

        # Send commands to robot if any control data was provided
        if joint_indices:
            p.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_positions,
                forces=[6.0] * len(joint_indices),
                positionGains=[0.2] * len(joint_indices),
                velocityGains=[1.0] * len(joint_indices)
            )

    def _handle_debug_visualization(self):
        """Handle debug visualization."""
        p.addUserDebugText(
            f'Debug Visualization Mode: {self.debug_visualizer.mode.next_mode().name}',
            textPosition=[-0.5, 0, 1],
            textColorRGB=[1, 0, 0],
            textSize=1.0,
            lifeTime=1.0)
        self.debug_visualizer.next_mode()

    def handle_keyboard_events(self):
        """Check for keyboard events and handle them."""
        keys = p.getKeyboardEvents()

        # Check for 'd' key press to toggle next debug visualization mode.
        if ord('d') in keys and keys[ord('d')] and p.KEY_WAS_TRIGGERED:
            self._handle_debug_visualization()


    def set_virtual_camera_view(self, view_name: str):
        """
        Set virtual camera view in simulation.

        Args:
            view_name: Name of the view ('front', 'side', 'top', 'robot')
        """
        if not self.sim_cameras:
            return

        camera_configs = {
            'front': {'distance': 2.0, 'yaw': 0, 'pitch': -30, 'target': [0, 0, 0.5]},
            'side': {'distance': 2.5, 'yaw': 90, 'pitch': -20, 'target': [0, 0, 0.5]},
            'top': {'distance': 3.0, 'yaw': 0, 'pitch': -90, 'target': [0, 0, 0.5]},
            'robot': {'distance': 1.5, 'yaw': 45, 'pitch': -15, 'target': [0, 0, 0.8]}
        }

        if view_name in camera_configs:
            config = camera_configs[view_name]
            p.resetDebugVisualizerCamera(
                cameraDistance=config['distance'],
                cameraYaw=config['yaw'],
                cameraPitch=config['pitch'],
                cameraTargetPosition=config['target']
            )
            _logger.debug("Camera view set to: %s", view_name)

    def get_robot_state(self) -> Dict:
        """
        Get current robot state information.

        Returns:
            Dictionary with robot state information
        """
        if self.robot_id is None:
            return {}

        # Get joint states
        num_joints = p.getNumJoints(self.robot_id)
        joint_states = p.getJointStates(self.robot_id, range(num_joints))

        # Extract positions and velocities
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'left_arm_positions': [joint_positions[i] for i in self.left_arm_joints.values()],
            'right_arm_positions': [joint_positions[i] for i in self.right_arm_joints.values()],
            'left_gripper_position': (
                joint_positions[self.left_gripper_joint]
                if self.left_gripper_joint else None),
            'right_gripper_position': (
                joint_positions[self.right_gripper_joint]
                if self.right_gripper_joint else None),
        }

    def reset_robot_pose(self):
        """Reset robot to initial pose."""
        _logger.debug("Resetting robot pose")
        self._setup_initial_positions()
        _logger.debug("Robot pose reset")

    def cleanup(self):
        """Clean up simulation resources."""
        _logger.debug("Cleaning up simulation")

        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        _logger.debug("Simulation cleanup complete")


def main():
    """Main function for testing the simulation setup only."""
    import argparse

    parser = argparse.ArgumentParser(description='Test Dual Arm Simulation Setup')
    parser.add_argument('--urdf', type=str, default='robot/Dual_S101_Assembly.urdf',
                        help='Path to robot URDF file')
    parser.add_argument('--no-cameras', action='store_true', help='Disable virtual cameras')

    args = parser.parse_args()

    sim = None
    try:
        # Create simulation (setup only)
        sim = DualArmTeleopSimulation(
            urdf_path=args.urdf,
            enable_virtual_cameras=not args.no_cameras
        )

        print("✅ Simulation setup completed successfully")

        # Keep simulation running for inspection
        print("Press Ctrl+C to exit...")
        while True:
            p.stepSimulation()
            time.sleep(1/240)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped")
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return 1
    finally:
        if sim:
            sim.cleanup()

    return 0


if __name__ == '__main__':
    exit(main())

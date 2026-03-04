"""TODO (isaac): write docs"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt

from lerobot.utils.robot_utils import busy_wait
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    so101_follower,
    make_robot_from_config
)
import pybullet as p

from vbhs.simulation.simulator import DualArmTeleopSimulation


class FakeRobot(so101_follower.SO101Follower):
    def __init__(self, config: RobotConfig):
        self.actions: list[npt.NDArray[np.float32]] = []

    def connect(self, calibrate: bool = True): ...

    def disconnect(self): ...

    def send_action(self, action: dict[str, float]):
        self.actions.append(action)

    @property
    def is_connected(self): ...

    @property
    def is_calibrated(self): ...

    def calibrate(self): ...

    def get_observation(self): ...

    def configure(self): ...

    def action_features(self): ...

    def observation_features(self): ...


class GripperMode(Enum):
    NORMAL = 'normal'
    BINARY = 'binary'
    OFFSET = 'offset'


@dataclass
class PlaybackConfig:
    robot: RobotConfig
    actions_file: str = ''
    urdf: str = "robot/Dual_S101_Assembly.urdf"  # Path to the URDF file.
    output_dir: str = "vbhs_output"
    # Limit the maximum frames per second.
    fps: int = 30
    simulator_fps: int = 30
    use_fake_robot: bool = False
    # Logging verbosity (0=WARNING, 1=INFO, 2=DEBUG)
    verbosity: int = 0
    # Acceleration limits for smooth motion.
    acceleration_limit: int = 50  #50 # Acceleration setting (0-254, lower = gentler)
    gripper_acceleration_limit: int = 100  # 0-254
    velocity_limit: int = 1500 # Velocity RPM
    gripper_velocity_limit: int = 3000  # Velocity RPM
    # PID configuration for smoother motion
    p_coefficient: int = 12  # Proportional gain (lower = less oscillation, default ~32)
    i_coefficient: int = 0   # Integral gain (0 = no integral windup)
    d_coefficient: int = 24  # Derivative gain (lower = smoother motion, default ~32)
    # Gripper offset to change closing behavior.
    gripper_mode: GripperMode = GripperMode.BINARY
    gripper_offset_deg: float = -30.0
    gripper_threshold_deg: float = 60.0
    gripper_open_deg: float = 90.0
    gripper_closed_deg: float = 0

    def get_actions(self) -> npt.NDArray[np.float32]:
        """Get the actions from the actions file."""
        if not self.actions_file:
            raise ValueError("actions_file is not set")
        actions = np.load(self.actions_file)
        match self.gripper_mode:
            case GripperMode.NORMAL:
                pass
            case GripperMode.BINARY:
                threshold_rad = self.gripper_threshold_deg * np.pi / 180.0
                open_rad = self.gripper_open_deg * np.pi / 180.0
                closed_rad = self.gripper_closed_deg * np.pi / 180.0
                actions[:, :, -1] = np.where(actions[:, :, -1] > threshold_rad, open_rad, closed_rad)
            case GripperMode.OFFSET:
                offset_rad = self.gripper_offset_deg * np.pi / 180.0
                actions[:, :, -1] += offset_rad
        return actions


JOINT_RANGES_RADS = np.array([
    [-1.91986, 1.91986],
    [-1.74533, 1.74533],
    [-1.74533, 1.5708],
    [-1.65806, 1.65806],
    [-2.79253, 2.79253],
    [-0.174533, 1.74533]
])


def configure_servos(robot: so101_follower.SO101Follower, config: PlaybackConfig):
    """Configure the servos in the robot."""
    robot.bus.disable_torque()
    for motor in robot.bus.motors :
        robot.bus.write("P_Coefficient", motor, config.p_coefficient)
        robot.bus.write("I_Coefficient", motor, config.i_coefficient)
        robot.bus.write("D_Coefficient", motor, config.d_coefficient)

    robot.bus.enable_torque()


# TODO (isaac): should .pos be in the name here?
ACTION_NAMES = ['shoulder_pan.pos',
                'shoulder_lift.pos',
                'elbow_flex.pos',
                'wrist_flex.pos',
                'wrist_roll.pos',
                'gripper.pos']


def send_home(robot: so101_follower.SO101Follower):
    """Send the robot to the home position."""
    robot.send_action({f'{motor}.pos': 0.0 for motor in robot.bus.motors})
    time.sleep(2)


def send_standby(robot: so101_follower.SO101Follower):
    """Send the robot to the standby position."""
    STANDBY_POSITION = {
        'shoulder_pan.pos': -1.20,
        'shoulder_lift.pos': -98.01,
        'elbow_flex.pos': 99.73,
        'wrist_flex.pos': 4.62,
        'wrist_roll.pos': -2.59,
        'gripper.pos': 0.0}
    robot.send_action(STANDBY_POSITION)
    time.sleep(3)


def get_action_dictionary(action: npt.NDArray[np.float32], action_names: list[str]) -> dict[str, float]:
    """
    Get the action dictionary for a given action.

    Args:
        action: The action to get the dictionary for.
        robot: The robot to get the dictionary for.

    Returns:
        The action dictionary.
    """
    assert len(action) == len(action_names)
    return {
        motor: float(action[i])
        for i, motor in enumerate(action_names)}


def normalized_action_to_motor_action(normalized_action: float, action_name: str) -> float:
    """ Convert a normalized action to a motor action."""
    if 'gripper' in action_name:
        return normalized_action * 100
    return (normalized_action - 0.5) * 200


def normalize_action(action: npt.NDArray[np.float32], min_angles: npt.NDArray[np.float32],
                     max_angles: npt.NDArray[np.float32]) -> Optional[npt.NDArray[np.float32]]:
    """Normalize an action to the range [0, 1]."""
    if np.isnan(action).any():
        return None
    action = np.clip(action, min_angles, max_angles)
    return (action - min_angles) / (max_angles - min_angles)


def prepare_action(action: npt.NDArray[np.float32], robot: so101_follower.SO101Follower) -> Optional[dict[str, float]]:
    """Prepare an action for the robot."""
    normalized_action = normalize_action(action, JOINT_RANGES_RADS[:, 0], JOINT_RANGES_RADS[:, 1])
    if normalized_action is None:
        return None
    # TODO (isaac): can read action names from robot.action_features
    normalized_action_dict = get_action_dictionary(normalized_action, ACTION_NAMES)
    motor_action_dict = {
        motor: normalized_action_to_motor_action(normalized_action, motor)
        for motor, normalized_action in normalized_action_dict.items()}
    return motor_action_dict


def wait_for_frame_time(loop_start: float, frame_time_s: float):
    """Wait for the frame time to pass."""
    dt_s = time.perf_counter() - loop_start
    busy_wait(frame_time_s - dt_s)


def setup_robot(cfg: PlaybackConfig, num_retries: int = 10):
    """Connect to the robot."""
    for _ in range(num_retries):
        try:
            robot: so101_follower.SO101Follower = (
                FakeRobot(cfg.robot) if cfg.use_fake_robot else make_robot_from_config(cfg.robot))
            robot.connect()
            configure_servos(robot, cfg)
            send_standby(robot)
            return robot
        except Exception as e:
            print(f"Failed to connect to robot: {type(e)}: {e}")
            time.sleep(1)
    raise RuntimeError(f"Failed to connect to robot after {num_retries} retries.")

def playback_on_robot(actions: npt.NDArray[np.float32], robot: so101_follower.SO101Follower, cfg: PlaybackConfig):
    """Playback actions on given robot."""
    num_frames = 0
    acceleration_dict = {joint: cfg.acceleration_limit for joint in robot.bus.motors}
    acceleration_dict['gripper'] = cfg.gripper_acceleration_limit
    speed_dict = {joint: cfg.velocity_limit for joint in robot.bus.motors}
    speed_dict['gripper'] = cfg.gripper_velocity_limit
    for _, right_action in actions:
        loop_start = time.perf_counter()
        # TODO (isaac): multi-arm support, for now only sending right action
        # left_action = prepare_action(left_action, robot)
        action = prepare_action(right_action, robot)

        print(f"Frame {num_frames}")
        num_frames += 1

        if action is None:
            wait_for_frame_time(loop_start, 1 / cfg.fps)
            continue

        # TODO (isaac): always sending right action for now
        robot.bus.sync_write("Acceleration", acceleration_dict)
        robot.bus.sync_write("Goal_Velocity", speed_dict)
        robot.send_action(action)
        wait_for_frame_time(loop_start, 1 / cfg.fps)


def playback_in_simulator(actions: npt.NDArray[np.float32], simulation: DualArmTeleopSimulation,
                          fps: int):
    """Playback actions in simulator."""
    num_frames = 0
    for action in actions:
        loop_start = time.perf_counter()

        left_action = list(action[0]) if not np.any(np.isnan(action[0])) else None
        right_action = list(action[1]) if not np.any(np.isnan(action[1])) else None

        print(f"Frame {num_frames}")
        num_frames += 1

        simulation.control_arms(
            left_control_data=left_action,
            right_control_data=right_action)

        # Handle keyboard events for dynamic interaction
        simulation.handle_keyboard_events()

        # Step simulation
        p.stepSimulation()

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

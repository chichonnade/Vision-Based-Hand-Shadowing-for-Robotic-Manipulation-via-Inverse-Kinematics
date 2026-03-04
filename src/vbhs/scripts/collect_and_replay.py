"""Record human demos and replay on robot.

Usage:
    sudo python -m vbhs.scripts.collect_and_replay \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem58FA0958041 \
        --robot.id=right_follower \
        --verbosity 1  # Optional: 0=WARNING (default), 1=INFO, 2=DEBUG
"""
import threading
import time
import os

import numpy as np
import numpy.typing as npt
import draccus
import logging
from pprint import pformat
from dataclasses import asdict
from datetime import datetime
import pybullet as p
# TODO (isaac): figure out why this is needed.
from lerobot.robots import (  # noqa: F401
    make_robot_from_config,
    so101_follower,
)

from vbhs.scripts import action_playback
from vbhs.scripts import record_from_glasses
from vbhs.simulation import simulator
from vbhs.pipeline import actions_pipeline
from vbhs.pipeline import camera_input  # TODO (isaac): seems like it should be moved.
from vbhs.utils import logging_config


def record_bag_file(bag_output_path: str, fps: int) -> bool:
    """Record a bag file from RealSense camera.

    Args:
        bag_output_path: Path where the bag file will be saved
        mp4_output_dir: Directory to save extracted actions
        fps: Frame rate at which to record.

    Returns:
        True if recording was successful, False otherwise
    """
    pipeline = actions_pipeline.HumanToRobotPipeline(
        simulation=None,
        source_type=camera_input.SourceType.LIVE,
        force_hand='right',
        align_frames=True,  # This is necessary for hand landmark detection.
        operation_mode=actions_pipeline.OperationMode.RECORD_BAG,
        bag_output_path=bag_output_path,
        fps=fps)

    done = threading.Event()
    def wait_for_enter():
        input()
        done.set()
    threading.Thread(target=wait_for_enter, daemon=True).start()

    print("✅ Recording started. Press [Enter] to stop recording...")
    while not done.is_set():
        if not pipeline.process_single_frame():
            return False
        # TODO (isaac): Magic number.
        time.sleep(1./240.)

    pipeline.cleanup()
    return True

def process_bag_to_actions(bag_path: str, actions_dir: str, mp4_output_dir: str,
                           cfg: action_playback.PlaybackConfig) -> str:
    """Process bag file to extract robot actions using TeleopPipeline.

    Args:
        bag_path: Path to the recorded bag file
        actions_dir: Directory to save extracted actions
        cfg: Playback configuration
        mp4_output_dir: Directory to save extracted mp4s

    Returns:
        Path to the generated actions file

    Raises:
        RuntimeError: If no actions file is generated
    """
    # Initialize simulation for processing
    simulation = simulator.DualArmTeleopSimulation(
        urdf_path=cfg.urdf,
        use_gui=False,  # No GUI needed for processing
        left_end_effector=6,
        right_end_effector=13
    )

    # Initialize pipeline with bag file and export actions
    pipeline = actions_pipeline.HumanToRobotPipeline(
        simulation=simulation,
        source_type=camera_input.SourceType.BAG,
        bag_file=bag_path,
        export_actions_dir=actions_dir,
        realtime_playback=False,
        loop_bag=False,  # Process once
        force_hand='right',  # TODO (isaac): make this configurable
        fps=cfg.fps,
        mp4_output_dir=mp4_output_dir
    )

    print('Processing bag file to extract actions...')
    try:
        frame_count = 0
        while True:
            if not pipeline.process_single_frame():
                break
            frame_count += 1
            if frame_count % 100 == 0:
                print(f'Processed {frame_count} frames...')

            # Step simulation
            p.stepSimulation()

            # Maintain real-time simulation speed
            time.sleep(1./240.)

        print(f'✅ Processed {frame_count} frames total.')

    except KeyboardInterrupt:
        print('⚠️  Processing interrupted by user')
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'❌ Error during processing: {e}')
        raise
    finally:
        pipeline.cleanup()
        simulation.cleanup()

    # Find the generated actions file
    actions_files = [f for f in os.listdir(actions_dir) if f.endswith('_actions.npy')]
    if not actions_files:
        raise RuntimeError("No actions file generated during processing")

    actions_file = os.path.join(actions_dir, actions_files[0])
    print(f'📄 Actions saved to: {actions_file}')
    return actions_file


def playback_actions_in_simulator(
        actions: npt.NDArray[np.float32], cfg: action_playback.PlaybackConfig):
    """Load and playback actions in PyBullet simulator.

    Args:
        actions: Actions to playback
        cfg: Playback configuration
    """
    print('Replaying actions in simulator...')
    print(f'Loaded {len(actions)} action frames.')

    # Create simulation for playback
    simulation = simulator.DualArmTeleopSimulation(
        urdf_path=cfg.urdf,
        use_gui=True,
        left_end_effector=6,
        right_end_effector=13
    )

    try:
        action_playback.playback_in_simulator(actions, simulation, cfg.simulator_fps)
        print("✅ Simulator playback completed")
    except KeyboardInterrupt:
        print('⚠️  Simulation aborted by user')
        return
    finally:
        simulation.cleanup()


def playback_actions_on_robot(actions: npt.NDArray[np.float32], mp4_output_dir: str, cfg: action_playback.PlaybackConfig):
    """Load and playback actions on the physical robot.

    Args:
        actions: Actions to playback
        mp4_output_dir: Directory to save extracted actions
        cfg: Playback configuration
    """
    print('Connecting to robot...')

    # Initialize robot
    robot = action_playback.setup_robot(cfg)

    print('Connected.')
    try:
        while True:
            action_playback.send_standby(robot)
            # Start video recording
            recorder = record_from_glasses.GlassesRecorder(output_directory=mp4_output_dir, fps=cfg.fps)
            with recorder:
                if not recorder.begin_recording():
                    print("❌ Failed to start video recording, continuing without recording...")
                else:
                    print("✅ Video recording started successfully")
                action_playback.playback_on_robot(actions, robot, cfg)
            action_playback.send_standby(robot)
            time.sleep(1.0)
            # Highlight yes to imply it is the default.
            print('✅ Robot playback completed.')
            if input('Would you like to repeat the demo? [(y)/n]: ') == 'n':
                break
    except KeyboardInterrupt:
        print('⚠️  Robot playback aborted by user')
        raise
    finally:
        action_playback.send_standby(robot)
        robot.disconnect()


def setup_output_directories(output_dir: str) -> tuple[str, str]:
    """Set up output directories and generate filenames.

    Args:
        cfg: Playback configuration

    Returns:
        Tuple of (bag_path, actions_dir)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    actions_dir = os.path.join(output_dir, "actions")
    os.makedirs(actions_dir, exist_ok=True)

    # Generate unique filenames with timestamp
    bag_path = os.path.join(output_dir, 'human_demo.bag')

    return bag_path, actions_dir

@draccus.wrap()
def collect_and_replay(cfg: action_playback.PlaybackConfig):
    """Record human demos, replay on simulator, then on robot."""
    logging_config.configure_logging(cfg.verbosity)
    logging.info(pformat(asdict(cfg)))

    print("Starting collect and replay pipeline...")
    # Save output in timestamped directory inside output_dir.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, timestamp)
    print(f'Output directory: {output_dir}')

    # Setup output directories and paths.
    bag_path, actions_dir = setup_output_directories(output_dir)

    try:
        # Step 1: Record human demonstration and save to bag file.
        input('Press [Enter] to begin recording human demonstration...')
        # TODO (isaac): add special config for recording framerate.
        bag_recorded = record_bag_file(bag_path, cfg.fps)
        if not bag_recorded:
            print('❌ Failed to record bag file. Exiting.')
            return
        print(f'✅ Bag file saved as: {bag_path}')

        # Step 2: Process bag file to extract actions.
        actions_file = process_bag_to_actions(bag_path, actions_dir, output_dir, cfg)
        cfg.actions_file = actions_file
        actions = cfg.get_actions()

        # Step 3: Replay actions in simulator.
        input('Press [Enter] to playback in simulator...')
        playback_actions_in_simulator(actions, cfg)

        # Step 4: Replay actions on robot.
        input('Press [Enter] to playback on robot...')
        playback_actions_on_robot(actions, output_dir, cfg)

        # Success summary
        print(f'Bag file saved as: {bag_path}')
        print(f'Actions file saved as: {actions_file}')
        print(f'Videos saved to: {output_dir}')

    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\n❌ Pipeline terminated: {e}")
        return


if __name__ == "__main__":
    # This is called by draccus which provides the cfg argument automatically
    collect_and_replay()  # pylint: disable=no-value-for-parameter

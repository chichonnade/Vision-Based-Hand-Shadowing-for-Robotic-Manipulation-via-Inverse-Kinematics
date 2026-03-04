"""Playback actions from file on robot.

Usage:
    python -m lerobot.vbhs.playback_on_robot \
        --actions_file path/to/actions.npy \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem58FA0958041 \
        --robot.id=right_follower \
        --verbosity 1  # Optional: 0=WARNING (default), 1=INFO, 2=DEBUG
"""
import draccus
import logging
from pprint import pformat
from dataclasses import asdict
# TODO (isaac): figure out why this is needed.
from lerobot.robots import (  # noqa: F401
    make_robot_from_config,
    so101_follower,
)

from vbhs.scripts import action_playback
from vbhs.scripts import record_from_glasses
from vbhs.simulation import simulator
from vbhs.utils import logging_config


@draccus.wrap()
def playback(cfg: action_playback.PlaybackConfig):
    """Playback actions from file on simulator then robot."""
    logging_config.configure_logging(cfg.verbosity)
    logging.info(pformat(asdict(cfg)))

    robot = action_playback.setup_robot(cfg)

    actions = cfg.get_actions()

    simulation = simulator.DualArmTeleopSimulation(
        urdf_path=cfg.urdf,
        use_gui=True,
        left_end_effector=6,
        right_end_effector=13)

    try:
        action_playback.playback_in_simulator(actions, simulation, cfg.simulator_fps)
    except KeyboardInterrupt:
        print('Simulation aborted. Replaying actions.')
    try:
        # Wait until user presses [enter] to playback on robot.
        input('Press [enter] to playback on robot...')
        recorder = record_from_glasses.GlassesRecorder(output_directory=cfg.output_dir, fps=cfg.fps)
        with recorder:
            if not recorder.begin_recording():
                print("❌ Failed to start video recording, continuing without recording...")
            else:
                print("✅ Video recording started successfully")
            action_playback.playback_on_robot(actions, robot, cfg)
    except KeyboardInterrupt:
        print('Playback on robot aborted. Shutting down.')
    action_playback.send_standby(robot)
    robot.disconnect()

if __name__ == "__main__":
    # This is called by draccus which provides the cfg argument automatically
    playback()  # pylint: disable=no-value-for-parameter

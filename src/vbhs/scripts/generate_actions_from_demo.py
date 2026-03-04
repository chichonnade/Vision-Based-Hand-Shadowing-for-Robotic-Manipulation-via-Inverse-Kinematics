"""Generate robot control commands from human demonstration, either live or from a bag file.

Usage:
    python -m vbhs.scripts.generate_actions_from_demo --urdf robot/Dual_S101_Assembly.urdf
    python -m vbhs.scripts.generate_actions_from_demo --urdf robot/Dual_S101_Assembly.urdf --bag-file data.bag
    python -m vbhs.scripts.generate_actions_from_demo --urdf robot/Dual_S101_Assembly.urdf -v 1  # INFO logging
"""

import argparse
import os
from typing import Optional
import time
import traceback

import pybullet as p

from vbhs.simulation import simulator
from vbhs.pipeline import actions_pipeline
from vbhs.pipeline import camera_input
from vbhs.utils import logging_config

def main():
    """Generate robot control commands from human demonstration, either live or from a bag file."""
    parser = argparse.ArgumentParser(description='Dual Arm Control from Demonstration')
    parser.add_argument('--urdf', type=str,
                        default='robot/Dual_S101_Assembly.urdf',
                        help='Path to URDF file of robot being controlled')
    parser.add_argument('--bag-file', type=str, default=None,
                        help='Path to bag file for recorded data playback')
    parser.add_argument('--export-actions', type=str, default=None,
                        help='Directory to export robot actions to. '
                             'Disables looping and requires a bag file.')
    parser.add_argument('--detection-confidence', type=float, default=0.5,
                        help='Hand detection confidence threshold')
    parser.add_argument('--tracking-confidence', type=float, default=0.5,
                        help='Hand tracking confidence threshold')
    parser.add_argument('--max-frames', type=int,
                        help='Maximum number of frames to process')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run simulation without GUI')
    parser.add_argument('--no-align', action='store_true',
                        help='Disable depth frame alignment')
    parser.add_argument('--display-landmarks', action='store_true',
                        help='Display hand landmarks in a separate window')
    parser.add_argument('--enable-mesh-visualization', action='store_true',
                        help='Display 3D hand mesh visualization (WiLoR only)')
    parser.add_argument('--realtime', action='store_true',
                        help='Set bag playback to real time')
    parser.add_argument('--loop', action='store_true',
                        help='Whether to loop the bag file')
    parser.add_argument('--force-hand', type=str, default=None, choices=['left', 'right'],
                        help='Force single detected hand to be classified as left or right.')
    parser.add_argument('--bag-output-path', type=str, default=None,
                        help='Path to save recorded bag file.')
    parser.add_argument('--mp4-output-dir', type=str, default=None,
                        help='Directory to save RGB and depth MP4 recordings')
    parser.add_argument('--perf-stats', action='store_true',
                        help='Print performance statistics')
    parser.add_argument('-v', '--verbosity', type=int, default=0,
                        help='Logging verbosity (0=WARNING, 1=INFO, 2=DEBUG)')
    args = parser.parse_args()

    # Configure logging based on verbosity level
    logging_config.configure_logging(args.verbosity)

    if args.bag_output_path and args.bag_file:
        parser.error("--bag-output-path and --bag-file cannot be used together.")

    if args.export_actions and not args.bag_file:
        parser.error("--export-actions requires --bag-file to be specified.")

    if args.loop and not args.bag_file:
        parser.error("--loop requires --bag-file to be specified.")

    if args.export_actions and args.loop:
        parser.error("--export-actions and --loop cannot be used together.")

    operation_mode = (actions_pipeline.OperationMode.HUMAN_TO_ROBOT if args.bag_output_path is None
                      else actions_pipeline.OperationMode.RECORD_BAG)

    source_type = (camera_input.SourceType.LIVE if args.bag_file is None
                   else camera_input.SourceType.BAG)

    print_performance_stats: bool = args.perf_stats

    if args.bag_file:
        if not os.path.exists(args.bag_file):
            raise FileNotFoundError(f"Bag file {args.bag_file} does not exist.")

    # Initialize simulation
    simulation: Optional[simulator.DualArmTeleopSimulation] = None
    if operation_mode != actions_pipeline.OperationMode.RECORD_BAG:
        simulation = simulator.DualArmTeleopSimulation(
            urdf_path=args.urdf,
            left_end_effector=6,
            right_end_effector=13,
            use_gui=not args.no_gui
        )

    # Initialize pipeline after simulation is set up
    pipeline = actions_pipeline.HumanToRobotPipeline(
        simulation=simulation,
        source_type=source_type,
        bag_file=args.bag_file,
        align_frames=not args.no_align,
        display_landmarks=args.display_landmarks,
        realtime_playback=args.realtime,
        export_actions_dir=args.export_actions,
        loop_bag=args.loop,
        force_hand=args.force_hand,
        operation_mode=operation_mode,
        bag_output_path=args.bag_output_path,
        mp4_output_dir=args.mp4_output_dir,
        enable_mesh_visualization=args.enable_mesh_visualization
    )

    try:
        # Main control loop
        while True:
            if not pipeline.process_single_frame():
                break  # Exit on error or end of data

            if pipeline.frame_count % 100 == 0 and print_performance_stats:
                pipeline.print_performance_stats()

            if simulation is not None:
                # Handle keyboard events for dynamic interaction
                simulation.handle_keyboard_events()

                # Step simulation
                p.stepSimulation()  # pylint: disable=c-extension-no-member

            # Maintain real-time simulation speed
            # TODO (isaac): Magic number.
            time.sleep(1./240.)

    except KeyboardInterrupt:
        print("🛑 User interrupted, shutting down...")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        # Clean up resources
        pipeline.cleanup()
        if simulation is not None:
            simulation.cleanup()

if __name__ == '__main__':
    main()

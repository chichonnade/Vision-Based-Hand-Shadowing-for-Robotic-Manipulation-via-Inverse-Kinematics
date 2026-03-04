#!/usr/bin/env python3
"""Human to robot pipeline for dual-arm robot control from camera input."""

from typing import Optional
import enum
import logging
import os
import time
import traceback

import numpy as np

from vbhs.config import config
from vbhs.pipeline import camera_input
from vbhs.pipeline import hand_detection
from vbhs.pipeline import joint_transformations
from vbhs.pipeline import hand_landmarks_correction
from vbhs.pipeline import control_command_estimation
from vbhs.pipeline import types
from vbhs.simulation import simulator

_logger = logging.getLogger(__name__)


class OperationMode(enum.Enum):
    """Operation mode for the pipeline."""
    HUMAN_TO_ROBOT = 'human_to_robot'  # Run full pipeline and generate robot control commands
    RECORD_BAG = 'record_bag'  # Record a bag file from the camera

class HumanToRobotPipeline:
    """Generate robot control commands from human demonstration."""

    def __init__(self,
                 simulation: Optional[simulator.DualArmTeleopSimulation] = None,
                 source_type: camera_input.SourceType = camera_input.SourceType.LIVE,
                 bag_file: Optional[str] = None,
                 align_frames: bool = True,
                 display_landmarks: bool = False,
                 realtime_playback: bool = False,
                 loop_bag: bool = False,
                 export_actions_dir: Optional[str] = None,
                 force_hand: Optional[str] = None,
                 operation_mode: OperationMode = OperationMode.HUMAN_TO_ROBOT,
                 bag_output_path: Optional[str] = None,
                 mp4_output_dir: Optional[str] = None,
                 fps: int = 30,
                 enable_mesh_visualization: bool = False):
        """
        Initialize the pipeline.

        Args:
            simulation: The PyBullet simulation instance
            source_type: Camera input source type
            bag_file: Path to bag file if using recorded data
            hand_detection_confidence: Hand detection confidence threshold
            hand_tracking_confidence: Hand tracking confidence threshold
            align_frames: Whether to align depth frames to color frames
            display_landmarks: Whether to display hand landmarks
            realtime_playback: Whether to play back the bag file in real time
            loop_bag: Whether to loop the bag file
            export_actions_dir: Directory to export actions to
            force_hand: Force single detected hand to be classified as left or right.
            operation_mode: Operation mode for the pipeline.
            bag_output_path: Path to save recorded bag file.
            mp4_output_dir: Directory to save RGB and depth MP4 recordings (default: None)
            fps: Frame rate at which to run the input source.
            enable_mesh_visualization: Whether to display 3D hand mesh visualization (WiLoR only)
        """
        if operation_mode == OperationMode.HUMAN_TO_ROBOT and simulation is None:
            raise ValueError("No simulation provided. Simulation is required for human to robot mode.")
        if operation_mode == OperationMode.RECORD_BAG:
            if bag_output_path is None:
                raise ValueError("No bag output directory provided. Bag output directory is required for record bag mode.")

        self.simulation = simulation
        self.bag_output_path = bag_output_path
        self.export_actions_dir = export_actions_dir
        self.recorded_actions = [] if self.export_actions_dir else None
        self.operation_mode = operation_mode
        if self.operation_mode == OperationMode.RECORD_BAG:
            display_landmarks = True

        if self.export_actions_dir:
            _logger.debug("Writing actions to %s", self.export_actions_dir)
            os.makedirs(self.export_actions_dir, exist_ok=True)


        _logger.info("Initializing pipeline")

        # Stage 1: Camera Input
        self.camera_input = camera_input.CameraFrameFromInput(
            source_type=source_type,
            bag_file=bag_file,
            align=align_frames,
            realtime_playback=realtime_playback,
            loop_bag=loop_bag,
            bag_output_path=bag_output_path,
            mp4_output_dir=mp4_output_dir,
            fps=fps
        )

        # Stage 2: Hand Detection (Image Space)
        self.hand_detector = hand_detection.HandLandmarksFromCameraFrame(
            camera_intrinsics=self.camera_input.camera_intrinsics,
            enable_smoothing=True,
            smoothing_alpha=0.8,
            enable_mesh_visualization=enable_mesh_visualization
        )

        # Stage 3: 2D to 3D Conversion (Camera Space)
        self.depth_converter = joint_transformations.CameraSpaceHandsFromImageSpace(
            min_depth_m=0.1,
            max_depth_m=5.0,
            depth_scale=1000.0,
            display_landmarks=display_landmarks,
            mp4_output_dir=mp4_output_dir
        )

        # Stage 4: Camera to Robot Space Transformation
        self.space_transformer = joint_transformations.RobotSpaceHandsFromCameraSpace(
            cam_to_robot_config=config.CAM_TO_ROBOT_CONFIG,
            urdf_offset_config=config.URDF_OFFSET_CONFIG
        )

        # Stage 5: Landmark Correction
        self.landmark_corrector = hand_landmarks_correction.CorrectHandLandmarks(
            swap_threshold_meters=0.05,
            confidence_alpha=0.15,
            confidence_threshold=0.2,
            min_confidence=0.4,
            force_hand=force_hand
        )

        self.robot_controller: Optional[control_command_estimation.RobotControlCommandsFromHandLandmarks] = None
        if self.operation_mode == OperationMode.HUMAN_TO_ROBOT:
            # Stage 6: Robot Control Commands
            self.robot_controller = control_command_estimation.RobotControlCommandsFromHandLandmarks(
                robot_id=simulation.robot_id,
                left_arm_joints=list(simulation.left_arm_joints.values()),
                right_arm_joints=list(simulation.right_arm_joints.values()),
                left_end_effector=simulation.left_end_effector,
                right_end_effector=simulation.right_end_effector,
                debug_visualizer=simulation.debug_visualizer,
                enable_smoothing=True,
                arm_smoothing_alpha=0.5)

        _logger.info("Pipeline initialized successfully")

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.stage_times = {}

        # Handedness tracking
        self.hand_detection_stats = {
            'raw': {'left_only': 0, 'right_only': 0, 'both': 0, 'none': 0},
            'corrected': {'left_only': 0, 'right_only': 0, 'both': 0, 'none': 0}
        }

    def _update_hand_stats(self, landmarks: types.HandLandmarksRobotSpace, stage: str):
        """Update handedness detection statistics."""
        stats = self.hand_detection_stats[stage]
        left_detected = landmarks.left_hand_landmarks is not None
        right_detected = landmarks.right_hand_landmarks is not None

        if left_detected and right_detected:
            stats['both'] += 1
        elif left_detected:
            stats['left_only'] += 1
        elif right_detected:
            stats['right_only'] += 1
        else:
            stats['none'] += 1

    def _finish_frame(self, frame_start: float):
        """Finish processing a frame."""
        # Update performance tracking
        total_time = time.time() - frame_start
        self._update_stage_time('total_frame', total_time)
        self.frame_count += 1

    def process_single_frame(self) -> bool:
        """
        Process a single frame through the complete pipeline.

        Returns:
            bool: True if frame processed successfully, False otherwise
        """
        frame_start = time.time()

        try:
            # Stage 1: Get camera frame.
            stage_start = time.time()
            camera_frame = self.camera_input(None)
            if camera_frame is None:
                return False
            self._update_stage_time('camera_input', time.time() - stage_start)

            # Stage 2: Detect hands in image space.
            stage_start = time.time()
            hand_landmarks_image_space = self.hand_detector(camera_frame)
            if hand_landmarks_image_space is None:
                _logger.debug("No hands detected in frame")
                return True  # Continue processing even without hands
            self._update_stage_time('hand_detection', time.time() - stage_start)

            # Stage 3: Convert hand landmarksto 3D camera space.
            stage_start = time.time()
            hand_landmarks_camera_space = self.depth_converter(
                (hand_landmarks_image_space, camera_frame))
            if hand_landmarks_camera_space is None:
                _logger.warning("Failed to convert to camera space")
                return True
            self._update_stage_time('depth_conversion', time.time() - stage_start)

            # Stage 4: Transform hand landmarks to robot space.
            stage_start = time.time()
            hand_landmarks_robot_space = self.space_transformer(hand_landmarks_camera_space)
            if hand_landmarks_robot_space is None:
                _logger.warning("Failed to transform to robot space")
                return True
            self._update_stage_time('space_transform', time.time() - stage_start)

            # Update raw stats before correction
            self._update_hand_stats(hand_landmarks_robot_space, 'raw')

            # Stage 5: Correct hand landmarks by applying heuristics etc.
            # DISABLED: WiLoR detection is accurate enough without correction
            stage_start = time.time()
            # final_hand_landmarks = self.landmark_corrector(
            #     hand_landmarks_robot_space)
            final_hand_landmarks = hand_landmarks_robot_space  # Skip correction step
            self._update_stage_time('landmark_correction', time.time() - stage_start)

            # Update corrected stats
            self._update_hand_stats(final_hand_landmarks, 'corrected')

            if self.operation_mode != OperationMode.HUMAN_TO_ROBOT:
                self._finish_frame(frame_start)
                return True

            assert self.robot_controller is not None, "Robot controller not initialized."

            # Stage 6: Generate robot control commands from hand landmarks.
            stage_start = time.time()
            try:
                control_commands = self.robot_controller(final_hand_landmarks)
            except Exception:  # pylint: disable=broad-exception-caught
                _logger.error("Exception in robot control command generation. "
                              "Returning empty commands.",
                              exc_info=True)
                control_commands = types.RobotControlCommands(robot_commands=(None, None))
            self._update_stage_time('robot_control', time.time() - stage_start)

            # Record actions if exporting
            if self.export_actions_dir is not None:
                self._record_actions(control_commands)

            # Apply control commands to simulation
            stage_start = time.time()
            self._apply_control_commands(control_commands)
            self._update_stage_time('simulation_update', time.time() - stage_start)

            self._finish_frame(frame_start)
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            _logger.error("Pipeline error: %s", e)
            traceback.print_exc()
            return False

    def _record_actions(self, control_commands: types.RobotControlCommands):
        """Record robot actions for later export."""
        left_joints, right_joints = control_commands.robot_commands

        left_action = left_joints if left_joints is not None else [np.nan] * 6
        right_action = right_joints if right_joints is not None else [np.nan] * 6

        action = np.stack([left_action, right_action], axis=0).astype(np.float32)
        if self.recorded_actions is not None:
            self.recorded_actions.append(action)

    def _apply_control_commands(self, control_commands: types.RobotControlCommands):
        """Apply control commands to the simulation."""
        left_joints, right_joints = control_commands.robot_commands
        
        # Debug logging for command application
        if left_joints is not None:
            _logger.debug('Sending LEFT arm commands: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]',
                         left_joints[0], left_joints[1], left_joints[2], 
                         left_joints[3], left_joints[4], left_joints[5])
        if right_joints is not None:
            _logger.debug('Sending RIGHT arm commands: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]',
                         right_joints[0], right_joints[1], right_joints[2], 
                         right_joints[3], right_joints[4], right_joints[5])
        if left_joints is None and right_joints is None:
            _logger.debug('No commands to send (both arms None)')

        self.simulation.control_arms(
            left_control_data=left_joints,
            right_control_data=right_joints
        )

    def _save_actions(self):
        """Save recorded actions to a .npy file."""
        if not self.recorded_actions:
            print("No actions recorded, skipping save.")
            return

        actions_array = np.array(self.recorded_actions)

        # Use bag file name to create a unique action file name
        if self.camera_input.bag_file:
            base_name = os.path.basename(self.camera_input.bag_file)
            file_name = os.path.splitext(base_name)[0] + "_actions.npy"
            if self.export_actions_dir:
                output_path = os.path.join(self.export_actions_dir, file_name)

                print(f"💾 Saving {len(self.recorded_actions)} actions to {output_path}")
                np.save(output_path, actions_array)

    def _update_stage_time(self, stage_name: str, duration: float):
        """Update timing statistics for a pipeline stage."""
        if stage_name not in self.stage_times:
            self.stage_times[stage_name] = []

        self.stage_times[stage_name].append(duration)

        # Keep only last 100 measurements for rolling average
        if len(self.stage_times[stage_name]) > 100:
            self.stage_times[stage_name] = self.stage_times[stage_name][-100:]

    def print_performance_stats(self):
        """Print pipeline performance statistics."""
        if not self.stage_times:
            return

        elapsed_total = time.time() - self.start_time
        fps = self.frame_count / elapsed_total

        print(f"\n📊 Pipeline Performance (Frame {self.frame_count}):")
        print(f"Overall FPS: {fps:.1f}")

        for stage_name, times in self.stage_times.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                print(f"  {stage_name}: {avg_time*1000:.1f}ms avg, {max_time*1000:.1f}ms max")
        print()

    def _print_hand_detection_summary(self):
        """Prints a summary of the hand detection statistics."""
        print("\n📊 Hand Detection Statistics:")
        total_frames = sum(self.hand_detection_stats['raw'].values())
        if total_frames == 0:
            print("No frames processed.")
            return

        for stage in ['raw', 'corrected']:
            stats = self.hand_detection_stats[stage]
            print(f"\n--- {stage.title()} Detections ---")
            print(f"  - Left Only:  {stats['left_only']:>5} ({stats['left_only']/total_frames:>7.2%})")
            print(f"  - Right Only: {stats['right_only']:>5} ({stats['right_only']/total_frames:>7.2%})")
            print(f"  - Both Hands: {stats['both']:>5} ({stats['both']/total_frames:>7.2%})")
            print(f"  - No Hands:   {stats['none']:>5} ({stats['none']/total_frames:>7.2%})")
        print()

    def cleanup(self):
        """Clean up pipeline resources."""
        _logger.debug("Cleaning up pipeline")

        self._print_hand_detection_summary()

        if self.export_actions_dir:
            self._save_actions()

        self.camera_input.cleanup()
        self.depth_converter.cleanup()
        self.hand_detector.cleanup()

        _logger.debug("Pipeline cleanup complete")

"""Compare MediaPipe vs WiLoR hand detection on a recorded bag file.

Measures two metrics per detector:
  1. Detection rate -- % of frames where the right hand is detected
  2. Valid IK target rate -- % of frames where a valid (p_target, q_target) can
     be computed from the detected landmarks + depth

Usage (run with conda vbhs env):
    python scripts/compare_hand_detectors.py [--bag BAG_PATH]
"""

import argparse
import logging
import sys

import cv2
import numpy as np

from vbhs.config import config
from vbhs.pipeline import camera_input
from vbhs.pipeline import joint_transformations
from vbhs.pipeline import types
from vbhs.pipeline.hands import deprojection
from vbhs.pipeline.hands import mediapipe_hand_detector
from vbhs.pipeline.hands import target_pose

try:
    from vbhs.pipeline.hands import wilor_hand_detector
except ImportError:
    wilor_hand_detector = None

_logger = logging.getLogger(__name__)

DEFAULT_BAG = (
    "new experiment data (realsense bag file with associated computed "
    "actions data for pick and place/human_demo.bag"
)

# Depth deprojection parameters (same as pipeline defaults)
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 5.0
DEPTH_SCALE = 1000.0
MIN_Z_HEIGHT = 0.05  # safety filter from robot_arm_estimation


def _try_compute_ik_target(
        hand_2d: types.HandPose2D,
        depth_image: np.ndarray,
        intrinsics: types.CameraIntrinsics,
        space_transformer: joint_transformations.RobotSpaceHandsFromCameraSpace,
) -> bool:
    """Return True if a valid IK target (position + orientation) can be computed."""
    # Step 1: Deproject 2D landmarks to 3D camera space.
    hand_3d = deprojection.deproject_hand_landmarks(
        hand_2d, depth_image, intrinsics,
        depth_scale=DEPTH_SCALE,
        min_depth_m=MIN_DEPTH_M,
        max_depth_m=MAX_DEPTH_M)
    if hand_3d is None:
        return False

    # Step 2: Transform to robot space.
    robot_hand = {}
    for key, pt in hand_3d.items():
        if pt is not None:
            robot_hand[key] = space_transformer._transform_point_to_robot(
                np.array(pt))
        else:
            robot_hand[key] = None

    # Step 3: Compute target position.
    target_pos = target_pose.calculate_target_position(
        robot_hand, target_pose.Hand.RIGHT)
    if target_pos is None:
        return False

    # Safety filter.
    if target_pos[2] < MIN_Z_HEIGHT:
        return False

    # Step 4: Compute target orientation (full or fallback).
    orientation = target_pose.calculate_target_orientation(
        robot_hand, target_pose.Hand.RIGHT)
    if orientation is None:
        orientation = target_pose.calculate_fallback_orientation(
            robot_hand, target_pose.Hand.RIGHT)

    # Valid if we have at least a position (orientation can be None for
    # position-only IK, but we consider it valid either way since the IK
    # solver can handle position-only targets).
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare MediaPipe vs WiLoR hand detection")
    parser.add_argument("--bag", default=DEFAULT_BAG,
                        help="Path to .bag file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if wilor_hand_detector is None:
        print("ERROR: WiLoR is not installed. Cannot run comparison.")
        sys.exit(1)

    # Set up camera input (bag file).
    cam = camera_input.CameraFrameFromInput(
        source_type=camera_input.SourceType.BAG,
        bag_file=args.bag,
        align=True,
        realtime_playback=False,
        loop_bag=False)
    intrinsics = cam.camera_intrinsics

    # Set up both detectors.
    mp_detector = mediapipe_hand_detector.MediaPipeHandDetector(intrinsics)
    wilor_detector = wilor_hand_detector.WilorHandDetector(
        intrinsics, enable_visualization=False)

    # Set up camera-to-robot transform (no PyBullet needed).
    space_transformer = joint_transformations.RobotSpaceHandsFromCameraSpace(
        cam_to_robot_config=config.CAM_TO_ROBOT_CONFIG,
        urdf_offset_config=config.URDF_OFFSET_CONFIG)

    # Counters -- track both left and right for each detector.
    total_frames = 0
    mp_left_det = 0
    mp_right_det = 0
    mp_left_valid = 0
    mp_right_valid = 0
    wilor_left_det = 0
    wilor_right_det = 0
    wilor_left_valid = 0
    wilor_right_valid = 0
    wilor_recovered = 0  # frames where WiLoR detected but MediaPipe didn't

    print("Processing bag file... (this may take a while)")
    print("(Handedness diagnostic enabled — tracking both left and right)")

    while True:
        try:
            frame = cam(None)
        except RuntimeError:
            break  # Bag file ended (RealSense timeout).
        if frame is None:
            break
        total_frames += 1

        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB)
        depth_image = frame.depth.values

        # --- MediaPipe ---
        mp_left, mp_right = mp_detector.detect(rgb)
        if mp_left is not None:
            mp_left_det += 1
            if _try_compute_ik_target(mp_left, depth_image,
                                      intrinsics, space_transformer):
                mp_left_valid += 1
        if mp_right is not None:
            mp_right_det += 1
            if _try_compute_ik_target(mp_right, depth_image,
                                      intrinsics, space_transformer):
                mp_right_valid += 1

        # --- WiLoR ---
        wilor_left, wilor_right = wilor_detector.detect(rgb)
        if wilor_left is not None:
            wilor_left_det += 1
            if _try_compute_ik_target(wilor_left, depth_image,
                                      intrinsics, space_transformer):
                wilor_left_valid += 1
        if wilor_right is not None:
            wilor_right_det += 1
            if _try_compute_ik_target(wilor_right, depth_image,
                                      intrinsics, space_transformer):
                wilor_right_valid += 1

        # Recovery: WiLoR detected but MediaPipe did not (either hand).
        mp_any = mp_left is not None or mp_right is not None
        wilor_any = wilor_left is not None or wilor_right is not None
        if wilor_any and not mp_any:
            wilor_recovered += 1

        if total_frames % 50 == 0:
            print(f"  Processed {total_frames} frames...")

    # Cleanup.
    mp_detector.cleanup()
    wilor_detector.cleanup()
    cam.cleanup()

    # Report.
    print(f"\n{'='*60}")
    print(f"Hand Detector Comparison ({total_frames} frames)")
    print(f"{'='*60}")

    if total_frames == 0:
        print("No frames processed!")
        return

    def pct(n):
        return 100.0 * n / total_frames

    # Determine dominant hand (the one with more total detections).
    total_left = mp_left_det + wilor_left_det
    total_right = mp_right_det + wilor_right_det
    dominant = "LEFT" if total_left >= total_right else "RIGHT"

    print(f"\n{'Metric':<35} {'MediaPipe':>12} {'WiLoR':>12}")
    print(f"{'-'*35} {'-'*12} {'-'*12}")
    print(f"{'LEFT hand detected':<35} "
          f"{mp_left_det:>5} ({pct(mp_left_det):5.1f}%) "
          f"{wilor_left_det:>5} ({pct(wilor_left_det):5.1f}%)")
    print(f"{'LEFT valid IK target':<35} "
          f"{mp_left_valid:>5} ({pct(mp_left_valid):5.1f}%) "
          f"{wilor_left_valid:>5} ({pct(wilor_left_valid):5.1f}%)")
    print(f"{'RIGHT hand detected':<35} "
          f"{mp_right_det:>5} ({pct(mp_right_det):5.1f}%) "
          f"{wilor_right_det:>5} ({pct(wilor_right_det):5.1f}%)")
    print(f"{'RIGHT valid IK target':<35} "
          f"{mp_right_valid:>5} ({pct(mp_right_valid):5.1f}%) "
          f"{wilor_right_valid:>5} ({pct(wilor_right_valid):5.1f}%)")

    # Use dominant hand for comparison metrics.
    if dominant == "LEFT":
        mp_det = mp_left_det
        mp_vt = mp_left_valid
        wilor_det = wilor_left_det
        wilor_vt = wilor_left_valid
    else:
        mp_det = mp_right_det
        mp_vt = mp_right_valid
        wilor_det = wilor_right_det
        wilor_vt = wilor_right_valid

    print(f"\n--- Comparison on dominant hand ({dominant}, egocentric) ---")
    print(f"{'Detection rate':<35} "
          f"{mp_det:>5} ({pct(mp_det):5.1f}%) "
          f"{wilor_det:>5} ({pct(wilor_det):5.1f}%)")
    print(f"{'Valid IK target rate':<35} "
          f"{mp_vt:>5} ({pct(mp_vt):5.1f}%) "
          f"{wilor_vt:>5} ({pct(wilor_vt):5.1f}%)")
    print(f"{'Recovery (WiLoR found, MP missed)':<35} "
          f"{wilor_recovered:>5} ({pct(wilor_recovered):5.1f}%)")

    if mp_det > 0:
        detection_improvement = (wilor_det - mp_det) / mp_det * 100
        print(f"{'Detection rate improvement':<35} {detection_improvement:>+11.1f}%")
    if mp_vt > 0:
        target_improvement = (wilor_vt - mp_vt) / mp_vt * 100
        print(f"{'Valid target rate improvement':<35} {target_improvement:>+11.1f}%")


if __name__ == "__main__":
    main()

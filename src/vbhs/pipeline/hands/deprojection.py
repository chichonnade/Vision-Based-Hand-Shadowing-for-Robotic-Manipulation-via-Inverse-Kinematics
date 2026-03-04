"""Deprojection of hand landmarks to 3D camera space."""
import logging
from typing import Optional

import numpy as np

from vbhs.pipeline import types

_logger = logging.getLogger(__name__)

class DeprojectionError(Exception):
    """Indicates an error occurred during deprojection."""

class DepthOutOfRange(DeprojectionError):
    """A depth value out of range was encountered."""
    def __init__(self, depth_m: float, min_depth_m: float, max_depth_m: float):
        super().__init__(f'Depth value {depth_m:.3f}m out of range '
                         f'({min_depth_m:.3f}-{max_depth_m:.3f})m')


def deproject_hand_landmarks(
        hand_landmarks: types.HandPose2D,
        depth_image: np.ndarray,
        intrinsics: types.CameraIntrinsics,
        depth_scale: float,
        min_depth_m: float,
        max_depth_m: float,
        percent_valid_landmarks: float = 0.5) -> Optional[types.HandPose3D]:
    """
    Deproject all 21 hand landmarks to 3D camera space.

    Args:
        hand_landmarks: List of 21 (u, v) coordinates in image space
        depth_image: Depth image array
        intrinsics: Camera intrinsics for deprojection
        depth_scale: Scale factor to convert depth values to meters TODO(isaac): rename this.
        min_depth_m: Minimum valid depth in meters
        max_depth_m: Maximum valid depth in meters
        percent_valid_landmarks: Percentage of landmarks that must be valid to return a valid pose

    Returns:
        List of 21 (x, y, z) coordinates in camera space or None if insufficient valid landmarks
    """
    landmarks_3d: types.HandPose3D = {}
    for key, landmark_uv in hand_landmarks.items():
        try:
            landmark_3d = _deproject_hand_landmark(
                landmark_uv, depth_image, intrinsics, depth_scale, min_depth_m, max_depth_m)
        except DepthOutOfRange as e:
            _logger.debug('Error deprojecting landmark %s: %s', key, e)
            landmark_3d = None
        except IndexError as e:
            _logger.debug('Error deprojecting landmark %s: %s', key, e)
            landmark_3d = None

        landmarks_3d[key] = landmark_3d

    # Fallback: If THUMB_MCP or INDEX_FINGER_MCP failed, use depth from the other
    _apply_depth_fallback(landmarks_3d, hand_landmarks, intrinsics, 
                         depth_scale, min_depth_m, max_depth_m)

    valid_count = sum(1 for landmark in landmarks_3d.values() if landmark is not None)
    if valid_count < len(hand_landmarks) * percent_valid_landmarks:
        # Check if critical landmarks for position calculation are valid
        thumb_mcp_valid = landmarks_3d.get('THUMB_MCP') is not None
        index_mcp_valid = landmarks_3d.get('INDEX_FINGER_MCP') is not None
        critical_valid = thumb_mcp_valid and index_mcp_valid
        
        _logger.debug(
            'Rejecting hand pose: only %d/%d landmarks valid (%.1f%% < %.1f%% threshold). '
            'Critical landmarks (THUMB_MCP, INDEX_MCP) valid: %s',
            valid_count, len(hand_landmarks),
            100.0 * valid_count / len(hand_landmarks),
            100.0 * percent_valid_landmarks,
            critical_valid)
        return None

    return landmarks_3d


def _apply_depth_fallback(
        landmarks_3d: types.HandPose3D,
        hand_landmarks_2d: types.HandPose2D,
        intrinsics: types.CameraIntrinsics,
        depth_scale: float,
        min_depth_m: float,
        max_depth_m: float) -> None:
    """
    Apply depth fallback for critical landmarks (THUMB_MCP, INDEX_FINGER_MCP).
    
    If one of these landmarks failed to deproject (invalid depth), but the other succeeded,
    use the depth from the successful one to reproject the failed one.
    
    Args:
        landmarks_3d: Dictionary of deprojected 3D landmarks (modified in-place)
        hand_landmarks_2d: Original 2D landmarks
        intrinsics: Camera intrinsics
        depth_scale: Depth scale factor
        min_depth_m: Minimum valid depth
        max_depth_m: Maximum valid depth
    """
    thumb_mcp_3d = landmarks_3d.get('THUMB_MCP')
    index_mcp_3d = landmarks_3d.get('INDEX_FINGER_MCP')
    
    # If both succeeded or both failed, no fallback needed
    if (thumb_mcp_3d is None and index_mcp_3d is None) or \
       (thumb_mcp_3d is not None and index_mcp_3d is not None):
        return
    
    # Fallback: THUMB_MCP failed but INDEX_FINGER_MCP succeeded
    if thumb_mcp_3d is None and index_mcp_3d is not None:
        thumb_mcp_2d = hand_landmarks_2d.get('THUMB_MCP')
        if thumb_mcp_2d is not None:
            try:
                # Use depth from INDEX_FINGER_MCP
                fallback_depth_m = index_mcp_3d[2]
                fallback_depth = fallback_depth_m * depth_scale
                
                x_px, y_px = int(thumb_mcp_2d[0]), int(thumb_mcp_2d[1])
                landmarks_3d['THUMB_MCP'] = deproject_point(
                    (x_px, y_px), fallback_depth, intrinsics, 
                    depth_scale, min_depth_m, max_depth_m)
                _logger.debug('Applied depth fallback for THUMB_MCP using INDEX_FINGER_MCP depth (%.3fm)', 
                             fallback_depth_m)
            except Exception as e:
                _logger.debug('Failed to apply depth fallback for THUMB_MCP: %s', e)
    
    # Fallback: INDEX_FINGER_MCP failed but THUMB_MCP succeeded
    elif index_mcp_3d is None and thumb_mcp_3d is not None:
        index_mcp_2d = hand_landmarks_2d.get('INDEX_FINGER_MCP')
        if index_mcp_2d is not None:
            try:
                # Use depth from THUMB_MCP
                fallback_depth_m = thumb_mcp_3d[2]
                fallback_depth = fallback_depth_m * depth_scale
                
                x_px, y_px = int(index_mcp_2d[0]), int(index_mcp_2d[1])
                landmarks_3d['INDEX_FINGER_MCP'] = deproject_point(
                    (x_px, y_px), fallback_depth, intrinsics, 
                    depth_scale, min_depth_m, max_depth_m)
                _logger.debug('Applied depth fallback for INDEX_FINGER_MCP using THUMB_MCP depth (%.3fm)', 
                             fallback_depth_m)
            except Exception as e:
                _logger.debug('Failed to apply depth fallback for INDEX_FINGER_MCP: %s', e)


def _deproject_hand_landmark(
        hand_uv: types.HandLandmark2D,
        depth_image: np.ndarray,
        intrinsics: types.CameraIntrinsics,
        depth_scale: float,
        min_depth_m: float,
        max_depth_m: float) -> types.HandLandmark3D:
    """
    Deproject a single 2D hand landmark to 3D camera space.

    Args:
        hand_uv: 2D hand position in image coordinates (u, v)
        depth_image: Depth image array
        intrinsics: Camera intrinsics for deprojection
        depth_scale: Scale factor to convert depth values to meters
        min_depth_m: Minimum valid depth in meters
        max_depth_m: Maximum valid depth in meters

    Returns:
        Tuple of (x, y, z) in camera coordinates (meters).

    Raises:
        IndexError: If the 2D landmark is out of bounds.
        DepthOutOfRange: If the depth value is out of provided range.
    """
    u, v = hand_uv
    x_px, y_px = int(u), int(v)
    if not 0 <= x_px < depth_image.shape[1] or not 0 <= y_px < depth_image.shape[0]:
        raise IndexError(f'2D landmark is out of bounds: {x_px}, {y_px} '
                         f'for depth map with dimensions: {depth_image.shape[1]}, {depth_image.shape[0]}')
    depth_value = depth_image[y_px, x_px].item()
    return deproject_point((x_px, y_px), depth_value, intrinsics,
                            depth_scale, min_depth_m, max_depth_m)


def deproject_point(
        image_coords: tuple[int, int],
        depth: float,
        intrinsics: types.CameraIntrinsics,
        depth_scale: float,
        min_depth_m: float,
        max_depth_m: float) -> types.HandLandmark3D:
    """
    Deproject a 2D point to 3D camera space.

    Args:
        image_coords: Image space coordinates of point to deproject
        depth: The depth of the point to deproject
        intrinsics: Camera intrinsics for deprojection
        depth_scale: Scale factor to convert depth values to meters
        min_depth_m: Minimum valid depth in meters
        max_depth_m: Maximum valid depth in meters

    Returns:
        Tuple of (x, y, z) in camera coordinates (meters).

    Raises:
        IndexError: If the image coordinates are out of bounds.
        DepthOutOfRange: If the depth value is out of provided range.

    Note:
        - Uses pinhole camera model for deprojection
        - Handles coordinate bounds checking
        - Filters invalid depth values
    """
    x_px, y_px = image_coords

    # Ensure coordinates are within image bounds
    if not 0 <= x_px < intrinsics.width or not 0 <= y_px < intrinsics.height:
        raise IndexError(f'Invalid image coordinates: {image_coords} '
                         f'for image with dimensions: {intrinsics.width, intrinsics.height}')

    # Convert depth to meters
    depth_m = depth / depth_scale

    # Validate depth range
    if depth_m < min_depth_m or depth_m > max_depth_m:
        raise DepthOutOfRange(depth_m, min_depth_m, max_depth_m)

    # Deproject using actual camera intrinsics
    x_3d = (x_px - intrinsics.ppx) / intrinsics.fx * depth_m
    y_3d = (y_px - intrinsics.ppy) / intrinsics.fy * depth_m
    z_3d = depth_m

    return (x_3d, y_3d, z_3d)

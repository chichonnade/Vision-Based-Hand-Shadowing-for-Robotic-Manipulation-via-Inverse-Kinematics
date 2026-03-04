"""Calculate target position and orientation from hand landmarks."""
import enum
from typing import Optional

import numpy as np
from scipy.spatial import transform

from vbhs.pipeline import types


class Hand(enum.Enum):
    """Enum for the two hands."""
    LEFT = 'left'
    RIGHT = 'right'


def target_position_distance(hand_landmarks_a: types.HandPose3D,
                             hand_landmarks_b: types.HandPose3D,
                             hand: Hand) -> Optional[float]:
    """Calculate the distance between the target positions of two hands."""
    target_pos_a = calculate_target_position(hand_landmarks_a, hand)
    if target_pos_a is None:
        return None
    target_pos_b = calculate_target_position(hand_landmarks_b, hand)
    if target_pos_b is None:
        return None
    return np.linalg.norm(np.array(target_pos_a) - np.array(target_pos_b))


def calculate_target_position(hand_landmarks: types.HandPose3D,
                              hand: Hand) -> Optional[tuple[float, float, float]]:
    """Calculate target position from hand landmarks."""
    _ = hand  # Unused for now.
    for key in ['THUMB_MCP', 'INDEX_FINGER_MCP']:
        if hand_landmarks[key] is None:
            return None

    thumb_mcp = np.array(hand_landmarks['THUMB_MCP'])
    index_mcp = np.array(hand_landmarks['INDEX_FINGER_MCP'])

    # Gripper position: between thumb and index bases (natural grip point)
    return (thumb_mcp + index_mcp) / 2.0


def calculate_fallback_orientation(hand_landmarks: types.HandPose3D,
                                   hand: Hand) -> Optional[tuple[float, float, float, float]]:
    """
    Calculate a fallback orientation when tip landmarks are unavailable.
    
    Uses THUMB_MCP, INDEX_FINGER_MCP, and WRIST to create a simplified orientation
    based on gripper width direction and wrist-to-gripper direction.
    
    Args:
        hand_landmarks: Hand landmarks in 3D space
        hand: Which hand (left or right)
        
    Returns:
        Orientation quaternion or None if critical landmarks missing
    """
    _ = hand  # Unused for now.
    
    # Need THUMB_MCP, INDEX_FINGER_MCP, and WRIST for fallback
    for key in ['THUMB_MCP', 'INDEX_FINGER_MCP', 'WRIST']:
        if hand_landmarks[key] is None:
            return None
    
    thumb_mcp = np.array(hand_landmarks['THUMB_MCP'])
    index_mcp = np.array(hand_landmarks['INDEX_FINGER_MCP'])
    wrist = np.array(hand_landmarks['WRIST'])
    
    # Origin: middle point between thumb_mcp and index_mcp (target position)
    origin = (thumb_mcp + index_mcp) / 2.0
    
    # 1st vector: gripper width direction (thumb to index)
    gripper_width_axis = index_mcp - thumb_mcp
    gripper_width_axis = gripper_width_axis / np.linalg.norm(gripper_width_axis)
    
    # 2nd vector: wrist to origin direction
    wrist_to_origin = origin - wrist
    wrist_to_origin_norm = np.linalg.norm(wrist_to_origin)
    
    if wrist_to_origin_norm < 1e-6:
        # Wrist and origin are at same position (degenerate case), use default
        wrist_to_origin = np.array([0.0, 0.0, 1.0])
    else:
        wrist_to_origin = wrist_to_origin / wrist_to_origin_norm
    
    # 3rd vector: orthogonal to both (using cross product)
    gripper_up_axis = np.cross(gripper_width_axis, wrist_to_origin)
    
    # Handle degenerate case where vectors are parallel
    if np.linalg.norm(gripper_up_axis) < 1e-6:
        # Vectors are parallel, create arbitrary perpendicular
        if abs(gripper_width_axis[2]) < 0.9:
            gripper_up_axis = np.cross(gripper_width_axis, np.array([0.0, 0.0, 1.0]))
        else:
            gripper_up_axis = np.cross(gripper_width_axis, np.array([1.0, 0.0, 0.0]))
    
    gripper_up_axis = gripper_up_axis / np.linalg.norm(gripper_up_axis)
    
    # Re-orthogonalize the 2nd vector (forward axis) to ensure orthogonal coordinate system
    gripper_forward_axis = np.cross(gripper_up_axis, gripper_width_axis)
    gripper_forward_axis = gripper_forward_axis / np.linalg.norm(gripper_forward_axis)
    
    # Build rotation matrix: [width_axis, forward_axis, up_axis] as columns
    rotation_matrix = np.column_stack([gripper_width_axis, gripper_forward_axis, gripper_up_axis])
    
    # Convert to quaternion
    rotation_obj = transform.Rotation.from_matrix(rotation_matrix)
    orientation_quaternion = rotation_obj.as_quat()
    
    return tuple[float, float, float, float](orientation_quaternion.tolist())


def calculate_target_orientation(hand_landmarks: types.HandPose3D,
                                 hand: Hand) -> Optional[tuple[float, float, float, float]]:
    """Calculate target orientation quaternion from hand landmarks."""
    _ = hand  # Unused for now.

    # If any of the key landmarks are None, return None.
    for key in ['WRIST', 'THUMB_MCP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_TIP']:
        if hand_landmarks[key] is None:
            return None

    wrist = np.array(hand_landmarks['WRIST'])
    thumb_mcp = np.array(hand_landmarks['THUMB_MCP'])
    thumb_tip = np.array(hand_landmarks['THUMB_TIP'])
    index_mcp = np.array(hand_landmarks['INDEX_FINGER_MCP'])
    index_tip = np.array(hand_landmarks['INDEX_FINGER_TIP'])

    # Define gripper coordinate system based on thumb-index relationship

    # Primary axis: gripper width direction (thumb to index)
    # This represents the "wide" part of the gripper opening
    gripper_width_axis = index_mcp - thumb_mcp
    gripper_width_axis = gripper_width_axis / np.linalg.norm(gripper_width_axis)

    # Secondary axis: average pointing direction of thumb and index
    # This represents the direction the gripper "points" towards objects
    thumb_direction = thumb_tip - thumb_mcp
    index_direction = index_tip - index_mcp
    avg_finger_direction = (thumb_direction + index_direction) / 2.0

    # Normalize the average direction
    if np.linalg.norm(avg_finger_direction) > 1e-6:
        avg_finger_direction = avg_finger_direction / np.linalg.norm(avg_finger_direction)
    else:
        # Fallback: use wrist to gripper center direction
        avg_finger_direction = np.array(calculate_target_position(hand_landmarks, hand)) - wrist
        avg_finger_direction = avg_finger_direction / np.linalg.norm(avg_finger_direction)

    # Third axis: perpendicular to both (gripper "up" direction)
    gripper_up_axis = np.cross(gripper_width_axis, avg_finger_direction)

    # Handle degenerate case where vectors are parallel
    if np.linalg.norm(gripper_up_axis) < 1e-6:
        # Create an arbitrary perpendicular vector
        if abs(gripper_width_axis[2]) < 0.9:
            gripper_up_axis = np.cross(gripper_width_axis, np.array([0, 0, 1]))
        else:
            gripper_up_axis = np.cross(gripper_width_axis, np.array([1, 0, 0]))

    gripper_up_axis = gripper_up_axis / np.linalg.norm(gripper_up_axis)

    # Re-orthogonalize the forward axis to ensure orthogonal coordinate system
    gripper_forward_axis = np.cross(gripper_up_axis, gripper_width_axis)
    gripper_forward_axis = gripper_forward_axis / np.linalg.norm(gripper_forward_axis)

    # Build rotation matrix: [width_axis, forward_axis, up_axis] as columns
    # This represents: X = gripper width, Y = gripper forward, Z = gripper up
    rotation_matrix = np.column_stack([gripper_width_axis, gripper_forward_axis, gripper_up_axis])

    # Convert to quaternion using scipy (robust and stable)
    rotation_obj = transform.Rotation.from_matrix(rotation_matrix)
    orientation_quaternion = rotation_obj.as_quat()  # Returns [x, y, z, w]

    return tuple[float, float, float, float](orientation_quaternion.tolist())

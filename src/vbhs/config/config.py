"""Config for the glasses, robot, and camera."""

# Camera to robot base transformation parameters
CAM_TO_ROBOT_CONFIG = {
    'translation': {
        'x': 0 + 0.04,      # a: distance in x (meters)
        'y': -0.0689 + 0.02,  # b: distance in y (meters)
        'z': 0.4799    # c: distance in z (meters)
    },
    'rotation_angle': 50  # degrees
}

# URDF offset transformation parameters
URDF_OFFSET_CONFIG = {
    'translation': {
        'x': 0.0,  # URDF offset in x (meters)
        'y': 0.0,  # URDF offset in y (meters)
        'z': 0.0   # URDF offset in z (meters)
    },
    'rpy': {
        'r': 0.0,  # roll (radians)
        'p': 0.0,  # pitch (radians)
        'y': 0.0   # yaw (radians)
    }
}

# Hand detection parameters
HAND_DETECTION_CONFIG = {
    'confidence_threshold': 0.5,
    'max_num_hands': 2,
    'model_complexity': 1
}

# Depth processing parameters
DEPTH_PROCESSING_CONFIG = {
    'min_depth_m': 0.1,    # Minimum valid depth in meters
    'max_depth_m': 5.0,    # Maximum valid depth in meters
    'depth_scale': 1000.0  # Scale factor (mm to meters)
}

# TODO (isaac): I think these are unused, and confusing.
# Camera parameters
CAMERA_CONFIG = {
    'live': {
        'width': 1920,
        'height': 1080,
        'fps': 6
    },
    'processing': {
        'target_fps': 30.0
    }
}

# MediaPipe hand landmark indices.
# Based on MediaPipe hand landmark model (21 landmarks per hand).
MEDIAPIPE_HAND_LANDMARKS = {
    'WRIST': 0,
    'THUMB_CMC': 1,
    'THUMB_MCP': 2,
    'THUMB_IP': 3,
    'THUMB_TIP': 4,
    'INDEX_FINGER_MCP': 5,
    'INDEX_FINGER_PIP': 6,
    'INDEX_FINGER_DIP': 7,
    'INDEX_FINGER_TIP': 8,
    'MIDDLE_FINGER_MCP': 9,
    'MIDDLE_FINGER_PIP': 10,
    'MIDDLE_FINGER_DIP': 11,
    'MIDDLE_FINGER_TIP': 12,
    'RING_FINGER_MCP': 13,
    'RING_FINGER_PIP': 14,
    'RING_FINGER_DIP': 15,
    'RING_FINGER_TIP': 16,
    'PINKY_MCP': 17,
    'PINKY_PIP': 18,
    'PINKY_DIP': 19,
    'PINKY_TIP': 20
}

# WiLoR hand landmark indices are the same as MediaPipe hand landmark indices.
WILOR_HAND_LANDMARKS = MEDIAPIPE_HAND_LANDMARKS

# Gripper control parameters
GRIPPER_SPEC = {
    'finger_distance': {
        'min_distance': 0.02,    # Minimum finger distance (closed) in meters
        'max_distance': 0.08     # Maximum finger distance (open) in meters
    },
    # TODO (isaac): load these from the URDF.
    'joint_angles': {
        'min_angle': 0.0873,  # Closed gripper angle in radians
        'max_angle': 1.6581   # Open gripper angle in radians
    },
}

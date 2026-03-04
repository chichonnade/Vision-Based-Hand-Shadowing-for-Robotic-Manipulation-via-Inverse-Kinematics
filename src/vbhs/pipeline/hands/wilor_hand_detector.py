"""MediaPipe-based hand detector implementation."""
from typing import Optional
import logging
import math

import cv2
import numpy as np
import torch  # TODO (isaac): eventually move to runtime to avoid loading on import.
from wilor_mini.pipelines import wilor_hand_pose3d_estimation_pipeline

from vbhs.config import config
from vbhs.pipeline import types
from vbhs.pipeline.hands import hand_detector
from vbhs.pipeline.hands import wilor_renderer

_logger = logging.getLogger(__name__)


def _get_torch_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """Get the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device('cuda'), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device('mps'), torch.float32
    else:
        return torch.device('cpu'), torch.float32


def _get_mano_faces(pipeline: wilor_hand_pose3d_estimation_pipeline.WiLorHandPose3dEstimationPipeline) -> np.ndarray:
    """Get MANO faces from the WiLoR pipeline.

    Args:
        pipeline: WiLoR pipeline instance

    Returns:
        MANO faces array of shape (F, 3)
    """
    faces = pipeline.wilor_model.mano.faces

    # Convert to numpy if needed
    if torch.is_tensor(faces):
        return faces.cpu().numpy().astype(np.int32)
    return np.array(faces, dtype=np.int32)


class WilorHandDetector(hand_detector.HandDetector):
    """WiLoR based hand detector."""

    def __init__(
            self,
            camera_intrinsics: types.CameraIntrinsics,
            enable_visualization: bool = True,
            visualization_window_name: str = "Hand Mesh"):
        """Initialize WiLoR hand detector.

        Args:
            camera_intrinsics: Camera intrinsics for hand detection
            enable_visualization: If True, displays rendered hand meshes in a cv2 window
            visualization_window_name: Name of the visualization window
        """
        super().__init__(camera_intrinsics)
        device, dtype = _get_torch_device_and_dtype()
        max_image_side_length = max(self._camera_intrinsics.width, self._camera_intrinsics.height)
        assert math.isclose(self._camera_intrinsics.fx, self._camera_intrinsics.fy, rel_tol=0.1), (
            'Assumes square pixels.')
        focal_length = self._camera_intrinsics.fx
        # NOTE: for now we disable the focal length scaling, since it throws off results.
        # From WiLoR documentation:
        # "scale the actual focal length by 256/max_image_side_length"
        focal_length_wilor = focal_length * 256 / max_image_side_length  # pylint: disable=unused-variable
        self._wilor_pipeline = (
            wilor_hand_pose3d_estimation_pipeline.WiLorHandPose3dEstimationPipeline(
                device=device,
                dtype=dtype,
                verbose=False
            )
        )

        # Visualization setup
        self._enable_visualization = enable_visualization
        self._visualization_window_name = visualization_window_name
        self._renderer: Optional[wilor_renderer.WilorRenderer] = None
        self._window_initialized = False
        self._display_width = 800
        self._display_height = 450

        if self._enable_visualization:
            # Initialize renderer with MANO faces
            try:
                faces = _get_mano_faces(self._wilor_pipeline)
                self._renderer = wilor_renderer.WilorRenderer(faces)
                self._setup_visualization_window()
            except RuntimeError as e:
                _logger.warning('Could not initialize WiLoR renderer: %s', e)
                self._enable_visualization = False

    def _setup_visualization_window(self):
        """Setup the visualization window for hand mesh rendering."""
        # pylint: disable=no-member
        # Create the named window
        cv2.namedWindow(self._visualization_window_name, cv2.WINDOW_NORMAL)

        # Set window to a reasonable size
        cv2.resizeWindow(self._visualization_window_name, self._display_width, self._display_height)

        self._window_initialized = True
        _logger.debug('Visualization window "%s" initialized (%dx%d)',
                    self._visualization_window_name, self._display_width, self._display_height)

    def detect(
            self, rgb_image: cv2.typing.MatLike
            ) -> tuple[Optional[types.HandPose2D], Optional[types.HandPose2D]]:
        """Detect hand landmarks from an RGB image."""
        results = self._wilor_pipeline.predict(rgb_image)
        left_hand_landmarks: Optional[types.HandPose2D] = None
        right_hand_landmarks: Optional[types.HandPose2D] = None

        for hand in results:
            keypoints_2d = hand['wilor_preds']['pred_keypoints_2d'][0]
            is_right = bool(hand['is_right'])
            landmarks = {
                key: (int(keypoints_2d[index][0]), int(keypoints_2d[index][1]))
                for key, index in config.WILOR_HAND_LANDMARKS.items()
            }
            if is_right:
                right_hand_landmarks = landmarks
            else:
                left_hand_landmarks = landmarks

        # Visualize if enabled
        if self._enable_visualization:
            self._visualize(results, rgb_image)

        return left_hand_landmarks, right_hand_landmarks

    def _visualize(self, results: list, rgb_image: np.ndarray):
        """Visualize hand mesh using the renderer.

        Args:
            results: WiLoR prediction results
            rgb_image: RGB image to render on
        """
        if self._renderer is None:
            _logger.error('Called _visualize without renderer')
            return
        elif not self._window_initialized:
            _logger.error('Called _visualize without window initialized')
            return

        if len(results) == 0:
            return

        # Create a copy of the image to render on
        vis_image = rgb_image.copy()

        # Render the meshes
        vis_image = self._renderer.render(results, vis_image)

        # Convert RGB to BGR for OpenCV display
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member

        # Resize the image to fit the display window
        display_image = cv2.resize(vis_image_bgr, (self._display_width, self._display_height))  # pylint: disable=no-member

        # Display in cv2 window
        cv2.imshow(self._visualization_window_name, display_image)  # pylint: disable=no-member
        cv2.waitKey(1)  # pylint: disable=no-member

    def cleanup(self):
        """Cleanup WiLoR resources."""
        # pylint: disable=no-member
        if self._window_initialized:
            cv2.destroyWindow(self._visualization_window_name)
            self._window_initialized = False

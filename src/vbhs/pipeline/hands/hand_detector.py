"""Abstract base class for hand detection."""
from typing import Optional
import abc

import cv2

from vbhs.pipeline import types


class HandDetector(abc.ABC):
    """Abstract base class for hand detector."""

    def __init__(self, camera_intrinsics: types.CameraIntrinsics):
        """Initialize the hand detector."""
        self._camera_intrinsics = camera_intrinsics

    @abc.abstractmethod
    def detect(
            self, rgb_image: cv2.typing.MatLike
            ) -> tuple[Optional[types.HandPose2D], Optional[types.HandPose2D]]:
        """Detect hand landmarks from an RGB image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def cleanup(self):
        """Cleanup the hand detector."""
        raise NotImplementedError()

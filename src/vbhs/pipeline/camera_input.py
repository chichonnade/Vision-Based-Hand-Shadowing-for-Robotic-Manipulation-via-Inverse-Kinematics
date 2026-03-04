"""
Camera Input Pipeline Step
==========================

This module handles camera input from various sources:
- Live RealSense camera streams
- Pre-recorded .bag files

The CameraFrameFromInput transformation captures RGB-D frames and converts them
to the standardized CameraFrame format used throughout the pipeline.

Key Features:
- Support for multiple input sources (live, bag)
- Automatic camera intrinsics extraction
- Frame alignment between RGB and depth streams
- Memory-efficient preloading for file sources
- Robust error handling and source detection

Usage:
    camera_input = CameraFrameFromInput(source_type='live')
    camera_frame = camera_input()  # Returns CameraFrame
"""

import enum
import logging
import os
import time
from typing import Any

import numpy as np
import pyrealsense2 as rs
import cv2

from vbhs.pipeline import transformations
from vbhs.pipeline import types

_logger = logging.getLogger(__name__)


class SourceType(enum.Enum):
    """Camera input source types."""
    LIVE = 'live'
    BAG = 'bag'


class CameraFrameFromInput(transformations.Transformation[None, types.CameraFrame]):
    """
    Camera input transformation that captures RGB and Depth frames from various sources.

    This transformation takes no input and outputs CameraFrame objects containing
    aligned RGB and depth data along with camera intrinsics.
    """

    def __init__(self,
                 source_type: SourceType = SourceType.LIVE,
                 bag_file: str | None = None,
                 rgb_width: int = 1920,
                 rgb_height: int = 1080,
                 depth_width: int = 1280,
                 depth_height: int = 720,
                 fps: int = 15,
                 align: bool = True,
                 realtime_playback: bool = False,
                 loop_bag: bool = True,
                 bag_output_path: str | None = None,
                 mp4_output_dir: str | None = None):
        """
        Initialize camera input source.

        Args:
            source_type: SourceType.LIVE or SourceType.BAG
            bag_file: Path to .bag file (if source_type=SourceType.BAG)
            rgb_width: RGB camera resolution width (only used for live cameras, default: 1920)
            rgb_height: RGB camera resolution height (only used for live cameras, default: 1080)
            depth_width: Depth camera resolution width (only used for live cameras, default: 1280)
            depth_height: Depth camera resolution height (only used for live cameras, default: 720)
            fps: Frame rate at which to run the input source.
            align: Whether to align depth frames to color frames (default: True)
            realtime_playback: Whether to playback bag file in real time
            loop: Whether to loop the bag file playback
            bag_output_path: Path to save recorded bag file if recording.
            mp4_output_dir: Directory to save RGB MP4 recording (default: None)

        Note:
            - For bag files: resolution auto-detected from recorded streams
            - For live cameras: uses specified rgb_width/rgb_height/depth_width/depth_height/fps parameters
            - align=False skips depth-to-color alignment (faster but may have misalignment)
            - mp4_output_dir creates human_rgb.mp4 file with timestamped name if provided
        """
        super().__init__()

        self.source_type = source_type
        self.bag_file = bag_file
        self.align = align
        self.realtime_playback = realtime_playback
        self.loop = loop_bag
        self.mp4_output_dir = mp4_output_dir

        # Only store live camera parameters when needed
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps


        # Internal state (private)
        # TODO (isaac): why all these optionals?
        self._bag_output_path = bag_output_path
        self._pipeline: Any | None = None
        self._config: Any | None = None
        self._align: Any | None = None
        self._color_intrinsics: types.CameraIntrinsics | None = None
        self._rgb_cap: Any | None = None
        self._depth_cap: Any | None = None
        self._recorder: Any | None = None
        self._total_frames: int = 0
        self._rgb_video_writer: cv2.VideoWriter | None = None  # pylint: disable=no-member
        self._rgb_output_filename: str | None = None


        # Initialize the appropriate input source
        self._setup_input_source()

    @property
    def camera_intrinsics(self) -> types.CameraIntrinsics:
        """Get the camera intrinsics."""
        if self._color_intrinsics is None:
            raise ValueError(
                'Camera intrinsics not available, please set up the camera input source first.')
        return self._color_intrinsics

    def _setup_input_source(self):
        """Setup the camera input source based on configuration."""
        if self.source_type == SourceType.BAG:
            if not self.bag_file:
                raise ValueError('bag_file must be provided when source_type=SourceType.BAG')
            self._setup_bag_playback()
        elif self.source_type == SourceType.LIVE:
            self._setup_live_camera()
        else:
            raise ValueError(f'Unknown source_type: {self.source_type}')

    def _setup_live_camera(self):
        """Setup live RealSense camera pipeline."""
        # pylint: disable=no-member
        for _ in range(10):
            try:
                self._pipeline = rs.pipeline()
                self._config = rs.config()

                # Configure streams using specified resolutions
                self._config.enable_stream(
                    rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.fps)
                self._config.enable_stream(
                    rs.stream.color, self.rgb_width, self.rgb_height, rs.format.bgr8, self.fps)

                self._pipeline.start(self._config)

                profile = self._pipeline.get_active_profile()
                # Start recording if bag output directory is provided
                if self._bag_output_path is not None:
                    self._recorder = rs.recorder(self._bag_output_path, profile.get_device())
                _logger.info('RealSense live camera started successfully')
                break
            except Exception as e:
                _logger.debug('Failed to start RealSense pipeline (retry): %s', e)
                time.sleep(0.1)
        else:
            raise RuntimeError('Failed to start RealSense pipeline')

        # Get camera intrinsics
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()

        # Calculate FOV from focal length
        fov = 2 * np.arctan(intrinsics.width / (2 * intrinsics.fx))

        self._color_intrinsics = types.CameraIntrinsics(
            fov=fov,
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            ppx=intrinsics.ppx,
            ppy=intrinsics.ppy
        )

        # Setup alignment (align depth TO color stream) if enabled
        if self.align:
            # Note: rs.align() is needed even if both streams have same resolution
            # because RGB and depth sensors are physically separate with different positions
            self._align = rs.align(rs.stream.color)
            _logger.debug('Camera intrinsics: fx=%.1f, fy=%.1f', self._color_intrinsics.fx, self._color_intrinsics.fy)
            _logger.debug('RGB Resolution: %dx%d, Depth Resolution: %dx%d (depth will be aligned to color)',
                        self.rgb_width, self.rgb_height, self.depth_width, self.depth_height)
        else:
            self._align = None
            _logger.debug('Camera intrinsics: fx=%.1f, fy=%.1f', self._color_intrinsics.fx, self._color_intrinsics.fy)
            _logger.debug('RGB Resolution: %dx%d, Depth Resolution: %dx%d (no alignment - depth and color may be misaligned)',
                        self.rgb_width, self.rgb_height, self.depth_width, self.depth_height)

        # Setup RGB video writer if mp4_output_dir is provided
        self._setup_rgb_video_writer()


    def _setup_bag_playback(self):
        """Setup .bag file playback with streaming."""
        if not self.bag_file or not os.path.exists(self.bag_file):
            raise FileNotFoundError(f'.bag file not found: {self.bag_file}')

        # Setup pipeline for bag file streaming
        # pylint: disable=no-member
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device_from_file(self.bag_file, repeat_playback=self.loop)

        try:
            self._pipeline.start(self._config)

            # Disable real-time playback for faster processing
            device = self._pipeline.get_active_profile().get_device()
            playback = device.as_playback()
            playback.set_real_time(self.realtime_playback)  # Process every frame.

            _logger.info('RealSense bag file streaming started: %s', self.bag_file)
        except Exception as e:
            _logger.error('Failed to start bag pipeline: %s', e)
            raise

        # Get camera intrinsics from bag file
        profile = self._pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        if color_stream:
            color_profile = rs.video_stream_profile(color_stream)
            intrinsics = color_profile.get_intrinsics()

            # Calculate FOV from focal length
            fov = 2 * np.arctan(intrinsics.width / (2 * intrinsics.fx))

            self._color_intrinsics = types.CameraIntrinsics(
                fov=fov,
                width=intrinsics.width,
                height=intrinsics.height,
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                ppx=intrinsics.ppx,
                ppy=intrinsics.ppy
            )
            _logger.debug('Bag intrinsics: fx=%.1f, fy=%.1f', self._color_intrinsics.fx, self._color_intrinsics.fy)
            if self.align:
                _logger.debug('Bag Resolution: %dx%d (auto-detected, depth will be aligned to color)',
                           self._color_intrinsics.width, self._color_intrinsics.height)
            else:
                _logger.debug('Bag Resolution: %dx%d (auto-detected, no alignment - depth and color may be misaligned)',
                           self._color_intrinsics.width, self._color_intrinsics.height)

        # Setup alignment (align depth TO color stream) if enabled
        if self.align:
            # Note: rs.align() is needed even if both streams have same resolution
            # because RGB and depth sensors are physically separate with different positions
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        # Setup RGB video writer if mp4_output_dir is provided
        self._setup_rgb_video_writer()

    def _transform(self, data_in: None) -> types.CameraFrame:
        """
        Capture and return a camera frame.

        Args:
            data_in: None (this transformation takes no input)

        Returns:
            CameraFrame: RGB-D frame with camera intrinsics
        """
        if self.source_type in [SourceType.LIVE, SourceType.BAG]:
            return self._get_live_frame()  # Both use RealSense pipeline
        else:
            raise RuntimeError(f'Unknown source type: {self.source_type}')

    def _get_live_frame(self) -> types.CameraFrame:
        """Get frame from live camera."""
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)

            if self.align and self._align is not None:
                # Apply alignment if enabled
                aligned_frames = self._align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
            else:
                # Use frames directly without alignment
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Write RGB frame to video if recording
            if self._rgb_video_writer is not None:
                self._rgb_video_writer.write(color_image)

            return self._create_camera_frame(color_image, depth_image)

        except Exception as e:
            raise RuntimeError(f'Error getting live frames: {e}')

    def _create_camera_frame(self, color_image: np.ndarray, depth_image: np.ndarray) -> types.CameraFrame:
        """Create a CameraFrame from RGB and depth images."""
        # Create depth map
        depth_map = types.DepthMap(values=depth_image)

        return types.CameraFrame(
            rgb=color_image,
            depth=depth_map,
            intrinsics=self._color_intrinsics
        )

    def _setup_rgb_video_writer(self):
        """Setup RGB video writer if mp4_output_dir is provided."""
        if self.mp4_output_dir is None:
            return

        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.mp4_output_dir, exist_ok=True)
            self._rgb_output_filename = os.path.join(self.mp4_output_dir, "human_rgb.mp4")

            # Get frame dimensions from camera intrinsics or defaults
            if self._color_intrinsics:
                width = self._color_intrinsics.width
                height = self._color_intrinsics.height
            elif self.source_type == SourceType.LIVE:
                width = self.rgb_width
                height = self.rgb_height
            else:
                # For bag files, try to get dimensions when intrinsics are available
                _logger.warning("Cannot determine frame dimensions for RGB recording")
                return

            # Setup video writer with MP4 codec
            # pylint: disable=no-member
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._rgb_video_writer = cv2.VideoWriter(
                self._rgb_output_filename,
                fourcc,
                self.fps,
                (width, height)
            )

            if not self._rgb_video_writer.isOpened():
                _logger.warning("Failed to open RGB video writer")
                self._rgb_video_writer = None
                return

            _logger.info("RGB recording started, saving to: %s", self._rgb_output_filename)

        except Exception as e:
            _logger.error("Failed to setup RGB video writer: %s", e)
            self._rgb_video_writer = None

    def cleanup(self):
        """Cleanup camera resources."""
        # Release RGB video writer
        if self._rgb_video_writer is not None:
            try:
                self._rgb_video_writer.release()
                if self._rgb_output_filename and os.path.exists(self._rgb_output_filename):
                    _logger.info('RGB recording saved to: %s', os.path.abspath(self._rgb_output_filename))
                else:
                    _logger.warning('RGB recording file not found')
            except Exception as e:
                _logger.error('Error releasing RGB video writer: %s', e)

        if self.source_type in [SourceType.LIVE, SourceType.BAG] and self._pipeline:
            try:
                self._pipeline.stop()
                _logger.debug('RealSense pipeline stopped')
            except Exception as e:
                _logger.error('Error stopping pipeline: %s', e)

        _logger.debug('Camera input cleanup completed')

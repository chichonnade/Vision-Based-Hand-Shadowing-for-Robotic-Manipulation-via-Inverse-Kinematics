import time
import pyrealsense2 as rs
import os
import cv2
import numpy as np
import threading
from datetime import datetime
from typing import Optional

from vbhs.scripts import action_playback


class GlassesRecorder:
    """
    A class to record video from RealSense glasses to MP4 files in a background thread.
    """

    def __init__(self, output_directory: str, width: int = 1920, height: int = 1080, fps: int = 30, show_preview: bool = False):
        """
        Initialize the GlassesRecorder.

        Args:
            output_directory: Directory where MP4 files will be saved
            width: Video width (default: 1920)
            height: Video height (default: 1080)
            fps: Frames per second (default: 15)
            show_preview: Whether to show live preview window (default: False)
        """
        self.output_directory = output_directory
        self.width = width
        self.height = height
        self.fps = fps
        self.show_preview = show_preview

        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

        # Recording state
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Setup status tracking
        self.setup_complete = threading.Event()
        self.setup_failed = threading.Event()
        self.setup_error_msg = ""

        # Frame tracking
        self.frames_captured = 0
        self.frames_captured_lock = threading.Lock()

        # RealSense components
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.output_filename: Optional[str] = None

    def _setup_pipeline(self) -> bool:
        """
        Setup the RealSense pipeline.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            return True
        except Exception as e:
            print(f"❌ Failed to setup RealSense pipeline: {e}")
            return False

    def _setup_video_writer(self) -> bool:
        """
        Setup the MP4 video writer.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_filename = os.path.join(self.output_directory, f"glasses_recording_{timestamp}.mp4")

            # Setup video writer with MP4 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_filename,
                fourcc,
                self.fps,
                (self.width, self.height)
            )

            if not self.video_writer.isOpened():
                error_msg = "Failed to open video writer - check output directory permissions and codec availability"
                print(f"❌ {error_msg}")
                self.setup_error_msg = error_msg
                return False

            print(f"✅ Video writer setup successful: {self.output_filename}")
            return True
        except Exception as e:
            error_msg = f"Failed to setup video writer: {e}"
            print(f"❌ {error_msg}")
            self.setup_error_msg = error_msg
            return False

    def _recording_loop(self):
        """
        Main recording loop that runs in the background thread.
        """
        setup_successful = False
        try:
            # Reset frame counter
            with self.frames_captured_lock:
                self.frames_captured = 0

            # Setup pipeline and video writer in the recording thread
            for attempt in range(10):
                if self._setup_pipeline():
                    print("✅ Pipeline setup successful")
                    break
                print(f"Pipeline setup attempt {attempt + 1}/10 failed, retrying...")
                time.sleep(0.1)
            else:
                self.setup_error_msg = "Failed to setup pipeline after 10 attempts"
                print(f"❌ {self.setup_error_msg}")
                self.setup_failed.set()
                return

            if not self._setup_video_writer():
                print("❌ Failed to setup video writer in recording thread")
                self.setup_failed.set()
                return

            # Start the pipeline
            try:
                self.pipeline.start(self.config)
                print(f"🔴 Recording started, saving to: {self.output_filename}")

                # Initialize preview window if enabled
                if self.show_preview:
                    cv2.namedWindow("RealSense Preview", cv2.WINDOW_AUTOSIZE)

                setup_successful = True
                self.setup_complete.set()  # Signal that setup is complete
            except Exception as e:
                self.setup_error_msg = f"Failed to start pipeline: {e}"
                print(f"❌ {self.setup_error_msg}")
                self.setup_failed.set()
                return

            # Main recording loop
            consecutive_failures = 0
            max_consecutive_failures = 30  # Allow ~3 seconds of failures at 10fps

            while not self.stop_event.is_set():
                try:
                    loop_start = time.perf_counter()

                    # Wait for frames with timeout
                    frames = self.pipeline.wait_for_frames(timeout_ms=100)
                    color_frame = frames.get_color_frame()

                    if not color_frame:
                        consecutive_failures += 1
                        if consecutive_failures > max_consecutive_failures:
                            print("❌ Too many consecutive frame failures, stopping recording")
                            break
                        continue

                    # Reset failure counter on successful frame
                    consecutive_failures = 0

                    # Convert to numpy array
                    color_image = np.asanyarray(color_frame.get_data())

                    # Write frame to video file
                    try:
                        self.video_writer.write(color_image)
                        with self.frames_captured_lock:
                            self.frames_captured += 1
                    except Exception as e:
                        print(f"❌ Failed to write frame to video file: {e}")

                    # Show preview if enabled
                    if self.show_preview:
                        # Resize for preview (smaller window)
                        preview_image = cv2.resize(color_image, (960, 540))

                        # Add recording indicator
                        cv2.circle(preview_image, (30, 30), 15, (0, 0, 255), -1)
                        cv2.putText(preview_image, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        cv2.imshow("RealSense Preview", preview_image)
                        cv2.waitKey(1)  # Non-blocking key check

                    # Wait for the frame time to pass
                    action_playback.wait_for_frame_time(loop_start, 1 / self.fps)

                except RuntimeError:
                    # Timeout or no frames available, continue
                    consecutive_failures += 1
                    if consecutive_failures > max_consecutive_failures:
                        print("❌ Too many consecutive runtime errors, stopping recording")
                        break
                    continue

        except Exception as e:
            error_msg = f"Error in recording loop: {e}"
            print(f"❌ {error_msg}")
            if not setup_successful:
                self.setup_error_msg = error_msg
                self.setup_failed.set()
        finally:
            # Cleanup
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    # Pipeline may not have been started
                    pass
            if self.video_writer:
                try:
                    # Ensure all frames are written before releasing
                    self.video_writer.release()
                    print(f"✅ Video writer released. Total frames captured: {self.frames_captured}")
                except Exception as e:
                    print(f"⚠️  Error releasing video writer: {e}")

            # Close preview window if it was opened
            if self.show_preview:
                try:
                    cv2.destroyWindow("RealSense Preview")
                except Exception:
                    pass

    def begin_recording(self, setup_timeout: float = 10.0) -> bool:
        """
        Start recording from the RealSense glasses in a background thread.

        Args:
            setup_timeout: Maximum time to wait for setup completion (seconds)

        Returns:
            True if recording started successfully, False otherwise
        """
        if self.is_recording:
            print("⚠️  Recording is already in progress")
            return False

        # Reset all events and state
        self.stop_event.clear()
        self.setup_complete.clear()
        self.setup_failed.clear()
        self.setup_error_msg = ""

        # Start recording thread (setup happens inside the thread)
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()

        # Wait for setup to complete or fail
        print("⏳ Waiting for recording setup...")
        setup_events = [self.setup_complete, self.setup_failed]
        finished_event = threading.Event()

        def wait_for_any_event():
            while not finished_event.is_set():
                for event in setup_events:
                    if event.wait(timeout=0.1):
                        finished_event.set()
                        break

        wait_thread = threading.Thread(target=wait_for_any_event, daemon=True)
        wait_thread.start()

        if not finished_event.wait(timeout=setup_timeout):
            print(f"❌ Setup timeout after {setup_timeout} seconds")
            self.stop_event.set()
            return False

        if self.setup_failed.is_set():
            print(f"❌ Setup failed: {self.setup_error_msg}")
            self.stop_event.set()
            return False

        if self.setup_complete.is_set():
            self.is_recording = True
            print("✅ Recording setup completed successfully")
            return True

        print("❌ Unknown setup status")
        return False

    def end_recording(self) -> str:
        """
        Stop recording and save the video file.

        Returns:
            Path to the saved video file, or empty string if no recording was active
        """
        if not self.is_recording:
            print("⚠️  No recording in progress")
            return ""

        print("🛑 Stopping recording...")

        # Get final frame count before stopping
        with self.frames_captured_lock:
            final_frame_count = self.frames_captured

        # Signal the recording thread to stop
        self.stop_event.set()

        # Wait for the recording thread to finish with longer timeout for cleanup
        if self.recording_thread and self.recording_thread.is_alive():
            print("⏳ Waiting for recording thread to finish...")
            self.recording_thread.join(timeout=10.0)

            if self.recording_thread.is_alive():
                print("⚠️  Recording thread did not finish within timeout")

        self.is_recording = False

        # Return the output filename
        output_file = self.output_filename or ""

        if output_file and os.path.exists(output_file):
            # Get file size to verify it's not empty
            file_size = os.path.getsize(output_file)
            print(f"✅ Recording saved to: {os.path.abspath(output_file)}")
            print(f"   File size: {file_size / (1024*1024):.2f} MB")
            print(f"   Frames captured: {final_frame_count}")

            if file_size < 1000:  # Less than 1KB is suspicious
                print("⚠️  Warning: Video file is very small, may be empty or corrupted")
            if final_frame_count == 0:
                print("⚠️  Warning: No frames were captured during recording")
        else:
            print("❌ Recording file not found")
            print(f"   Expected path: {output_file}")

        return output_file

    def is_recording_active(self) -> bool:
        """
        Check if recording is currently active.

        Returns:
            True if recording is active, False otherwise
        """
        return self.is_recording

    def get_recording_status(self) -> dict:
        """
        Get detailed recording status information.

        Returns:
            Dictionary with recording status details
        """
        with self.frames_captured_lock:
            frames_captured = self.frames_captured

        status = {
            "is_recording": self.is_recording,
            "setup_complete": self.setup_complete.is_set(),
            "setup_failed": self.setup_failed.is_set(),
            "setup_error_msg": self.setup_error_msg,
            "frames_captured": frames_captured,
            "thread_alive": self.recording_thread.is_alive() if self.recording_thread else False,
            "output_filename": self.output_filename,
        }
        return status

    def is_recording_healthy(self) -> bool:
        """
        Check if recording is healthy (actively capturing frames).

        Returns:
            True if recording appears to be working properly, False otherwise
        """
        if not self.is_recording:
            return False

        status = self.get_recording_status()

        # Check if setup completed successfully
        if not status["setup_complete"] or status["setup_failed"]:
            return False

        # Check if thread is still alive
        if not status["thread_alive"]:
            return False

        # Check if we're capturing frames (at least some frames should be captured)
        # This is a basic health check - in a real scenario you might want to check
        # if frames are being captured recently
        return status["frames_captured"] > 0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure recording is stopped."""
        if self.is_recording:
            self.end_recording()

if __name__ == "__main__":
    recorder = GlassesRecorder(output_directory="video_out", show_preview=True)
    with recorder:
        recorder.begin_recording()
        time.sleep(10)

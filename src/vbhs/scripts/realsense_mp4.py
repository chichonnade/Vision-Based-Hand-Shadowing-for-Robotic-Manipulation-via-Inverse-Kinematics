import os

from vbhs.scripts import record_from_glasses


def main():
    """Simple script to record MP4 from RealSense glasses."""

    # Create output directory
    output_dir = "vbhs_output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize recorder with preview enabled
    recorder = record_from_glasses.GlassesRecorder(output_directory=output_dir, show_preview=True)

    print("Press Enter to start recording (preview window will open), Enter again to stop, 'q' to quit")

    recording = False

    while True:
        user_input = input().strip().lower()

        if user_input == 'q':
            break
        elif user_input == '':  # Enter key
            if not recording:
                if recorder.begin_recording():
                    recording = True
                    print("Recording started...")
            else:
                output_file = recorder.end_recording()
                recording = False
                if output_file:
                    print(f"Recording saved: {output_file}")

    # Cleanup
    if recording:
        recorder.end_recording()


if __name__ == "__main__":
    main()

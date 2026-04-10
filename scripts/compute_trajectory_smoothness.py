"""Compute trajectory smoothness metrics (jerk) from .npy action files.

Loads IK-generated joint angle trajectories, extracts contiguous non-NaN
segments, and computes RMS jerk per joint for both raw and EMA-smoothed
outputs. Results are saved as a CSV table and a comparison bar chart.

Usage:
    python scripts/compute_trajectory_smoothness.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

NPY_DIR = pathlib.Path(
    "new experiment data (pick and place action files)"
)

OUTPUT_DIR = pathlib.Path("paper_data")

DT = 1.0 / 5.0  # pipeline runs at ~5 FPS
EMA_ALPHA = 0.5  # IK smoothing alpha used in the pipeline
RIGHT_ARM_BLOCK = 1
MIN_SEGMENT_LENGTH = 6  # need at least 6 frames for 3rd derivative


def extract_contiguous_segments(
    joint_angles: np.ndarray,
) -> list[np.ndarray]:
    """Extract contiguous non-NaN segments from a (T, 6) array."""
    valid_mask = ~np.isnan(joint_angles).any(axis=1)
    segments: list[np.ndarray] = []
    start = None

    for i, valid in enumerate(valid_mask):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            if i - start >= MIN_SEGMENT_LENGTH:
                segments.append(joint_angles[start:i])
            start = None

    if start is not None and len(joint_angles) - start >= MIN_SEGMENT_LENGTH:
        segments.append(joint_angles[start:])

    return segments


def apply_ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential moving average along axis 0."""
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def compute_rms_jerk(segment: np.ndarray, dt: float) -> np.ndarray:
    """Compute RMS jerk per joint for a (T, 6) segment. Returns shape (6,)."""
    jerk = np.diff(segment, n=3, axis=0) / (dt ** 3)
    return np.sqrt(np.mean(jerk ** 2, axis=0))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    npy_files = sorted(NPY_DIR.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {NPY_DIR}")

    all_raw_jerks: list[np.ndarray] = []
    all_smoothed_jerks: list[np.ndarray] = []
    total_valid_frames = 0

    print(f"Found {len(npy_files)} action files\n")

    for fpath in npy_files:
        data = np.load(fpath)  # (T, 2, 6)
        right_arm = data[:, RIGHT_ARM_BLOCK, :]  # (T, 6)
        segments = extract_contiguous_segments(right_arm)

        n_valid = sum(len(s) for s in segments)
        total_valid_frames += n_valid
        print(
            f"  {fpath.name}: {data.shape[0]} frames, "
            f"{len(segments)} segments, {n_valid} valid frames"
        )

        for seg in segments:
            raw_jerk = compute_rms_jerk(seg, DT)
            all_raw_jerks.append(raw_jerk)

            smoothed_seg = apply_ema(seg, EMA_ALPHA)
            smoothed_jerk = compute_rms_jerk(smoothed_seg, DT)
            all_smoothed_jerks.append(smoothed_jerk)

    if not all_raw_jerks:
        raise ValueError("No valid contiguous segments found")

    raw_mean = np.mean(all_raw_jerks, axis=0)
    smoothed_mean = np.mean(all_smoothed_jerks, axis=0)
    reduction_pct = (1 - smoothed_mean / raw_mean) * 100

    print(f"\nTotal valid frames across all files: {total_valid_frames}")
    print(f"Total contiguous segments: {len(all_raw_jerks)}")
    print(f"\n{'Joint':<16} {'Raw RMS Jerk':>14} {'Smoothed RMS Jerk':>18} {'Reduction':>10}")
    print("-" * 62)
    for i, name in enumerate(JOINT_NAMES):
        print(
            f"{name:<16} {raw_mean[i]:>14.2f} {smoothed_mean[i]:>18.2f} "
            f"{reduction_pct[i]:>9.1f}%"
        )

    csv_path = OUTPUT_DIR / "trajectory_smoothness.csv"
    with open(csv_path, "w") as f:
        f.write("joint,raw_rms_jerk_rad_per_s3,smoothed_rms_jerk_rad_per_s3,reduction_pct\n")
        for i, name in enumerate(JOINT_NAMES):
            f.write(f"{name},{raw_mean[i]:.4f},{smoothed_mean[i]:.4f},{reduction_pct[i]:.1f}\n")
    print(f"\nCSV saved to {csv_path}")

    # Bar chart
    x = np.arange(len(JOINT_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_raw = ax.bar(x - width / 2, raw_mean, width, label="Raw IK output")
    bars_smooth = ax.bar(
        x + width / 2, smoothed_mean, width, label=f"EMA smoothed (α={EMA_ALPHA})"
    )

    ax.set_ylabel("RMS Jerk (rad/s³)")
    ax.set_title("Trajectory Smoothness: Raw vs. EMA-Smoothed IK Output")
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, pct in zip(bars_smooth, reduction_pct):
        ax.annotate(
            f"−{pct:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="green",
        )

    fig.tight_layout()
    chart_path = OUTPUT_DIR / "trajectory_smoothness.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

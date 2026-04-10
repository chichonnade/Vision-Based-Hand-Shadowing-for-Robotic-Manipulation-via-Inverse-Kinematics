"""EMA smoothing ablation study.

Applies different EMA alpha values to the raw IK joint angle trajectories
from the .npy action files and measures the effect on:
  - Joint-space jerk (RMS, rad/s^3)
  - Cartesian end-effector jitter (mm, via FK)
  - Orientation jitter (deg)

This simulates what would happen if the pipeline used different smoothing
parameters, without needing to re-run the full pipeline on .bag files.

Usage (run in WSL with conda vbhs env):
    python scripts/compute_ema_ablation.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

URDF_PATH = "robot/Dual_S101_Assembly.urdf"
NPY_DIR = pathlib.Path("new experiment data (pick and place action files)")
OUTPUT_DIR = pathlib.Path("paper_data")

RIGHT_ARM_JOINTS = [8, 9, 10, 11, 12]
RIGHT_GRIPPER_JOINT = 13
RIGHT_END_EFFECTOR = 12

DT = 1.0 / 5.0
ALPHA_VALUES = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]  # 1.0 = no smoothing (raw)
MIN_SEGMENT_LENGTH = 6


def extract_contiguous_segments(joint_angles):
    valid_mask = ~np.isnan(joint_angles).any(axis=1)
    segments = []
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


def apply_ema(data, alpha):
    if alpha >= 1.0:
        return data.copy()
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def setup_pybullet():
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    robot_id = p.loadURDF(
        URDF_PATH, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
        useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    return cid, robot_id


def set_joint_angles(robot_id, angles_6dof):
    for joint_idx, angle in zip(RIGHT_ARM_JOINTS + [RIGHT_GRIPPER_JOINT], angles_6dof):
        p.resetJointState(robot_id, joint_idx, angle)


def get_ee_pose(robot_id):
    link_state = p.getLinkState(robot_id, RIGHT_END_EFFECTOR,
                                computeForwardKinematics=True)
    return np.array(link_state[4]), np.array(link_state[5])


def quat_angular_dist(q1, q2):
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def compute_metrics_for_alpha(robot_id, segments, alpha):
    """Compute jerk and jitter metrics for a given EMA alpha."""
    all_jerks = []
    all_pos_jitters = []
    all_orn_jitters = []

    for seg in segments:
        smoothed = apply_ema(seg, alpha)

        # Joint-space jerk
        jerk = np.diff(smoothed, n=3, axis=0) / (DT ** 3)
        rms_jerk = np.sqrt(np.mean(jerk ** 2, axis=0))
        all_jerks.append(rms_jerk)

        # Cartesian jitter via FK
        positions = []
        orientations = []
        for angles in smoothed:
            set_joint_angles(robot_id, angles)
            pos, orn = get_ee_pose(robot_id)
            positions.append(pos)
            orientations.append(orn)

        positions = np.array(positions)
        diffs = np.diff(positions, axis=0)
        pos_changes = np.linalg.norm(diffs, axis=1) * 1000  # mm
        all_pos_jitters.extend(pos_changes)

        orn_changes = [quat_angular_dist(orientations[i], orientations[i + 1])
                       for i in range(len(orientations) - 1)]
        all_orn_jitters.extend(orn_changes)

    mean_jerk = np.mean(all_jerks, axis=0)  # per-joint
    total_rms_jerk = np.mean(mean_jerk)  # average across joints
    pos_jitter = np.array(all_pos_jitters)
    orn_jitter = np.array(all_orn_jitters)

    return {
        "alpha": alpha,
        "mean_rms_jerk": total_rms_jerk,
        "per_joint_jerk": mean_jerk,
        "pos_jitter_mean_mm": pos_jitter.mean(),
        "pos_jitter_median_mm": np.median(pos_jitter),
        "pos_jitter_std_mm": pos_jitter.std(),
        "orn_jitter_mean_deg": orn_jitter.mean(),
        "orn_jitter_median_deg": np.median(orn_jitter),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cid, robot_id = setup_pybullet()

    npy_files = sorted(NPY_DIR.glob("*.npy"))
    all_segments = []
    for fpath in npy_files:
        data = np.load(fpath)
        right_arm = data[:, 1, :]
        all_segments.extend(extract_contiguous_segments(right_arm))

    total_frames = sum(len(s) for s in all_segments)
    print(f"Loaded {len(all_segments)} segments, {total_frames} frames\n")

    results = []
    for alpha in ALPHA_VALUES:
        label = "raw (no smoothing)" if alpha >= 1.0 else f"α={alpha}"
        print(f"  Computing metrics for {label}...", end=" ", flush=True)
        r = compute_metrics_for_alpha(robot_id, all_segments, alpha)
        results.append(r)
        print(f"jerk={r['mean_rms_jerk']:.1f} rad/s³, "
              f"pos_jitter={r['pos_jitter_mean_mm']:.2f} mm, "
              f"orn_jitter={r['orn_jitter_mean_deg']:.2f}°")

    p.disconnect(cid)

    # Print table
    print(f"\n{'='*75}")
    print(f"EMA Smoothing Ablation Results")
    print(f"{'='*75}")
    print(f"{'Alpha':<8} {'RMS Jerk':>12} {'Pos Jitter':>12} {'Pos Median':>12} "
          f"{'Orn Jitter':>12}")
    print(f"{'':8} {'(rad/s³)':>12} {'mean (mm)':>12} {'(mm)':>12} {'mean (deg)':>12}")
    print("-" * 75)
    for r in results:
        label = "1.0 (raw)" if r["alpha"] >= 1.0 else f"{r['alpha']}"
        print(f"{label:<8} {r['mean_rms_jerk']:>12.2f} {r['pos_jitter_mean_mm']:>12.2f} "
              f"{r['pos_jitter_median_mm']:>12.2f} {r['orn_jitter_mean_deg']:>12.2f}")

    # Save CSV
    csv_path = OUTPUT_DIR / "ema_ablation.csv"
    with open(csv_path, "w") as f:
        f.write("alpha,mean_rms_jerk_rad_s3,pos_jitter_mean_mm,pos_jitter_median_mm,"
                "pos_jitter_std_mm,orn_jitter_mean_deg,orn_jitter_median_deg\n")
        for r in results:
            f.write(f"{r['alpha']},{r['mean_rms_jerk']:.4f},"
                    f"{r['pos_jitter_mean_mm']:.4f},{r['pos_jitter_median_mm']:.4f},"
                    f"{r['pos_jitter_std_mm']:.4f},{r['orn_jitter_mean_deg']:.4f},"
                    f"{r['orn_jitter_median_deg']:.4f}\n")
    print(f"\nCSV saved to {csv_path}")

    # Plots
    alphas = [r["alpha"] for r in results]
    alpha_labels = ["raw" if a >= 1.0 else str(a) for a in alphas]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) RMS Jerk vs alpha
    jerks = [r["mean_rms_jerk"] for r in results]
    axes[0].plot(alphas, jerks, "o-", color="blue", markersize=8)
    axes[0].set_xlabel("EMA α (1.0 = no smoothing)")
    axes[0].set_ylabel("Mean RMS Jerk (rad/s³)")
    axes[0].set_title("Joint-Space Smoothness vs. α")
    axes[0].grid(alpha=0.3)
    axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="α=0.5 (used)")
    axes[0].legend()

    # 2) Position jitter vs alpha
    pos_j = [r["pos_jitter_mean_mm"] for r in results]
    axes[1].plot(alphas, pos_j, "s-", color="green", markersize=8)
    axes[1].set_xlabel("EMA α (1.0 = no smoothing)")
    axes[1].set_ylabel("Mean Position Jitter (mm)")
    axes[1].set_title("Cartesian Position Jitter vs. α")
    axes[1].grid(alpha=0.3)
    axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="α=0.5 (used)")
    axes[1].legend()

    # 3) Orientation jitter vs alpha
    orn_j = [r["orn_jitter_mean_deg"] for r in results]
    axes[2].plot(alphas, orn_j, "^-", color="orange", markersize=8)
    axes[2].set_xlabel("EMA α (1.0 = no smoothing)")
    axes[2].set_ylabel("Mean Orientation Jitter (deg)")
    axes[2].set_title("Orientation Jitter vs. α")
    axes[2].grid(alpha=0.3)
    axes[2].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="α=0.5 (used)")
    axes[2].legend()

    fig.tight_layout()
    chart_path = OUTPUT_DIR / "ema_ablation.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

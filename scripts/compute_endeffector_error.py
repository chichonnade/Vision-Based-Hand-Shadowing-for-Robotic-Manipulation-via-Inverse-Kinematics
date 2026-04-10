"""Compute end-effector trajectory metrics from .npy action files via FK.

Loads the robot URDF in PyBullet (headless), sets joint angles from each
valid frame, queries FK for the achieved end-effector pose, and reports:
  - Workspace envelope (XYZ min/max/range)
  - Frame-to-frame position and orientation change (Cartesian jitter)
  - Per-trajectory path length and duration
  - Comparison of raw vs EMA-smoothed Cartesian jitter

Usage (run in WSL with conda vbhs env):
    python scripts/compute_endeffector_error.py
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

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

DT = 1.0 / 5.0
EMA_ALPHA = 0.5
MIN_SEGMENT_LENGTH = 4


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


def get_ee_pose(robot_id):
    link_state = p.getLinkState(robot_id, RIGHT_END_EFFECTOR,
                                computeForwardKinematics=True)
    return np.array(link_state[4]), np.array(link_state[5])


def set_joint_angles(robot_id, angles_6dof):
    for joint_idx, angle in zip(RIGHT_ARM_JOINTS + [RIGHT_GRIPPER_JOINT], angles_6dof):
        p.resetJointState(robot_id, joint_idx, angle)


def quat_angular_dist(q1, q2):
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def compute_ee_trajectory(robot_id, segment):
    """Compute FK for each frame, return positions and orientations."""
    positions = []
    orientations = []
    for angles in segment:
        set_joint_angles(robot_id, angles)
        pos, orn = get_ee_pose(robot_id)
        positions.append(pos)
        orientations.append(orn)
    return np.array(positions), np.array(orientations)


def trajectory_metrics(positions, orientations):
    """Compute per-trajectory metrics."""
    diffs = np.diff(positions, axis=0)
    pos_changes = np.linalg.norm(diffs, axis=1)
    orn_changes = np.array([
        quat_angular_dist(orientations[i], orientations[i + 1])
        for i in range(len(orientations) - 1)])
    path_length = pos_changes.sum()
    return pos_changes, orn_changes, path_length


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cid, robot_id = setup_pybullet()
    print(f"PyBullet initialized, robot id={robot_id}")

    npy_files = sorted(NPY_DIR.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {NPY_DIR}")

    # Collect metrics for raw and smoothed trajectories
    raw_pos_changes_all = []
    raw_orn_changes_all = []
    smooth_pos_changes_all = []
    smooth_orn_changes_all = []
    all_raw_positions = []
    all_smooth_positions = []
    trajectory_summaries = []

    print(f"\nProcessing {len(npy_files)} files...\n")

    for fpath in npy_files:
        data = np.load(fpath)
        right_arm = data[:, 1, :]
        segments = extract_contiguous_segments(right_arm)
        n_valid = sum(len(s) for s in segments)
        print(f"  {fpath.name}: {data.shape[0]} frames, "
              f"{len(segments)} segments, {n_valid} valid")

        for seg_idx, seg in enumerate(segments):
            # Raw trajectory
            raw_pos, raw_orn = compute_ee_trajectory(robot_id, seg)
            raw_pc, raw_oc, raw_path = trajectory_metrics(raw_pos, raw_orn)
            raw_pos_changes_all.extend(raw_pc)
            raw_orn_changes_all.extend(raw_oc)
            all_raw_positions.append(raw_pos)

            # EMA-smoothed trajectory
            smoothed_seg = apply_ema(seg, EMA_ALPHA)
            sm_pos, sm_orn = compute_ee_trajectory(robot_id, smoothed_seg)
            sm_pc, sm_oc, sm_path = trajectory_metrics(sm_pos, sm_orn)
            smooth_pos_changes_all.extend(sm_pc)
            smooth_orn_changes_all.extend(sm_oc)
            all_smooth_positions.append(sm_pos)

            trajectory_summaries.append({
                "file": fpath.name, "segment": seg_idx,
                "frames": len(seg), "duration_s": len(seg) * DT,
                "raw_path_mm": raw_path * 1000,
                "smooth_path_mm": sm_path * 1000,
                "raw_mean_jitter_mm": raw_pc.mean() * 1000,
                "smooth_mean_jitter_mm": sm_pc.mean() * 1000,
            })

    p.disconnect(cid)

    raw_pc = np.array(raw_pos_changes_all) * 1000  # mm
    raw_oc = np.array(raw_orn_changes_all)
    sm_pc = np.array(smooth_pos_changes_all) * 1000
    sm_oc = np.array(smooth_orn_changes_all)
    all_raw_pos = np.vstack(all_raw_positions)
    all_sm_pos = np.vstack(all_smooth_positions)

    # Print results
    print(f"\n{'='*65}")
    print(f"End-Effector Metrics ({len(all_raw_pos)} frames, "
          f"{len(raw_pc)} transitions)")
    print(f"{'='*65}")

    print(f"\n--- Workspace Envelope ---")
    for i, axis in enumerate(["X", "Y", "Z"]):
        print(f"  {axis}: [{all_raw_pos[:, i].min()*1000:.1f}, "
              f"{all_raw_pos[:, i].max()*1000:.1f}] mm  "
              f"(range: {(all_raw_pos[:, i].max() - all_raw_pos[:, i].min())*1000:.1f} mm)")

    print(f"\n--- Frame-to-Frame Cartesian Jitter ---")
    print(f"  {'Metric':<28} {'Raw':>10} {'Smoothed':>10} {'Reduction':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Pos mean (mm)':<28} {raw_pc.mean():>10.2f} {sm_pc.mean():>10.2f} "
          f"{(1 - sm_pc.mean()/raw_pc.mean())*100:>9.1f}%")
    print(f"  {'Pos std (mm)':<28} {raw_pc.std():>10.2f} {sm_pc.std():>10.2f} "
          f"{(1 - sm_pc.std()/raw_pc.std())*100:>9.1f}%")
    print(f"  {'Pos max (mm)':<28} {raw_pc.max():>10.2f} {sm_pc.max():>10.2f}")
    print(f"  {'Pos median (mm)':<28} {np.median(raw_pc):>10.2f} {np.median(sm_pc):>10.2f}")
    print(f"  {'Orn mean (deg)':<28} {raw_oc.mean():>10.2f} {sm_oc.mean():>10.2f} "
          f"{(1 - sm_oc.mean()/raw_oc.mean())*100:>9.1f}%")
    print(f"  {'Orn std (deg)':<28} {raw_oc.std():>10.2f} {sm_oc.std():>10.2f}")
    print(f"  {'Orn max (deg)':<28} {raw_oc.max():>10.2f} {sm_oc.max():>10.2f}")

    print(f"\n--- Per-Trajectory Summaries ---")
    for ts in trajectory_summaries:
        print(f"  {ts['file']} seg{ts['segment']}: {ts['frames']} frames "
              f"({ts['duration_s']:.1f}s), path={ts['raw_path_mm']:.0f}mm "
              f"(smooth: {ts['smooth_path_mm']:.0f}mm), "
              f"jitter={ts['raw_mean_jitter_mm']:.1f}mm "
              f"(smooth: {ts['smooth_mean_jitter_mm']:.1f}mm)")

    # Save CSV
    csv_path = OUTPUT_DIR / "endeffector_accuracy.csv"
    with open(csv_path, "w") as f:
        f.write("metric,raw,smoothed,reduction_pct\n")
        f.write(f"pos_jitter_mean_mm,{raw_pc.mean():.4f},{sm_pc.mean():.4f},"
                f"{(1 - sm_pc.mean()/raw_pc.mean())*100:.1f}\n")
        f.write(f"pos_jitter_std_mm,{raw_pc.std():.4f},{sm_pc.std():.4f},"
                f"{(1 - sm_pc.std()/raw_pc.std())*100:.1f}\n")
        f.write(f"pos_jitter_max_mm,{raw_pc.max():.4f},{sm_pc.max():.4f},\n")
        f.write(f"pos_jitter_median_mm,{np.median(raw_pc):.4f},{np.median(sm_pc):.4f},\n")
        f.write(f"orn_jitter_mean_deg,{raw_oc.mean():.4f},{sm_oc.mean():.4f},"
                f"{(1 - sm_oc.mean()/raw_oc.mean())*100:.1f}\n")
        f.write(f"orn_jitter_std_deg,{raw_oc.std():.4f},{sm_oc.std():.4f},\n")
        f.write(f"orn_jitter_max_deg,{raw_oc.max():.4f},{sm_oc.max():.4f},\n")
        f.write(f"\nworkspace_x_range_mm,{(all_raw_pos[:, 0].max() - all_raw_pos[:, 0].min())*1000:.1f}\n")
        f.write(f"workspace_y_range_mm,{(all_raw_pos[:, 1].max() - all_raw_pos[:, 1].min())*1000:.1f}\n")
        f.write(f"workspace_z_range_mm,{(all_raw_pos[:, 2].max() - all_raw_pos[:, 2].min())*1000:.1f}\n")
    print(f"\nCSV saved to {csv_path}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1) Position jitter: raw vs smoothed
    bins = np.linspace(0, min(raw_pc.max(), 80), 50)
    axes[0, 0].hist(raw_pc, bins=bins, alpha=0.6, label="Raw", edgecolor="black")
    axes[0, 0].hist(sm_pc, bins=bins, alpha=0.6, label=f"EMA (α={EMA_ALPHA})", edgecolor="black")
    axes[0, 0].axvline(raw_pc.mean(), color="blue", linestyle="--",
                        label=f"Raw mean={raw_pc.mean():.1f} mm")
    axes[0, 0].axvline(sm_pc.mean(), color="orange", linestyle="--",
                        label=f"Smooth mean={sm_pc.mean():.1f} mm")
    axes[0, 0].set_xlabel("Frame-to-Frame Position Change (mm)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("End-Effector Position Jitter")
    axes[0, 0].legend(fontsize=8)

    # 2) Orientation jitter: raw vs smoothed
    bins_orn = np.linspace(0, min(raw_oc.max(), 40), 50)
    axes[0, 1].hist(raw_oc, bins=bins_orn, alpha=0.6, label="Raw", edgecolor="black")
    axes[0, 1].hist(sm_oc, bins=bins_orn, alpha=0.6,
                     label=f"EMA (α={EMA_ALPHA})", edgecolor="black")
    axes[0, 1].axvline(raw_oc.mean(), color="blue", linestyle="--",
                        label=f"Raw mean={raw_oc.mean():.1f}°")
    axes[0, 1].axvline(sm_oc.mean(), color="orange", linestyle="--",
                        label=f"Smooth mean={sm_oc.mean():.1f}°")
    axes[0, 1].set_xlabel("Frame-to-Frame Orientation Change (deg)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("End-Effector Orientation Jitter")
    axes[0, 1].legend(fontsize=8)

    # 3) 3D workspace (raw)
    ax3d = fig.add_subplot(2, 2, 3, projection="3d")
    ax3d.scatter(all_raw_pos[:, 0] * 1000, all_raw_pos[:, 1] * 1000,
                 all_raw_pos[:, 2] * 1000, s=1, alpha=0.3, c="blue")
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("End-Effector Workspace (Raw)")
    axes[1, 0].set_visible(False)

    # 4) Summary bar chart
    metrics = ["Pos Jitter\n(mm)", "Orn Jitter\n(deg)"]
    raw_vals = [raw_pc.mean(), raw_oc.mean()]
    sm_vals = [sm_pc.mean(), sm_oc.mean()]
    x = np.arange(len(metrics))
    w = 0.35
    bars_r = axes[1, 1].bar(x - w / 2, raw_vals, w, label="Raw")
    bars_s = axes[1, 1].bar(x + w / 2, sm_vals, w, label=f"EMA (α={EMA_ALPHA})")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylabel("Mean Value")
    axes[1, 1].set_title("Jitter Reduction from EMA Smoothing")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)
    for bar_r, bar_s, rv, sv in zip(bars_r, bars_s, raw_vals, sm_vals):
        pct = (1 - sv / rv) * 100
        axes[1, 1].annotate(f"−{pct:.0f}%",
                             xy=(bar_s.get_x() + bar_s.get_width() / 2, bar_s.get_height()),
                             xytext=(0, 4), textcoords="offset points",
                             ha="center", fontsize=9, color="green")

    fig.tight_layout()
    chart_path = OUTPUT_DIR / "endeffector_accuracy.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

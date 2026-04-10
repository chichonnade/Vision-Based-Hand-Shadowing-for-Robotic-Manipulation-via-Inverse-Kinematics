"""Compute true IK target-vs-achieved end-effector error from real data.

Loads the target poses (extracted from .bag via pipeline) and the
corresponding IK joint angles, sets them in PyBullet, queries FK,
and measures the error between FK(theta_ik) and the original target.

Target poses .npy format: (T, 14) -- per frame:
  [left_pos(3), left_orn(4), right_pos(3), right_orn(4)]

Usage (run in WSL with conda vbhs env):
    python scripts/compute_ik_target_error.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

URDF_PATH = "robot/Dual_S101_Assembly.urdf"
ACTIONS_PATH = "paper_data/human_demo_actions.npy"
TARGETS_PATH = "paper_data/human_demo_target_poses.npy"
OUTPUT_DIR = pathlib.Path("paper_data")

RIGHT_ARM_JOINTS = [8, 9, 10, 11, 12]
RIGHT_GRIPPER_JOINT = 13
RIGHT_END_EFFECTOR = 12


def quat_angular_dist(q1, q2):
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    actions = np.load(ACTIONS_PATH)   # (T, 2, 6)
    targets = np.load(TARGETS_PATH)   # (T, 14)

    print(f"Actions shape: {actions.shape}")
    print(f"Targets shape: {targets.shape}")

    right_actions = actions[:, 1, :]  # (T, 6)
    # Right arm target: positions at [7:10], orientations at [10:14]
    right_tgt_pos = targets[:, 7:10]
    right_tgt_orn = targets[:, 10:14]

    # Valid frame: both actions and targets are non-NaN
    valid = (~np.isnan(right_actions).any(axis=1) &
             ~np.isnan(right_tgt_pos).any(axis=1) &
             ~np.isnan(right_tgt_orn).any(axis=1))

    print(f"Total frames: {len(actions)}")
    print(f"Valid frames (actions + targets both non-NaN): {valid.sum()}")

    if valid.sum() == 0:
        print("No valid frames!")
        return

    va = right_actions[valid]
    vt_pos = right_tgt_pos[valid]
    vt_orn = right_tgt_orn[valid]

    # Set up PyBullet
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(
        URDF_PATH, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
        useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    pos_errors = []
    orn_errors = []
    per_axis = []

    for i in range(len(va)):
        # Set IK-solved joint angles
        for joint_idx, angle in zip(RIGHT_ARM_JOINTS, va[i, :5]):
            p.resetJointState(robot_id, joint_idx, angle)
        p.resetJointState(robot_id, RIGHT_GRIPPER_JOINT, va[i, 5])

        # Query FK
        link_state = p.getLinkState(robot_id, RIGHT_END_EFFECTOR,
                                    computeForwardKinematics=True)
        achieved_pos = np.array(link_state[4])
        achieved_orn = np.array(link_state[5])

        pos_err = np.linalg.norm(achieved_pos - vt_pos[i]) * 1000
        orn_err = quat_angular_dist(achieved_orn, vt_orn[i])
        axis_err = np.abs(achieved_pos - vt_pos[i]) * 1000

        pos_errors.append(pos_err)
        orn_errors.append(orn_err)
        per_axis.append(axis_err)

    p.disconnect(cid)

    pos_errors = np.array(pos_errors)
    orn_errors = np.array(orn_errors)
    per_axis = np.array(per_axis)

    print(f"\n{'='*60}")
    print(f"IK Target-vs-Achieved Error ({len(pos_errors)} real frames)")
    print(f"{'='*60}")
    print(f"\nPosition Error ||FK(θ_ik) − target_pos|| (mm):")
    print(f"  Mean:   {pos_errors.mean():.3f}")
    print(f"  Std:    {pos_errors.std():.3f}")
    print(f"  Max:    {pos_errors.max():.3f}")
    print(f"  Median: {np.median(pos_errors):.3f}")
    print(f"  95th:   {np.percentile(pos_errors, 95):.3f}")

    print(f"\nPer-Axis Position Error (mm):")
    for i, axis in enumerate(["X", "Y", "Z"]):
        print(f"  {axis}: mean={per_axis[:, i].mean():.3f}, "
              f"std={per_axis[:, i].std():.3f}, "
              f"max={per_axis[:, i].max():.3f}")

    print(f"\nOrientation Error (deg):")
    print(f"  Mean:   {orn_errors.mean():.3f}")
    print(f"  Std:    {orn_errors.std():.3f}")
    print(f"  Max:    {orn_errors.max():.3f}")
    print(f"  Median: {np.median(orn_errors):.3f}")
    print(f"  95th:   {np.percentile(orn_errors, 95):.3f}")

    # Save CSV
    csv_path = OUTPUT_DIR / "ik_target_error.csv"
    with open(csv_path, "w") as f:
        f.write("metric,mean,std,max,median,p95\n")
        f.write(f"position_error_mm,{pos_errors.mean():.4f},{pos_errors.std():.4f},"
                f"{pos_errors.max():.4f},{np.median(pos_errors):.4f},"
                f"{np.percentile(pos_errors, 95):.4f}\n")
        f.write(f"orientation_error_deg,{orn_errors.mean():.4f},{orn_errors.std():.4f},"
                f"{orn_errors.max():.4f},{np.median(orn_errors):.4f},"
                f"{np.percentile(orn_errors, 95):.4f}\n")
        for i, axis in enumerate(["x", "y", "z"]):
            f.write(f"pos_error_{axis}_mm,{per_axis[:, i].mean():.4f},"
                    f"{per_axis[:, i].std():.4f},{per_axis[:, i].max():.4f},"
                    f"{np.median(per_axis[:, i]):.4f},"
                    f"{np.percentile(per_axis[:, i], 95):.4f}\n")
        f.write(f"\nn_frames,{len(pos_errors)}\n")
    print(f"\nCSV saved to {csv_path}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(pos_errors, bins=40, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(pos_errors.mean(), color="red", linestyle="--",
                        label=f"mean={pos_errors.mean():.2f} mm")
    axes[0, 0].axvline(np.median(pos_errors), color="orange", linestyle="--",
                        label=f"median={np.median(pos_errors):.2f} mm")
    axes[0, 0].set_xlabel("Position Error (mm)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"IK Position Error: ||FK(θ) − target|| (N={len(pos_errors)})")
    axes[0, 0].legend()

    axes[0, 1].hist(orn_errors, bins=40, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].axvline(orn_errors.mean(), color="red", linestyle="--",
                        label=f"mean={orn_errors.mean():.2f}°")
    axes[0, 1].axvline(np.median(orn_errors), color="blue", linestyle="--",
                        label=f"median={np.median(orn_errors):.2f}°")
    axes[0, 1].set_xlabel("Orientation Error (deg)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"IK Orientation Error (N={len(orn_errors)})")
    axes[0, 1].legend()

    axes[1, 0].boxplot([per_axis[:, i] for i in range(3)],
                        tick_labels=["X", "Y", "Z"])
    axes[1, 0].set_ylabel("Position Error (mm)")
    axes[1, 0].set_title("Per-Axis IK Position Error")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].plot(pos_errors, alpha=0.6, linewidth=0.8, label="Position error")
    axes[1, 1].axhline(pos_errors.mean(), color="red", linestyle="--",
                         alpha=0.7, label=f"mean={pos_errors.mean():.1f} mm")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Position Error (mm)")
    axes[1, 1].set_title("IK Position Error Over Time")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    chart_path = OUTPUT_DIR / "ik_target_error.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

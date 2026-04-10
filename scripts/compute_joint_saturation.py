"""Compute joint saturation analysis from .npy action files.

Reports what percentage of valid frames each joint is at or near its
URDF-defined limits, indicating workspace boundary effects.

Usage:
    python scripts/compute_joint_saturation.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

NPY_DIR = pathlib.Path("new experiment data (pick and place action files)")
OUTPUT_DIR = pathlib.Path("paper_data")

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# Right arm joint limits from URDF (radians)
JOINT_LIMITS = [
    (-1.9199, 1.9199),   # shoulder_pan
    (-1.7453, 1.7453),   # shoulder_lift
    (-1.7453, 1.5708),   # elbow_flex
    (-1.6581, 1.6581),   # wrist_flex
    (-2.7925, 2.7925),   # wrist_roll
    (-0.1745, 1.7453),   # gripper
]

SATURATION_MARGIN = 0.05  # within 0.05 rad of limit counts as "near limit"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    npy_files = sorted(NPY_DIR.glob("*.npy"))
    all_valid_frames = []

    for fpath in npy_files:
        data = np.load(fpath)
        right_arm = data[:, 1, :]  # (T, 6)
        valid_mask = ~np.isnan(right_arm).any(axis=1)
        all_valid_frames.append(right_arm[valid_mask])

    all_angles = np.vstack(all_valid_frames)
    n_frames = len(all_angles)
    print(f"Total valid frames: {n_frames}\n")

    print(f"{'Joint':<16} {'Lower':>8} {'Upper':>8} {'At Lower':>10} {'At Upper':>10} "
          f"{'Near Limit':>11} {'Mean':>8} {'Std':>8}")
    print("-" * 90)

    at_lower_pcts = []
    at_upper_pcts = []
    near_limit_pcts = []

    for i, (name, (lo, hi)) in enumerate(zip(JOINT_NAMES, JOINT_LIMITS)):
        col = all_angles[:, i]
        at_lower = np.sum(col <= lo + SATURATION_MARGIN)
        at_upper = np.sum(col >= hi - SATURATION_MARGIN)
        near_limit = at_lower + at_upper

        at_lower_pct = at_lower / n_frames * 100
        at_upper_pct = at_upper / n_frames * 100
        near_limit_pct = near_limit / n_frames * 100

        at_lower_pcts.append(at_lower_pct)
        at_upper_pcts.append(at_upper_pct)
        near_limit_pcts.append(near_limit_pct)

        print(f"{name:<16} {lo:>8.4f} {hi:>8.4f} {at_lower_pct:>9.1f}% "
              f"{at_upper_pct:>9.1f}% {near_limit_pct:>10.1f}% "
              f"{col.mean():>8.4f} {col.std():>8.4f}")

    # Save CSV
    csv_path = OUTPUT_DIR / "joint_saturation.csv"
    with open(csv_path, "w") as f:
        f.write("joint,lower_limit_rad,upper_limit_rad,at_lower_pct,at_upper_pct,"
                "near_limit_pct,mean_rad,std_rad\n")
        for i, (name, (lo, hi)) in enumerate(zip(JOINT_NAMES, JOINT_LIMITS)):
            col = all_angles[:, i]
            f.write(f"{name},{lo:.4f},{hi:.4f},{at_lower_pcts[i]:.2f},"
                    f"{at_upper_pcts[i]:.2f},{near_limit_pcts[i]:.2f},"
                    f"{col.mean():.4f},{col.std():.4f}\n")
    print(f"\nCSV saved to {csv_path}")

    # Plot: stacked bar of lower/upper saturation per joint
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(JOINT_NAMES))
    w = 0.6
    axes[0].bar(x, at_lower_pcts, w, label="At lower limit")
    axes[0].bar(x, at_upper_pcts, w, bottom=at_lower_pcts, label="At upper limit")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(JOINT_NAMES, rotation=30, ha="right")
    axes[0].set_ylabel("% of Frames Near Joint Limit")
    axes[0].set_title(f"Joint Saturation (within {SATURATION_MARGIN} rad of limit)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Plot: violin/box plot of joint angle distributions with limits
    parts = axes[1].violinplot(
        [all_angles[:, i] for i in range(6)],
        positions=x, showmeans=True, showmedians=True)
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        axes[1].plot([i - 0.3, i + 0.3], [lo, lo], "r-", linewidth=2,
                     alpha=0.7, label="Joint limits" if i == 0 else "")
        axes[1].plot([i - 0.3, i + 0.3], [hi, hi], "r-", linewidth=2, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(JOINT_NAMES, rotation=30, ha="right")
    axes[1].set_ylabel("Joint Angle (rad)")
    axes[1].set_title("Joint Angle Distributions vs. URDF Limits")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    chart_path = OUTPUT_DIR / "joint_saturation.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

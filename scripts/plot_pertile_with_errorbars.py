"""Generate per-tile success rate bar chart with std error bars."""
import pathlib
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = pathlib.Path("paper_data")

TILE_LABELS = ["#1", "#2", "#3", "#4", "#5"]
RUNS = {
    "Run 1": [10, 10, 9, 9, 7],
    "Run 2": [10, 9, 7, 9, 6],
    "Run 3": [10, 9, 8, 10, 7],
}

means = np.array([np.mean([r[i] for r in RUNS.values()]) for i in range(5)])
stds = np.array([np.std([r[i] for r in RUNS.values()], ddof=1) for i in range(5)])

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(TILE_LABELS))
bars = ax.bar(x, means, yerr=stds, capsize=5, edgecolor="black", alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(TILE_LABELS)
ax.set_xlabel("Tile")
ax.set_ylabel("Successes (out of 10)")
ax.set_ylim(0, 11)
ax.set_title("IK Retargeting Per-Tile Success (3 runs, mean ± std)")
ax.grid(axis="y", alpha=0.3)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.3, f"{m:.1f}", ha="center", fontsize=9)

total_mean = means.sum()
ax.axhline(total_mean / 5, color="red", linestyle="--", alpha=0.5,
           label=f"Mean: {total_mean:.0f}/50 = {total_mean/50*100:.1f}%")
ax.legend()

fig.tight_layout()
out = OUTPUT_DIR / "chart_pertile_errorbars.png"
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.close(fig)

if __name__ == "__main__":
    pass

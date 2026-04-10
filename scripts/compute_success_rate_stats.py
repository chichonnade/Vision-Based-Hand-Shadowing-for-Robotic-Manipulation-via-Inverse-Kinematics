"""Compute success rate statistics with confidence intervals from benchmark runs.

Usage:
    python scripts/compute_success_rate_stats.py
"""

import pathlib

import numpy as np

OUTPUT_DIR = pathlib.Path("paper_data")

RUNS = {
    "Run 1 (original)": [10, 10, 9, 9, 7],
    "Run 2": [10, 9, 7, 9, 6],
    "Run 3": [10, 9, 8, 10, 7],
}

TILE_LABELS = ["#1", "#2", "#3", "#4", "#5"]
GRASPS_PER_TILE = 10
TILES = 5
EPISODES_PER_RUN = GRASPS_PER_TILE * TILES  # 50


def wilson_ci(successes, trials, z=1.96):
    """Wilson score interval for binomial proportion (95% CI)."""
    p_hat = successes / trials
    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Pick-and-Place Success Rate Statistics (3 runs)")
    print("=" * 70)

    # Per-run totals
    run_totals = []
    run_rates = []
    for name, tiles in RUNS.items():
        total = sum(tiles)
        rate = total / EPISODES_PER_RUN
        run_totals.append(total)
        run_rates.append(rate)
        lo, hi = wilson_ci(total, EPISODES_PER_RUN)
        print(f"  {name}: {tiles} = {total}/{EPISODES_PER_RUN} "
              f"({rate*100:.0f}%), 95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]")

    # Aggregate across runs
    mean_rate = np.mean(run_rates)
    std_rate = np.std(run_rates, ddof=1)
    total_successes = sum(run_totals)
    total_trials = EPISODES_PER_RUN * len(RUNS)
    agg_lo, agg_hi = wilson_ci(total_successes, total_trials)

    print(f"\n--- Aggregate ---")
    print(f"  Mean success rate: {mean_rate*100:.1f}% ± {std_rate*100:.1f}% (std)")
    print(f"  Pooled: {total_successes}/{total_trials} = {total_successes/total_trials*100:.1f}%")
    print(f"  Pooled 95% Wilson CI: [{agg_lo*100:.1f}%, {agg_hi*100:.1f}%]")

    # Per-tile statistics
    print(f"\n--- Per-Tile Breakdown ---")
    print(f"{'Tile':<8}", end="")
    for name in RUNS:
        print(f"{name:>20}", end="")
    print(f"{'Mean':>10} {'Std':>8}")
    print("-" * 76)

    tile_means = []
    tile_stds = []
    for t in range(TILES):
        tile_scores = [RUNS[name][t] for name in RUNS]
        mean_t = np.mean(tile_scores)
        std_t = np.std(tile_scores, ddof=1)
        tile_means.append(mean_t)
        tile_stds.append(std_t)
        print(f"{TILE_LABELS[t]:<8}", end="")
        for s in tile_scores:
            print(f"{s:>20}/10", end="")
        print(f"{mean_t:>10.1f} {std_t:>8.2f}")

    # Save CSV
    csv_path = OUTPUT_DIR / "success_rate_stats.csv"
    with open(csv_path, "w") as f:
        f.write("run,tile_1,tile_2,tile_3,tile_4,tile_5,total,rate\n")
        for name, tiles in RUNS.items():
            total = sum(tiles)
            f.write(f"{name},{','.join(str(t) for t in tiles)},"
                    f"{total},{total/EPISODES_PER_RUN:.4f}\n")
        f.write(f"\nmean_rate,{mean_rate:.4f}\n")
        f.write(f"std_rate,{std_rate:.4f}\n")
        f.write(f"pooled_successes,{total_successes}\n")
        f.write(f"pooled_trials,{total_trials}\n")
        f.write(f"pooled_rate,{total_successes/total_trials:.4f}\n")
        f.write(f"wilson_ci_lower,{agg_lo:.4f}\n")
        f.write(f"wilson_ci_upper,{agg_hi:.4f}\n")
        f.write(f"\ntile_means,{','.join(f'{m:.1f}' for m in tile_means)}\n")
        f.write(f"tile_stds,{','.join(f'{s:.2f}' for s in tile_stds)}\n")
    print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()

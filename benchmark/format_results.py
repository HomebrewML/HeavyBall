import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Rectangle


def parse_loss(loss_str):
    if loss_str.strip() == "inf":
        return float("inf")
    try:
        if "e" in loss_str:
            base, exp = loss_str.split("e")
            return float(base) * 10 ** float(exp)
        return float(loss_str)
    except (ValueError, IndexError):
        return float("nan")


def read_benchmark_results(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    details_section = re.search(r"## Details\n\n(.*?)(?=\n\n|$)", content, re.DOTALL)
    if not details_section:
        raise ValueError("Could not find Details section")

    lines = details_section.group(1).strip().split("\n")[2:]
    data = []
    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 8:
            continue
        data.append({
            "benchmark": parts[0],
            "optimizer": parts[1],
            "cautious": parts[2] == "Yes",
            "mars": parts[3] == "Yes",
            "success": parts[4] == "✓",
            "runtime": float(parts[5].replace("s", "")),
            "loss": parse_loss(parts[6]),
            "attempts": int(parts[7]),
        })

    return pd.DataFrame(data)


def create_result_matrix(df):
    benchmarks = sorted(df["benchmark"].unique())
    optimizers = sorted(df["optimizer"].unique())

    success_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    attempts_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    runtime_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    loss_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)

    for _, row in df.iterrows():
        success_matrix.loc[row["benchmark"], row["optimizer"]] = row["success"]
        attempts_matrix.loc[row["benchmark"], row["optimizer"]] = row["attempts"]
        runtime_matrix.loc[row["benchmark"], row["optimizer"]] = row["runtime"]
        loss_matrix.loc[row["benchmark"], row["optimizer"]] = row["loss"]

    return success_matrix, attempts_matrix, runtime_matrix, loss_matrix


def normalize_row_attempts(row_attempts, row_success):
    """Normalize attempts within a row, considering only successful runs"""
    # Convert to boolean and handle NaN
    success_mask = to_bool(row_success)
    successful_attempts = row_attempts[success_mask]

    if len(successful_attempts) == 0:
        return pd.Series(np.nan, index=row_attempts.index)

    min_attempts = successful_attempts.min()
    max_attempts = successful_attempts.max()

    if max_attempts == min_attempts:
        # all successful attempts are the same, so they should be 0
        normalized = pd.Series(0, index=row_attempts.index)
    else:
        normalized = (row_attempts - min_attempts) / (max_attempts - min_attempts)

    # Set unsuccessful attempts to NaN
    normalized[~success_mask] = np.nan
    return normalized


def get_color_for_cell(normalized_value, success, best_in_row=False):
    """Generate color for a cell based on normalized value and success"""
    if pd.isna(normalized_value) or not success:
        return "#FF3B30"  # Failure color (red)

    # Create a gradient from light green to dark blue
    light_green = np.array([0.7, 1.0, 0.7])  # Light green
    dark_blue = np.array([0.0, 0.3, 0.8])  # Dark blue

    # Interpolate between light green and dark blue based on normalized value (inverted)
    # Invert normalized value because lower attempts are better
    color = light_green * normalized_value + dark_blue * (1 - normalized_value)

    # If this is the best in row, add a slight golden tint
    if best_in_row:
        color = color * 0.9 + np.array([0.8, 0.6, 0.0]) * 0.1  # Golden tint

    return tuple(color)


def to_bool(x):
    return x.fillna(False).astype(bool)


def create_visual_matrix(success_matrix, attempts_matrix, runtime_matrix, loss_matrix):
    plt.style.use("default")
    fig = plt.figure(figsize=(20, 15), facecolor="white")

    # Create grid for multiple panels with adjusted width ratios
    gs = plt.GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[4, 0.1, 1.5, 1.5],
        height_ratios=[1, 1],
        wspace=0.3,
        hspace=0.3,
    )

    # Main heatmap
    main_ax = fig.add_subplot(gs[:, 0])

    # Side panels (Optimizer Statistics)
    stats_ax1 = fig.add_subplot(gs[0, 2])
    stats_ax2 = fig.add_subplot(gs[0, 3])
    stats_ax3 = fig.add_subplot(gs[1, 2])
    stats_ax4 = fig.add_subplot(gs[1, 3])

    # Normalize attempts per row
    normalized_attempts = pd.DataFrame(index=success_matrix.index, columns=success_matrix.columns)

    for idx in success_matrix.index:
        normalized_attempts.loc[idx] = normalize_row_attempts(
            attempts_matrix.loc[idx], success_matrix.loc[idx]
        )

    # Calculate best performers per benchmark
    best_performers = pd.DataFrame(index=success_matrix.index, columns=success_matrix.columns)

    for idx in success_matrix.index:
        row_success = success_matrix.loc[idx]
        row_attempts = attempts_matrix.loc[idx]
        row_runtime = runtime_matrix.loc[idx]

        # Find best performer (successful with minimum attempts, then minimum runtime)
        successful_mask = to_bool(row_success)
        if successful_mask.any():
            min_attempts = row_attempts[successful_mask].min()
            min_attempts_mask = (row_attempts == min_attempts) & successful_mask

            if min_attempts_mask.sum() > 1:
                # If multiple with same attempts, use runtime as tiebreaker
                best_idx = row_runtime[min_attempts_mask].idxmin()
            else:
                best_idx = min_attempts_mask[min_attempts_mask].index[0]

            best_performers.loc[idx, best_idx] = True

    # Plot main heatmap
    for i in range(success_matrix.shape[0]):
        for j in range(success_matrix.shape[1]):
            success = success_matrix.iloc[i, j]
            attempts = attempts_matrix.iloc[i, j]
            runtime = runtime_matrix.iloc[i, j]
            normalized = normalized_attempts.iloc[i, j]
            is_best = to_bool(best_performers.iloc[i, j])

            # Get cell color
            color = get_color_for_cell(normalized, success, is_best)

            # Create cell rectangle
            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=color,
                alpha=1.0,
                edgecolor="white",
                linewidth=1,
            )
            main_ax.add_patch(rect)

            if pd.notna(attempts):
                # Add attempt count and runtime
                attempts_text = f"{int(attempts)}"
                runtime_text = f"{runtime:.1f}s"

                # Determine text color based on background brightness
                if isinstance(color, tuple):
                    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    text_color = "white" if brightness < 0.65 else "black"
                else:
                    text_color = "white" if color == "#FF3B30" else "black"

                # Add text with better formatting
                main_ax.text(
                    j,
                    i - 0.15,
                    attempts_text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                )
                main_ax.text(
                    j,
                    i + 0.15,
                    runtime_text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

                # Add star for best performer
                if is_best:
                    main_ax.text(
                        j - 0.4,
                        i - 0.4,
                        "★",
                        ha="center",
                        va="center",
                        color="#FFD700",
                        fontsize=14,
                        fontweight="bold",
                    )

    # Add grid lines
    for i in range(success_matrix.shape[0] + 1):
        main_ax.axhline(y=i - 0.5, color="#DDD", linewidth=0.5, alpha=0.5)
    for j in range(success_matrix.shape[1] + 1):
        main_ax.axvline(x=j - 0.5, color="#DDD", linewidth=0.5, alpha=0.5)

    # Format axis labels
    main_ax.set_xticks(range(len(success_matrix.columns)))
    main_ax.set_yticks(range(len(success_matrix.index)))
    main_ax.set_xticklabels(
        success_matrix.columns, rotation=45, ha="right", fontsize=10, fontweight="bold"
    )
    main_ax.set_yticklabels(success_matrix.index, fontsize=10, fontweight="bold")

    # Create statistics panels
    def create_stats_panel(ax, title, data, is_percentage=False, cmap=plt.cm.RdYlGn):
        ax.clear()
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

        # Add value labels
        for i, v in enumerate(data.values):
            text = f"{v:.1%}" if is_percentage else f"{v:.1f}"
            ax.text(v + max(data.values) * 0.02, i, text, va="center", fontsize=8)

        # Format axis
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, fontsize=8)
        ax.set_xlim(0, max(data.values) * 1.15)

        # Remove frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax

    # Calculate and plot optimizer success rates
    success_rates = success_matrix.mean()
    create_stats_panel(
        stats_ax1,
        "Optimizer Success Rates (↑)",
        success_rates.sort_values(ascending=True),
        is_percentage=True,
        cmap=plt.cm.Greens,
    )

    # Calculate and plot average attempts for successful runs
    avg_attempts = pd.Series(index=success_matrix.columns, dtype=float)
    for col in success_matrix.columns:
        success_mask = to_bool(success_matrix[col])
        successful_attempts = attempts_matrix[success_mask][col]
        avg_attempts[col] = successful_attempts.mean() if len(successful_attempts) > 0 else np.nan
    create_stats_panel(
        stats_ax2,
        "Avg Attempts Needed (↓)",
        avg_attempts.sort_values(ascending=False),
        cmap=plt.cm.GnBu,
    )

    # Calculate and plot average runtime for successful runs
    avg_runtime = pd.Series(index=success_matrix.columns, dtype=float)
    for col in success_matrix.columns:
        success_mask = to_bool(success_matrix[col])
        successful_runtime = runtime_matrix[success_mask][col]
        avg_runtime[col] = successful_runtime.mean() if len(successful_runtime) > 0 else np.nan
    create_stats_panel(
        stats_ax3,
        "Avg Runtime Needed (↓)",
        avg_runtime.sort_values(ascending=False),
        cmap=plt.cm.YlOrBr,
    )

    # Calculate and plot average loss for successful runs
    avg_best = pd.Series(index=runtime_matrix.columns, dtype=float)
    avg_best[:] = 0
    for (_, ru), (_, su), (_, at) in zip(
        runtime_matrix.iterrows(), success_matrix.iterrows(), attempts_matrix.iterrows()
    ):
        score = ru + at * 1000  # minimize attempt count. if tie, use runtime
        score = score - su * 1e12  # only count successful runs
        avg_best[runtime_matrix.columns[score.argmin()]] += 1
    avg_best /= avg_best.sum() / 100
    create_stats_panel(
        stats_ax4, "Best Optimizer% (↑)", avg_best.sort_values(ascending=True), cmap=plt.cm.YlGn
    )

    # Add title and subtitle
    plt.suptitle("Optimizer Performance Matrix", y=0.98, fontsize=16, fontweight="bold")
    fig.text(
        0.25,
        0.94,
        "Color intensity shows relative number of attempts per benchmark (row-normalized)\n"
        + "★ indicates best performer per benchmark",
        ha="center",
        fontsize=11,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])

    return fig


def main(file: str = "benchmark_results.md"):
    df = read_benchmark_results(file)
    success_matrix, attempts_matrix, runtime_matrix, loss_matrix = create_result_matrix(df)

    # Create the enhanced visual matrix
    _fig = create_visual_matrix(success_matrix, attempts_matrix, runtime_matrix, loss_matrix)

    # Save with high quality
    plt.savefig(
        "benchmark_matrix.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.5
    )
    plt.close()

    # Print text summary
    print("\nSummary Statistics:")
    success_rates = success_matrix.mean().sort_values(ascending=False)
    avg_attempts = attempts_matrix.mean()
    median_attempts = attempts_matrix.median()
    avg_runtime = runtime_matrix.mean()
    median_runtime = runtime_matrix.median()
    avg_loss = loss_matrix.mean()
    median_loss = loss_matrix.median()

    print("\nSuccess Rates by Optimizer:")
    for optimizer in success_rates.index:
        rate = success_rates[optimizer] * 100
        print(f"{optimizer}: {rate:.1f}%")

    print("\nAverage Attempts by Optimizer (Mean, Median):")
    for optimizer in avg_attempts.index:
        mean_val = avg_attempts[optimizer]
        median_val = median_attempts[optimizer]
        print(f"{optimizer}: Mean={mean_val:.1f}, Median={median_val:.1f}")

    print("\nAverage Runtime by Optimizer (Mean, Median):")
    for optimizer in avg_runtime.index:
        mean_val = avg_runtime[optimizer]
        median_val = median_runtime[optimizer]
        print(f"{optimizer}: Mean={mean_val:.1f}s, Median={median_val:.1f}s")

    print("\nAverage Loss by Optimizer (Mean, Median):")
    for optimizer in avg_loss.index:
        mean_val = avg_loss[optimizer]
        median_val = median_loss[optimizer]
        print(f"{optimizer}: Mean={mean_val:.2e}, Median={median_val:.2e}")


if __name__ == "__main__":
    typer.run(main)

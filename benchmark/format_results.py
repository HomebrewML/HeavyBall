import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Configure matplotlib for better rendering
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


class BenchmarkParser:
    """Handles parsing of benchmark results from markdown files."""

    @staticmethod
    def parse_loss(loss_str: str) -> float:
        """Parse loss string to float, handling special cases."""
        loss_str = loss_str.strip().lower()

        if not loss_str or loss_str == "nan":
            return float("nan")
        if loss_str == "inf":
            return float("inf")

        try:
            if "e" in loss_str:
                base, exp = loss_str.split("e")
                return float(base) * (10 ** float(exp))
            return float(loss_str)
        except (ValueError, AttributeError):
            return float("nan")

    @staticmethod
    def process_option(value: str, label: str) -> str:
        """Process Yes/No/Other options into formatted strings."""
        if value == "No":
            return ""
        if value == "Yes":
            return f"{label}-"
        return f"{value}-"

    def read_benchmark_results(self, file_path: str) -> pd.DataFrame:
        """Read and parse benchmark results from a markdown file."""
        with open(file_path, "r") as f:
            content = f.read()

        # Extract details section
        details_match = re.search(r"## Details\n\n(.*?)(?=\n##|\n\Z)", content, re.DOTALL | re.IGNORECASE)

        if not details_match:
            raise ValueError("Details section not found in the file.")

        table_content = details_match.group(1).strip()

        # Extract data rows
        table_match = re.search(r"\|:?-+:(.*?)\|\n(.*)", table_content, re.DOTALL)
        if not table_match:
            raise ValueError("Table format not recognized.")

        lines = table_match.group(2).strip().split("\n")

        data = []
        for line in lines:
            if not line.strip() or line.startswith("|---"):
                continue

            parts = [p.strip() for p in line.split("|")[1:-1]]

            if len(parts) < 8:
                continue

            try:
                # Process optimizer name
                caution = self.process_option(parts[2], "cautious")
                mars = self.process_option(parts[3], "mars")
                optimizer = f"{caution}{mars}{parts[1]}"
                optimizer = optimizer.replace("Foreach", "").replace("Cached", "").strip()

                data.append({
                    "benchmark": parts[0],
                    "optimizer": optimizer,
                    "success": parts[4] == "✓",
                    "runtime": float(parts[5].replace("s", "")) if parts[5] else float("nan"),
                    "loss": self.parse_loss(parts[6]) if parts[6] else float("nan"),
                    "attempts": int(parts[7]) if parts[7].isdigit() else 0,
                })
            except (IndexError, ValueError):
                print(f"Warning: Skipping malformed line: {line}")
                continue

        return pd.DataFrame(data)


class BenchmarkAnalyzer:
    """Analyzes benchmark results and creates matrices for visualization."""

    @staticmethod
    def create_success_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Create a matrix of success counts by task and optimizer."""
        if df.empty:
            return pd.DataFrame()

        benchmarks = sorted(df["benchmark"].unique())
        optimizers = sorted(df["optimizer"].unique())

        # Create initial success matrix
        success_matrix = pd.DataFrame(0, index=benchmarks, columns=optimizers, dtype=int)

        # Fill success matrix
        for _, row in df.iterrows():
            if row["success"]:
                if row["benchmark"] in success_matrix.index and row["optimizer"] in success_matrix.columns:
                    success_matrix.loc[row["benchmark"], row["optimizer"]] = 1

        # Aggregate by base task
        base_tasks = sorted(set(b.split("-")[0] for b in benchmarks))
        success_total_matrix = pd.DataFrame(0, index=base_tasks, columns=optimizers, dtype=int)

        for benchmark in success_matrix.index:
            base_task = benchmark.split("-")[0]
            if base_task in success_total_matrix.index:
                success_total_matrix.loc[base_task] += success_matrix.loc[benchmark]

        return success_total_matrix

    @staticmethod
    def calculate_normalized_scores(matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized scores for visualization."""
        if matrix.empty:
            return pd.DataFrame()

        # Filter out empty rows
        tasks_to_keep = matrix.sum(axis=1) > 0
        filtered_matrix = matrix[tasks_to_keep].copy()

        if filtered_matrix.empty:
            return pd.DataFrame()

        # Normalize by row maximum
        normalized = filtered_matrix.copy()
        for idx in filtered_matrix.index:
            row_max = filtered_matrix.loc[idx].max()
            if row_max > 0:
                normalized.loc[idx] = (filtered_matrix.loc[idx] / row_max) * 100

        return normalized


class BenchmarkVisualizer:
    """Creates visualizations for benchmark results."""

    def __init__(self):
        # Create custom colormap for better accessibility
        colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
        self.cmap = LinearSegmentedColormap.from_list("custom_blues", colors)

    def create_heatmap(self, success_matrix: pd.DataFrame, normalized_matrix: pd.DataFrame) -> plt.Figure:
        """Create an improved heatmap visualization."""
        if normalized_matrix.empty:
            return None

        # Calculate scores for ordering
        optimizer_scores = (normalized_matrix**0.5).mean(axis=0)
        optimizer_order = optimizer_scores.sort_values(ascending=False).index

        # Reorder matrix
        plot_matrix = normalized_matrix[optimizer_order].copy()

        # Add summary statistics
        optimizer_means = plot_matrix.mean(axis=0)
        task_means = plot_matrix.mean(axis=1)
        overall_mean = optimizer_means.mean()

        # Add average row and column
        plot_matrix.loc["Average"] = optimizer_means
        plot_matrix["Task Avg"] = pd.concat([task_means, pd.Series([overall_mean], index=["Average"])])

        # Create figure
        fig_height = max(8, len(plot_matrix.index) * 0.5)
        fig_width = max(10, len(plot_matrix.columns) * 0.6)

        fig, (ax_main, ax_legend) = plt.subplots(
            1, 2, figsize=(fig_width + 3, fig_height), gridspec_kw={"width_ratios": [fig_width, 3]}
        )

        # Draw heatmap
        self._draw_heatmap(ax_main, plot_matrix, success_matrix, optimizer_order)

        # Add legend
        self._add_legend(ax_legend, success_matrix)

        plt.tight_layout()
        return fig

    def _draw_heatmap(
        self, ax: plt.Axes, plot_matrix: pd.DataFrame, success_matrix: pd.DataFrame, optimizer_order: pd.Index
    ) -> None:
        """Draw the main heatmap."""
        # Create color array
        color_array = plot_matrix.values.copy()

        # Create mask for cells with no successes
        mask = np.zeros_like(color_array, dtype=bool)
        for i, task in enumerate(plot_matrix.index[:-1]):  # Exclude average row
            for j, opt in enumerate(plot_matrix.columns[:-1]):  # Exclude average column
                if task in success_matrix.index and opt in optimizer_order:
                    if success_matrix.loc[task, opt] == 0:
                        mask[i, j] = True

        # Draw heatmap
        im = ax.imshow(color_array, cmap=self.cmap, aspect="auto", vmin=0, vmax=100, alpha=0.9)

        # Add text annotations
        for i in range(len(plot_matrix.index)):
            for j in range(len(plot_matrix.columns)):
                value = plot_matrix.iloc[i, j]

                # Determine text color based on background
                text_color = "white" if value > 60 else "black"

                # Add count for non-summary cells
                if i < len(plot_matrix.index) - 1 and j < len(plot_matrix.columns) - 1 and not mask[i, j]:
                    task = plot_matrix.index[i]
                    opt = plot_matrix.columns[j]
                    if task in success_matrix.index and opt in success_matrix.columns:
                        count = success_matrix.loc[task, opt]
                        text = f"{value:.0f}%\n({count})"
                    else:
                        text = f"{value:.0f}%"
                else:
                    text = f"{value:.0f}%"

                # Gray out cells with no successes
                if mask[i, j]:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="lightgray", alpha=0.7)
                    ax.add_patch(rect)
                    text_color = "darkgray"

                ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9, fontweight="normal")

        # Add divider lines
        ax.axhline(len(plot_matrix.index) - 1.5, color="black", linewidth=2)
        ax.axvline(len(plot_matrix.columns) - 1.5, color="black", linewidth=2)

        # Set ticks and labels
        ax.set_xticks(range(len(plot_matrix.columns)))
        ax.set_yticks(range(len(plot_matrix.index)))
        ax.set_xticklabels(plot_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(plot_matrix.index)

        # Labels
        ax.set_xlabel("Optimizer", fontweight="bold")
        ax.set_ylabel("Task", fontweight="bold")
        ax.set_title(
            "Benchmark Success Rates\n(Normalized to best performer per task)", fontsize=16, fontweight="bold", pad=20
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Success Rate (%)", rotation=270, labelpad=20)

        # Style
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)

    def _add_legend(self, ax: plt.Axes, success_matrix: pd.DataFrame) -> None:
        """Add a legend explaining the visualization."""
        ax.axis("off")

        legend_text = [
            "Legend:",
            "",
            "• Percentages show relative success",
            "  (100% = best for that task)",
            "",
            "• Numbers in parentheses show",
            "  absolute success count",
            "",
            "• Gray cells indicate no successes",
            "",
            f"• Total tasks: {len(success_matrix.index)}",
            f"• Total optimizers: {len(success_matrix.columns)}",
        ]

        for i, text in enumerate(legend_text):
            weight = "bold" if i == 0 else "normal"
            ax.text(0.1, 0.9 - i * 0.08, text, transform=ax.transAxes, fontsize=10, fontweight=weight, va="top")

    def create_summary_plots(self, df: pd.DataFrame, normalized_matrix: pd.DataFrame) -> plt.Figure:
        """Create additional summary visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Success rate by optimizer
        self._plot_optimizer_success_rates(axes[0, 0], df)

        # 2. Runtime distribution
        self._plot_runtime_distribution(axes[0, 1], df)

        # 3. Task difficulty
        self._plot_task_difficulty(axes[1, 0], normalized_matrix)

        # 4. Optimizer consistency
        self._plot_optimizer_consistency(axes[1, 1], normalized_matrix)

        plt.suptitle("Benchmark Analysis Summary", fontsize=16, fontweight="bold")
        plt.tight_layout()

        return fig

    def _plot_optimizer_success_rates(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot overall success rates by optimizer."""
        success_rates = df.groupby("optimizer")["success"].mean() * 100
        success_rates = success_rates.sort_values(ascending=True)

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(success_rates)))
        bars = ax.barh(range(len(success_rates)), success_rates.values, color=colors)

        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, success_rates.values)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%", va="center")

        ax.set_yticks(range(len(success_rates)))
        ax.set_yticklabels(success_rates.index)
        ax.set_xlabel("Success Rate (%)")
        ax.set_title("Overall Success Rate by Optimizer")
        ax.set_xlim(0, max(100, success_rates.max() * 1.1))
        ax.grid(axis="x", alpha=0.3)

    def _plot_runtime_distribution(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot runtime distribution for successful runs."""
        successful_runs = df[df["success"] & df["runtime"].notna()]

        if not successful_runs.empty:
            optimizers = successful_runs["optimizer"].unique()
            runtime_data = [
                successful_runs[successful_runs["optimizer"] == opt]["runtime"].values for opt in optimizers
            ]

            # Create violin plot
            parts = ax.violinplot(runtime_data, positions=range(len(optimizers)), showmeans=True, showmedians=True)

            # Color the violins
            colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(optimizers)))
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            ax.set_xticks(range(len(optimizers)))
            ax.set_xticklabels(optimizers, rotation=45, ha="right")
            ax.set_ylabel("Runtime (seconds)")
            ax.set_title("Runtime Distribution for Successful Runs")
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3)

    def _plot_task_difficulty(self, ax: plt.Axes, normalized_matrix: pd.DataFrame) -> None:
        """Plot task difficulty based on average success rates."""
        if "Task Avg" in normalized_matrix.columns:
            task_avg = normalized_matrix["Task Avg"][:-1]  # Exclude average row
        else:
            task_avg = normalized_matrix.mean(axis=1)

        task_avg = task_avg.sort_values()

        colors = plt.cm.RdYlGn(task_avg.values / 100)
        bars = ax.barh(range(len(task_avg)), task_avg.values, color=colors)

        # Add value labels
        for i, (bar, avg) in enumerate(zip(bars, task_avg.values)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{avg:.1f}%", va="center")

        ax.set_yticks(range(len(task_avg)))
        ax.set_yticklabels(task_avg.index)
        ax.set_xlabel("Average Success Rate (%)")
        ax.set_title("Task Difficulty (Lower = Harder)")
        ax.set_xlim(0, 105)
        ax.grid(axis="x", alpha=0.3)

    def _plot_optimizer_consistency(self, ax: plt.Axes, normalized_matrix: pd.DataFrame) -> None:
        """Plot optimizer consistency (std dev of success rates)."""
        # Remove summary rows/columns
        data_matrix = normalized_matrix.copy()
        if "Task Avg" in data_matrix.columns:
            data_matrix = data_matrix.drop("Task Avg", axis=1)
        if "Average" in data_matrix.index:
            data_matrix = data_matrix.drop("Average", axis=0)

        consistency = data_matrix.std(axis=0).sort_values()

        colors = plt.cm.coolwarm(consistency.values / consistency.max())
        _bars = ax.bar(range(len(consistency)), consistency.values, color=colors)

        ax.set_xticks(range(len(consistency)))
        ax.set_xticklabels(consistency.index, rotation=45, ha="right")
        ax.set_ylabel("Standard Deviation")
        ax.set_title("Optimizer Consistency\n(Lower = More Consistent)")
        ax.grid(axis="y", alpha=0.3)


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    file: str = typer.Argument("benchmark_results.md", help="Path to benchmark results file"),
    output_dir: str = typer.Option(".", help="Directory to save output files"),
    create_summary: bool = typer.Option(True, help="Create additional summary plots"),
    dpi: int = typer.Option(300, help="DPI for saved images"),
):
    """Generate visualizations from benchmark results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Parse benchmark results
    parser = BenchmarkParser()
    try:
        df = parser.read_benchmark_results(file)
        if df.empty:
            print("Error: No data found in benchmark file.")
            return
    except Exception as e:
        print(f"Error reading benchmark file: {e}")
        return

    print(f"Loaded {len(df)} benchmark results")

    # Analyze results
    analyzer = BenchmarkAnalyzer()
    success_matrix = analyzer.create_success_matrix(df)

    if success_matrix.empty:
        print("Error: No successful runs found.")
        return

    normalized_matrix = analyzer.calculate_normalized_scores(success_matrix)

    print(f"Analyzed {len(success_matrix.index)} tasks and {len(success_matrix.columns)} optimizers")

    # Create visualizations
    visualizer = BenchmarkVisualizer()

    # Main heatmap
    heatmap_fig = visualizer.create_heatmap(success_matrix, normalized_matrix)
    if heatmap_fig:
        heatmap_path = output_path / "benchmark_heatmap.png"
        heatmap_fig.savefig(heatmap_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Saved heatmap to: {heatmap_path}")
        plt.close(heatmap_fig)

    # Summary plots
    if create_summary:
        summary_fig = visualizer.create_summary_plots(df, normalized_matrix)
        if summary_fig:
            summary_path = output_path / "benchmark_summary.png"
            summary_fig.savefig(summary_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"Saved summary plots to: {summary_path}")
            plt.close(summary_fig)

    # Save processed data
    csv_path = output_path / "benchmark_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved processed data to: {csv_path}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    app()

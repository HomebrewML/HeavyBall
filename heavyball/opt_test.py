import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heavyball import ForeachAdaLomo, ForeachAdamW, ForeachLaProp, SophiaH


class SimpleModel(nn.Module):
    """A simple model for testing optimizers"""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def test_optimizers():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create synthetic data
    input_dim = 10
    n_samples = 1000
    X = torch.randn(n_samples, input_dim)

    # Create a target function with heterogeneous curvature
    w = torch.zeros(input_dim)
    w[0], w[1] = 10.0, 0.1  # Dimension : sharp, flat
    y = X @ w + 0.1 * torch.randn(n_samples)

    # Split data into training batches
    batch_size = 32

    # Define loss function
    criterion = nn.MSELoss()

    # Function to train for one epoch and return losses
    def train_epoch(model, optimizer, criterion, n_steps=100, log_interval=1000):
        model.train()
        losses = []
        times = []

        for step in range(n_steps):
            # Randomly sample a batch
            indices = torch.randperm(n_samples)[:batch_size]
            X_batch, y_batch = X[indices], y[indices]

            # Forward pass
            optimizer.zero_grad()

            # Measure time per step
            start_time = time.time()

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            end_time = time.time()
            times.append(end_time - start_time)

            # Record loss
            losses.append(loss.item())

            if step % log_interval == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")

        return losses, sum(times) / len(times)

    # Create models
    models = {
        "ForeachAdamW": SimpleModel(input_dim=input_dim),
        "ForeachLaProp": SimpleModel(input_dim=input_dim),
        "SophiaH": SimpleModel(input_dim=input_dim),
        "ForeachAdaLomo": SimpleModel(input_dim=input_dim),
    }

    # Create optimizers
    optimizers = {
        "ForeachAdamW": ForeachAdamW(
            models["ForeachAdamW"].parameters(),
            lr=0.001,
            weight_decay=0.01
        ),
        "ForeachLaProp": ForeachLaProp(
            models["ForeachLaProp"].parameters(),
            lr=0.001,
            weight_decay=0.01
        ),
        "SophiaH": SophiaH(
            models["SophiaH"].parameters(),
            lr=0.001,
            weight_decay=0.01
        ),
        "ForeachAdaLomo": ForeachAdaLomo(
            models["ForeachAdaLomo"].parameters(),
            lr=0.0005,
            beta=0.99,
            eps=1e-8,
            weight_decay=0.01,
            warmup_steps=10
        )
    }

    # Training settings
    n_steps, log_interval, results = 11000, 1000, {}

    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name}:")
        losses, avg_time = train_epoch(models[name], optimizer, criterion, n_steps, log_interval)
        results[name] = {
            "losses": losses,
            "avg_time": avg_time,
            "final_loss": losses[-1]
        }

    # Print timing information
    print("\nAverage time per step (ms):")
    for name, data in results.items():
        print(f"{name}: {data['avg_time']*1000:.3f} ms")

    # Print final losses
    print("\nFinal losses:")
    for name, data in results.items():
        print(f"{name}: {data['final_loss']:.6f}")

    # Memory usage comparison
    try:
        # Measure memory usage (requires GPU)
        if torch.cuda.is_available():
            memory_usage = {}

            X_cuda = X.cuda()
            y_cuda = y.cuda()

            # Function to measure memory usage
            def measure_memory(model, optimizer):
                model.cuda()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                indices = torch.randperm(n_samples)[:batch_size].cuda()
                X_batch, y_batch = X_cuda[indices], y_cuda[indices]

                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                mem_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                model.cpu()  # Move model back to CPU
                return mem_usage

            # Measure memory usage for each optimizer
            for name, optimizer in optimizers.items():
                memory_usage[name] = measure_memory(models[name], optimizer)

            print("\nPeak memory usage (MB):")
            for name, usage in memory_usage.items():
                print(f"{name}: {usage:.2f}")

    except Exception as e:
        print("\nSkipping memory usage comparison (requires GPU)")
        print(f"Error: {e}")

    # Plot loss curves with improved line styles
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))

    for i, (name, data) in enumerate(results.items()):
        plt.plot(data["losses"],
                 label=name,
                 color=colors[i],
                 linewidth=1.5,      # 얇은 선
                 alpha=0.8,          # 적절한 투명도
                 zorder=3)           # 그리드 위에 표시

    plt.title("Optimizer Comparison - Loss Curves", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')  # Log scale often works better for loss curves

    # Add loss values to the plot with improved styling
    for i, (name, data) in enumerate(results.items()):
        plt.annotate(f"{data['final_loss']:.6f}",
                    xy=(len(data["losses"])-1, data["losses"][-1]),
                    xytext=(10, 10+(i*20)),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", alpha=0.7),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                    fontsize=9)

    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=300)
    print("\nPlot saved as 'optimizer_comparison.png'")

    # Compare convergence speed (steps to reach certain loss threshold)
    thresholds = [0.5, 0.1, 0.05, 0.01]
    print("\nSteps to reach loss thresholds:")

    headers = ["Optimizer"] + [f"Loss < {t}" for t in thresholds]
    print(" | ".join(headers))
    print("-" * (sum(len(h) for h in headers) + len(headers) - 1))

    for name, data in results.items():
        steps = []
        for threshold in thresholds:
            try:
                step = next(i for i, loss in enumerate(data["losses"]) if loss < threshold)
                steps.append(str(step))
            except StopIteration:
                steps.append("N/A")
        print(f"{name} | {' | '.join(steps)}")

    # Combined bar chart for final loss and average time
    plt.figure(figsize=(14, 8))

    # Setup for subplot
    names = list(results.keys())
    final_losses = [results[name]["final_loss"] for name in names]
    avg_times = [results[name]["avg_time"]*1000 for name in names]  # Convert to ms

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Final Loss
    bars1 = ax1.bar(names, final_losses, color=colors, alpha=0.8)
    ax1.set_title("Final Loss Comparison", fontsize=14)
    ax1.set_xlabel("Optimizer", fontsize=12)
    ax1.set_ylabel("Final Loss (MSE)", fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels above bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.6f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

    # Plot 2: Average Step Time
    bars2 = ax2.bar(names, avg_times, color=colors, alpha=0.8)
    ax2.set_title("Average Step Time", fontsize=14)
    ax2.set_xlabel("Optimizer", fontsize=12)
    ax2.set_ylabel("Time per Step (ms)", fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels above bars
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f"{height:.3f} ms",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig("optimizer_performance.png", dpi=300)
    print("Combined performance comparison plot saved as 'optimizer_performance.png'")

    return results


if __name__ == "__main__":
    test_optimizers()

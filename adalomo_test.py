import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import optimizers from heavyball
# Assuming the heavyball package is installed or in PYTHONPATH
from heavyball import AdaLomo, AdamW


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


def test_adalomo():
    print("Testing AdaLomo optimizer...")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create synthetic data
    input_dim = 10
    n_samples = 1000
    X = torch.randn(n_samples, input_dim)
    # Create a target function with heterogeneous curvature
    w = torch.zeros(input_dim)
    w[0] = 10.0  # Sharp dimension
    w[1] = 0.1   # Flat dimension
    y = X @ w + 0.1 * torch.randn(n_samples)

    # Split data into training batches
    batch_size = 32

    # Define loss function
    criterion = nn.MSELoss()

    # Function to train for one epoch and return losses
    def train_epoch(model, optimizer, criterion, n_steps=100):
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

            if step % 1000 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")

        return losses, sum(times) / len(times)

    # Create models and optimizers
    model_adam = SimpleModel(input_dim=input_dim)
    model_lomo = SimpleModel(input_dim=input_dim)
    model_adalomo = SimpleModel(input_dim=input_dim)

    # Initialize models with the same weights
    for p_adam, p_lomo, p_adalomo in zip(model_adam.parameters(),
                                        model_lomo.parameters(),
                                        model_adalomo.parameters()):
        # Use the same initialization
        p_lomo.data.copy_(p_adam.data)
        p_adalomo.data.copy_(p_adam.data)

    # Create optimizers
    adam_opt = AdamW(
        model_adam.parameters(),
        lr=0.001,
        weight_decay=0.01
    )

    adalomo_opt = AdaLomo(
        model_adalomo.parameters(),
        lr=0.0005,  # AdaLomo often works well with slightly lower learning rate
        beta=0.99,   # Higher beta for second moment estimation
        eps=1e-8,
        weight_decay=0.01,
        warmup_steps=10
    )

    # Train models
    n_steps = 11000
    print("\nTraining with AdamW:")
    adam_losses, adam_time = train_epoch(model_adam, adam_opt, criterion, n_steps)

    print("\nTraining with AdaLomo:")
    adalomo_losses, adalomo_time = train_epoch(model_adalomo, adalomo_opt, criterion, n_steps)

    # Print timing information
    print("\nAverage time per step:")
    print(f"AdamW:   {adam_time*1000:.3f} ms")
    print(f"AdaLomo: {adalomo_time*1000:.3f} ms")

    # Print final losses
    print("\nFinal losses:")
    print(f"AdamW:   {adam_losses[-1]:.6f}")
    print(f"AdaLomo: {adalomo_losses[-1]:.6f}")

    # Memory usage comparison
    try:
        # Measure memory usage (requires GPU)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

            # Models to GPU
            model_adam.cuda()
            model_adalomo.cuda()

            X_cuda = X.cuda()
            y_cuda = y.cuda()

            # Test memory usage
            def measure_memory(model, optimizer):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                indices = torch.randperm(n_samples)[:batch_size].cuda()
                X_batch, y_batch = X_cuda[indices], y_cuda[indices]

                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                return torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

            adam_mem = measure_memory(model_adam, adam_opt)
            adalomo_mem = measure_memory(model_adalomo, adalomo_opt)

            print("\nPeak memory usage (MB):")
            print(f"AdamW:   {adam_mem:.2f}")
            print(f"AdaLomo: {adalomo_mem:.2f}")

            # Memory efficiency
            print("\nMemory efficiency (relative to AdamW):")
            print(f"AdaLomo: {adam_mem/adalomo_mem:.2f}x")
    except:
        print("\nSkipping memory usage comparison (requires GPU)")


if __name__ == "__main__":
    test_adalomo()

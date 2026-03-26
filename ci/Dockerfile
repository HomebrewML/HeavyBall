FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir opt-einsum "numpy>=2.3.4" pytest hypothesis lightbench pandas optuna optunahub gpytorch scikit-learn typer --break-system-packages

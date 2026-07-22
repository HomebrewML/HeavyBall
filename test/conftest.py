"""Pin each pytest-xdist worker to its own GPU so GPU tests (notably the feature matrix) fan out across the
node's GPUs. No-op without xdist: a serial run keeps the default device."""
import os


def pytest_configure(config):
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(int(worker.removeprefix("gw")) % torch.cuda.device_count())

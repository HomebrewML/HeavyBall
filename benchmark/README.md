# HeavyBall Benchmark Suite

This repository contains a suite of benchmark problems designed to test the capabilities and robustness of `torch.optim` optimizers. The framework is designed to be modular and extensible, allowing for the easy addition of new benchmarks and optimizers.

## Setup

To get started, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the entire benchmark suite with a specific set of optimizers, use the `run_all_benchmarks.py` script.

For example, to run the benchmarks for Adam and SGD on easy and medium difficulties:
```bash
python run_all_benchmarks.py --opt Adam --opt SGD --difficulties easy medium
```

You can specify multiple optimizers and difficulties. The results will be written to `benchmark_results.md`.

For more options, see:
```bash
python run_all_benchmarks.py --help
```

### Core Philosophy and Design Principles

Based on the analysis of [`benchmark_template.py`](benchmark_template.py), [`optimizer_template.py`](optimizer_template.py), and [`utils.py`](utils.py), the core framework is a modular and extensible system designed for benchmarking `torch` optimizers. It is composed of three primary components:

1.  **Benchmarks**: Each benchmark is a self-contained optimization problem defined by a class. It provides a loss function to be minimized, the parameters to be optimized, and a specific success condition.
2.  **Optimizers**: These are expected to adhere to the `torch.optim.Optimizer` interface. A template is provided to facilitate the creation of new custom optimizers.
3.  **Benchmarking Harness**: The [`utils.py`](utils.py) script contains the engine that runs the benchmarks. It includes a sophisticated `Validator` class for monitoring convergence and detecting failures, an `Objective` class that encapsulates the entire process of running a benchmark (including hyperparameter search with `optuna`), and a high-level `trial` function to orchestrate the whole process.

#### Design Principles

The framework is built upon the following design principles:

*   **Modularity**: The separation of benchmarks, optimizers, and the runner logic allows for easy extension. New benchmarks or optimizers can be added by implementing simple, well-defined interfaces without needing to alter the core execution logic.
*   **Automation**: The framework automates the process of running benchmarks, including a hyperparameter search for learning rate and momentum coefficients. This simplifies the process of evaluating an optimizer's performance across various problems.
*   **Robustness**: The `Validator` class provides sophisticated logic to terminate runs that are not making progress, saving computational resources. It checks for multiple failure conditions, such as stagnating loss and lack of improvement over time.
*   **PyTorch Native**: The framework is built on PyTorch and leverages its core components like `autograd` for gradient computation and the standard `Optimizer` class structure.

#### Benchmark Structure

A benchmark is structured as a class, as defined in [`benchmark_template.py`](benchmark_template.py:4), with the following key methods:

*   **`__init__(self, device)`**: Initializes the benchmark's state. This is where the model parameters and any necessary data are created. The parameters to be optimized are stored in `self.params`.
*   **`__call__(self) -> torch.Tensor`**: This is the core of the benchmark, defining the objective function. It computes and returns a scalar loss tensor that the optimizer will try to minimize.
*   **`has_succeeded(self) -> bool`**: Defines the "win condition" for the benchmark. It returns `True` if the optimization is considered successful, typically based on the loss value falling below a predefined threshold.
*   **`get_params(self) -> list[torch.Tensor]`**: Returns the list of `torch.Tensor` parameters that will be optimized.

### Existing Benchmarks

| File | Category | Description |
|---|---|---|
| [`adversarial_gradient.py`](adversarial_gradient.py) | Noise Robustness | Tests optimizer's robustness to an oscillating adversarial component added to the gradient, simulating adversarial noise. |
| [`batch_size_scaling.py`](batch_size_scaling.py) | Noise Robustness | Tests how optimizers handle noise at different scales, which are modulated by a simulated, randomly changing batch size. |
| [`beale.py`](beale.py) | General/Classic | Implements the Beale function, a classic optimization benchmark with sharp valleys, to test general performance. Also includes visualization. |
| [`char_rnn.py`](char_rnn.py) | Sequence & Memory (RNN/LSTM) | A character-level language model using an LSTM on text data, testing performance on a sequence task requiring memory. |
| [`constrained_optimization.py`](constrained_optimization.py) | Landscape Traversal | A simple quadratic objective with a penalty-based constraint, creating a sharp ridge in the loss landscape for the optimizer to navigate. |
| [`discontinuous_gradient.py`](discontinuous_gradient.py) | Gradient Characteristics | Tests optimizer robustness to non-smooth landscapes by using an objective function with a discontinuous gradient at the origin. |
| [`dynamic_landscape.py`](dynamic_landscape.py) | Dynamic Environments | Tests an optimizer's ability to track a continuously shifting target in a non-stationary loss landscape. |
| [`exploding_gradient.py`](exploding_gradient.py) | Gradient Characteristics | Tests an optimizer's numerical stability and handling of extreme gradient values by using an exponential function that causes gradients to grow rapidly. |
| [`gradient_delay.py`](gradient_delay.py) | Gradient Characteristics | Tests an optimizer's ability to handle asynchronous or delayed updates by using gradients from previous steps. |
| [`gradient_noise_scale.py`](gradient_noise_scale.py) | Noise Robustness | Tests an optimizer's ability to handle dynamically changing noise levels, where the noise scale anneals over time. |
| [`grokking.py`](grokking.py) | Landscape Traversal | Tests for 'grokking' by training a model on a modular arithmetic task, examining if the optimizer can find a generalizable solution after a long period of memorization. |
| [`layer_wise_scale.py`](layer_wise_scale.py) | Multi-Scale & Conditioning | Tests an optimizer's ability to handle parameters with vastly different gradient scales by scaling the loss contribution of different layers. |
| [`minimax.py`](minimax.py) | Landscape Traversal | Implements a minimax objective function which creates a saddle point, testing the optimizer's ability to escape such points. |
| [`momentum_utilization.py`](momentum_utilization.py) | Landscape Traversal | Tests the effective use of momentum by creating an oscillating loss landscape with many local minima. |
| [`noisy_matmul.py`](noisy_matmul.py) | Gradient Characteristics | Tests optimizer stability in deep networks by performing a sequence of matrix multiplications, which can lead to exploding or vanishing gradients. |
| [`parameter_scale.py`](parameter_scale.py) | Multi-Scale & Conditioning | Tests an optimizer's ability to handle parameters initialized at widely different scales, creating a poorly conditioned problem. |
| [`plateau_navigation.py`](plateau_navigation.py) | Landscape Traversal | Tests an optimizer's ability to navigate a loss landscape with a large, flat plateau region surrounded by a steep cliff. |
| [`powers_varying_target.py`](powers_varying_target.py) | Multi-Scale & Conditioning | Creates a complex, poorly conditioned landscape by raising parameters to different powers against a non-zero, varying target. |
| [`powers.py`](powers.py) | Multi-Scale & Conditioning | Creates a poorly conditioned problem by raising parameters to various powers, resulting in gradients of different magnitudes. |
| [`quadratic_varying_scale.py`](quadratic_varying_scale.py) | Multi-Scale & Conditioning | Tests handling of ill-conditioned problems by creating a quadratic objective where each parameter's gradient has a different scale. |
| [`quadratic_varying_target.py`](quadratic_varying_target.py) | General/Classic | A simple quadratic bowl (sphere) benchmark where the minimum is at a non-zero target vector. |
| [`rastrigin.py`](rastrigin.py) | General/Classic | Implements the Rastrigin function, a classic highly multi-modal benchmark for testing global optimization capabilities. |
| [`rosenbrock.py`](rosenbrock.py) | General/Classic | Implements the Rosenbrock function, a classic benchmark known for its narrow, banana-shaped valley that is difficult to navigate. |
| [`saddle_point.py`](saddle_point.py) | Landscape Traversal | Tests an optimizer's ability to escape a classic saddle point (x^2 - y^2) and visualizes the optimization paths. |
| [`scale_invariant.py`](scale_invariant.py) | Multi-Scale & Conditioning | Tests an optimizer's handling of parameters at different scales by using a logarithmic objective on parameters initialized across many orders of magnitude. |
| [`sparse_gradient.py`](sparse_gradient.py) | Gradient Characteristics | Tests an optimizer's performance when gradients are sparse, which is simulated by randomly masking parameter updates. |
| [`wide_linear.py`](wide_linear.py) | Multi-Scale & Conditioning | Tests optimizer performance on a model with a very wide linear layer, which can present conditioning challenges. |
| [`xor_digit_rnn.py`](xor_digit_rnn.py) | Sequence & Memory (RNN/LSTM) | Tests an RNN's ability to solve the parity task (XOR sum) on a sequence of bits, requiring it to maintain a memory state. |
| [`xor_digit.py`](xor_digit.py) | Sequence & Memory (RNN/LSTM) | Tests an LSTM's ability to solve the parity task (XOR sum) on a sequence of bits, a classic test of sequence memory. |
| [`xor_sequence_rnn.py`](xor_sequence_rnn.py) | Sequence & Memory (RNN/LSTM) | A sequence-to-sequence task where an RNN must learn to compute the element-wise XOR of two input sequences. |
| [`xor_sequence.py`](xor_sequence.py) | Sequence & Memory (RNN/LSTM) | A sequence-to-sequence task where an LSTM must learn to compute the element-wise XOR of two input sequences. |
| [`xor_spot_rnn.py`](xor_spot_rnn.py) | Sequence & Memory (RNN/LSTM) | Tests an RNN's ability to learn a pointwise forget mechanism by predicting the XOR of two values at randomly marked spots in a sequence. |
| [`xor_spot.py`](xor_spot.py) | Sequence & Memory (RNN/LSTM) | Tests an LSTM's ability to learn a pointwise forget mechanism by predicting the XOR of two values at randomly marked spots in a sequence. |

### Contributing

This framework is designed for extensibility. You can contribute by adding new benchmarks to test optimizers against, or by implementing new optimizers to evaluate.

#### Adding a New Benchmark

To add a new benchmark, follow these steps:

1.  **Create a new Python file** for your benchmark (e.g., `my_new_benchmark.py`).
2.  **Implement the benchmark class.** Your class should follow the structure provided in [`benchmark_template.py`](benchmark_template.py:4).
    *   **`__init__(self, device)`**: Initializes the benchmark's state. This is where the model parameters and any necessary data are created. The parameters to be optimized are stored in `self.params`.
    *   **`__call__(self) -> torch.Tensor`**: This is the core of the benchmark, defining the objective function. It computes and returns a scalar loss tensor that the optimizer will try to minimize.
    *   **`has_succeeded(self) -> bool`**: Defines the "win condition" for the benchmark. It returns `True` if the optimization is considered successful, typically based on the loss value falling below a predefined threshold.
    *   **`get_params(self) -> list[torch.Tensor]`**: Returns the list of `torch.Tensor` parameters that will be optimized.
3.  **Add your benchmark to `run_all_benchmarks.py`**: To include your new benchmark in the full suite, you will need to import it in [`run_all_benchmarks.py`](run_all_benchmarks.py) and add it to the list of benchmarks to be run.
4.  **(Optional) Add a description to `README.md`**: Add a row to the "Benchmark Tests" table in this `README.md` to document your new benchmark.

#### Adding a New Optimizer

To add a new optimizer:

1.  **Implement your optimizer class.** Your optimizer must follow the `torch.optim.Optimizer` interface. You can use [`optimizer_template.py`](optimizer_template.py:1) as a starting point.
2.  **Make your optimizer available to the benchmark runner.** This is typically done by adding it to the `optimizer_mapping` dictionary in [`utils.py`](utils.py:1).
3.  **Run the benchmarks.** You can now run your optimizer against the benchmarks using the `run_all_benchmarks.py` script. For example:
    ```bash
    python run_all_benchmarks.py --opt YourNewOptimizerName --difficulties easy

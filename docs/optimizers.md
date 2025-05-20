# `heavyball`: Efficient Optimizers

* [Public API](#Public-API)
    - [Foreach Optimizers](#Foreach-Optimizers)
    - [`heavyball.utils`](#heavyball.utils)
    - [Example Usage](#Example-Usage)

* [`heavyball.chainable`](##heavyball.chainable)
    - [Core Concept](#Core-Concept)
    - [`FunctionTransform` and Guards](#FunctionTransform-and-Guards)
    - [Chaining Transformations](#Chaining-Transformations)
    - [Building Optimizers](#Building-Optimizers)
    - [Creating New Transformations](#Creating-New-Transformations)

* [Optimizer Recommendations](#Optimizer-Recommendations)
    - [Choosing the Right Optimizer](#Choosing-the-Right-Optimizer)

---

The `heavyball` library provides a collection of efficient optimizers designed for deep learning. It leverages
techniques like preconditioning, momentum, and adaptive learning rates to accelerate training and improve convergence.
The library's core strength lies in its `chainable` API, which allows for flexible composition of optimizers, enabling
users to build custom optimization strategies.

## Public API

The `heavyball` library exposes the following optimizers through its main namespace:

### Foreach Optimizers

These optimizers are designed to be efficient by operating on batches of parameters simultaneously using `foreach`
operations whenever possible.

#### `ForeachAdamW`

```python
class ForeachAdamW(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
# ...
```

A foreach implementation of the AdamW optimizer. It incorporates weight decay into the update rule and uses adaptive
learning rates based on the first and second moments of the gradients.

**Key Parameters:**

* **`lr`**: Learning rate.
* **`betas`**: Coefficients used for computing running averages of the gradient and its square.
* **`eps`**: A small constant for numerical stability.
* **`weight_decay`**: Weight decay coefficient.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`storage_dtype`**: The floating-point type to be used for internal state. `"float32"` or `"bfloat16"`.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.
* **`palm`**: Enables/disables PaLM's beta2 schedule.
* **`beta2_scale`**: if we're using the PaLM schedule, `beta2 = step ** -beta2_scale`

#### `ForeachRMSprop`

```python
class ForeachRMSprop(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-6, weight_decay=0, warmup_steps=0, r=0.0,
                 weight_lr_power=2.0, foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False,
                 caution: bool = False, mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
# ...
```

A foreach implementation of a debiased RMSprop optimizer (Note: this is different from `torch.optim.RMSprop`). It uses
adaptive learning rates based on the second moment of the gradients.

**Key Parameters:**

* **`lr`**: Learning rate.
* **`betas`**: Coefficients used for computing running averages of the squared gradient.
* **`eps`**: A small constant for numerical stability.
* **`weight_decay`**: Weight decay coefficient.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`r`**: Schedule-Free coefficient that controls dependence of the learning rate on step count.
* **`weight_lr_power`**: Schedule-Free coefficient that controls the sensitivity of `r` to the learning rate.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`storage_dtype`**: The floating-point type to be used for internal state. `"float32"` or `"bfloat16"`.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.
* **`palm`**: Enables/disables PaLM's beta2 schedule.
* **`beta2_scale`**: if we're using the PaLM schedule, `beta2 = step ** -beta2_scale`

#### `ForeachSFAdamW`

```python
class ForeachSFAdamW(C.ScheduleFree):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-6, weight_decay=0, warmup_steps=0, r=0.0,
                 weight_lr_power=2.0, foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False,
                 caution: bool = False, mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
# ...
```

A foreach implementation of the Schedule-Free AdamW optimizer. It combines the benefits of AdamW with the Schedule-Free
approach, which dynamically adjusts the learning rate based on the current state of optimization.

**Key Parameters:**

* **`lr`**: Base learning rate. The effective learning rate at each step depends on `lr`, `r`, and `weight_lr_power`.
* **`betas`**: Coefficients used for computing running averages of the gradient and its square.
* **`eps`**: A small constant for numerical stability.
* **`weight_decay`**: Weight decay coefficient.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`r`**: Schedule-Free coefficient that controls dependence of the learning rate on step count.
* **`weight_lr_power`**: Schedule-Free coefficient that controls the sensitivity of `r` to the learning rate.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`storage_dtype`**: The floating-point type to be used for internal state. `"float32"` or `"bfloat16"`.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.
* **`palm`**: Enables/disables PaLM's beta2 schedule.
* **`beta2_scale`**: if we're using the PaLM schedule, `beta2 = step ** -beta2_scale`

#### `PaLMForeachSFAdamW`

```python
class PaLMForeachSFAdamW(ForeachSFAdamW):
    palm: bool = True
```

A specialized version of `ForeachSFAdamW` with PaLM's beta2 schedule enabled by default.

#### `ForeachADOPT`

```python
class ForeachADOPT(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
# ...
```

A foreach implementation of the ADOPT optimizer, which uses a debiased estimate of the second moment of the gradients.

**Key Parameters:**

* **`lr`**: Learning rate.
* **`betas`**: Coefficients used for computing running averages of the gradient and its square.
* **`eps`**: A small constant for numerical stability.
* **`weight_decay`**: Weight decay coefficient.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`storage_dtype`**: The floating-point type to be used for internal state. `"float32"` or `"bfloat16"`.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.
* **`palm`**: Enables/disables PaLM's beta2 schedule.
* **`beta2_scale`**: if we're using the PaLM schedule, `beta2 = step ** -beta2_scale`

#### `ForeachMuon`

```python
class ForeachMuon(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8,
                 nesterov: bool = True):
# ...
```

A foreach implementation of the Muon optimizer, incorporating orthogonal updates via the `orthogonalize_update`
transformation.

**Key Parameters:**

* **`lr`**: Learning rate.
* **`betas`**: Coefficients used for computing running averages of the gradient and its square.
* **`eps`**: A small constant for numerical stability.
* **`weight_decay`**: Weight decay coefficient.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`storage_dtype`**: The floating-point type to be used for internal state. `"float32"` or `"bfloat16"`.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.
* **`palm`**: Enables/disables PaLM's beta2 schedule.
* **`beta2_scale`**: if we're using the PaLM schedule, `beta2 = step ** -beta2_scale`
* **`nesterov`**: Enables/disables Nesterov momentum.

#### `ForeachLaProp`

```python
class ForeachLaProp(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
# ...
```

A foreach implementation of the LaProp optimizer.

**Key Parameters:**

* **`lr`**: Learning rate.
* **`betas`**: Coefficients used for computing running averages of the gradient and its square.
* **`eps`**: A small constant for numerical stability.
* **`weight_decay`**: Weight decay coefficient.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`storage_dtype`**: The floating-point type to be used for internal state. `"float32"` or `"bfloat16"`.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.
* **`palm`**: Enables/disables PaLM's beta2 schedule.
* **`beta2_scale`**: if we're using the PaLM schedule, `beta2 = step ** -beta2_scale`

#### `MuonLaProp`

```python
class MuonLaProp(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
# ...
```

A variant of LaProp that incorporates orthogonal updates via the `orthogonalize_update` transformation.

**Key Parameters:**

* **`lr`**: Learning rate.
* **`betas`**: Coefficients used for computing running averages of the gradient and its square.
* **`eps`**: A small constant for numerical stability.
* **`weight_decay`**: Weight decay coefficient.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`storage_dtype`**: The floating-point type to be used for internal state. `"float32"` or `"bfloat16"`.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.
* **`palm`**: Enables/disables PaLM's beta2 schedule.
* **`beta2_scale`**: if we're using the PaLM schedule, `beta2 = step ** -beta2_scale`

#### `ForeachSOAP`

```python
class ForeachSOAP(C.BaseOpt):
    use_precond_schedule: bool = False

    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.95), shampoo_beta: float = 0.95, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 2048,  #
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 correct_bias: bool = True, warmup_steps: int = 1,
                 split: bool = False, foreach: bool = True, mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, palm: bool = C.use_default, precond_scheduler=(1 / 3, 9),
                 beta2_scale: float = 0.8, use_precond_schedule: bool = C.use_default,
                 gradient_clipping: C.str_or_fn = C.use_default, update_clipping: C.str_or_fn = C.use_default):
# ...
```

A foreach implementation of the SOAP (Second-Order Adaptive Preconditioner) optimizer. It uses a preconditioner based on
the second-order statistics of the gradients to accelerate convergence.

**Key Parameters:**

* **`lr`**: Learning rate.
* **`betas`**: Coefficients used for computing running averages of the gradient.
* **`shampoo_beta`**: Coefficient used for computing running average of the preconditioner.
* **`eps`**: A small constant for numerical stability.
* **`weight_decay`**: Weight decay coefficient.
* **`precondition_frequency`**: Frequency of preconditioner updates. If using `use_precond_schedule`, this parameter is
  ignored.
* **`max_precond_dim`**: Maximum dimension of the preconditioner.
* **`merge_dims`**: Whether to merge dimensions when forming the preconditioner.
* **`precondition_1d`**: Whether to use a 1D preconditioner for 1D parameters.
* **`normalize_grads`**: Whether to normalize gradients before applying SOAP.
* **`correct_bias`**: Enables/disables bias correction for the running averages.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`split`**: Whether to split large dimensions when forming the preconditioner.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`palm`**: Enables/disables PaLM's beta2 schedule.
* **`precond_scheduler`**: A tuple `(power, log_base)` specifying the preconditioner update schedule, where the update
  probability is `1 / (step ** power * log_base)`. This parameter is only used if `use_precond_schedule` is `True`.
* **`beta2_scale`**: if we're using the PaLM schedule, `beta2 = step ** -beta2_scale`
* **`use_precond_schedule`**: Whether to use a dynamic preconditioner update schedule instead of a fixed frequency.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.

#### `PaLMForeachSOAP`

```python
class PaLMForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = False
    palm: bool = True
```

A specialized version of `ForeachSOAP` with PaLM's beta2 schedule enabled by default.

#### `PrecondScheduleForeachSOAP`

```python
class PrecondScheduleForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = True
```

A specialized version of `ForeachSOAP` that uses a dynamic preconditioner update schedule.

#### `PrecondSchedulePaLMForeachSOAP`

```python
class PrecondSchedulePaLMForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = True
    palm: bool = True
```

A specialized version of `ForeachSOAP` with both PaLM-specific modifications and a dynamic preconditioner update
schedule enabled by default.

#### `ForeachPSGDKron`

```python
class ForeachPSGDKron(C.BaseOpt):
    delayed: bool = False
    cached: bool = False
    exp_avg_input: bool = True

    def __init__(self, params, lr=0.001, beta=0.9, weight_decay=0.0, preconditioner_update_probability=None,
                 max_size_triangular=2048, min_ndim_triangular=2, memory_save_mode=None,
                 momentum_into_precond_update=True, warmup_steps: int = 1, merge_dims: bool = False,
                 split: bool = False, store_triu_as_line: bool = True, foreach: bool = True, q_dtype='float32',
                 stochastic_schedule: bool = True, storage_dtype: str = 'float32', mars: bool = False,
                 caution: bool = False, mars_gamma: float = 0.0025, delayed: Optional[bool] = C.use_default,
                 cached: Optional[bool] = C.use_default, exp_avg_input: Optional[bool] = C.use_default,
                 gradient_clipping: C.str_or_fn = C.use_default, update_clipping: C.str_or_fn = C.use_default,  #
                 # expert parameters
                 precond_init_scale=1.0, precond_lr=0.1):
# ...
```

A foreach implementation of the PSGD (Preconditioned Stochastic Gradient Descent) optimizer with Kronecker-factored
preconditioners.

**Key Parameters:**

* **`lr`**: Learning rate.
* **`beta`**: Coefficient used for computing running average of the gradient.
* **`weight_decay`**: Weight decay coefficient.
* **`preconditioner_update_probability`**: Probability of updating the preconditioner at each step. If `None`, a default
  schedule is used.
* **`max_size_triangular`**: Maximum size of triangular matrices used in the preconditioner.
* **`min_ndim_triangular`**: Minimum number of dimensions for a tensor to be considered for triangular preconditioner.
* **`memory_save_mode`**: Memory saving mode for the preconditioner. Can be `None`, `"one_diag"`, or `"all_diag"`.
* **`momentum_into_precond_update`**: Whether to use momentum in the preconditioner update.
* **`warmup_steps`**: Number of steps for linear learning rate warmup.
* **`merge_dims`**: Whether to merge dimensions when forming the preconditioner.
* **`split`**: Whether to split large dimensions when forming the preconditioner.
* **`store_triu_as_line`**: Whether to store the upper triangular part of the preconditioner as a 1D vector.
* **`foreach`**: Enables/disables the use of `foreach` operations.
* **`q_dtype`**: The floating-point type to be used for the preconditioner. `"float32"` or `"bfloat16"`.
* **`stochastic_schedule`**: Whether to use a stochastic schedule for updating the preconditioner.
* **`storage_dtype`**: The floating-point type to be used for internal state. `"float32"` or `"bfloat16"`.
* **`mars`**: Enables/disables Mars correction.
* **`caution`**: Enables/disables the use of a cautious update rule, avoiding updates that point in the opposite
  direction to the gradients.
* **`mars_gamma`**: Mars correction coefficient.
* **`delayed`**: Enables/disables delayed preconditioner updates.
* **`cached`**: Enables/disables caching of preconditioner-related computations.
* **`exp_avg_input`**: Whether to apply `exp_avg` to the input before calculating the preconditioner.
* **`gradient_clipping`**: Gradient clipping function or method. See `heavyball.utils` for available options.
* **`update_clipping`**: Update clipping function or method. See `heavyball.utils` for available options.
* **`precond_init_scale`**: Initial scale of the preconditioner.
* **`precond_lr`**: Learning rate for preconditioner updates.

#### `ForeachPurePSGD`

```python
class ForeachPurePSGD(ForeachPSGDKron):
    exp_avg_input: bool = False
```

A specialized version of `ForeachPSGDKron` that does not apply `exp_avg` to the input before calculating the
preconditioner.

#### `ForeachCachedDelayedPSGDKron`

```python
class ForeachCachedDelayedPSGDKron(ForeachPSGDKron):
    delayed: bool = True
    cached: bool = True
```

A specialized version of `ForeachPSGDKron` with both delayed preconditioner updates and caching enabled by default.

#### `ForeachCachedPSGDKron`

```python
class ForeachCachedPSGDKron(ForeachPSGDKron):
    cached: bool = True
```

A specialized version of `ForeachPSGDKron` with caching enabled by default.

#### `ForeachDelayedPSGD`

```python
class ForeachDelayedPSGD(ForeachPSGDKron):
    delayed: bool = True
```

A specialized version of `ForeachPSGDKron` with delayed preconditioner updates enabled by default.

## `heavyball.utils`

The `heavyball.utils` module provides several important functions and settings that users may find useful:

### Settings

* **`compile_mode`**:  (defaults to `"max-autotune-no-cudagraphs"`) Controls the compilation mode used by
  `torch.compile`. Setting this to `"default"` or `"max-autotune-no-cudagraphs"` improves performance at the cost of
  increasd compile time. Setting it to `None` disables compilation.
* **`dynamic`**: (defaults to `False`) Enables/disables dynamic shapes during compilation. Enabling this reduces
  compilation time but may lead to slower execution.
* **`zeroth_power_mode`**: (defaults to `"qr"`) Controls the method used for computing the zeroth power of a matrix (
  orthogonalization) in certain preconditioners. Options include:
    * `"qr"`: Uses QR decomposition.
    * `"svd"`: Uses singular value decomposition.
    * `"newtonschulz"`: Uses Newton-Schulz iteration.

### Gradient/Update Clipping

The following functions are used for gradient and update clipping. They can be passed to the `gradient_clipping` or
`update_clipping` arguments of the optimizers:

* **`l2_clip_`**: Clips the gradient/update by its L2 norm.
* **`rmsnorm_clip_`**: Clips the gradient/update by its RMS norm.
* **`trust_region_clip_`**: Clips the gradient/update using a trust region method.
* **`mu_law_compress`**: Compresses the gradient/update using the µ-law algorithm.
* **`a_law_compress`**: Compresses the gradient/update using the A-law algorithm.
* **`identity`**: Does not modify the gradient/update (no clipping).

### Other Utilities

* **`set_torch`**: Sets recommended PyTorch settings for performance, including enabling cuDNN benchmark mode, disabling
  deterministic algorithms, setting the precision of float32 matrix multiplications, and enabling opt-einsum with the "
  auto-hq" strategy.
* **`clean`**: Clears the CUDA cache.
* **`hook_optimizer_into_model`**: Hooks an optimizer into a model's `post_accumulate_grad_hook`.
* **`fused_hook`**: Hooks an optimizer into a model's `post_accumulate_grad_hook`, fusing multiple parameter updates
  into a single step.
* **`disable_caution_scaling`**: Disables the scaling factor applied when `caution` is enabled in optimizers.

## Example Usage

```python
import torch
from torch import nn
import heavyball

# Define a simple model
model = nn.Linear(10, 2)

# Create an optimizer
optimizer = heavyball.ForeachAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
# alternative:
optimizer = heavyball.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Generate some dummy data
input = torch.randn(1, 10)
target = torch.randn(1, 2)

# Training loop
for _ in range(100):
    # Forward pass
    output = model(input)
    loss = (output - target).sum()

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    # optional: zero gradients; optimizer.step() already does this, which is different from torch.optim
    optimizer.zero_grad()
```

This example demonstrates how to create an `AdamW` optimizer and use it to train a simple linear model. You can easily
replace `AdamW` with any other optimizer from the `heavyball` library and customize its behavior using the various
available parameters and settings.

By using `heavyball`'s optimizers and understanding the options in `heavyball.utils`, users can achieve better
performance, control over training, and easier experimentation with advanced optimization techniques.


---

# `heavyball.chainable`: A Composable Optimizer API

The `heavyball.chainable` module provides a powerful and flexible way to build optimizers through function composition,
similar to Optax. It allows you to chain together a sequence of transformations to create custom optimization algorithms
tailored to your specific needs. This modular approach makes it easy to experiment with different optimization
strategies and build complex optimizers from simple, reusable components.

## Core Concept

At the heart of `heavyball.chainable` lies the concept of gradient transformations. A gradient transformation is simply
a function that takes a state dictionary, a group dictionary, an update tensor, a gradient tensor, and a parameter
tensor as input, and returns a new (or modified) update tensor. These transformations can be chained together to form an
optimization algorithm.

The state dictionary stores any persistent state needed by the transformation, such as momentum buffers or
preconditioners. The group dictionary contains hyperparameters specific to a group of parameters. The update tensor is
the current update being processed, the gradient tensor is the gradient of the loss with respect to the parameter, and
the parameter tensor is the parameter itself.

### Function Signature

A typical gradient transformation function has the following signature:

```python

def my_transformation(state: dict, group: dict, update: List[torch.Tensor], grad: List[torch.Tensor],
                      param: List[torch.Tensor]) -> torch.Tensor:
    # ... transformation logic ...
    return update
```

or

```python
@C.no_state_no_foreach
def my_transformation(group: dict, update: torch.Tensor, grad: torch.Tensor, param: torch.Tensor, *args,
                      **kwargs) -> torch.Tensor:
    # ... transformation logic ...
    return update
```

Note that the second version has no state and processes updates one by one, while the first version processes updates
in parallel.

These functions modify the `update` in place or return a new tensor.

### Example: Scaling by the learning rate

```python
from heavyball import chainable as C


@C.no_state_no_foreach
def scale_by_learning_rate(group: dict, update: torch.Tensor, grad: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    return update * group["lr"]
```

## `FunctionTransform` and Guards

To make it easier to create gradient transformations, `heavyball.chainable` provides the `FunctionTransform` class and a
set of "guard" decorators.

### `FunctionTransform`

`FunctionTransform` is a base class for gradient transformations that provides a common interface and helper methods. It
takes a function `fn` as input and stores it along with its name.

```python
class FunctionTransform:
    def __init__(self, fn):
        self.fn = fn
        self.fn_name = self.get_fn().__name__

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        raise NotImplementedError

    def get_fn(self):
        if hasattr(self.fn, 'get_fn'):
            return self.fn.get_fn()
        return self.fn

    def val_name(self, name):
        return f"{self.fn_name}_{name}"
```

### Guards

Guards are decorators that help manage the state dictionary and ensure that transformations are applied correctly. They
handle common tasks like initializing state variables and preventing redundant computations.

#### `zero_guard`

The `zero_guard` decorator ensures that a specific variable in the state dictionary is initialized to zero if it doesn't
exist.

```python
@C.zero_guard("momentum")
def my_transformation(state, group, update, grad, param, momentum):
    # ... momentum will be initialized to zero if it doesn't exist in state ...
    return update
```

#### `copy_guard`

The `copy_guard` decorator creates a copy of a specified input (update, grad, or param) and stores it in the state
dictionary.

```python
@C.copy_guard(0, "update_copy")  # 0 refers to the 'update' argument
def my_transformation(state, group, update, grad, param, update_copy):
    # ... update_copy will be a copy of the update tensor ...
    return update
```

#### `general_guard`

The `general_guard` decorator provides a more flexible way to manage state. It allows you to specify a custom
initialization function that is called if a specific variable is not found in the state.

```python
def init_preconditioner(state, group, update, grad, param, **kwargs):


# ... initialize preconditioner ...

@C.general_guard("precond", init_fn=init_preconditioner)
def my_transformation(state, group, update, grad, param, precond):
    # ... precond will be initialized using init_preconditioner if it doesn't exist ...
    return update
```

#### `no_state`

The `no_state` decorator indicates that a transformation does not use or modify any state.

#### `no_state_no_foreach`

The `no_state_no_foreach` decorator indicates that a transformation does not use or modify any state and also does not
support `foreach` implementations.

## Chaining Transformations

The power of `heavyball.chainable` comes from its ability to chain transformations together. This is achieved through
the `chain` function.

```python
def chain(state: Union[callable, dict], group, grad, param, *fns):
    update = [torch.clone(g, memory_format=torch.preserve_format) for g in grad]
    skip_update = False
    for fn in fns:
        try:
            update = fn(state, group, update, grad, param)
        except SkipUpdate:
            skip_update = True
            continue
        if update is None:
            break
    if not skip_update and update is not None:
        utils.update_param_(param, update, group['lr'], group['weight_decay'], caution=group['caution'], grad=grad)
```

The `chain` function takes a state dictionary, a group dictionary, a gradient tensor, a parameter tensor, and a sequence
of gradient transformations as input. It applies each transformation in order, passing the output of one transformation
as the input to the next.

## Building Optimizers

The `ChainOpt` class provides a convenient way to build optimizers from chained transformations.

```python
class ChainOpt(utils.StatefulOptimizer):
    # ...
    def __init__(self, params, defaults, foreach: bool, *fns):
        # ...
        self.fns = tuple(fns)

    def _step(self, group):
        # ...
        if not group['foreach'] or len(p) == 1:
            for param, grad in zip(p, g):
                chain(self.state_, group, [grad], [param], *self.fns)
        else:
            chain(self.state_, group, g, p, *self.fns)
        # ...
```

### BaseOpt

The `BaseOpt` class extends `ChainOpt` and provides additional features like gradient clipping, update clipping, and
optional PaLM beta2 schedule.

```python
class BaseOpt(ChainOpt):
    # ...
    def __init__(self, params, defaults, foreach: bool, gradient_clipping: str_or_fn, update_clipping: str_or_fn,
                 palm: bool = use_default, *fns, compile_step: bool = use_default, promote: bool = use_default):
# ...
```

### `ScheduleFree`

The `ScheduleFree` class provides a convenient interface for using the `update_by_schedule_free` transformation.

### Predefined Transformations

`heavyball.chainable` provides a number of predefined gradient transformations, including:

* `exp_avg`: Calculates the exponential moving average of the gradients.
* `scale_by_exp_avg_sq`: Scales the updates by the inverse square root of the exponential moving average of squared
  gradients.
* `scale_by_adam`: Scales the updates using the Adam algorithm.
* `update_by_adam`: Updates the parameters using the Adam algorithm.
* `scale_by_laprop`: Scales the updates using the LaProp algorithm.
* `update_by_laprop`: Updates the parameters using the LaProp algorithm.
* `update_by_schedule_free`: Updates the parameters using the Schedule-Free algorithm.
* `update_by_adopt`: Updates the parameters using the ADOPT algorithm.
* `scale_by_adopt`: Scales the updates using the ADOPT algorithm.
* `orthogonalize_update`: Orthogonalizes the update tensor.
* `nesterov_momentum`: Applies Nesterov momentum to the updates.
* `heavyball_momentum`: Applies heavy-ball momentum to the updates.
* `scale_by_soap`: Scales the updates using the SOAP preconditioner.
* `scale_by_psgd`: Scales the updates using the PSGD preconditioner.
* `scale_by_delayed_psgd`: Scales the updates using the delayed PSGD preconditioner.
* `update_by_psgd`: Updates the parameters using the PSGD preconditioner.
* `update_by_delayed_psgd`: Updates the parameters using the delayed PSGD preconditioner.
* `palm_beta2`: Modifies the beta2 parameter for PaLM optimizers.

## Creating New Transformations

You can easily create new gradient transformations by following the function signature and using the provided guards and
`FunctionTransform` class.

### Example: Clipping gradients by norm

```python
from heavyball import chainable as C
from heavyball import utils


@C.no_state
def clip_by_global_norm(group: dict, update: torch.Tensor, grad: torch.Tensor, param: torch.Tensor,
                        max_norm: float) -> torch.Tensor:
    """Clips the gradient by its global norm."""
    total_norm = torch.norm(torch.stack([torch.norm(g) for g in grad]))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return [u * clip_coef for u in update]
    return update
```

### Example: L2-Normalization of updates

```python
from heavyball import chainable as C
from heavyball import utils


@C.no_state_no_foreach
def l2_normalize_updates(group: dict, update: torch.Tensor, grad: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """L2-normalizes the updates."""
    norm = update.norm()
    if norm > 0:
        return update / norm
    return update
```

---

## Optimizer Recommendations

This hierarchy ranks optimizers from most recommended (top) to least recommended (bottom) for general deep learning
tasks. However, the best choice always depends on your specific model, dataset, and computational resources.

**1. Preconditioned Optimizers (SOAP and PSGD):**

- **Recommendation:** **Start here.** These are generally the most powerful and efficient optimizers in `heavyball`.
- **`ForeachSOAP`** (and its variants: `PaLMForeachSOAP`, `PrecondScheduleForeachSOAP`,
  `PrecondSchedulePaLMForeachSOAP`):
    - **Strengths:**
        - **Adaptive Preconditioning:** SOAP dynamically adapts to the curvature of the loss landscape using
          second-order information, leading to faster convergence, especially in ill-conditioned problems.
        - **Robustness:** Less sensitive to hyperparameter choices compared to Adam.
        - **Strong Empirical Performance:** Often outperforms other optimizers across various tasks and architectures.
    - **Weaknesses:**
        - **Computational Cost:** Higher per-step cost due to preconditioner computation and updates.
        - **Memory Usage:** Can use more memory than simpler optimizers, particularly for large models.
        - **`precondition_frequency` or `precond_scheduler`:** Needs to be tuned, though the default schedule usually
          works well.
    - **When to use:**
        - **Complex models and datasets:** Where optimization is challenging.
        - **When training stability is crucial.**
        - **When you can't retune hyperparameters.**
    - **Variants:**
        - `PaLMForeachSOAP`: Enables PaLM's beta2 schedule by default.
        - `PrecondScheduleForeachSOAP`: Uses a dynamic schedule for preconditioner updates.
        - `PrecondSchedulePaLMForeachSOAP`: Combines the PaLM schedule with a dynamic preconditioner schedule.

- **`ForeachPSGDKron`** (and its variants: `ForeachPurePSGD`, `ForeachCachedDelayedPSGDKron`, `ForeachCachedPSGDKron`,
  `ForeachDelayedPSGD`):
    - **Strengths:**
        - **Preconditioning:** Uses Kronecker-factored approximations to capture curvature information, providing many
          of the benefits of second-order methods at a lower cost than full curvature methods.
        - **Efficiency:** Relatively efficient in terms of computation.
        - **Tunability:** Offers many options for customization.
        - **Convergence:** Tends to converge faster than SOAP.
    - **Weaknesses:**
        - **No baseline:** SOAP can copy Adam's hyperparameters - PSGD requires more tuning.
        - **Complexity:** Has many hyperparameters to tune.
    - **When to use:**
        - **Large models:** Where memory is a constraint.
        - **When `ForeachSOAP` is too computationally expensive.**
        - **When you want potentially the best performance regardless of computational cost.**
    - **Variants:**
        - `ForeachPurePSGD`: Disables exponential averaging of the input when calculating the preconditioner.
        - `ForeachCachedDelayedPSGDKron`: Caches preconditioner-related computations and uses delayed preconditioner
          updates.
        - `ForeachCachedPSGDKron`: Caches preconditioner-related computations.
        - `ForeachDelayedPSGD`: Uses delayed preconditioner updates.

**2. Muon:**

- **`ForeachMuon`** (and `MuonLaProp`):
    - **Strengths:**
        - **Momentum with Orthogonal Updates:** Combines momentum with orthogonalized updates, which can
          improve stability and exploration.
        - **Good Generalization:** Often leads to better generalization performance compared to Adam.
    - **Weaknesses:**
        - **Performance:** Generally outperformed by SOAP and PSGD.
        - **Computational Cost:** Higher overheads than SOAP and PSGD.
    - **When to use:**
        - **When generalization is a primary concern.**
        - **When you want an optimizer less prone to finding sharp minima.**

**3. Adam-Based Optimizers:**

- **`ForeachLaProp`**:
    - **Strengths:**
        - **Backward Compatibility:** Can use Adam's hyperparameters, but allows a larger range of betas.
        - **Stability:** More stable than Adam.
    - **Weaknesses:**
        - **Performance:** Generally outperformed by SOAP, PSGD, and Muon.
    - **When to use:**
        - **When you want less risk or better losses than Adam, but can't run advanced methods.**

- **`ForeachAdamW`** (and `ForeachSFAdamW`, `PaLMForeachSFAdamW`):
    - **Strengths:**
        - **Widely Used:** A popular and well-established optimizer.
    - **Weaknesses:**
        - **Performance:** Often outperformed by preconditioned optimizers (SOAP, PSGD) and Muon.
        - **Sensitivity to Hyperparameters:** Can be sensitive to the choice of learning rate and beta parameters.
    - **When to use:**
        - **As a strong baseline.**
        - **When you are familiar with Adam and want a robust starting point.**
        - **When computational cost is a major concern (compared to second-order methods).**
    - **Variants:**
        - `ForeachSFAdamW`: A Schedule-Free version of AdamW that dynamically adjusts the learning rate.
        - `PaLMForeachSFAdamW`: A PaLM version of Schedule-Free AdamW.

## Choosing the Right Optimizer

1. **Start with Preconditioning:** Begin with either `ForeachSOAP` or `ForeachPSGDKron`. If computational resources are
   a major constraint, lean towards `ForeachPSGDKron`. If performance is paramount, try `ForeachSOAP` first.

2. **Consider Muon:** If preconditioned optimizers are not feasible or if you want to explore alternatives that
   incorporate momentum and orthogonal updates, try `ForeachMuon`.

3. **Use LaProp or Adam as Baselines:** `ForeachLaProp` can serve as a simple adaptive baseline. `ForeachAdamW` is a
   strong and widely used baseline that you should always compare against.

4. **Experiment and Tune:** The best optimizer ultimately depends on your specific problem. It's crucial to experiment
   with different optimizers and carefully tune their hyperparameters (especially the learning rate).

## Important Notes

* **Learning Rate:** The learning rate is the most important hyperparameter. You'll likely need to adjust it when
  switching between optimizers.
* **Warmup:** Consider using a learning rate warmup, especially for more complex optimizers like SOAP and PSGD.
* **Weight Decay:** Weight decay can improve generalization for many optimizers, especially AdamW.
* **`foreach`:** Use `foreach` versions of the optimizers when possible for better performance.
* **`heavyball.utils`:** Remember to utilize the settings and functions in `heavyball.utils` (e.g., `set_torch`,
  `compile_mode`, `zeroth_power_mode`, clipping functions) to optimize performance and experiment with different
  configurations.
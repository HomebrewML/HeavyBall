# HeavyBall - Benchmark

This test suite is meant to have fast execution, with each test case covering different optimization properties.

## Categories

Each test is tagged with categories describing specific optimization challenges:

- **Gradient Properties**
  - Magnitude: Exploding/vanishing gradients
  - Quality: Weak/noisy/sparse/delayed signals
  - Direction: Adversarial/misleading patterns

- **Landscape Navigation**
  - Topology: Valleys, plateaus, saddle points
  - Complexity: Multimodality, nonconvexity
  - Dynamics: Moving targets, shifting optima

- **Numerical Challenges**
  - Conditioning: Ill-conditioned problems
  - Scaling: Parameter scale variations
  - Stability: Numerical overflow/underflow

- **Learning Dynamics**
  - Memory: Long-term dependencies
  - Interaction: Multiplicative effects
  - Compression: Information bottlenecks

### Benchmark Tests

| Test | Description | Primary Challenge | In Suite |
|------|-------------|-------------------|:---------:|
| Adversarial Gradient | Tests optimizer robustness to misleading gradients | Gradient Direction & Stability | ✓ |
| Batch Size Scaling | Tests adaptation to batch size changes | Noise & Batch Effects | ✓ |
| Beale | Tests navigation of complex valley structure | Deep Valleys & Nonconvexity | ✓ |
| Char RNN | Tests sequential pattern learning | Long Sequences & Weak Gradients | |
| Discontinuous Gradient | Tests non-smooth landscape navigation | Discontinuities & Convergence | ✓ |
| Dynamic Landscape | Tests adaptation to shifting optima | Moving Targets & Changing Gradients | ✓ |
| Exploding Gradient | Tests handling of growing gradients | Gradient Explosion & Overflow | ✓ |
| Gradient Delay | Tests async update handling | Delayed Gradients & Async Updates | ✓ |
| Gradient Noise Scale | Tests noise level adaptation | Variable Noise & Scaling | ✓ |
| Grokking | Tests sudden learning after memorization | Phase Transitions & Memory | |
| Layer-wise Scale | Tests multi-layer gradient scaling | Layer Variation & Balance | ✓ |
| Loss Contour | Tests complex landscape navigation | Surface Complexity & Visualization | |
| Momentum Utilization | Tests momentum in oscillating landscapes | Oscillations & Momentum | ✓ |
| Noisy Matmul | Tests heavy noise robustness | Noise Robustness & Stability | ✓ |
| Plateau Navigation | Tests small gradient regions | Plateaus & Weak Gradients | ✓ |
| Powers | Tests polynomial optimization | Polynomial Surfaces & Curvature | ✓ |
| Powers Varying Target | Tests changing polynomial targets | Moving Targets & Surface Changes | ✓ |
| Quadratic Varying Scale | Tests dynamic parameter scaling | Scale Adaptation & Stability | ✓ |
| Quadratic Varying Target | Tests moving quadratic minima | Moving Targets & Quadratics | ✓ |
| Rastrigin | Tests multimodal landscape escape | Multiple Minima & Escape | ✓ |
| ReLU Boundaries | Tests ReLU decision boundary learning | Sharp Boundaries & Zero Gradients | |
| Rosenbrock | Tests curved valley navigation | Valley Navigation & Sharp Turns | ✓ |
| Saddle Point | Tests saddle point escape | Saddle Points & Escape | ✓ |
| Scale Invariant | Tests scale-invariant optimization | Scale Invariance & Robustness | ✓ |
| Sparse Gradient | Tests sparse update handling | Sparse Gradients & Embeddings | ✓ |
| XOR Digit | Tests RNN weak gradient learning (~RL) | Weak Gradients & Memory | ✓ |
| XOR Sequence | Tests RNN state compression (~AE, LLM) | Compression & Sequential | ✓ |
| XOR Spot | Tests RNN multiplicative effects (~RL) | Multiplicative & Memory | ✓ |

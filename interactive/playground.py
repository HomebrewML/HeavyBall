import functools
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

import heavyball.chainable as C
import heavyball.utils

# TensorFlow Playground inspired colors
COLORS = {
    "bg": "#ffffff",
    "surface": "#f7f7f7",
    "primary": "#ff6f00",  # Orange
    "secondary": "#0d47a1",  # Blue
    "positive": "#4caf50",  # Green
    "negative": "#f44336",  # Red
    "text": "#212121",
    "text_light": "#757575",
    "border": "#e0e0e0",
    # Component categories
    "gradient": "#9c27b0",  # Purple - Gradient input
    "momentum": "#2196f3",  # Blue - Momentum transforms
    "scaling": "#ff5722",  # Deep Orange - Scaling transforms
    "regularization": "#009688",  # Teal - Regularization
    "normalization": "#795548",  # Brown - Normalization
    "update": "#4caf50",  # Green - Update rules
}

# Test functions
PROBLEMS = {
    "Simple Bowl": {
        "func": lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2,
        "bounds": [(-3, 5), (-2, 6)],
        "init": [-1.0, 0.0],
        "optimal": [1.0, 2.0],
    },
    "Ravine": {
        "func": lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 100,
        "bounds": [(-3, 5), (-2, 6)],
        "init": [-1.0, 1.0],
        "optimal": [1.0, 2.0],
    },
    "Rosenbrock": {
        "func": lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
        "bounds": [(-2, 2), (-1, 3)],
        "init": [-1.0, 1.0],
        "optimal": [1.0, 1.0],
    },
    "Himmelblau": {
        "func": lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,
        "bounds": [(-5, 5), (-5, 5)],
        "init": [0.0, 0.0],
        "optimal": [3.0, 2.0],
    },
}

# Chainable optimizer components
OPTIMIZER_BLOCKS = {
    "gradient": {
        "name": "🎯 Gradient Input",
        "color": COLORS["gradient"],
        "components": [
            {
                "id": "gradient_input",
                "name": "Raw Gradient",
                "icon": "∇",
                "description": "Start of the pipeline - raw gradients from backprop",
                "func": None,  # This is just a placeholder
                "params": {},
            }
        ],
    },
    "momentum": {
        "name": "⚡ Momentum",
        "color": COLORS["momentum"],
        "components": [
            {
                "id": "heavyball",
                "name": "Heavy Ball",
                "icon": "🏐",
                "description": "Classic momentum: v = βv - g",
                "func": C.heavyball_momentum,
                "params": {"beta": 0.9},
            },
            {
                "id": "nesterov",
                "name": "Nesterov",
                "icon": "🚀",
                "description": "Look-ahead momentum",
                "func": C.nesterov_momentum,
                "params": {"beta": 0.9},
            },
            {
                "id": "nesterov_ema",
                "name": "Nesterov EMA",
                "icon": "📈",
                "description": "Exponential moving average variant",
                "func": C.nesterov_ema,
                "params": {"beta": 0.9},
            },
            {
                "id": "basic_ema",
                "name": "Basic EMA",
                "icon": "📉",
                "description": "Simple exponential moving average",
                "func": C.exp_avg,
                "params": {"betas": (0.9, 0.999)},
            },
        ],
    },
    "scaling": {
        "name": "📊 Adaptive Scaling",
        "color": COLORS["scaling"],
        "components": [
            {
                "id": "adam_scale",
                "name": "Adam Scaling",
                "icon": "👑",
                "description": "Adaptive per-parameter learning rates",
                "func": C.scale_by_adam,
                "params": {"betas": (0.9, 0.999), "eps": 1e-8},
            },
            {
                "id": "rmsprop_scale",
                "name": "RMSprop Scaling",
                "icon": "📉",
                "description": "Root mean square normalization",
                "func": C.scale_by_exp_avg_sq,
                "params": {"beta2": 0.999, "eps": 1e-8},
            },
            {
                "id": "adagrad_scale",
                "name": "AdaGrad Scaling",
                "icon": "📐",
                "description": "Accumulate all past gradients",
                "func": C.scale_by_exp_avg_sq,
                "params": {"beta2": 1.0, "eps": 1e-8},
            },
        ],
    },
    "regularization": {
        "name": "⚖️ Regularization",
        "color": COLORS["regularization"],
        "components": [
            {
                "id": "weight_decay",
                "name": "Weight Decay",
                "icon": "🪶",
                "description": "L2 regularization (AdamW style)",
                "func": C.weight_decay_to_ema,
                "params": {"weight_decay_to_ema": 0.01, "ema_beta": 0.999},
            },
            {
                "id": "weight_decay_init",
                "name": "Decay to Init",
                "icon": "🎯",
                "description": "Pull weights toward initialization",
                "func": C.weight_decay_to_init,
                "params": {"weight_decay_to_init": 0.01},
            },
            {
                "id": "l1_weight_decay",
                "name": "L1 Weight Decay",
                "icon": "⚡",
                "description": "L1 regularization to EMA",
                "func": C.l1_weight_decay_to_ema,
                "params": {"weight_decay_to_ema": 0.01, "ema_beta": 0.999},
            },
        ],
    },
    "normalization": {
        "name": "🔧 Gradient Processing",
        "color": COLORS["normalization"],
        "components": [
            {
                "id": "grad_clip",
                "name": "Gradient Clipping",
                "icon": "✂️",
                "description": "Clip gradient norm",
                "func": functools.partial(C.global_clip, clip_fn=heavyball.utils.l2_clip_),
                "params": {"max_norm": 1.0},
            },
            {
                "id": "sign",
                "name": "Sign SGD",
                "icon": "±",
                "description": "Use only gradient signs",
                "func": C.sign,
                "params": {"graft": True},
            },
            {
                "id": "orthogonalize",
                "name": "Orthogonalize",
                "icon": "⊥",
                "description": "Orthogonalize gradient to parameter",
                "func": C.orthogonalize_grad_to_param,
                "params": {"eps": 1e-8},
            },
            {
                "id": "orthogonalize_update",
                "name": "Orthogonalize Update",
                "icon": "⊗",
                "description": "Orthogonalize the update itself",
                "func": C.orthogonalize_update,
                "params": {},
            },
        ],
    },
    "advanced_scaling": {
        "name": "🚀 Advanced Optimizers",
        "color": COLORS["scaling"],
        "components": [
            {
                "id": "laprop",
                "name": "Laprop",
                "icon": "🌊",
                "description": "Layerwise adaptive propagation",
                "func": C.scale_by_laprop,
                "params": {"betas": (0.9, 0.999), "eps": 1e-8},
            },
            {
                "id": "adopt",
                "name": "ADOPT",
                "icon": "🎯",
                "description": "Adaptive gradient methods",
                "func": C.scale_by_adopt,
                "params": {"betas": (0.9, 0.9999), "eps": 1e-12},
            },
        ],
    },
    "preconditioning": {
        "name": "🔮 Preconditioning",
        "color": COLORS["gradient"],
        "components": [
            {
                "id": "soap",
                "name": "SOAP",
                "icon": "🧼",
                "description": "Shampoo-based preconditioning",
                "func": functools.partial(C.scale_by_soap, inner="adam"),
                "params": {
                    "shampoo_beta": 0.99,
                    "max_precond_dim": 10000,
                    "precondition_1d": False,
                    "is_preconditioning": True,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                },
            },
            {
                "id": "psgd",
                "name": "PSGD",
                "icon": "🎲",
                "description": "Preconditioned SGD",
                "func": functools.partial(C.scale_by_psgd, cached=False),
                "params": {
                    "precond_lr": 0.1,
                    "max_size_triangular": 1024,
                    "precondition_frequency": 10,
                    "adaptive": False,
                    "store_triu_as_line": True,
                    "q_dtype": "float32",
                    "inverse_free": False,
                    "precond_init_scale": 1.0,
                    "precond_init_scale_scale": 0.0,
                    "precond_init_scale_power": 1.0,
                    "min_ndim_triangular": 2,
                    "memory_save_mode": None,
                    "dampening": 1.0,
                    "is_preconditioning": True,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "ortho_method": "qr",
                    "lower_bound_beta": 0.999,
                    "precond_update_power_iterations": 1,
                },
            },
            {
                "id": "psgd_lra",
                "name": "PSGD LRA",
                "icon": "📐",
                "description": "Low-rank approximation PSGD",
                "func": C.scale_by_psgd_lra,
                "params": {
                    "precond_lr": 0.1,
                    "rank": 4,
                    "param_count": 10000,
                    "precondition_frequency": 10,
                    "precond_init_scale": 1.0,
                    "precond_init_scale_scale": 0.0,
                    "precond_init_scale_power": 1.0,
                    "q_dtype": "float32",
                    "is_preconditioning": True,
                    "eps": 1e-8,
                    "betas": (0.9, 0.999),
                },
            },
            {
                "id": "delayed_psgd",
                "name": "Delayed PSGD",
                "icon": "⏱️",
                "description": "PSGD with delayed preconditioner updates",
                "func": functools.partial(C.scale_by_delayed_psgd, cached=False),
                "params": {
                    "precond_lr": 0.1,
                    "max_size_triangular": 1024,
                    "precondition_frequency": 10,
                    "adaptive": False,
                    "store_triu_as_line": True,
                    "q_dtype": "float32",
                    "inverse_free": False,
                    "precond_init_scale": 1.0,
                    "precond_init_scale_scale": 0.0,
                    "precond_init_scale_power": 1.0,
                    "min_ndim_triangular": 2,
                    "memory_save_mode": None,
                    "dampening": 1.0,
                    "is_preconditioning": True,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "ortho_method": "qr",
                    "lower_bound_beta": 0.999,
                    "precond_update_power_iterations": 1,
                },
            },
            {
                "id": "delayed_psgd_lra",
                "name": "Delayed PSGD LRA",
                "icon": "⏰",
                "description": "Delayed low-rank PSGD",
                "func": C.scale_by_delayed_psgd_lra,
                "params": {
                    "precond_lr": 0.1,
                    "rank": 4,
                    "param_count": 10000,
                    "precondition_frequency": 10,
                    "precond_init_scale": 1.0,
                    "precond_init_scale_scale": 0.0,
                    "precond_init_scale_power": 1.0,
                    "q_dtype": "float32",
                    "is_preconditioning": True,
                    "eps": 1e-8,
                    "betas": (0.9, 0.999),
                },
            },
        ],
    },
    "adaptive_lr": {
        "name": "🎛️ Adaptive Learning Rate",
        "color": COLORS["regularization"],
        "components": [
            {
                "id": "d_adapt",
                "name": "D-Adaptation",
                "icon": "📈",
                "description": "Automatic learning rate adaptation",
                "func": C.scale_by_d_adaptation,
                "params": {"initial_d": 1.0},
            },
            {
                "id": "lr_adapt",
                "name": "LR Adaptation",
                "icon": "🔄",
                "description": "Learning rate adaptation",
                "func": C.scale_by_lr_adaptation,
                "params": {"initial_d": 1.0, "lr_lr": 0.1},
            },
            {
                "id": "pointwise_lr_adapt",
                "name": "Pointwise LR",
                "icon": "🎚️",
                "description": "Per-parameter learning rate",
                "func": C.scale_by_pointwise_lr_adaptation,
                "params": {"initial_d": 1.0, "lr_lr": 0.1},
            },
        ],
    },
    "special": {
        "name": "✨ Special Methods",
        "color": COLORS["update"],
        "components": [
            {
                "id": "schedule_free",
                "name": "Schedule-Free",
                "icon": "🗓️",
                "description": "No learning rate schedule needed",
                "func": C.update_by_schedule_free,
                "params": {"r": 0.0, "weight_lr_power": 2.0},
            },
            {
                "id": "msam",
                "name": "MSAM",
                "icon": "🏔️",
                "description": "Momentum SAM optimizer",
                "func": C.update_by_msam,
                "params": {"sam_step_size": 0.05},
            },
            {
                "id": "mup",
                "name": "μP Approx",
                "icon": "📏",
                "description": "Maximal update parametrization",
                "func": C.mup_approx,
                "params": {},
            },
            {
                "id": "palm_beta2",
                "name": "PALM β₂",
                "icon": "🌴",
                "description": "Dynamic β₂ scheduling for PALM",
                "func": C.palm_beta2,
                "params": {"beta2_scale": 0.8},
            },
            {
                "id": "identity",
                "name": "Identity",
                "icon": "🔄",
                "description": "Pass-through (no operation)",
                "func": C.identity,
                "params": {},
            },
        ],
    },
}

# Pre-built optimizer recipes
RECIPES = {
    "SGD": ["gradient_input"],
    "SGD + Momentum": ["gradient_input", "heavyball"],
    "Adam": ["gradient_input", "adam_scale"],
    "AdamW": ["gradient_input", "adam_scale", "weight_decay"],
    "RMSprop": ["gradient_input", "rmsprop_scale"],
    "Nesterov SGD": ["gradient_input", "nesterov"],
    "SOAP": ["gradient_input", "soap"],
    "Laprop": ["gradient_input", "laprop"],
    "ADOPT": ["gradient_input", "adopt"],
    "Sign SGD": ["gradient_input", "sign"],
    "AdamW + Clipping": ["gradient_input", "grad_clip", "adam_scale", "weight_decay"],
    "D-Adapted Adam": ["gradient_input", "adam_scale", "d_adapt"],
    "PSGD": ["gradient_input", "psgd"],
    "Schedule-Free AdamW": ["gradient_input", "adam_scale", "weight_decay", "schedule_free"],
    "EMA SGD": ["gradient_input", "basic_ema"],
    "Orthogonal Adam": ["gradient_input", "orthogonalize_update", "adam_scale"],
    "PALM": ["gradient_input", "palm_beta2", "adam_scale"],
    "Delayed PSGD": ["gradient_input", "delayed_psgd"],
    "μP Adam": ["gradient_input", "mup", "adam_scale"],
}


def get_component_info(comp_id):
    """Get component info by ID"""
    for category in OPTIMIZER_BLOCKS.values():
        for comp in category["components"]:
            if comp["id"] == comp_id:
                return comp, category["color"]
    return None, "#757575"


def create_pipeline_display(pipeline):
    """Create HTML display for the current pipeline"""
    if not pipeline or pipeline == ["gradient_input"]:
        return """
        <div style="min-height: 120px; background: linear-gradient(135deg, #f5f5f5 25%, transparent 25%), linear-gradient(225deg, #f5f5f5 25%, transparent 25%), linear-gradient(45deg, #f5f5f5 25%, transparent 25%), linear-gradient(315deg, #f5f5f5 25%, #fafafa 25%); background-size: 20px 20px; background-position: 0 0, 10px 0, 10px -10px, 0px 10px; border: 2px dashed #bdbdbd; border-radius: 12px; padding: 20px; display: flex; align-items: center; justify-content: center; color: #9e9e9e; font-style: italic;">
            Drop components here to build your optimizer pipeline
        </div>
        """

    blocks_html = ""
    for i, comp_id in enumerate(pipeline):
        comp_info, color = get_component_info(comp_id)
        if comp_info:
            show_arrow = i < len(pipeline) - 1
            blocks_html += f"""
            <div style="display: inline-block; position: relative; margin: 0 10px;">
                <div style="background: white; border: 3px solid {color}; border-radius: 8px; padding: 16px; min-width: 100px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); position: relative;">
                    <div style="font-size: 32px; margin-bottom: 8px;">{comp_info["icon"]}</div>
                    <div style="font-weight: 600; font-size: 13px;">{comp_info["name"]}</div>
                    <button onclick="window.removePipelineBlock({i})" style="position: absolute; top: 4px; right: 4px; width: 24px; height: 24px; border-radius: 50%; background: #f44336; color: white; border: none; cursor: pointer; opacity: 0.7; font-size: 16px; line-height: 1; padding: 0;">×</button>
                </div>
                {'<div style="position: absolute; right: -30px; top: 50%; transform: translateY(-50%); font-size: 24px; color: #ff6f00;">→</div>' if show_arrow else ""}
            </div>
            """

    return f"""
    <div style="min-height: 120px; background: white; border: 2px solid #e0e0e0; border-radius: 12px; padding: 20px; display: flex; align-items: center; overflow-x: auto; gap: 10px;">
        {blocks_html}
    </div>
    """


def build_optimizer_from_pipeline(pipeline: List[str], params):
    """Build optimizer from pipeline of component IDs"""
    fns = []
    opt_params = {
        "lr": 0.001,
        "step": 1,  # Required for many functions
        "caution": False,
        "weight_decay": 0.0,
    }

    for comp_id in pipeline:
        if comp_id == "gradient_input":
            continue  # Skip the input block

        # Find component
        comp_info, _ = get_component_info(comp_id)
        if comp_info and comp_info["func"] is not None:
            fns.append(comp_info["func"])
            # Update parameters, handling special cases
            params_to_add = comp_info["params"].copy()

            # Handle special parameter mappings
            if "beta" in params_to_add and "betas" not in params_to_add:
                # Convert single beta to betas tuple for functions expecting it
                beta = params_to_add.pop("beta")
                if "betas" in opt_params:
                    opt_params["betas"] = (beta, opt_params["betas"][1])
                else:
                    opt_params["betas"] = (beta, 0.999)

            opt_params.update(params_to_add)

    if not fns:
        # Default to simple gradient descent
        return C.BaseOpt(params, opt_params, fns=[C.identity])

    return C.BaseOpt(params, opt_params, fns=fns)


def run_optimization(problem_name: str, pipeline: List[str], steps: int, lr: float, **kwargs) -> Tuple[Any, Dict]:
    """Run optimization with the custom pipeline"""
    problem = PROBLEMS[problem_name]
    func = problem["func"]
    init = problem["init"]

    # Initialize
    x = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
    params = [x]

    # Build optimizer from pipeline
    optimizer = build_optimizer_from_pipeline(pipeline, params)

    # Override learning rate
    if hasattr(optimizer, "param_groups"):
        for group in optimizer.param_groups:
            group["lr"] = lr
            # Update other params from kwargs
            for key, value in kwargs.items():
                if key in group:
                    group[key] = value

    # Run optimization
    trajectory = []
    losses = []
    gradients = []

    for i in range(steps):
        trajectory.append(x.detach().numpy().copy())

        def closure():
            optimizer.zero_grad()
            loss = func(x)
            loss.backward()
            if x.grad is not None:
                gradients.append(x.grad.detach().numpy().copy())
            return loss

        loss = optimizer.step(closure)
        losses.append(loss.item())

    trajectory = np.array(trajectory)

    # Create visualization
    fig = create_visualization(problem_name, trajectory, losses, gradients, pipeline)

    return fig, {
        "trajectory": trajectory.tolist(),
        "losses": losses,
        "final_loss": losses[-1],
        "steps_to_converge": find_convergence(losses),
    }


def find_convergence(losses, threshold=1e-6):
    """Find when optimization converged"""
    if len(losses) < 10:
        return len(losses)

    for i in range(10, len(losses)):
        if abs(losses[i] - losses[i - 5]) < threshold:
            return i
    return len(losses)


def create_visualization(problem_name, trajectory, losses, gradients, pipeline):
    """Create integrated visualization"""
    problem = PROBLEMS[problem_name]
    func = problem["func"]
    bounds = problem["bounds"]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Optimization Landscape", "Pipeline Architecture", "Loss Curve", "Learning Dynamics"),
        column_widths=[0.6, 0.4],
        row_heights=[0.6, 0.4],
        specs=[[{"type": "contour"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "scatter"}]],
    )

    # 1. Optimization landscape
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    # Contour plot
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=Z,
            colorscale=[[0, "#e3f2fd"], [0.5, "#2196f3"], [1, "#0d47a1"]],
            showscale=False,
            contours=dict(
                start=0,
                end=Z.max(),
                size=Z.max() / 15,
            ),
        ),
        row=1,
        col=1,
    )

    # Add optimization path
    if len(trajectory) > 0:
        colors = np.linspace(0, 1, len(trajectory))

        for i in range(1, len(trajectory)):
            fig.add_trace(
                go.Scatter(
                    x=[trajectory[i - 1, 0], trajectory[i, 0]],
                    y=[trajectory[i - 1, 1], trajectory[i, 1]],
                    mode="lines",
                    line=dict(color=f"rgba(255, {int(111 * (1 - colors[i]))}, 0, {0.3 + 0.7 * colors[i]})", width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # Start and end points
        fig.add_trace(
            go.Scatter(
                x=[trajectory[0, 0]],
                y=[trajectory[0, 1]],
                mode="markers+text",
                marker=dict(size=12, color="#4caf50", line=dict(color="white", width=2)),
                text=["Start"],
                textposition="top center",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                mode="markers+text",
                marker=dict(size=12, color="#ff6f00", line=dict(color="white", width=2)),
                text=["End"],
                textposition="top center",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # 2. Pipeline visualization
    block_x = []
    block_y = []
    block_text = []
    block_colors = []

    for i, comp_id in enumerate(pipeline):
        block_x.append(i / (len(pipeline) - 1) if len(pipeline) > 1 else 0.5)
        block_y.append(0.5)

        comp_info, comp_color = get_component_info(comp_id)
        block_text.append(comp_info["icon"] if comp_info else "?")
        block_colors.append(comp_color)

    fig.add_trace(
        go.Scatter(
            x=block_x,
            y=block_y,
            mode="markers+text",
            marker=dict(size=40, color=block_colors, line=dict(color="white", width=2)),
            text=block_text,
            textposition="middle center",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Add arrows between blocks
    for i in range(len(pipeline) - 1):
        fig.add_annotation(
            x=block_x[i + 1],
            y=0.5,
            ax=block_x[i],
            ay=0.5,
            xref="x2",
            yref="y2",
            axref="x2",
            ayref="y2",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#ff6f00",
            row=1,
            col=2,
        )

    # 3. Loss curve
    fig.add_trace(
        go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            mode="lines",
            line=dict(color="#ff6f00", width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 111, 0, 0.1)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. Learning dynamics (gradient norm)
    if gradients:
        grad_norms = [np.linalg.norm(g) for g in gradients]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(grad_norms))),
                y=grad_norms,
                mode="lines",
                line=dict(color="#2196f3", width=2),
                fill="tozeroy",
                fillcolor="rgba(33, 150, 243, 0.1)",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(
        height=700,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    # Pipeline plot
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], row=1, col=2)

    # Loss plot
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Loss", type="log", row=2, col=1)

    # Gradient plot
    fig.update_xaxes(title_text="Iteration", row=2, col=2)
    fig.update_yaxes(title_text="Gradient Norm", row=2, col=2)

    return fig


def create_app():
    with gr.Blocks(theme=gr.themes.Base()) as app:
        # Custom CSS
        gr.HTML("""
        <style>
        .component-block {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 4px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
            display: inline-block;
        }
        .component-block:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-color: #ff6f00;
        }
        .category-header {
            background: #f5f5f5;
            padding: 8px 16px;
            margin: 16px 0 8px 0;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            color: #616161;
        }
        </style>
        """)

        # Header
        gr.Markdown("""
        # 🧩 HeavyBall Chainable Optimizer Playground

        ### Build custom optimizers by combining components like LEGO blocks!

        Click components to add them to your pipeline. Each component transforms the gradient in a specific way - stack them to create powerful optimization algorithms!
        """)

        # Hidden state for pipeline
        pipeline_state = gr.State(["gradient_input"])

        with gr.Row():
            # Left column - Component palette
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 Component Palette")
                gr.Markdown("*Click blocks to add to pipeline*")

                # Component buttons organized by category
                for cat_id, category in OPTIMIZER_BLOCKS.items():
                    if cat_id == "gradient":  # Skip gradient input in palette
                        continue

                    gr.HTML(f'<div class="category-header">{category["name"]}</div>')

                    with gr.Row():
                        for comp in category["components"]:
                            btn = gr.Button(
                                value=f"{comp['icon']} {comp['name']}", elem_id=f"btn_{comp['id']}", size="sm"
                            )
                            # Store component ID in button for click handler
                            btn.click(
                                fn=lambda p, cid=comp["id"]: p + [cid],
                                inputs=[pipeline_state],
                                outputs=[pipeline_state],
                            )

                # Recipe selector
                gr.Markdown("### 📚 Preset Recipes")
                recipe_dropdown = gr.Dropdown(choices=list(RECIPES.keys()), value=None, label="Load a preset optimizer")
                load_recipe_btn = gr.Button("Load Recipe", size="sm")

            # Center column - Main visualization
            with gr.Column(scale=2):
                # Pipeline builder
                gr.Markdown("### 🔧 Pipeline Builder")
                pipeline_display = gr.HTML()

                with gr.Row():
                    clear_pipeline_btn = gr.Button("🗑️ Clear Pipeline", size="sm", variant="secondary")
                    refresh_btn = gr.Button("🔄 Refresh Display", size="sm")

                # Visualization
                viz_plot = gr.Plot(label="")

                # Run button
                run_btn = gr.Button("🚀 Run Optimization", variant="primary", size="lg")

            # Right column - Parameters and metrics
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Parameters")

                problem_select = gr.Dropdown(choices=list(PROBLEMS.keys()), value="Rosenbrock", label="Problem")

                lr_slider = gr.Slider(minimum=0.0001, maximum=0.1, value=0.01, step=0.0001, label="Learning Rate")

                steps_slider = gr.Slider(minimum=10, maximum=500, value=200, step=10, label="Steps")

                # Component-specific parameters
                with gr.Accordion("Advanced Parameters", open=False):
                    beta_slider = gr.Slider(minimum=0.0, maximum=0.999, value=0.9, step=0.001, label="Momentum β")
                    beta2_slider = gr.Slider(minimum=0.0, maximum=0.999, value=0.999, step=0.001, label="Adam β₂")
                    eps_slider = gr.Slider(minimum=1e-8, maximum=1e-4, value=1e-8, step=1e-8, label="Epsilon")

                gr.Markdown("### 📊 Metrics")

                final_loss_display = gr.Textbox(label="Final Loss", value="-")
                convergence_display = gr.Textbox(label="Steps to Converge", value="-")

        # Footer
        gr.Markdown("""
        ---
        ### 💡 How it works:

        1. **Start with Gradient** - Every pipeline begins with raw gradients
        2. **Add Transforms** - Click components to add them to your pipeline
        3. **Order Matters** - Components are applied in sequence
        4. **Run & Compare** - See how different combinations perform!

        **Example combinations:**
        - Adam = Gradient → Adam Scaling
        - AdamW = Gradient → Adam Scaling → Weight Decay
        - Momentum SGD = Gradient → Heavy Ball
        """)

        # Event handlers
        def update_display(pipeline):
            return create_pipeline_display(pipeline)

        def load_recipe(recipe_name):
            if recipe_name in RECIPES:
                return RECIPES[recipe_name].copy()
            return ["gradient_input"]

        def clear_pipeline():
            return ["gradient_input"]

        def remove_block(pipeline, index):
            """Remove a block from the pipeline"""
            if 0 <= index < len(pipeline) and pipeline[index] != "gradient_input":
                new_pipeline = pipeline.copy()
                new_pipeline.pop(index)
                return new_pipeline
            return pipeline

        def run_optimization_handler(problem, pipeline, steps, lr, beta, beta2, eps):
            """Run optimization with current pipeline"""
            if not pipeline or len(pipeline) == 1:
                pipeline = ["gradient_input"]

            # Run optimization
            fig, metrics = run_optimization(
                problem, pipeline, steps, lr, beta=beta, betas=(beta, beta2), beta2=beta2, eps=eps
            )

            return fig, f"{metrics['final_loss']:.2e}", str(metrics["steps_to_converge"])

        # Connect events
        # Update display when pipeline changes
        pipeline_state.change(fn=update_display, inputs=[pipeline_state], outputs=[pipeline_display])

        # Recipe loading
        load_recipe_btn.click(fn=load_recipe, inputs=[recipe_dropdown], outputs=[pipeline_state])

        # Clear pipeline
        clear_pipeline_btn.click(fn=clear_pipeline, outputs=[pipeline_state])

        # Refresh display
        refresh_btn.click(fn=update_display, inputs=[pipeline_state], outputs=[pipeline_display])

        # Run optimization
        run_btn.click(
            fn=run_optimization_handler,
            inputs=[problem_select, pipeline_state, steps_slider, lr_slider, beta_slider, beta2_slider, eps_slider],
            outputs=[viz_plot, final_loss_display, convergence_display],
        )

        # Add JavaScript for removing blocks
        app.load(
            None,
            None,
            None,
            js="""
            function() {
                // Function to remove pipeline blocks
                window.removePipelineBlock = function(index) {
                    // This would need to trigger a Gradio event
                    console.log('Remove block at index:', index);
                    // In a real implementation, this would update the pipeline state
                };

                console.log('Playground initialized');
            }
            """,
        )

        # Initialize display
        app.load(
            fn=lambda: (
                create_pipeline_display(["gradient_input"]),
                run_optimization("Rosenbrock", ["gradient_input", "adam_scale"], 200, 0.01)[0],
            ),
            outputs=[pipeline_display, viz_plot],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)

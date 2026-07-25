from unittest.mock import patch

import torch

import heavyball
import heavyball.core
from heavyball.numerics import stochastic_copy_
from heavyball.transforms import _STATE_PHILOX_ROUNDS, Tempo


def _fp32_state_trajectory(state_rounds: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    initial = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).bfloat16()
    generator = torch.Generator().manual_seed(731)
    gradients = [torch.randn(initial.shape, generator=generator).bfloat16() for _ in range(20)]

    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(initial.clone())
    with patch("heavyball.core._STATE_PHILOX_ROUNDS", state_rounds):
        optimizer = heavyball.AdamW([parameter], storage_dtype=None)
        trajectory = []
        for gradient in gradients:
            parameter.grad.copy_(gradient)
            optimizer.step()
            trajectory.append(parameter.detach().clone())

    state = tuple(
        value.detach().clone()
        for engine in optimizer._engines
        for group in engine.groups
        for slots in (*group.states, group.commit_state)
        for value in slots.values()
    )
    return torch.stack(trajectory), state


def test_param_rounds_default_seven():
    tempo = Tempo(
        torch.zeros((), dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.ones(1, dtype=torch.bool),
        object(),
        False,
        base_seed=torch.tensor((1, 2), dtype=torch.int64),
        leaf_indices=torch.zeros(1, dtype=torch.int64),
    )

    assert tempo.rounds == 7
    assert torch.equal(tempo.random_like(torch.zeros(1, 8)), tempo._replace(rounds=7).random_like(torch.zeros(1, 8)))


def test_fp32_state_bit_exact_unchanged():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        reference_trajectory, reference_state = _fp32_state_trajectory(7)
        trajectory, state = _fp32_state_trajectory(_STATE_PHILOX_ROUNDS)

    assert torch.equal(trajectory.view(torch.uint8), reference_trajectory.view(torch.uint8))
    assert len(state) == len(reference_state)
    assert all(
        torch.equal(value.view(torch.uint8), reference.view(torch.uint8))
        for value, reference in zip(state, reference_state, strict=True)
    )


def test_fp32_state_shared_noise_plumbing_bit_exact_unchanged():
    def legacy_stochastic_copy(target, source, tempo, shared_noise=None):
        stochastic_copy_(target, source, tempo)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        trajectory, state = _fp32_state_trajectory(_STATE_PHILOX_ROUNDS)
        with patch("heavyball.core.stochastic_copy_", legacy_stochastic_copy):
            reference_trajectory, reference_state = _fp32_state_trajectory(_STATE_PHILOX_ROUNDS)

    assert torch.equal(trajectory.view(torch.uint8), reference_trajectory.view(torch.uint8))
    assert len(state) == len(reference_state)
    assert all(
        torch.equal(value.view(torch.uint8), reference.view(torch.uint8))
        for value, reference in zip(state, reference_state, strict=True)
    )


def test_bf16_state_uses_fewer_rounds(monkeypatch):
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        torch.manual_seed(1234)
        parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 32).bfloat16())
        optimizer = heavyball.AdamW([parameter], storage_dtype=torch.bfloat16)
        state_targets = {
            value.data_ptr()
            for engine in optimizer._engines
            for group in engine.groups
            for slots in (*group.states, group.commit_state)
            for value in slots.values()
            if value.is_floating_point()
        }
        state_rounds = []
        parameter_rounds = []
        state_noises = []
        parameter_noises = []
        random_draws = []

        original_random_like = Tempo.random_like

        def record_random_like(tempo, value, *, _stream=0):
            random = original_random_like(tempo, value, _stream=_stream)
            random_draws.append(random.detach().clone())
            return random

        def record_stochastic_copy(target, source, tempo, shared_noise=None):
            if target.data_ptr() in state_targets:
                state_rounds.append(tempo.rounds)
                state_noises.append(shared_noise.detach().clone())
            else:
                parameter_rounds.append(tempo.rounds)
                parameter_noises.append(shared_noise.detach().clone())
            stochastic_copy_(target, source, tempo, shared_noise=shared_noise)

        monkeypatch.setattr(Tempo, "random_like", record_random_like)
        monkeypatch.setattr(heavyball.core, "stochastic_copy_", record_stochastic_copy)
        parameter.grad.copy_(torch.linspace(0.5, -0.5, 32).bfloat16())
        optimizer.step()

    assert state_rounds
    assert set(state_rounds) == {_STATE_PHILOX_ROUNDS}
    assert _STATE_PHILOX_ROUNDS < 7
    assert parameter_rounds == [7]
    assert len(random_draws) == 1
    assert len(parameter_noises) == 1
    assert torch.equal(parameter_noises[0], random_draws[0])
    assert state_noises
    assert all(torch.equal(noise, parameter_noises[0]) for noise in state_noises)

import pytest
import torch

from heavyball.codecs import decode, encode


def _random_values() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1729)
    return [torch.randn(4096, generator=generator) * scale for scale in (2.0**-20, 2.0**-7, 1.0, 2.0**9)]


def test_correction_recovers_precision():
    values = _random_values()
    value = torch.cat(values)

    torch.manual_seed(100)
    narrow8, correction8 = encode(value, torch.bfloat16, torch.int8)
    torch.manual_seed(101)
    narrow16, correction16 = encode(value, torch.bfloat16, torch.int16)

    bf16_error = (value - value.bfloat16().float()).abs().mean()
    int8_error = (value - decode(narrow8, correction8, torch.int8)).abs().mean()
    int16_error = (value - decode(narrow16, correction16, torch.int16)).abs().mean()

    print(
        f"reconstruction MAE: bf16={bf16_error.item():.9e}, "
        f"bf16+int8={int8_error.item():.9e}, bf16+int16={int16_error.item():.9e}"
    )
    assert int8_error < bf16_error / 8
    assert int16_error < bf16_error / 100


def test_stochastic_encode_is_unbiased():
    generator = torch.Generator().manual_seed(2718)
    value = torch.randn(256, generator=generator) * torch.logspace(-3, 3, 256)

    torch.manual_seed(200)
    narrow, correction = encode(value, torch.bfloat16, torch.int8)
    single_error = (decode(narrow, correction) - value).abs().mean().double()

    trials = 8192
    repeated = value.expand(trials, -1)
    torch.manual_seed(201)
    repeated_narrow, repeated_correction = encode(repeated, torch.bfloat16, torch.int8)
    mean_decode = decode(repeated_narrow, repeated_correction).double().mean(dim=0)
    mean_error = (mean_decode - value.double()).abs().mean()

    assert mean_error < single_error / 20


@pytest.mark.parametrize("correction_dtype", (torch.int8, torch.int16))
def test_decode_is_pure(correction_dtype):
    value = torch.tensor([0.0, 2.0**-134, -(2.0**-133), 1.0e-10, -1.5, 4096.25], dtype=torch.float32)
    torch.manual_seed(300)
    narrow, correction = encode(value, torch.bfloat16, correction_dtype)

    first = decode(narrow, correction)
    second = decode(narrow, correction)

    assert narrow.dtype == torch.bfloat16
    assert correction.dtype == correction_dtype
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_ecc16_reconstructs_fp32_exactly():
    wide_range = torch.logspace(-44, 38, 257, dtype=torch.float32)
    specific = torch.tensor(
        [
            float("-inf"),
            -123456.789,
            -1.0000001192092896,
            -1.0,
            -(2.0**-148),
            0.0,
            -0.0,
            2.0**-149,
            0.1,
            1.0,
            1.0000001192092896,
            123456.789,
            float("inf"),
        ],
        dtype=torch.float32,
    )
    value = torch.cat((-wide_range.flip(0), specific, wide_range))

    decoded = decode(*encode(value, torch.bfloat16, torch.int16))

    assert torch.equal(decoded, value)
    assert torch.equal(decoded.view(torch.int32), value.view(torch.int32))

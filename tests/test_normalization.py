import torch

from src.amor.brain.normalization import RMSNorm


def test_rmsnorm_shape():
    norm = RMSNorm(256)

    x = torch.randn(
        2,
        16,
        256,
    )

    output = norm(x)

    assert output.shape == x.shape


def test_rmsnorm_parameters():
    norm = RMSNorm(256)

    assert norm.weight.shape == (256,)
    assert isinstance(
        norm.weight,
        torch.nn.Parameter,
    )


def test_rmsnorm_finite():
    norm = RMSNorm(256)

    x = torch.randn(
        2,
        16,
        256,
    )

    output = norm(x)

    assert torch.isfinite(output).all()


def test_rmsnorm_backward():
    norm = RMSNorm(256)

    x = torch.randn(
        2,
        16,
        256,
        requires_grad=True,
    )

    output = norm(x)
    loss = output.mean()

    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
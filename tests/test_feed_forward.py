import torch

from src.amor.brain.feed_forward import FeedForward


def test_feed_forward_shape():
    ffn = FeedForward(
        dim=256,
        hidden_dim=1024,
    )

    x = torch.randn(
        2,
        16,
        256,
    )

    output = ffn(x)

    assert output.shape == x.shape


def test_feed_forward_finite():
    ffn = FeedForward(
        dim=256,
        hidden_dim=1024,
    )

    x = torch.randn(
        2,
        16,
        256,
    )

    output = ffn(x)

    assert torch.isfinite(output).all()


def test_feed_forward_backward():
    ffn = FeedForward(
        dim=256,
        hidden_dim=1024,
    )

    x = torch.randn(
        2,
        16,
        256,
        requires_grad=True,
    )

    output = ffn(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None

    for parameter in ffn.parameters():
        assert parameter.grad is not None


def test_feed_forward_different_sequence_lengths():
    ffn = FeedForward(
        dim=256,
        hidden_dim=1024,
    )

    for seq_len in [1, 4, 16, 64]:
        x = torch.randn(
            2,
            seq_len,
            256,
        )

        output = ffn(x)

        assert output.shape == x.shape
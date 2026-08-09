import pytest
import torch

from src.amor.brain.attention import CausalSelfAttention


def test_attention_shape():
    attention = CausalSelfAttention(
        dim=256,
        num_heads=8,
        max_seq_len=512,
    )

    x = torch.randn(
        2,
        16,
        256,
    )

    output = attention(x)

    assert output.shape == x.shape


def test_attention_finite():
    attention = CausalSelfAttention(
        dim=256,
        num_heads=8,
        max_seq_len=512,
    )

    x = torch.randn(
        2,
        16,
        256,
    )

    output = attention(x)

    assert torch.isfinite(output).all()


def test_attention_backward():
    attention = CausalSelfAttention(
        dim=256,
        num_heads=8,
        max_seq_len=512,
    )

    x = torch.randn(
        2,
        16,
        256,
        requires_grad=True,
    )

    output = attention(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None


def test_attention_requires_divisible_dimensions():
    with pytest.raises(ValueError):
        CausalSelfAttention(
            dim=250,
            num_heads=8,
            max_seq_len=512,
        )


def test_attention_different_sequence_lengths():
    attention = CausalSelfAttention(
        dim=256,
        num_heads=8,
        max_seq_len=512,
    )

    for seq_len in [1, 4, 16, 64]:
        x = torch.randn(
            2,
            seq_len,
            256,
        )

        output = attention(x)

        assert output.shape == x.shape
        
def test_causal_mask_blocks_future_tokens():
    attention = CausalSelfAttention(
        dim=32,
        num_heads=4,
        max_seq_len=16,
    )

    x = torch.randn(
        1,
        5,
        32,
    )

    _, weights = attention(
        x,
        return_attention=True,
    )

    # Future-token attention weights must be zero.
    for position in range(5):
        future_weights = weights[
            0,
            :,
            position,
            position + 1:,
        ]

        assert torch.allclose(
            future_weights,
            torch.zeros_like(future_weights),
            atol=1e-6,
        )
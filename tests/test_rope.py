import pytest
import torch

from src.amor.brain.rope import RotaryEmbedding


def test_rope_shape():
    rope = RotaryEmbedding(
        dim=32,
        max_seq_len=512,
    )

    x = torch.randn(
        2,
        8,
        16,
        32,
    )

    output = rope(x)

    assert output.shape == x.shape


def test_rope_finite():
    rope = RotaryEmbedding(
        dim=32,
        max_seq_len=512,
    )

    x = torch.randn(
        2,
        8,
        16,
        32,
    )

    output = rope(x)

    assert torch.isfinite(output).all()


def test_rope_changes_nonzero_position():
    rope = RotaryEmbedding(
        dim=32,
        max_seq_len=512,
    )

    x = torch.randn(
        1,
        1,
        8,
        32,
    )

    output = rope(x)

    assert not torch.allclose(
        x,
        output,
    )


def test_rope_position_offset():
    rope = RotaryEmbedding(
        dim=32,
        max_seq_len=512,
    )

    x = torch.randn(
        1,
        1,
        8,
        32,
    )

    output_0 = rope(
        x,
        position_offset=0,
    )

    output_10 = rope(
        x,
        position_offset=10,
    )

    assert not torch.allclose(
        output_0,
        output_10,
    )


def test_rope_rejects_odd_dimension():
    with pytest.raises(ValueError):
        RotaryEmbedding(
            dim=31,
            max_seq_len=512,
        )


def test_rope_rejects_long_sequence():
    rope = RotaryEmbedding(
        dim=32,
        max_seq_len=16,
    )

    x = torch.randn(
        1,
        1,
        17,
        32,
    )

    with pytest.raises(ValueError):
        rope(x)
        
def test_rope_matches_manual_rotation():
    rope = RotaryEmbedding(
        dim=4,
        max_seq_len=8,
        base=10_000.0,
    )

    x = torch.tensor(
        [[[
            [1.0, 2.0, 3.0, 4.0],
        ]]]
    )

    output = rope(x)

    # Position 0 has angle 0.
    # Therefore:
    # cos(0) = 1
    # sin(0) = 0
    #
    # Rotation should leave the vector unchanged.
    expected = x

    assert torch.allclose(
        output,
        expected,
        atol=1e-6,
    )
import torch

from src.amor.brain.transformer_block import TransformerBlock


def create_block() -> TransformerBlock:
    return TransformerBlock(
        dim=256,
        num_heads=8,
        ff_hidden_dim=1024,
        max_seq_len=512,
    )


def test_transformer_block_shape():
    block = create_block()

    x = torch.randn(
        2,
        16,
        256,
    )

    output = block(x)

    assert output.shape == x.shape


def test_transformer_block_finite():
    block = create_block()

    x = torch.randn(
        2,
        16,
        256,
    )

    output = block(x)

    assert torch.isfinite(output).all()


def test_transformer_block_backward():
    block = create_block()

    x = torch.randn(
        2,
        16,
        256,
        requires_grad=True,
    )

    output = block(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None

    for parameter in block.parameters():
        assert parameter.grad is not None


def test_transformer_block_sequence_lengths():
    block = create_block()

    for seq_len in [1, 4, 16, 64]:
        x = torch.randn(
            2,
            seq_len,
            256,
        )

        output = block(x)

        assert output.shape == x.shape
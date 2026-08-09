import torch

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel


def create_model() -> AMORModel:
    config = AMORConfig(
        vocab_size=1_000,
        dim=64,
        num_heads=4,
        num_layers=2,
        ff_hidden_dim=256,
        max_seq_len=128,
    )

    return AMORModel(config)


def test_model_output_shape():
    model = create_model()

    input_ids = torch.randint(
        0,
        1_000,
        (2, 16),
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        16,
        1_000,
    )


def test_model_output_finite():
    model = create_model()

    input_ids = torch.randint(
        0,
        1_000,
        (2, 16),
    )

    logits = model(input_ids)

    assert torch.isfinite(logits).all()


def test_model_backward():
    model = create_model()

    input_ids = torch.randint(
        0,
        1_000,
        (2, 16),
    )

    logits = model(input_ids)

    loss = logits.mean()

    loss.backward()

    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None


def test_model_weight_tying():
    model = create_model()

    assert (
        model.token_embedding.weight
        is model.lm_head.weight
    )


def test_model_sequence_lengths():
    model = create_model()

    for seq_len in [1, 4, 16, 64]:
        input_ids = torch.randint(
            0,
            1_000,
            (2, seq_len),
        )

        logits = model(input_ids)

        assert logits.shape == (
            2,
            seq_len,
            1_000,
        )
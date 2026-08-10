from collections.abc import Iterable

import torch


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    Create the AdamW optimizer for AMOR.

    Parameters:
        model:
            AMOR neural network.

        learning_rate:
            Initial learning rate.

        weight_decay:
            Decoupled weight decay coefficient.
    """

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative."
        )

    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        # Biases and normalization parameters should
        # generally not receive weight decay.
        if (
            parameter.ndim == 1
            or name.endswith(".bias")
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
    )
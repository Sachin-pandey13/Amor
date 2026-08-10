from pathlib import Path
from typing import Any

import torch


def _get_scheduler_state(
    scheduler: Any,
) -> dict:
    """
    Extract scheduler state.

    Supports both:
    - PyTorch-style schedulers implementing state_dict()
    - Custom AMOR schedulers without state_dict()
    """

    if hasattr(scheduler, "state_dict"):
        return {
            "type": "state_dict",
            "state": scheduler.state_dict(),
        }

    state = {}

    for key, value in vars(scheduler).items():
        # The optimizer is already stored separately.
        if key == "optimizer":
            continue

        # Only checkpoint simple state values.
        if isinstance(
            value,
            (
                int,
                float,
                str,
                bool,
                type(None),
            ),
        ):
            state[key] = value

    return {
        "type": "attributes",
        "state": state,
    }


def _restore_scheduler_state(
    scheduler: Any,
    scheduler_state: dict,
) -> None:
    """
    Restore scheduler state for either a standard
    PyTorch scheduler or a custom AMOR scheduler.
    """

    state_type = scheduler_state.get(
        "type"
    )

    state = scheduler_state.get(
        "state",
        {},
    )

    if state_type == "state_dict":
        if not hasattr(
            scheduler,
            "load_state_dict",
        ):
            raise ValueError(
                "Checkpoint contains a PyTorch-style "
                "scheduler state, but the supplied "
                "scheduler does not implement "
                "load_state_dict()."
            )

        scheduler.load_state_dict(state)
        return

    if state_type == "attributes":
        for key, value in state.items():
            setattr(
                scheduler,
                key,
                value,
            )

        return

    raise ValueError(
        "Invalid scheduler state in checkpoint."
    )


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config=None,
) -> None:
    """
    Save a complete AMOR training checkpoint.

    Stores:
        - model parameters
        - optimizer state
        - scheduler state
        - training step
        - optional training configuration
    """

    if step < 0:
        raise ValueError(
            "step must be greater than or equal to zero."
        )

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state": (
            _get_scheduler_state(
                scheduler
            )
        ),
        "step": step,
        "config": config,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Load an AMOR training checkpoint.

    Model state is always restored.

    Optimizer and scheduler states are restored
    when corresponding objects are supplied.

    Returns:
        Dictionary containing checkpoint metadata.
    """

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state",
        "step",
        "config",
    }

    missing_keys = (
        required_keys
        - checkpoint.keys()
    )

    if missing_keys:
        raise ValueError(
            "Invalid checkpoint. Missing keys: "
            f"{sorted(missing_keys)}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if scheduler is not None:
        _restore_scheduler_state(
            scheduler,
            checkpoint[
                "scheduler_state"
            ],
        )

    return {
        "step": checkpoint["step"],
        "config": checkpoint["config"],
    }
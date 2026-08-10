import math

import torch


class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        if warmup_steps >= max_steps:
            raise ValueError(
                "warmup_steps must be smaller "
                "than max_steps."
            )

        if min_lr < 0:
            raise ValueError(
                "min_lr cannot be negative."
            )

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr

        self.base_lrs = [
            group["lr"]
            for group in optimizer.param_groups
        ]

        self.step_count = 0

    def get_lr(
        self,
        step: int,
    ) -> list[float]:
        """
        Calculate learning rates for a given step.

        Schedule:

            Linear warmup
                ↓
            Cosine decay
                ↓
            Minimum learning rate
        """

        # -------------------------
        # Linear warmup
        # -------------------------
        if step < self.warmup_steps:
            warmup_factor = (
                (step + 1)
                / self.warmup_steps
            )

            return [
                base_lr * warmup_factor
                for base_lr in self.base_lrs
            ]

        # -------------------------
        # Final training step
        # -------------------------
        if step >= self.max_steps - 1:
            return [
                self.min_lr
                for _ in self.base_lrs
            ]

        # -------------------------
        # Cosine decay
        # -------------------------
        progress = (
            step - self.warmup_steps
        ) / (
            self.max_steps
            - self.warmup_steps
            - 1
        )

        cosine_factor = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

        return [
            self.min_lr
            + (
                base_lr - self.min_lr
            )
            * cosine_factor
            for base_lr in self.base_lrs
        ]

    def step(self) -> None:
        """
        Advance the scheduler by one step.
        """

        learning_rates = self.get_lr(
            self.step_count
        )

        for group, learning_rate in zip(
            self.optimizer.param_groups,
            learning_rates,
        ):
            group["lr"] = learning_rate

        self.step_count += 1

    def get_last_lr(self) -> list[float]:
        """
        Return the current learning rates.
        """

        return [
            group["lr"]
            for group in self.optimizer.param_groups
        ]
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.amor.training.loss import NextTokenLoss
from src.amor.training.scheduler import WarmupCosineScheduler


@dataclass
class TrainingStepResult:
    step: int
    loss: float
    learning_rate: float


class Trainer:
    """
    Minimal AMOR training loop.

    Handles:

        forward pass
        next-token loss
        backward pass
        gradient clipping
        optimizer step
        learning-rate scheduler
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        device: torch.device,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        if gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be greater than zero."
            )

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm

        self.loss_fn = NextTokenLoss()

        self.step_count = 0

    def train_step(
        self,
        batch: torch.Tensor,
    ) -> TrainingStepResult:
        self.model.train()

        batch = batch.to(self.device)

        # ---------------------------------
        # Next-token prediction
        # ---------------------------------
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        # ---------------------------------
        # Forward pass
        # ---------------------------------
        logits = self.model(input_ids)

        # ---------------------------------
        # Loss
        # ---------------------------------
        loss = self.loss_fn(
            logits,
            targets,
        )

        # ---------------------------------
        # Backward pass
        # ---------------------------------
        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # ---------------------------------
        # Gradient clipping
        # ---------------------------------
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip_norm,
        )

        # ---------------------------------
        # Parameter update
        # ---------------------------------
        self.optimizer.step()

        # ---------------------------------
        # Learning-rate update
        # ---------------------------------
        self.scheduler.step()

        self.step_count += 1

        learning_rate = (
            self.scheduler.get_last_lr()[0]
        )

        return TrainingStepResult(
            step=self.step_count,
            loss=loss.item(),
            learning_rate=learning_rate,
        )

    def train(
        self,
        dataloader: DataLoader,
        max_steps: int,
    ) -> list[TrainingStepResult]:
        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        results: list[TrainingStepResult] = []

        while self.step_count < max_steps:
            for batch in dataloader:
                result = self.train_step(batch)

                results.append(result)

                if self.step_count >= max_steps:
                    break

        return results
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
    AMOR training loop.

    Handles:

        - forward pass
        - next-token loss
        - automatic mixed precision
        - gradient accumulation
        - backward pass
        - gradient clipping
        - optimizer step
        - learning-rate scheduler
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        device: torch.device,
        gradient_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
    ) -> None:
        if gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be greater than zero."
            )

        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be "
                "greater than zero."
            )

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )

        # AMP is useful primarily on CUDA.
        self.use_amp = (
            use_amp
            and device.type == "cuda"
        )

        self.loss_fn = NextTokenLoss()

        self.step_count = 0
        self.accumulation_count = 0

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

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

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
        ):
            logits = self.model(input_ids)

            # ---------------------------------
            # Loss
            # ---------------------------------

            loss = self.loss_fn(
                logits,
                targets,
            )

            # Scale loss for gradient accumulation.
            scaled_loss = (
                loss
                / self.gradient_accumulation_steps
            )

        # ---------------------------------
        # Backward pass
        # ---------------------------------

        self.scaler.scale(
            scaled_loss
        ).backward()

        self.accumulation_count += 1

        # ---------------------------------
        # Optimizer step
        # ---------------------------------

        should_update = (
            self.accumulation_count
            >= self.gradient_accumulation_steps
        )

        if should_update:
            # Unscale gradients before clipping.
            self.scaler.unscale_(
                self.optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            self.optimizer.zero_grad(
                set_to_none=True
            )

            self.scheduler.step()

            self.step_count += 1
            self.accumulation_count = 0

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
                previous_step = self.step_count

                result = self.train_step(batch)

                # Only record actual optimizer updates.
                if self.step_count > previous_step:
                    results.append(result)

                if self.step_count >= max_steps:
                    break

        return results
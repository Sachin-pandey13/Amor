from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    # Reproducibility
    seed: int = 42

    # Data
    batch_size: int = 4
    sequence_length: int = 512

    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1

    # Training duration
    max_steps: int = 10_000
    warmup_steps: int = 500

    # Stability
    gradient_clip_norm: float = 1.0

    # Memory / performance
    gradient_accumulation_steps: int = 4
    use_amp: bool = True

    # Checkpointing
    checkpoint_interval: int = 500

    # Logging
    log_interval: int = 10
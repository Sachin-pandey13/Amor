from dataclasses import dataclass


@dataclass
class TokenBudget:
    """
    Tracks how many tokens have been acquired.
    """

    target_tokens: int
    current_tokens: int = 0

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError(
                "target_tokens must be greater than zero."
            )

        if self.current_tokens < 0:
            raise ValueError(
                "current_tokens cannot be negative."
            )

        if self.current_tokens > self.target_tokens:
            raise ValueError(
                "current_tokens cannot exceed target_tokens."
            )

    @property
    def remaining_tokens(self) -> int:
        return max(
            0,
            self.target_tokens - self.current_tokens,
        )

    @property
    def is_complete(self) -> bool:
        return self.current_tokens >= self.target_tokens

    def add(self, token_count: int) -> None:
        if token_count < 0:
            raise ValueError(
                "token_count cannot be negative."
            )

        self.current_tokens += token_count

        if self.current_tokens > self.target_tokens:
            self.current_tokens = self.target_tokens

    def can_accept(self, token_count: int) -> bool:
        if token_count < 0:
            raise ValueError(
                "token_count cannot be negative."
            )

        return (
            self.current_tokens + token_count
            <= self.target_tokens
        )
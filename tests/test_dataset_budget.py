import pytest

from src.amor.data.acquisition.budget import (
    TokenBudget,
)


def test_budget_starts_at_zero():
    budget = TokenBudget(
        target_tokens=1000
    )

    assert budget.current_tokens == 0
    assert budget.remaining_tokens == 1000
    assert budget.is_complete is False


def test_budget_adds_tokens():
    budget = TokenBudget(
        target_tokens=1000
    )

    budget.add(300)

    assert budget.current_tokens == 300
    assert budget.remaining_tokens == 700
    assert budget.is_complete is False


def test_budget_reaches_target():
    budget = TokenBudget(
        target_tokens=1000
    )

    budget.add(1000)

    assert budget.current_tokens == 1000
    assert budget.remaining_tokens == 0
    assert budget.is_complete is True


def test_budget_does_not_exceed_target():
    budget = TokenBudget(
        target_tokens=1000
    )

    budget.add(1500)

    assert budget.current_tokens == 1000
    assert budget.remaining_tokens == 0
    assert budget.is_complete is True


def test_can_accept_tokens():
    budget = TokenBudget(
        target_tokens=1000
    )

    budget.add(600)

    assert budget.can_accept(400) is True
    assert budget.can_accept(401) is False


def test_invalid_target():
    with pytest.raises(ValueError):
        TokenBudget(target_tokens=0)


def test_negative_current_tokens():
    with pytest.raises(ValueError):
        TokenBudget(
            target_tokens=1000,
            current_tokens=-1,
        )


def test_current_tokens_above_target():
    with pytest.raises(ValueError):
        TokenBudget(
            target_tokens=1000,
            current_tokens=1001,
        )


def test_negative_addition():
    budget = TokenBudget(
        target_tokens=1000
    )

    with pytest.raises(ValueError):
        budget.add(-1)


def test_negative_can_accept():
    budget = TokenBudget(
        target_tokens=1000
    )

    with pytest.raises(ValueError):
        budget.can_accept(-1)
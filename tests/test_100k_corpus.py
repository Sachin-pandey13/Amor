from src.amor.data.acquisition.balanced import (
    allocate_token_budget,
    validate_budget_allocation,
)


def test_allocate_token_budget_preserves_total():
    budgets = allocate_token_budget(
        100_000,
        {
            "fineweb": 0.35,
            "fineweb_edu": 0.25,
            "stackv2": 0.15,
            "finemath": 0.15,
            "aya": 0.10,
        },
    )

    assert sum(budgets.values()) == 100_000


def test_allocate_token_budget_uses_expected_distribution():
    budgets = allocate_token_budget(
        100_000,
        {
            "fineweb": 0.35,
            "fineweb_edu": 0.25,
            "stackv2": 0.15,
            "finemath": 0.15,
            "aya": 0.10,
        },
    )

    assert budgets["fineweb"] == 35_000
    assert budgets["fineweb_edu"] == 25_000
    assert budgets["stackv2"] == 15_000
    assert budgets["finemath"] == 15_000
    assert budgets["aya"] == 10_000


def test_allocate_budget_rejects_invalid_ratio():
    try:
        allocate_token_budget(
            100_000,
            {
                "fineweb": 0.50,
                "fineweb_edu": 0.50,
                "aya": 0.10,
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_allocate_budget_rejects_zero_target():
    try:
        allocate_token_budget(
            0,
            {
                "fineweb": 1.0,
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_validate_budget_allocation_accepts_valid_budget():
    budgets = {
        "fineweb": 35_000,
        "fineweb_edu": 25_000,
        "stackv2": 15_000,
        "finemath": 15_000,
        "aya": 10_000,
    }

    assert validate_budget_allocation(
        budgets,
        100_000,
    )


def test_validate_budget_allocation_rejects_mismatch():
    budgets = {
        "fineweb": 35_000,
        "fineweb_edu": 25_000,
        "stackv2": 15_000,
        "finemath": 15_000,
        "aya": 9_000,
    }

    assert not validate_budget_allocation(
        budgets,
        100_000,
    )
def allocate_token_budget(
    total_tokens: int,
    source_ratios: dict[str, float],
) -> dict[str, int]:
    """
    Allocate a total token budget across datasets.

    The returned integer budgets sum exactly to
    total_tokens.
    """

    if total_tokens <= 0:
        raise ValueError(
            "total_tokens must be greater than zero."
        )

    if not source_ratios:
        raise ValueError(
            "source_ratios cannot be empty."
        )

    for source, ratio in source_ratios.items():
        if ratio <= 0:
            raise ValueError(
                f"Invalid ratio for {source}: {ratio}"
            )

    ratio_sum = sum(
        source_ratios.values()
    )

    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "Source ratios must sum to 1.0."
        )

    raw_budgets = {
        source: total_tokens * ratio
        for source, ratio in source_ratios.items()
    }

    budgets = {
        source: int(value)
        for source, value in raw_budgets.items()
    }

    remainder = (
        total_tokens
        - sum(budgets.values())
    )

    # Distribute rounding remainder to the
    # largest fractional parts.
    fractional_parts = sorted(
        source_ratios,
        key=lambda source: (
            raw_budgets[source]
            - budgets[source]
        ),
        reverse=True,
    )

    for source in fractional_parts[
        :remainder
    ]:
        budgets[source] += 1

    return budgets


def validate_budget_allocation(
    budgets: dict[str, int],
    expected_total: int,
) -> bool:
    """
    Validate that an allocation is internally
    consistent.
    """

    if expected_total <= 0:
        return False

    if not budgets:
        return False

    if any(
        value < 0
        for value in budgets.values()
    ):
        return False

    return (
        sum(budgets.values())
        == expected_total
    )
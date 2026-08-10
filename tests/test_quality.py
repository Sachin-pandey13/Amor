from src.amor.data.quality import (
    QualityResult,
    check_quality,
)


def test_accept_normal_text():
    result = check_quality(
        "AMOR is a small language model project."
    )

    assert result == QualityResult(
        accepted=True,
        reason=None,
    )


def test_reject_empty_text():
    result = check_quality("")

    assert result.accepted is False
    assert result.reason == "empty"


def test_reject_whitespace_only():
    result = check_quality("   \n\t  ")

    assert result.accepted is False
    assert result.reason == "empty"


def test_reject_short_text():
    result = check_quality(
        "Too short",
        min_characters=20,
    )

    assert result.accepted is False
    assert result.reason == "too_short"


def test_accept_long_code():
    text = (
        "def calculate(x):\n"
        "    result = x * 2\n"
        "    return result\n"
    )

    result = check_quality(text)

    assert result.accepted is True


def test_reject_repeated_lines():
    text = "\n".join(
        ["same line"] * 10
    )

    result = check_quality(
        text,
        min_characters=1,
    )

    assert result.accepted is False
    assert result.reason == "repeated_lines"


def test_reject_url_heavy_text():
    text = (
        "https://example.com/"
        + "a" * 100
    )

    result = check_quality(
        text,
        min_characters=1,
        max_url_ratio=0.3,
    )

    assert result.accepted is False
    assert result.reason == "url_heavy"


def test_reject_control_character_heavy_text():
    text = "This is bad" + "\x01" * 10

    result = check_quality(
        text,
        min_characters=1,
    )

    assert result.accepted is False
    assert result.reason == "control_character_heavy"
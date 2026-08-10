import pytest

from src.amor.data.normalization import (
    normalize_text,
)


def test_normalize_nfc_unicode():
    text = "Cafe\u0301"

    result = normalize_text(text)

    assert result == "Café"


def test_normalize_line_endings():
    text = (
        "hello\r\n"
        "world\r"
        "again"
    )

    result = normalize_text(text)

    assert result == (
        "hello\n"
        "world\n"
        "again"
    )


def test_remove_trailing_whitespace():
    text = (
        "hello   \n"
        "world\t"
    )

    result = normalize_text(text)

    assert result == (
        "hello\n"
        "world"
    )


def test_collapse_excessive_blank_lines():
    text = (
        "hello\n"
        "\n"
        "\n"
        "\n"
        "world"
    )

    result = normalize_text(text)

    assert result == (
        "hello\n"
        "\n"
        "world"
    )


def test_collapse_spaces():
    text = "hello     world"

    result = normalize_text(text)

    assert result == (
        "hello world"
    )


def test_collapse_tabs_between_words():
    text = "hello\t\tworld"

    result = normalize_text(text)

    assert result == (
        "hello world"
    )


def test_strip_outer_whitespace():
    text = (
        "   hello world   "
    )

    result = normalize_text(text)

    assert result == (
        "hello world"
    )


def test_preserve_code_structure():
    text = (
        "def foo():\n"
        "    return 42"
    )

    result = normalize_text(text)

    assert result == (
        "def foo():\n"
        "    return 42"
    )


def test_preserve_code_indentation():
    text = (
        "def foo():\n"
        "    if True:\n"
        "        return 42"
    )

    result = normalize_text(text)

    assert result == (
        "def foo():\n"
        "    if True:\n"
        "        return 42"
    )


def test_never_concatenate_words():
    text = (
        "machine     learning "
        "is useful"
    )

    result = normalize_text(text)

    assert result == (
        "machine learning is useful"
    )


def test_preserve_punctuation_spacing():
    text = (
        "Hello,     world! "
        "How are you?"
    )

    result = normalize_text(text)

    assert result == (
        "Hello, world! "
        "How are you?"
    )


def test_preserve_markdown():
    text = (
        "# AMOR\n"
        "\n"
        "This is **important**.\n"
        "\n"
        "- Item one\n"
        "- Item two"
    )

    result = normalize_text(text)

    assert result == (
        "# AMOR\n"
        "\n"
        "This is **important**.\n"
        "\n"
        "- Item one\n"
        "- Item two"
    )


def test_repair_common_mojibake():
    # These escapes represent UTF-8 text that was incorrectly
    # decoded as Windows-1252.
    text = (
        "It\u00e2\u20ac\u2122s a test "
        "\u00e2\u20ac\u201d "
        "with \u00e2\u20ac\u0153quotes"
        "\u00e2\u20ac\u009d."
    )

    result = normalize_text(text)

    assert result == (
        "It\u2019s a test "
        "\u2014 "
        "with \u201cquotes"
        "\u201d."
    )


def test_repair_mojibake_does_not_change_normal_text():
    text = (
        "This is normal English text."
    )

    result = normalize_text(text)

    assert result == text


def test_non_string_rejected():
    with pytest.raises(TypeError):
        normalize_text(123)
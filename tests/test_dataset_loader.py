from src.amor.data.acquisition.loader import (
    extract_instruction_pair,
    extract_text,
)


def test_extract_text_from_text_field():
    record = {
        "text": "Hello AMOR."
    }

    assert extract_text(record) == "Hello AMOR."


def test_extract_text_from_content_field():
    record = {
        "content": "Python is useful."
    }

    assert extract_text(record) == (
        "Python is useful."
    )


def test_extract_text_strips_whitespace():
    record = {
        "text": "   Hello AMOR.   "
    }

    assert extract_text(record) == "Hello AMOR."


def test_extract_text_returns_empty_for_missing_text():
    record = {
        "title": "No content"
    }

    assert extract_text(record) == ""


def test_extract_text_ignores_empty_values():
    record = {
        "text": "",
        "content": "Actual content",
    }

    assert extract_text(record) == (
        "Actual content"
    )


def test_extract_instruction_pair():
    record = {
        "inputs": "What is Python?",
        "targets": "Python is a programming language.",
    }

    result = extract_instruction_pair(record)

    assert "User:" in result
    assert "What is Python?" in result
    assert "Assistant:" in result
    assert "Python is a programming language." in result


def test_extract_instruction_pair_strips_whitespace():
    record = {
        "inputs": "  What is Python?  ",
        "targets": "  A programming language.  ",
    }

    result = extract_instruction_pair(record)

    assert result == (
        "User:\n"
        "What is Python?\n\n"
        "Assistant:\n"
        "A programming language."
    )


def test_extract_instruction_pair_missing_input():
    record = {
        "targets": "Some answer.",
    }

    assert extract_instruction_pair(record) == ""


def test_extract_instruction_pair_missing_target():
    record = {
        "inputs": "Some question.",
    }

    assert extract_instruction_pair(record) == ""
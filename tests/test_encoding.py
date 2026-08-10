import pytest

from src.amor.data.encoding import (
    repair_encoding,
)


def test_repair_mojibake_middle_dot():
    text = "DAYS: News Â· Discussion"

    result = repair_encoding(text)

    assert result == "DAYS: News · Discussion"


def test_repair_mojibake_copyright():
    text = "Facts for FamiliesÂ©"

    result = repair_encoding(text)

    assert result == "Facts for Families©"


def test_repair_multiple_mojibake_sequences():
    text = "Copyright Â© 2012 Â· AMOR"

    result = repair_encoding(text)

    assert result == "Copyright © 2012 · AMOR"


def test_preserve_normal_ascii():
    text = "Hello AMOR. This is normal text."

    result = repair_encoding(text)

    assert result == text


def test_preserve_unicode():
    text = "Hello 世界 नमस्ते 🌍"

    result = repair_encoding(text)

    assert result == text


def test_non_string_rejected():
    with pytest.raises(TypeError):
        repair_encoding(123)
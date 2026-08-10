from __future__ import annotations

from typing import Final


# Common mojibake markers produced when UTF-8 bytes
# are incorrectly decoded as Latin-1 / Windows-1252.
_MOJIBAKE_MARKERS: Final[tuple[str, ...]] = (
    "Ã",
    "Â",
    "â",
    "ð",
    "�",
)


def _mojibake_score(text: str) -> int:
    """
    Estimate how likely text is to contain mojibake.

    A higher score means the text contains more common
    encoding-corruption markers.
    """

    return sum(
        text.count(marker)
        for marker in _MOJIBAKE_MARKERS
    )


def repair_encoding(text: str) -> str:
    """
    Conservatively repair common UTF-8 mojibake.

    The function attempts a Latin-1 round-trip only when
    the input contains strong evidence of encoding
    corruption.

    Normal ASCII, valid Unicode, emoji, and multilingual
    text are preserved.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if not text:
        return text

    original_score = _mojibake_score(text)

    # No evidence of mojibake.
    if original_score == 0:
        return text

    try:
        repaired = (
            text.encode("latin-1")
            .decode("utf-8")
        )
    except (
        UnicodeEncodeError,
        UnicodeDecodeError,
    ):
        # The text cannot safely be repaired using
        # this transformation.
        return text

    repaired_score = _mojibake_score(
        repaired
    )

    # Only accept the transformation when it
    # actually reduces the evidence of corruption.
    if repaired_score < original_score:
        return repaired

    return text
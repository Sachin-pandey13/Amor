from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reason: str | None = None


def check_quality(
    text: str,
    min_characters: int = 20,
    max_repeated_line_ratio: float = 0.5,
    max_url_ratio: float = 0.3,
) -> QualityResult:
    """
    Perform conservative quality checks on a document.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = text.strip()

    if not text:
        return QualityResult(
            accepted=False,
            reason="empty",
        )

    if len(text) < min_characters:
        return QualityResult(
            accepted=False,
            reason="too_short",
        )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        unique_lines = set(lines)

        repeated_ratio = (
            1
            - len(unique_lines) / len(lines)
        )

        if repeated_ratio > max_repeated_line_ratio:
            return QualityResult(
                accepted=False,
                reason="repeated_lines",
            )

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        text,
        flags=re.IGNORECASE,
    )

    if urls:
        url_characters = sum(
            len(url)
            for url in urls
        )

        url_ratio = (
            url_characters / len(text)
        )

        if url_ratio > max_url_ratio:
            return QualityResult(
                accepted=False,
                reason="url_heavy",
            )

    control_characters = sum(
        1
        for char in text
        if ord(char) < 32
        and char not in "\n\t"
    )

    control_ratio = (
        control_characters / len(text)
    )

    if control_ratio > 0.01:
        return QualityResult(
            accepted=False,
            reason="control_character_heavy",
        )

    return QualityResult(
        accepted=True,
    )
import re
import unicodedata


def _encode_possible_cp1252_mojibake(
    text: str,
) -> bytes:
    """
    Convert text that may contain UTF-8 decoded as CP1252
    back into the original byte sequence.

    CP1252 has undefined mappings for some bytes such as 0x9D.
    Those bytes can nevertheless appear in real-world mojibake,
    so they are handled explicitly.
    """

    output = bytearray()

    for char in text:
        codepoint = ord(char)

        # C1 control characters correspond directly to bytes
        # 0x80-0x9F. Some of these are undefined in CP1252,
        # but they can still occur in corrupted UTF-8 text.
        if 0x80 <= codepoint <= 0x9F:
            output.append(codepoint)
            continue

        try:
            encoded = char.encode("cp1252")
        except UnicodeEncodeError:
            # Character cannot belong to a CP1252 mojibake
            # sequence. Re-raise so the caller can safely
            # abandon this repair attempt.
            raise

        output.extend(encoded)

    return bytes(output)


def _repair_mojibake(text: str) -> str:
    """
    Repair common UTF-8 text that was incorrectly decoded
    using Windows-1252.

    Examples:

        Itâ€™s
        ->
        It’s

        â€”
        ->
        —

        â€œquotesâ€
        ->
        “quotes”
    """

    mojibake_markers = (
        "Ã",
        "Â",
        "â",
        "ð",
        "�",
    )

    # Normal text should pass through untouched.
    if not any(
        marker in text
        for marker in mojibake_markers
    ):
        return text

    try:
        raw_bytes = (
            _encode_possible_cp1252_mojibake(
                text
            )
        )

        repaired = raw_bytes.decode(
            "utf-8"
        )

    except (
        UnicodeEncodeError,
        UnicodeDecodeError,
    ):
        return text

    original_markers = sum(
        text.count(marker)
        for marker in mojibake_markers
    )

    repaired_markers = sum(
        repaired.count(marker)
        for marker in mojibake_markers
    )

    # Only accept the repair if it actually reduces
    # the amount of mojibake.
    if repaired_markers < original_markers:
        return repaired

    return text


def normalize_text(text: str) -> str:
    """
    Conservatively normalize dataset text.

    Guarantees:
    - common UTF-8 mojibake is repaired
    - NFC Unicode normalization
    - CRLF/CR converted to LF
    - trailing whitespace removed from lines
    - excessive blank lines collapsed
    - excessive horizontal spaces collapsed
    - code indentation preserved
    - outer whitespace removed
    - separate words are never concatenated
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    # ---------------------------------------------------------
    # 1. Repair common encoding corruption
    # ---------------------------------------------------------

    text = _repair_mojibake(text)

    # ---------------------------------------------------------
    # 2. Unicode normalization
    # ---------------------------------------------------------

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    # ---------------------------------------------------------
    # 3. Normalize line endings
    # ---------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # ---------------------------------------------------------
    # 4. Normalize horizontal whitespace
    #
    # Preserve indentation while collapsing spaces/tabs
    # inside actual content.
    # ---------------------------------------------------------

    lines = text.split("\n")

    normalized_lines = []

    for line in lines:
        # Remove trailing spaces/tabs only.
        line = line.rstrip(" \t")

        if not line:
            normalized_lines.append("")
            continue

        # Separate indentation from content.
        indentation_match = re.match(
            r"^[ \t]*",
            line,
        )

        indentation = (
            indentation_match.group(0)
            if indentation_match
            else ""
        )

        content = line[
            len(indentation):
        ]

        # Collapse horizontal whitespace inside
        # actual content.
        content = re.sub(
            r"[ \t]+",
            " ",
            content,
        )

        normalized_lines.append(
            indentation + content
        )

    text = "\n".join(
        normalized_lines
    )

    # ---------------------------------------------------------
    # 5. Collapse excessive blank lines
    #
    # Maximum: one empty line between content lines.
    # ---------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # ---------------------------------------------------------
    # 6. Remove outer whitespace
    # ---------------------------------------------------------

    text = text.strip()

    return text
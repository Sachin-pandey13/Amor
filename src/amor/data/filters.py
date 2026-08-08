def normalize_text(text: str) -> str:
    return " ".join(text.split())


def is_valid_document(
    document: dict,
    min_chars: int = 20,
    max_chars: int = 100_000,
) -> bool:
    text = document.get("text", "")

    if not isinstance(text, str):
        return False

    text = normalize_text(text)

    if len(text) < min_chars:
        return False

    if len(text) > max_chars:
        return False

    return True


def clean_document(document: dict) -> dict:
    cleaned = document.copy()
    cleaned["text"] = normalize_text(cleaned["text"])

    return cleaned
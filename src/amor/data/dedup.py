import hashlib


def document_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def deduplicate_documents(
    documents: list[dict],
) -> list[dict]:

    seen = set()
    unique_documents = []

    for document in documents:
        fingerprint = document_hash(document["text"])

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        unique_documents.append(document)

    return unique_documents
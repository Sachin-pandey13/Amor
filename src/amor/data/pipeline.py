from pathlib import Path
import json

from .sources import load_jsonl
from .filters import clean_document, is_valid_document
from .dedup import deduplicate_documents


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
) -> None:

    documents = load_jsonl(input_path)

    valid_documents = []

    for document in documents:
        if is_valid_document(document):
            valid_documents.append(
                clean_document(document)
            )

    unique_documents = deduplicate_documents(
        valid_documents
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for document in unique_documents:
            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Input documents: {len(documents)}")
    print(f"Valid documents: {len(valid_documents)}")
    print(f"Unique documents: {len(unique_documents)}")
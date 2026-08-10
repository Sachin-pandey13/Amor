from pathlib import Path

from src.amor.data.acquisition.config import (
    DATASET_SOURCES,
)
from src.amor.data.acquisition.loader import (
    stream_dataset,
)


OUTPUT_PATH = Path(
    "data/raw/tokenizer_bootstrap.txt"
)

DOCUMENTS_PER_SOURCE = 100


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_documents = 0
    total_characters = 0

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:

        for source in DATASET_SOURCES:
            print(
                f"Acquiring: {source.name}"
            )

            count = 0

            records = stream_dataset(
                source=source,
                max_documents=DOCUMENTS_PER_SOURCE,
            )

            for record in records:
                text = record["text"].strip()

                if not text:
                    continue

                output.write(text)
                output.write("\n\n")

                count += 1
                total_documents += 1
                total_characters += len(text)

            print(
                f"  Documents: {count}"
            )

    print()
    print("Tokenizer Bootstrap Corpus")
    print("==========================")
    print(
        f"Documents:   {total_documents}"
    )
    print(
        f"Characters:  {total_characters:,}"
    )
    print(
        f"Output:      {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import json

from tokenizers import Tokenizer

from src.amor.data.acquisition.budget import TokenBudget
from src.amor.data.acquisition.config import DATASET_SOURCES
from src.amor.data.acquisition.loader import stream_dataset


TOKENIZER_PATH = (
    "data/tokenizer/amor_tokenizer.json"
)

OUTPUT_PATH = Path(
    "data/raw/smoke_corpus.jsonl"
)

MANIFEST_PATH = Path(
    "data/raw/smoke_manifest.json"
)

TOKENS_PER_SOURCE = 1_000
DOCUMENTS_PER_SOURCE = 50


def main() -> None:
    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_stats = []
    total_documents = 0
    total_tokens = 0

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:

        for source in DATASET_SOURCES:
            print()
            print(f"Acquiring: {source.name}")

            budget = TokenBudget(
                target_tokens=TOKENS_PER_SOURCE
            )

            source_documents = 0

            records = stream_dataset(
                source=source,
                max_documents=DOCUMENTS_PER_SOURCE,
            )

            for record in records:
                if budget.is_complete:
                    break

                text = record["text"].strip()

                if not text:
                    continue

                token_count = len(
                    tokenizer.encode(text).ids
                )

                if token_count == 0:
                    continue

                # Don't partially truncate documents.
                if not budget.can_accept(
                    token_count
                ):
                    continue

                output_record = {
                    "id": record["id"],
                    "text": text,
                    "source": record["source"],
                    "dataset_id": record["dataset_id"],
                    "config": record["config"],
                    "split": record["split"],
                    "token_count": token_count,
                }

                output.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                budget.add(token_count)

                source_documents += 1
                total_documents += 1
                total_tokens += token_count

            source_stats.append(
                {
                    "dataset_name": source.name,
                    "dataset_id": source.dataset_id,
                    "config": source.config,
                    "split": source.split,
                    "target_tokens": TOKENS_PER_SOURCE,
                    "documents": source_documents,
                    "actual_tokens": budget.current_tokens,
                    "complete": budget.is_complete,
                }
            )

            print(
                f"  Documents: {source_documents}"
            )
            print(
                f"  Tokens:    {budget.current_tokens}"
            )

    manifest = {
        "corpus_name": "AMOR-smoke-corpus",
        "tokenizer": {
            "path": TOKENIZER_PATH,
            "vocab_size": tokenizer.get_vocab_size(),
        },
        "total_target_tokens": (
            TOKENS_PER_SOURCE
            * len(DATASET_SOURCES)
        ),
        "total_actual_tokens": total_tokens,
        "total_documents": total_documents,
        "tokens_per_source": TOKENS_PER_SOURCE,
        "sources": source_stats,
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("AMOR Balanced Smoke Corpus")
    print("==========================")
    print(
        f"Target tokens: "
        f"{manifest['total_target_tokens']}"
    )
    print(
        f"Actual tokens: "
        f"{total_tokens}"
    )
    print(
        f"Documents: "
        f"{total_documents}"
    )


if __name__ == "__main__":
    main()
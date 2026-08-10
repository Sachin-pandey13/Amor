from collections import Counter
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from .encoding import repair_encoding
from .normalization import normalize_text
from .quality import check_quality


DEFAULT_TOKENIZER_PATH = (
    "data/tokenizer/amor_tokenizer.json"
)


def _text_hash(text: str) -> str:
    """
    Generate a stable SHA-256 hash for normalized text.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _load_tokenizer(
    tokenizer_path: str,
) -> Tokenizer | None:
    """
    Load the tokenizer once for the entire
    processing run.

    Returns None when the tokenizer file does
    not exist.
    """

    path = Path(tokenizer_path)

    if not path.exists():
        return None

    return Tokenizer.from_file(
        str(path)
    )


def process_jsonl(
    input_path: str,
    output_path: str,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
) -> dict:
    """
    Process a JSONL corpus.

    Pipeline:

        JSONL
          ↓
        encoding repair
          ↓
        text normalization
          ↓
        quality filtering
          ↓
        exact deduplication
          ↓
        token counting
          ↓
        processed JSONL

    The tokenizer is loaded exactly once per
    processing run.

    Returns processing statistics.
    """

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(
            "Input corpus does not exist: "
            f"{input_file}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = _load_tokenizer(
        tokenizer_path
    )

    total = 0
    accepted = 0
    rejected = 0
    duplicates = 0

    reasons = Counter()

    input_tokens = 0
    output_tokens = 0

    seen_hashes: set[str] = set()

    with (
        input_file.open(
            "r",
            encoding="utf-8",
        ) as source,
        output_file.open(
            "w",
            encoding="utf-8",
        ) as destination,
    ):
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(record, dict):
                rejected += 1
                reasons["invalid_record"] += 1
                continue

            text = record.get(
                "text",
                "",
            )

            if not isinstance(text, str):
                rejected += 1
                reasons["invalid_text"] += 1
                continue

            # -------------------------------------------------
            # 1. Encoding repair
            # -------------------------------------------------

            text = repair_encoding(text)

            # -------------------------------------------------
            # 2. Conservative normalization
            # -------------------------------------------------

            normalized = normalize_text(
                text
            )

            # -------------------------------------------------
            # 3. Quality filtering
            # -------------------------------------------------

            result = check_quality(
                normalized
            )

            if not result.accepted:
                rejected += 1
                reasons[
                    result.reason
                ] += 1
                continue

            # -------------------------------------------------
            # 4. Exact deduplication
            # -------------------------------------------------

            text_hash = _text_hash(
                normalized
            )

            if text_hash in seen_hashes:
                duplicates += 1
                reasons["duplicate"] += 1
                continue

            seen_hashes.add(text_hash)

            # -------------------------------------------------
            # 5. Token counting
            # -------------------------------------------------

            original_token_count = record.get(
                "token_count",
                0,
            )

            if isinstance(
                original_token_count,
                int,
            ):
                input_tokens += (
                    original_token_count
                )

            if tokenizer is not None:
                token_count = len(
                    tokenizer.encode(
                        normalized
                    ).ids
                )
            elif isinstance(
                original_token_count,
                int,
            ):
                token_count = (
                    original_token_count
                )
            else:
                token_count = 0

            # -------------------------------------------------
            # 6. Update record
            # -------------------------------------------------

            record["text"] = normalized
            record["token_count"] = (
                token_count
            )
            record["text_hash"] = (
                text_hash
            )

            destination.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            accepted += 1
            output_tokens += token_count

    return {
        "total_records": total,
        "accepted_records": accepted,
        "rejected_records": rejected,
        "duplicate_records": duplicates,
        "unique_records": accepted,
        "acceptance_rate": (
            accepted / total
            if total
            else 0.0
        ),
        "rejection_reasons": dict(
            reasons
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokenizer_path": tokenizer_path,
        "tokenizer_loaded": (
            tokenizer is not None
        ),
    }
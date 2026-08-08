from pathlib import Path
import json


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    documents = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            documents.append(json.loads(line))

    return documents
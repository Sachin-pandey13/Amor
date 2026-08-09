from pathlib import Path

from src.amor.data.tokenizer import train_tokenizer


def test_tokenizer_training(tmp_path: Path):

    corpus = tmp_path / "sample.txt"

    corpus.write_text(
        "The Earth orbits the Sun.\n"
        "Python is a programming language.\n"
        "2 + 2 = 4.\n",
        encoding="utf-8",
    )

    output = tmp_path / "tokenizer.json"

    train_tokenizer(
        [str(corpus)],
        str(output),
    )

    assert output.exists()
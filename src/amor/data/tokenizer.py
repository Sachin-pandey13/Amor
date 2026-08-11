from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import trainers
from tokenizers import decoders

from .tokenizer_config import VOCAB_SIZE, SPECIAL_TOKENS


def create_tokenizer() -> tuple[Tokenizer, trainers.BpeTrainer]:
    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>",
            byte_fallback=True,
        )
    )

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFC(),
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    return tokenizer, trainer


def train_tokenizer(
    input_files: list[str],
    output_path: str,
) -> None:
    tokenizer, trainer = create_tokenizer()

    tokenizer.train(
        input_files,
        trainer,
    )

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(output))

    print(f"Tokenizer saved to: {output}")
    print(
        f"Vocabulary size: "
        f"{tokenizer.get_vocab_size()}"
    )
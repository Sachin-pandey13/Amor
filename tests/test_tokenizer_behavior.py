from tokenizers import Tokenizer
from src.amor.data.tokenizer import create_tokenizer


def train_test_tokenizer() -> Tokenizer:
    tokenizer, trainer = create_tokenizer()

    corpus = [
        "The Earth orbits the Sun.",
        "Python is a programming language.",
        "def calculate(x): return x * 2",
        "2 + 2 = 4",
        "Quantum computing uses qubits.",
        '{"tool":"youtube","action":"play"}',
        "D:\\Amor\\src\\amor",
        "Hello 世界 🚀",
    ]

    tokenizer.train_from_iterator(corpus, trainer)

    return tokenizer


def test_english():
    tokenizer = train_test_tokenizer()

    text = "What is quantum computing?"
    encoding = tokenizer.encode(text)

    assert len(encoding.ids) > 0


def test_code():
    tokenizer = train_test_tokenizer()

    text = "def calculate(x): return x * 2"
    encoding = tokenizer.encode(text)

    assert len(encoding.ids) > 0


def test_math():
    tokenizer = train_test_tokenizer()

    text = "2 + 2 = 4"
    encoding = tokenizer.encode(text)

    assert len(encoding.ids) > 0


def test_json_tool_command():
    tokenizer = train_test_tokenizer()

    text = '{"tool":"youtube","action":"play"}'
    encoding = tokenizer.encode(text)

    assert len(encoding.ids) > 0


def test_unicode():
    tokenizer = train_test_tokenizer()

    text = "Hello 世界 🚀"
    encoding = tokenizer.encode(text)

    assert len(encoding.ids) > 0


def test_decode_round_trip():
    tokenizer = train_test_tokenizer()

    text = "Python calculates 2 + 2."

    encoding = tokenizer.encode(text)
    decoded = tokenizer.decode(encoding.ids)

    assert decoded == text
from src.amor.data.tokenizer import create_tokenizer


def train_test_tokenizer():
    tokenizer, trainer = create_tokenizer()

    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language used for machine learning.",
        "def calculate(x): return x * 2",
        "2 + 2 = 4",
        "Quantum computing uses quantum bits called qubits.",
        '{"tool":"youtube","action":"play","query":"music"}',
        r"D:\Amor\src\amor\brain",
        "Hello 世界 🚀 नमस्ते",
    ]

    tokenizer.train_from_iterator(corpus, trainer)

    return tokenizer


def evaluate_tokenizer(tokenizer, name, text):
    encoding = tokenizer.encode(text)

    token_count = len(encoding.ids)
    character_count = len(text)

    compression = character_count / token_count

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Characters : {character_count}")
    print(f"Tokens     : {token_count}")
    print(f"Chars/token: {compression:.2f}")
    print(f"Tokens     : {encoding.tokens}")


def main():
    tokenizer = train_test_tokenizer()

    samples = {
        "English": "What is quantum computing?",
        "Code": "def calculate(x): return x * 2",
        "Math": "2 + 2 = 4",
        "JSON": '{"tool":"youtube","action":"play"}',
        "Path": r"D:\Amor\src\amor\brain",
        "Unicode": "Hello 世界 🚀 नमस्ते",
    }

    for name, text in samples.items():
        evaluate_tokenizer(tokenizer, name, text)


if __name__ == "__main__":
    main()
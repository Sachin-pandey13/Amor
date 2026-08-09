import time

import torch

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel


def benchmark(
    model: AMORModel,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    iterations: int = 20,
) -> None:
    model = model.to(device)
    model.eval()

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    average_ms = (
        elapsed / iterations
    ) * 1000

    tokens = (
        batch_size
        * seq_len
        * iterations
    )

    tokens_per_second = tokens / elapsed

    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Average forward pass: {average_ms:.2f} ms")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")

    if device.type == "cuda":
        memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(
            f"Peak allocated GPU memory: "
            f"{memory:.2f} MB"
        )


def main() -> None:
    config = AMORConfig()
    model = AMORModel(config)

    print("AMOR-B0 GPU Benchmark")
    print("=====================")

    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        benchmark(
            model,
            torch.device("cuda"),
            batch_size=1,
            seq_len=128,
        )

        benchmark(
            model,
            torch.device("cuda"),
            batch_size=1,
            seq_len=512,
        )

    benchmark(
        model,
        torch.device("cpu"),
        batch_size=1,
        seq_len=128,
    )


if __name__ == "__main__":
    main()
from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel


def count_parameters(model: AMORModel) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    config = AMORConfig()

    model = AMORModel(config)

    total_parameters = count_parameters(model)

    print("AMOR-B0 Configuration")
    print("---------------------")
    print(f"Vocabulary size : {config.vocab_size:,}")
    print(f"Embedding dim   : {config.dim}")
    print(f"Attention heads : {config.num_heads}")
    print(f"Layers          : {config.num_layers}")
    print(f"FFN dimension   : {config.ff_hidden_dim}")
    print(f"Context length  : {config.max_seq_len}")
    print()
    print(f"Trainable parameters: {total_parameters:,}")


if __name__ == "__main__":
    main()
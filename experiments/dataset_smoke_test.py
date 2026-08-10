from src.amor.data.acquisition.config import DATASET_SOURCES
from src.amor.data.acquisition.loader import stream_dataset


def main() -> None:
    source = DATASET_SOURCES[0]

    print("AMOR Dataset Smoke Test")
    print("========================")
    print(f"Dataset: {source.dataset_id}")
    print(f"Config:  {source.config}")
    print(f"Split:   {source.split}")
    print()

    records = stream_dataset(
        source=source,
        max_documents=5,
    )

    for index, record in enumerate(records, start=1):
        print(f"Record {index}")
        print("-" * 60)
        print(f"ID:       {record['id']}")
        print(f"Source:   {record['source']}")
        print(f"Dataset:  {record['dataset_id']}")
        print(f"Config:   {record['config']}")
        print(f"Text:     {record['text'][:500]}")
        print()


if __name__ == "__main__":
    main()
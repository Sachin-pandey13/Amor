from src.amor.data.acquisition.config import DATASET_SOURCES
from src.amor.data.acquisition.loader import stream_dataset


def main() -> None:
    print("AMOR Dataset Sources Smoke Test")
    print("================================")
    print()

    for source in DATASET_SOURCES:
        print(f"Dataset: {source.name}")
        print(f"ID:      {source.dataset_id}")
        print(f"Config:  {source.config}")
        print(f"Split:   {source.split}")

        try:
            records = stream_dataset(
                source=source,
                max_documents=2,
            )

            count = 0

            for record in records:
                count += 1

                print(
                    f"  Record {count}: "
                    f"{len(record['text'])} characters"
                )

            print(
                f"  Status: OK ({count} records)"
            )

        except Exception as exc:
            print(
                f"  Status: FAILED"
            )
            print(
                f"  Error: {exc}"
            )

        print()


if __name__ == "__main__":
    main()
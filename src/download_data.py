"""Download the official IMDB splits, keeping the test split separate."""
from datasets import load_dataset

from src.utils import load_config, resolve_project_path


def main():
    config = load_config("params.yaml")["data"]
    dataset = load_dataset(config["source"])
    output = resolve_project_path("data/raw/imdb")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        dataset[split].to_pandas()[["text", "label"]].to_csv(output / f"{split}.csv", index=False)
    print(f"Downloaded official IMDB train/test splits to {output}")


if __name__ == "__main__":
    main()

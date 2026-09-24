"""Create data visualizations from the training partition only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import resolve_project_path


def main():
    frame = pd.read_csv(resolve_project_path("data/processed/imdb/train.csv"))
    output = resolve_project_path("reports")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    frame.label.value_counts().sort_index().rename(index={0: "negative", 1: "positive"}).plot.bar(ax=axes[0], rot=0)
    axes[0].set(title="Training sentiment distribution", ylabel="Reviews")
    frame.text.str.split().str.len().plot.hist(bins=40, ax=axes[1])
    axes[1].set(title="Training review lengths", xlabel="Words", ylabel="Reviews")
    fig.tight_layout()
    fig.savefig(output / "data_exploration.png")
    plt.close(fig)
    print(frame.text.str.split().str.len().describe())


if __name__ == "__main__":
    main()

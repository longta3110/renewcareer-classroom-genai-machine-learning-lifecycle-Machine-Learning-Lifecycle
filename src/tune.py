"""Compare two learning rates using validation only; keep the winning checkpoint."""
import copy
import json
import shutil
import tempfile

from src.transfer_model import train
from src.utils import load_config, resolve_project_path, save_json


def main():
    config = load_config("params.yaml")
    base_rate = config["train"]["learning_rate"]
    results, best = [], -1.0
    output = resolve_project_path("models/transformer")
    with tempfile.TemporaryDirectory() as temporary:
        winner = resolve_project_path(temporary) / "winner"
        for rate in (base_rate, base_rate * 2):
            trial = copy.deepcopy(config)
            trial["train"]["learning_rate"] = rate
            train(trial)
            info = json.loads((output / "training.json").read_text())
            score = info["best_validation_f1"]
            results.append({"learning_rate": rate, "validation_f1": score, "run_id": info["run_id"]})
            if score > best:
                best = score
                shutil.copytree(output, winner, dirs_exist_ok=True)
        shutil.copytree(winner, output, dirs_exist_ok=True)
    save_json({"trials": results, "selected_validation_f1": best}, "reports/tuning.json")
    save_json({"best_validation_f1": best}, "reports/train_metrics.json")
    print(results)


if __name__ == "__main__":
    main()

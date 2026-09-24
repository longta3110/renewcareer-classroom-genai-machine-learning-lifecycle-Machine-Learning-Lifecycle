# Solution Review: Machine Learning Lifecycle

## Repository comparison

The personal GenAI-named repository and the RenewCareer classroom repository
started at the same commit, `bfc703d`. Their assignment README is identical.
The task is movie-review sentiment classification with HuggingFace transfer
learning, not a chatbot or text-generation application.

The earlier copied solution was a TF-IDF/logistic-regression baseline on 16
example reviews. Its seven passing tests did not establish completion of the
assignment. The baseline remains available through `src.model` for comparison.

## New solution

1. `src.download_data` downloads the official IMDB train and test partitions.
2. `src.transfer_data` removes empty reviews, duplicates, conflicting labels,
   and cross-partition overlap. It samples by label with a fixed seed and splits
   the training partition into training and validation. The official test
   partition stays separate from model selection.
3. `src.transfer_model` loads pretrained DistilBERT, tokenizes reviews, and trains
   its classification head. The frozen encoder retains pretrained language
   features and reduces CPU cost. Set `freeze_encoder: false` to fine-tune all
   layers. Validation F1 selects the checkpoint; test data is not used here.
4. MLflow records configuration, per-epoch loss and validation metrics, and
   the saved HuggingFace model/tokenizer files as run artifacts.
5. `src.transfer_evaluate` computes held-out accuracy, precision, recall, and F1,
   writes misclassified reviews to CSV, and draws a confusion matrix.
6. DVC tracks the dataset and the prepare/train/evaluate dependency graph.
7. The Flask API serves the saved transformer with input limits, health status,
   and Prometheus request/latency metrics. It returns 503 if the model is missing.
8. CI runs tests. Automatic-on-training-changes and manual training workflows prepare
   model artifacts and describe a private Google Cloud Run deployment.

## What the important code does

The tokenizer turns a review into token IDs and an attention mask:

```python
tokens = tokenizer(["This movie was excellent!"], truncation=True,
                   padding="max_length", max_length=128, return_tensors="pt")
```

The attention mask distinguishes real tokens from padding. Truncation limits
CPU and memory usage, but it also discards the ends of long reviews. The
training reviews average approximately 227 words, so this limitation matters.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2)
for parameter in model.base_model.parameters():
    parameter.requires_grad = False
```

The encoder starts with language knowledge learned before this assignment.
The sentiment classification head starts with newly initialized weights.
Freezing the encoder means this run updates only the head, not every layer.

```python
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

Labels are 0 for negative and 1 for positive. The model computes classification
loss, backpropagation calculates gradients, and AdamW changes trainable weights.
The implementation resets gradients before each batch and clips large gradients.

Validation F1 decides when to save a checkpoint. Accuracy measures the fraction
of all reviews classified correctly. Precision measures how often positive
predictions are correct; recall measures how many actual positive reviews were
found. F1 balances precision and recall. The confusion matrix separates false
positives from false negatives, and errors.csv contains examples to investigate.

MLflow records each run's settings and epoch metrics. The current run compares
checkpoints across epochs; broader learning-rate tuning remains an experiment
you can run with `python -m src.tune` (two learning rates, selected using validation
only), then `python -m src.transfer_evaluate`. To make tuning part of DVC, change
the train stage command to `python -m src.tune` and include `src/tune.py` in its
dependencies. DVC notices parameter changes and
reruns training and evaluation while reusing unchanged prepared data.

## Run locally in PowerShell

```powershell
cd 'C:\Users\tahoa\Documents\Renew Career\machine-learning-lifecycle-personal'
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ml.txt
python -m pytest -q
python -m src.download_data
dvc repro
python -m src.explore
python -m app.api
```

API: POST `/predict` with `{"text":"A wonderful movie"}`; POST
`/predict-batch` with `{"texts":["A wonderful movie","A terrible movie"]}`.
GET `/health` reports the MLflow run ID as model version; GET `/metrics`
exposes operational metrics, not accuracy monitoring without new ground truth.

MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`.
For notebooks, install `requirements-notebooks.txt` and select the local
`.venv` Python interpreter as the notebook kernel.
Use `params.yaml` to compare learning rates or encoder freezing. Compare
validation F1 in MLflow before evaluating the selected configuration on test.
The default uses a stratified subset for CPU feasibility; set train/test limits
to zero for all available deduplicated reviews. These subset results must not
be presented as a benchmark on the complete 50,000-review dataset.

## Deployment and remaining external setup

Build from the repository root after training:
`docker build -f app/Dockerfile -t sentiment-api .`
Then run `docker run --rm -p 5000:5000 sentiment-api`.

Cloud deployment requires a Google Cloud project, Artifact Registry repository,
Workload Identity Federation, a service account with the relevant deployment
permissions, and GitHub secrets `GCP_WORKLOAD_IDENTITY_PROVIDER` and
`GCP_SERVICE_ACCOUNT`. The deploy workflow takes a successful training run ID.
Docker and cloud deployment have not been executed on this machine.

DVC has a local cache. Shared artifact storage still requires your chosen
bucket and credentials. For example, install `dvc[gs]`, then configure
`dvc remote add -d storage gs://YOUR_BUCKET/ml-lifecycle` and `dvc push`.
Do not place credentials in Git. Local model files are excluded from Git.

## Review gate

### Verified locally on September 24, 2026

| Check | Result |
| --- | --- |
| Unit/integration tests | 19 passed |
| DVC prepare, train, evaluate | Completed; up to date |
| Training / validation / test reviews | 1600 / 400 / 1000 |
| Selected epoch | 2 |
| Validation F1 | 0.78298 |
| Test accuracy | 0.714 |
| Test precision | 0.65429 |
| Test recall | 0.91235 |
| Test F1 | 0.76206 |
| Notebook execution | All three executed successfully |
| Real saved-model health / batch / metrics endpoints | HTTP 200 |
| Docker and cloud deployment | Not run; Docker unavailable, cloud not configured |
| Two-rate tuning utility | Selection logic tested; full tuning not run |

These results show a working pipeline, not a high-performing production model.
High positive recall with lower precision means it predicts positive too often.
Possible next experiments include full-encoder fine-tuning, more training data,
and longer token sequences, selected using validation rather than test results.
Executed notebooks are in `reports/executed_notebooks`; evaluation charts and
misclassified examples are in `reports`. MLflow run ID:
`2da8e2ba68b445e6a4262bdd626b8aa0`.

The solution was reviewed locally on `codex/complete-transfer-learning` and
approved for submission by the repository owner on September 24, 2026.
The earlier baseline push remains in Git history.

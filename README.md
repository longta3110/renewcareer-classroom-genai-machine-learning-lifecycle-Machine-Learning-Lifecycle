# ML Lifecycle & MLOps Assignment: Sentiment Analysis with HuggingFace Transfer Learning

## 📋 Assignment Overview

Implementation and local run instructions: [Solution Review](SOLUTION_REVIEW.md).
The review distinguishes implemented features, verified results, and external deployment setup.

This comprehensive project guides you through the complete ML lifecycle using industry-standard MLOps tools. You'll build, version, track, and deploy a sentiment analysis model using transfer learning from HuggingFace, with automated CI/CD pipelines and production-ready deployment.

### Learning Objectives

1. **Data Lifecycle Management** - Use DVC for version control of datasets and ML artifacts
2. **Model Development** - Fine-tune pre-trained transformer models from HuggingFace
3. **Pipeline Orchestration** - Build reproducible ML pipelines with parameter tracking
4. **Experiment Tracking** - Use MLflow to log metrics, parameters, and models
5. **CI/CD Automation** - Implement GitHub Actions for automated testing and deployment
6. **Production Deployment** - Deploy models to cloud with monitoring capabilities

---

## 🎯 Project Dataset

**Dataset**: [Movie Review Sentiment Dataset (Microsoft/reviews)]([https://github.com/microsoft/xDNN-Review-Sentiment](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews))
- **Size**: ~50,000 movie reviews (train/test split)
- **Labels**: Binary sentiment classification (positive/negative)
- **Format**: CSV with review text and sentiment labels
- **Why this dataset**: Realistic size, clear labels, perfect for transfer learning with transformers

**Alternative datasets you can explore:**
- [IMDB Dataset](https://huggingface.co/datasets/imdb)
- [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full)
- [SST-2 (Stanford Sentiment Treebank)](https://huggingface.co/datasets/sst2)

---

## 🛠️ Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Data Versioning | DVC | Track datasets and model artifacts |
| Experiment Tracking | MLflow | Log parameters, metrics, and models |
| Model Hub | HuggingFace | Pre-trained transformers, tokenizers |
| Orchestration | Python Scripts | Reproducible training pipelines |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Cloud Deployment | Docker + Cloud Run/EC2 | Production serving |
| Monitoring | Flask API + Prometheus | Model performance tracking |

---

## 📁 Project Structure

```
ml-lifecycle-project/
├── README.md                          # This file
├── .github/
│   └── workflows/
│       ├── train.yml                  # Automated training on push
│       ├── test.yml                   # Unit and integration tests
│       └── deploy.yml                 # Automated deployment
├── data/
│   ├── raw/                           # Original dataset (DVC tracked)
│   │   └── .gitignore                 # DVC ignores large files
│   ├── processed/                     # Processed/tokenized data
│   └── data.dvc                       # DVC metadata file
├── models/
│   ├── trained/                       # Final model artifacts (DVC tracked)
│   └── models.dvc                     # DVC metadata
├── src/
│   ├── __init__.py
│   ├── config.yaml                    # Pipeline configuration
│   ├── data_loader.py                 # Data loading and preprocessing
│   ├── model.py                       # Model definition and training
│   ├── evaluate.py                    # Evaluation metrics
│   ├── inference.py                   # Batch and single prediction
│   └── utils.py                       # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and data analysis
│   ├── 02_model_development.ipynb     # Iterative development
│   └── 03_results_analysis.ipynb      # Results and interpretation
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py            # Data pipeline tests
│   ├── test_model.py                  # Model training tests
│   └── test_inference.py              # API endpoint tests
├── app/
│   ├── api.py                         # Flask API for serving
│   ├── requirements.txt               # Python dependencies
│   └── Dockerfile                     # Container for deployment
├── dvc.yaml                           # DVC pipeline definition
├── params.yaml                        # Experiment parameters
├── .dvc/                              # DVC configuration (auto-created)
├── mlruns/                            # MLflow artifacts (auto-created)
├── requirements.txt                   # Main dependencies
└── .gitignore                         # Git ignore rules

```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git and GitHub account
- Docker (for local deployment testing)
- ~5GB disk space (for datasets and models)
- GPU recommended (CUDA 11.8+) but CPU works too

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-lifecycle-project.git
cd ml-lifecycle-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC and MLflow
dvc init --no-scm  # Skip if using Git
mlflow experiments create -n "sentiment-analysis"
```

### Step 2: Download and Version Dataset

```bash
# Download the dataset
python src/download_data.py

# Add to DVC for version control
dvc add data/raw/reviews.csv
git add data/raw/reviews.csv.dvc .gitignore
git commit -m "Add dataset v1.0"

# Push to remote storage (if configured)
dvc push
```

### Step 3: Configure Experiment

Edit `params.yaml` to set your experiment parameters:

```yaml
train:
  model_name: "distilbert-base-uncased"  # HuggingFace model
  batch_size: 32
  epochs: 3
  learning_rate: 2e-5
  max_length: 128
  validation_split: 0.2

data:
  dataset_path: "data/raw/reviews.csv"
  test_size: 0.2
```

### Step 4: Run Pipeline Locally

```bash
# Train model with DVC pipeline
dvc repro

# View experiment metrics in MLflow
mlflow ui
# Open http://localhost:5000

# Compare runs
mlflow experiments compare --experiment-ids 1 2 3
```

### Step 5: Test and Validate

```bash
# Run unit tests
pytest tests/ -v

# Test the trained model
python -c "from src.inference import predict; predict('This movie was amazing!')"
```

### Step 6: Deploy Model

```bash
# Build Docker image
cd app
docker build -t sentiment-api:latest .

# Test locally
docker run -p 5000:5000 sentiment-api:latest

# Test endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I loved this movie"}'
```

---

## 📊 DVC Pipeline (Data Versioning & Workflows)

### Understanding DVC

DVC (Data Version Control) tracks:
- Large datasets and model files
- Pipeline definitions and dependencies
- Experiment reproducibility

### Pipeline Stages

Your `dvc.yaml` defines the ML workflow:

```yaml
stages:
  prepare:
    cmd: python src/data_loader.py
    deps:
      - data/raw/reviews.csv
      - src/data_loader.py
    outs:
      - data/processed/train.pkl
      - data/processed/test.pkl

  train:
    cmd: python src/model.py
    deps:
      - data/processed/train.pkl
      - src/model.py
    params:
      - train.model_name
      - train.batch_size
    outs:
      - models/trained/model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/trained/model.pkl
      - data/processed/test.pkl
    metrics:
      - eval_metrics.json:
          cache: false
```

### Key DVC Commands

```bash
# Run entire pipeline (respects dependencies)
dvc repro

# Run specific stage
dvc repro prepare
dvc repro train

# Check what changed
dvc diff

# Compare metrics across experiments
dvc metrics show
dvc plots show

# Visualize pipeline
dvc dag
```

---

## 📈 MLflow Experiment Tracking

### Logging During Training

The `src/model.py` logs experiments automatically:

```python
import mlflow
from transformers import Trainer

# Start MLflow run
mlflow.start_run()
mlflow.log_param("model_name", "distilbert-base-uncased")
mlflow.log_param("batch_size", 32)

# Train model
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()

# Log metrics
mlflow.log_metrics({
    "train_loss": train_loss,
    "val_accuracy": val_accuracy,
    "val_f1": val_f1
})

# Log model
mlflow.transformers.log_model(model, "model")
mlflow.end_run()
```

### Viewing Results

```bash
# Start MLflow UI
mlflow ui

# Navigate to http://localhost:5000
# Compare runs, metrics, and artifacts
```

### Accessing Logged Models

```python
import mlflow.transformers

# Load best model
model = mlflow.transformers.load_model(
    "runs:/RUN_ID/model",
    return_type="huggingface"
)
```

---

## 🔄 GitHub Actions CI/CD Pipeline

### Workflow 1: Training (`.github/workflows/train.yml`)

Triggers on code push:
1. Checkout code
2. Install dependencies
3. Download data (DVC)
4. Run DVC pipeline (training)
5. Log metrics to MLflow
6. Commit metrics back to repo

```yaml
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          dvc remote add -d myremote s3://my-bucket/dvc-storage
      
      - name: Pull data
        run: dvc pull
      
      - name: Run training pipeline
        run: dvc repro
      
      - name: Push metrics
        run: dvc push
```

### Workflow 2: Testing (`.github/workflows/test.yml`)

Triggers on pull requests:
1. Run unit tests
2. Validate data pipeline
3. Check code quality
4. Test API endpoints

### Workflow 3: Deployment (`.github/workflows/deploy.yml`)

Triggers on release tags:
1. Build Docker image
2. Push to container registry
3. Deploy to production
4. Run smoke tests

---

## 🎓 Task Breakdown for Students

### Task 1: Data Exploration & Preparation (Week 1)
**File**: `notebooks/01_data_exploration.ipynb`, `src/data_loader.py`

**Objectives**:
- [ ] Load and explore the dataset
- [ ] Analyze sentiment distribution
- [ ] Implement train/test split
- [ ] Create data processing pipeline
- [ ] Track data with DVC

**TODO Code Sections**:
```python
# In src/data_loader.py
class DataPreprocessor:
    def __init__(self, data_path, max_length=128):
        # TODO: Load data from CSV
        pass
    
    def preprocess(self):
        # TODO: Clean text (lowercase, remove special chars)
        # TODO: Tokenize and create train/test splits
        # TODO: Save to processed/ directory
        pass
```

### Task 2: Model Development & Training (Week 2-3)
**File**: `src/model.py`, `notebooks/02_model_development.ipynb`

**Objectives**:
- [ ] Load pre-trained HuggingFace model
- [ ] Set up training arguments
- [ ] Implement training loop
- [ ] Log experiments to MLflow
- [ ] Track model with DVC

**TODO Code Sections**:
```python
# In src/model.py
class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased"):
        # TODO: Load tokenizer from HuggingFace
        # TODO: Load pre-trained model
        # TODO: Freeze/unfreeze layers as needed
        pass
    
    def train(self, train_data, val_data, epochs=3):
        # TODO: Create Trainer with training arguments
        # TODO: Log hyperparameters to MLflow
        # TODO: Train and validate
        # TODO: Save model and log to MLflow
        pass
```

### Task 3: Evaluation & Analysis (Week 3)
**File**: `src/evaluate.py`, `notebooks/03_results_analysis.ipynb`

**Objectives**:
- [ ] Calculate accuracy, precision, recall, F1
- [ ] Generate confusion matrix
- [ ] Perform error analysis
- [ ] Create visualizations
- [ ] Document findings

**TODO Code Sections**:
```python
# In src/evaluate.py
def evaluate_model(model, test_data):
    # TODO: Generate predictions
    # TODO: Calculate metrics (accuracy, precision, recall, F1)
    # TODO: Create confusion matrix
    # TODO: Save metrics to eval_metrics.json
    # TODO: Log to MLflow
    pass
```

### Task 4: API Development & Testing (Week 4)
**File**: `app/api.py`, `tests/test_inference.py`

**Objectives**:
- [ ] Build Flask API for predictions
- [ ] Implement batch prediction endpoint
- [ ] Add input validation
- [ ] Write integration tests
- [ ] Document API endpoints

**TODO Code Sections**:
```python
# In app/api.py
from flask import Flask, request, jsonify
from src.inference import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # TODO: Get text from request
    # TODO: Validate input
    # TODO: Call predict() function
    # TODO: Return JSON response with sentiment and confidence
    pass

@app.route('/health', methods=['GET'])
def health():
    # TODO: Return model status and version
    pass
```

### Task 5: CI/CD Pipeline Setup (Week 4-5)
**Files**: `.github/workflows/*.yml`

**Objectives**:
- [ ] Configure GitHub Actions for training
- [ ] Set up DVC remote storage (AWS S3/Google Cloud)
- [ ] Implement automated testing
- [ ] Create deployment workflow
- [ ] Test end-to-end pipeline

**TODO**:
- [ ] Create AWS S3 bucket for DVC storage
- [ ] Add GitHub secrets for AWS credentials
- [ ] Implement `.github/workflows/train.yml`
- [ ] Implement `.github/workflows/test.yml`
- [ ] Implement `.github/workflows/deploy.yml`

### Task 6: Production Deployment (Week 5-6)
**Files**: `app/Dockerfile`, `app/requirements.txt`

**Objectives**:
- [ ] Containerize model service
- [ ] Create health check endpoint
- [ ] Set up monitoring
- [ ] Deploy to cloud platform
- [ ] Document deployment process

**TODO Code Sections**:
```dockerfile
# In app/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# TODO: Copy requirements and install
# TODO: Copy model and source code
# TODO: Expose port 5000
# TODO: Set entrypoint to run Flask app
```

---

## 💡 Key Concepts Explained

### Transfer Learning with HuggingFace

We use `DistilBERT` (smaller, faster BERT variant):
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # Binary classification
)

# Fine-tune only the top layers
for param in model.distilbert.parameters():
    param.requires_grad = False  # Freeze base model

for param in model.pre_classifier.parameters():
    param.requires_grad = True   # Unfreeze classifier
```

### DVC Workflow Example

```bash
# 1. Modify training parameters
vim params.yaml  # Change learning_rate, batch_size

# 2. Run pipeline (only re-runs affected stages)
dvc repro

# 3. Compare metrics across runs
dvc metrics show --all
dvc plots show training_loss

# 4. Version results
git add dvc.lock metrics.json
git commit -m "Train v2: improved learning rate"

# 5. Share artifacts
dvc push  # Push to remote storage
```

### MLflow Experiment Comparison

```python
import mlflow.pyfunc

# Register best model
best_run = mlflow.search_runs(
    experiment_ids=[exp_id],
    order_by=["metrics.val_f1 DESC"],
    max_results=1
).iloc[0]

mlflow.register_model(
    f"runs:/{best_run.run_id}/model",
    "sentiment-classifier"
)
```

---

## 📋 Deliverables Checklist

### Phase 1: Data & Exploration
- [ ] Dataset downloaded and tracked with DVC
- [ ] EDA notebook with visualizations
- [ ] Data processing pipeline implemented
- [ ] Train/test splits created

### Phase 2: Model Development
- [ ] HuggingFace model integrated
- [ ] Training script with MLflow logging
- [ ] Experiments logged and compared
- [ ] Best model selected and saved

### Phase 3: Evaluation & Testing
- [ ] Evaluation metrics calculated
- [ ] Unit tests for data pipeline
- [ ] Unit tests for model training
- [ ] Integration tests for inference

### Phase 4: API & Deployment
- [ ] Flask API with /predict endpoint
- [ ] Health check endpoint
- [ ] Docker container created
- [ ] Local deployment tested

### Phase 5: CI/CD & Production
- [ ] GitHub Actions workflows created
- [ ] DVC remote storage configured
- [ ] Automated training enabled
- [ ] Production deployment tested
- [ ] Documentation completed

---

## 🔗 Useful Resources

### HuggingFace
- [Transformers Documentation](https://huggingface.co/transformers/)
- [Model Hub](https://huggingface.co/models)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Course: NLP with Transformers](https://huggingface.co/course)

### DVC
- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorials](https://dvc.org/learn)
- [DVC Pipelines](https://dvc.org/doc/user-guide/pipelines)

### MLflow
- [MLflow Documentation](https://mlflow.org/docs/)
- [Experiment Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)

### GitHub Actions
- [Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Examples](https://github.com/actions)

### MLOps Best Practices
- [Made With ML - MLOps Fundamentals](https://madewithml.com/)
- [MLOps.community](https://mlops.community/)

---

## 🐛 Troubleshooting

### DVC Pull Fails
```bash
# Check DVC remote configuration
dvc remote list

# Manually set remote
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc config core.autostage true
```

### Model Training OOM (Out of Memory)
```yaml
# Reduce batch size in params.yaml
train:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 2  # Maintain effective batch size
```

### MLflow UI Not Opening
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Start fresh
mlflow ui --host 0.0.0.0 --port 5000
```

### GitHub Actions Secret Issues
```bash
# Add secrets via CLI or GitHub UI
# Settings → Secrets → New repository secret
# Add: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
```

---

## 📞 Support

- **Office Hours**: Day Time
- **Discussion Forum**: GitHub Discussions and WhatsApp Group
- **Email**: desmondonam@gmail.com
- **Slack**: #ml-lifecycle-help

---

## 📝 Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Data Pipeline | 15 | DVC versioning, EDA, preprocessing |
| Model Development | 25 | Transfer learning, MLflow logging, hyperparameter tuning |
| Evaluation | 15 | Metrics, error analysis, visualizations |
| API & Testing | 20 | Flask API, unit tests, integration tests |
| CI/CD & Deployment | 20 | GitHub Actions, Docker, cloud deployment |
| Documentation | 5 | README clarity, code comments, deployment guide |
| **Total** | **100** | |

---

## 📄 License

This project is provided for educational purposes. Feel free to extend and modify for your learning.

---

**Last Updated**: January 2026
**Maintained by**: ML Engineering Team

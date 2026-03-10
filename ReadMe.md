# Titanic MLOps Pipeline 🚢

End-to-end Machine Learning pipeline using **Apache Airflow** for orchestration and **MLflow** for experiment tracking and model registry. Predicts Titanic passenger survival.

---

## Project Structure

```
titanic_mlops/
├── mlops_airflow_mlflow_pipeline.py   # Main Airflow DAG
├── run_experiments.py                 # Task 10: Run 3 experiments
├── setup.sh                           # One-command environment setup
├── requirements.txt                   # Python dependencies
├── technical_report.docx              # Full technical report
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10 or 3.11
- pip3
- curl (for dataset download)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/titanic-mlops-pipeline.git
cd titanic-mlops-pipeline
```

### 2. Run Setup Script (one command)
```bash
bash setup.sh
```
This will:
- Create a Python virtual environment at `~/titanic_mlops/venv`
- Install all dependencies (Airflow 2.9.1, MLflow 2.13.0, scikit-learn, etc.)
- Download the Titanic dataset to `~/airflow/data/titanic.csv`
- Initialize the Airflow SQLite database
- Create an admin user (username: `admin`, password: `admin`)
- Copy the DAG to `~/airflow/dags/`

### 3. Start Services (3 separate terminals)

**Terminal 1 — MLflow server:**
```bash
source ~/titanic_mlops/venv/bin/activate
export AIRFLOW_HOME=~/airflow
mlflow server --host 127.0.0.1 --port 5000
```

**Terminal 2 — Airflow webserver:**
```bash
source ~/titanic_mlops/venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow webserver --port 8080
```

**Terminal 3 — Airflow scheduler:**
```bash
source ~/titanic_mlops/venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow scheduler
```

### 4. Access UIs
| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI  | http://localhost:5000 | (no auth) |

---

## Triggering the Pipeline

### Single manual run
```bash
source ~/titanic_mlops/venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow dags trigger titanic_mlops_pipeline
```

### Run all 3 experiments (Task 10)
```bash
source ~/titanic_mlops/venv/bin/activate
export AIRFLOW_HOME=~/airflow
python run_experiments.py
```
This runs 3 different hyperparameter configurations automatically:
- Experiment 1: Random Forest (n_estimators=50, max_depth=3)
- Experiment 2: Random Forest (n_estimators=200, max_depth=10)
- Experiment 3: Logistic Regression (C=0.5, max_iter=500)

---

## Changing Hyperparameters Manually

Edit the `PIPELINE_CONFIG` block at the top of `mlops_airflow_mlflow_pipeline.py`:

```python
PIPELINE_CONFIG = {
    "model_type":    "RandomForest",   # "LogisticRegression" | "RandomForest"
    "n_estimators":  100,
    "max_depth":     5,
    "C":             1.0,
    "max_iter":      200,
    "test_size":     0.2,
    "random_state":  42,
    "accuracy_threshold": 0.80,
}
```

Then copy the updated DAG to Airflow and trigger:
```bash
cp mlops_airflow_mlflow_pipeline.py ~/airflow/dags/
airflow dags trigger titanic_mlops_pipeline
```

---

## Pipeline Tasks

| Task | ID | Description |
|---|---|---|
| 1 | `start` | Anchor node |
| 2 | `data_ingestion` | Load CSV, print shape, log missing, XCom path |
| 3 | `data_validation` | Validate missing %, retry on failure |
| 4a | `handle_missing` | Fill Age (median), Embarked (mode) — **parallel** |
| 4b | `feature_engineering` | Create FamilySize, IsAlone — **parallel** |
| 5 | `data_encoding` | Encode Sex/Embarked, drop irrelevant cols |
| 6 | `model_training` | Train + log to MLflow |
| 7 | `model_evaluation` | Compute metrics, log to MLflow, XCom accuracy |
| 8 | `branch_on_accuracy` | BranchPythonOperator (≥0.80 → register, else reject) |
| 9a | `register_model` | Register in MLflow Model Registry |
| 9b | `reject_model` | Log rejection reason to MLflow |
| — | `end` | Final anchor |

---

## Retry Demo

Task `data_validation` deliberately fails on its first attempt to demonstrate Airflow's retry mechanism. It will:
1. Fail with `RuntimeError` on attempt 1
2. Wait 10 seconds
3. Succeed on attempt 2

This is controlled by a sentinel file at `~/airflow/tmp_pipeline/.validation_retried`.

---

## MLflow Experiment Comparison

After running all 3 experiments:
1. Open http://localhost:5000
2. Select experiment **Titanic_Survival_Prediction**
3. Select all 3 runs → click **Compare**
4. Use the Chart view to compare `accuracy`, `f1_score`, `precision`, `recall`

---

## Requirements

See `requirements.txt`. Key packages:
- `apache-airflow==2.9.1`
- `mlflow==2.13.0`
- `scikit-learn==1.5.0`
- `pandas==2.2.2`
- `numpy==1.26.4`

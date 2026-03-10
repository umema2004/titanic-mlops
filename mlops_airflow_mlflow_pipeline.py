# =============================================================================
# mlops_airflow_mlflow_pipeline.py
# Titanic Survival Prediction - End-to-End MLOps Pipeline
# Apache Airflow + MLflow
# =============================================================================

import os
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# CONFIGURATION — Edit these before each run to vary hyperparameters (Task 10)
# =============================================================================
PIPELINE_CONFIG = {
    "model_type":    "RandomForest",   # "LogisticRegression" | "RandomForest"
    "n_estimators":  100,              # RandomForest only
    "max_depth":     5,                # RandomForest only
    "C":             1.0,              # LogisticRegression only
    "max_iter":      200,              # LogisticRegression only
    "test_size":     0.2,
    "random_state":  42,
    "accuracy_threshold": 0.80,
}

# Path to the Titanic CSV (place it in ~/airflow/data/ or update path)
TITANIC_CSV_PATH = os.path.expanduser("~/airflow/data/titanic.csv")

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT   = "Titanic_Survival_Prediction"

# =============================================================================
# DEFAULT ARGS
# =============================================================================
default_args = {
    "owner":            "student",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,                        # Task 3 retry evidence
    "retry_delay":      timedelta(seconds=10),
    "start_date":       datetime(2024, 1, 1),
}

# =============================================================================
# TASK 1 – helper: shared temp directory
# =============================================================================
TMP_DIR = os.path.expanduser("~/airflow/tmp_pipeline")
os.makedirs(TMP_DIR, exist_ok=True)


# =============================================================================
# TASK 2 – DATA INGESTION
# =============================================================================
def task_data_ingestion(**context):
    """Load Titanic CSV, print shape, log missing values, push path via XCom."""
    logging.info("=== TASK 2: Data Ingestion ===")

    if not os.path.exists(TITANIC_CSV_PATH):
        raise FileNotFoundError(
            f"Titanic CSV not found at {TITANIC_CSV_PATH}. "
            "Run the setup script to download it."
        )

    df = pd.read_csv(TITANIC_CSV_PATH)

    # Print dataset shape
    logging.info(f"Dataset shape: {df.shape}")
    print(f"[INGESTION] Dataset shape: {df.shape}")

    # Log missing values
    missing = df.isnull().sum()
    logging.info(f"Missing values per column:\n{missing}")
    print(f"[INGESTION] Missing values:\n{missing.to_string()}")

    # Save raw copy to tmp
    raw_path = os.path.join(TMP_DIR, "raw_data.csv")
    df.to_csv(raw_path, index=False)

    # Push dataset path via XCom
    context["ti"].xcom_push(key="raw_data_path", value=raw_path)
    context["ti"].xcom_push(key="dataset_rows",  value=int(df.shape[0]))
    logging.info(f"Raw data saved to {raw_path} and path pushed via XCom.")


# =============================================================================
# TASK 3 – DATA VALIDATION  (intentional failure on attempt 1 for retry demo)
# =============================================================================
def task_data_validation(**context):
    """
    Validate missing % for Age and Embarked.
    Intentionally fails on the first attempt to demonstrate retry behavior.
    """
    logging.info("=== TASK 3: Data Validation ===")

    raw_path = context["ti"].xcom_pull(
        task_ids="data_ingestion", key="raw_data_path"
    )
    df = pd.read_csv(raw_path)

    # ── Intentional failure on first try (retry demo) ──
    retry_flag = os.path.join(TMP_DIR, ".validation_retried")
    if not os.path.exists(retry_flag):
        open(retry_flag, "w").close()          # create flag so next try passes
        raise RuntimeError(
            "[RETRY DEMO] First attempt fails intentionally. "
            "Airflow will retry automatically."
        )

    # ── Real validation ──
    total_rows = len(df)
    cols_to_check = {"Age": 30.0, "Embarked": 30.0}

    for col, threshold in cols_to_check.items():
        missing_pct = (df[col].isnull().sum() / total_rows) * 100
        logging.info(f"{col}: {missing_pct:.2f}% missing")
        print(f"[VALIDATION] {col}: {missing_pct:.2f}% missing")

        if missing_pct > threshold:
            raise ValueError(
                f"[VALIDATION FAILED] Column '{col}' has {missing_pct:.2f}% "
                f"missing values — exceeds threshold of {threshold}%."
            )

    logging.info("All validation checks passed.")
    context["ti"].xcom_push(key="validation_passed", value=True)


# =============================================================================
# TASK 4a – HANDLE MISSING VALUES  (runs in parallel with 4b)
# =============================================================================
def task_handle_missing(**context):
    """Fill missing Age with median; fill missing Embarked with mode."""
    logging.info("=== TASK 4a: Handle Missing Values ===")

    raw_path = context["ti"].xcom_pull(
        task_ids="data_ingestion", key="raw_data_path"
    )
    df = pd.read_csv(raw_path)

    # Fill Age with median
    age_median = df["Age"].median()
    df["Age"].fillna(age_median, inplace=True)
    logging.info(f"Age missing filled with median: {age_median:.1f}")

    # Fill Embarked with mode
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"].fillna(embarked_mode, inplace=True)
    logging.info(f"Embarked missing filled with mode: {embarked_mode}")

    # Save
    cleaned_path = os.path.join(TMP_DIR, "data_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    context["ti"].xcom_push(key="cleaned_data_path", value=cleaned_path)
    logging.info(f"Cleaned data saved to {cleaned_path}")


# =============================================================================
# TASK 4b – FEATURE ENGINEERING  (runs in parallel with 4a)
# =============================================================================
def task_feature_engineering(**context):
    """
    Create FamilySize = SibSp + Parch + 1
    Create IsAlone    = 1 if FamilySize == 1 else 0
    These features are saved separately and merged before encoding.
    """
    logging.info("=== TASK 4b: Feature Engineering ===")

    raw_path = context["ti"].xcom_pull(
        task_ids="data_ingestion", key="raw_data_path"
    )
    df = pd.read_csv(raw_path)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    logging.info(f"FamilySize stats:\n{df['FamilySize'].describe()}")
    logging.info(f"IsAlone distribution:\n{df['IsAlone'].value_counts()}")

    features_path = os.path.join(TMP_DIR, "data_features.csv")
    df[["PassengerId", "FamilySize", "IsAlone"]].to_csv(features_path, index=False)
    context["ti"].xcom_push(key="features_path", value=features_path)
    logging.info(f"Engineered features saved to {features_path}")


# =============================================================================
# TASK 5 – DATA ENCODING
# =============================================================================
def task_data_encoding(**context):
    """Merge cleaned data + features, encode Sex/Embarked, drop irrelevant cols."""
    logging.info("=== TASK 5: Data Encoding ===")

    cleaned_path = context["ti"].xcom_pull(
        task_ids="handle_missing", key="cleaned_data_path"
    )
    features_path = context["ti"].xcom_pull(
        task_ids="feature_engineering", key="features_path"
    )

    df_clean    = pd.read_csv(cleaned_path)
    df_features = pd.read_csv(features_path)

    # Merge on PassengerId
    df = df_clean.merge(df_features, on="PassengerId", how="left")

    # Encode Sex: male → 0, female → 1
    le_sex = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])

    # Encode Embarked: S → 2, C → 0, Q → 1 (alphabetical)
    le_emb = LabelEncoder()
    df["Embarked"] = le_emb.fit_transform(df["Embarked"])

    # Drop irrelevant / high-cardinality columns
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    logging.info(f"Encoded dataframe shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")

    encoded_path = os.path.join(TMP_DIR, "data_encoded.csv")
    df.to_csv(encoded_path, index=False)
    context["ti"].xcom_push(key="encoded_data_path", value=encoded_path)
    logging.info(f"Encoded data saved to {encoded_path}")


# =============================================================================
# TASK 6 – MODEL TRAINING WITH MLflow
# =============================================================================
def task_model_training(**context):
    """Train model, log params & artifact to MLflow."""
    logging.info("=== TASK 6: Model Training ===")

    encoded_path = context["ti"].xcom_pull(
        task_ids="data_encoding", key="encoded_data_path"
    )
    df = pd.read_csv(encoded_path)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PIPELINE_CONFIG["test_size"],
        random_state=PIPELINE_CONFIG["random_state"],
        stratify=y,
    )

    # ── MLflow ──
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:

        model_type = PIPELINE_CONFIG["model_type"]
        mlflow.log_param("model_type",    model_type)
        mlflow.log_param("test_size",     PIPELINE_CONFIG["test_size"])
        mlflow.log_param("random_state",  PIPELINE_CONFIG["random_state"])

        if model_type == "RandomForest":
            mlflow.log_param("n_estimators", PIPELINE_CONFIG["n_estimators"])
            mlflow.log_param("max_depth",    PIPELINE_CONFIG["max_depth"])
            model = RandomForestClassifier(
                n_estimators = PIPELINE_CONFIG["n_estimators"],
                max_depth    = PIPELINE_CONFIG["max_depth"],
                random_state = PIPELINE_CONFIG["random_state"],
            )
        else:  # LogisticRegression
            mlflow.log_param("C",        PIPELINE_CONFIG["C"])
            mlflow.log_param("max_iter", PIPELINE_CONFIG["max_iter"])
            model = LogisticRegression(
                C        = PIPELINE_CONFIG["C"],
                max_iter = PIPELINE_CONFIG["max_iter"],
                random_state = PIPELINE_CONFIG["random_state"],
            )

        model.fit(X_train, y_train)

        # Log dataset sizes
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size",  len(X_test))

        # Compute and log ALL metrics inside the same run
        y_pred    = model.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)

        logging.info(f"Metrics: Accuracy={accuracy:.4f} Precision={precision:.4f} "
                     f"Recall={recall:.4f} F1={f1:.4f}")

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        logging.info(f"MLflow run_id: {run_id}")

    # Save results for evaluation task
    import pickle
    test_data_path = os.path.join(TMP_DIR, "test_split.pkl")
    with open(test_data_path, "wb") as f:
        pickle.dump({
            "X_test": X_test, "y_test": y_test, "model": model,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
        }, f)

    context["ti"].xcom_push(key="mlflow_run_id",  value=run_id)
    context["ti"].xcom_push(key="test_data_path", value=test_data_path)
    context["ti"].xcom_push(key="accuracy",       value=float(accuracy))
    logging.info("Model trained, evaluated, and fully logged to MLflow.")


# =============================================================================
# TASK 7 – MODEL EVALUATION
# =============================================================================
def task_model_evaluation(**context):
    """Read saved metrics from training task, log to console, push accuracy via XCom."""
    logging.info("=== TASK 7: Model Evaluation ===")

    import pickle
    # Accuracy is already pushed by training task via XCom
    accuracy = context["ti"].xcom_pull(task_ids="model_training", key="accuracy")

    # Also load full results from pickle for detailed logging
    test_data_path = context["ti"].xcom_pull(task_ids="model_training", key="test_data_path")
    with open(test_data_path, "rb") as f:
        data = pickle.load(f)

    accuracy  = data["accuracy"]
    precision = data["precision"]
    recall    = data["recall"]
    f1        = data["f1"]

    logging.info(f"Accuracy : {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall   : {recall:.4f}")
    logging.info(f"F1-Score : {f1:.4f}")
    print(f"[EVALUATION] Accuracy={accuracy:.4f}  Precision={precision:.4f}  "
          f"Recall={recall:.4f}  F1={f1:.4f}")

    # Push accuracy via XCom for branching
    context["ti"].xcom_push(key="accuracy", value=float(accuracy))
    logging.info(f"Accuracy pushed via XCom: {accuracy:.4f}")


# =============================================================================
# TASK 8 – BRANCHING LOGIC
# =============================================================================
def task_branch_on_accuracy(**context):
    """
    BranchPythonOperator:
      accuracy >= threshold → register_model
      accuracy <  threshold → reject_model
    """
    accuracy  = context["ti"].xcom_pull(task_ids="model_evaluation", key="accuracy")
    threshold = PIPELINE_CONFIG["accuracy_threshold"]

    logging.info(f"[BRANCH] Accuracy={accuracy:.4f}  Threshold={threshold}")

    if accuracy >= threshold:
        logging.info("[BRANCH] → register_model")
        return "register_model"
    else:
        logging.info("[BRANCH] → reject_model")
        return "reject_model"


# =============================================================================
# TASK 9a – MODEL REGISTRATION
# =============================================================================
def task_register_model(**context):
    """Register approved model in MLflow Model Registry."""
    logging.info("=== TASK 9a: Model Registration ===")

    run_id   = context["ti"].xcom_pull(task_ids="model_training", key="mlflow_run_id")
    accuracy = context["ti"].xcom_pull(task_ids="model_evaluation", key="accuracy")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri  = f"runs:/{run_id}/model"
    model_name = "TitanicSurvivalModel"

    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    logging.info(
        f"Model registered: {model_name} | Version: {result.version} | "
        f"Accuracy: {accuracy:.4f}"
    )
    print(f"[REGISTRATION] Model '{model_name}' v{result.version} registered. "
          f"Accuracy={accuracy:.4f}")


# =============================================================================
# TASK 9b – MODEL REJECTION
# =============================================================================
def task_reject_model(**context):
    """Log rejection reason when accuracy is below threshold."""
    logging.info("=== TASK 9b: Model Rejection ===")

    run_id    = context["ti"].xcom_pull(task_ids="model_training", key="mlflow_run_id")
    accuracy  = context["ti"].xcom_pull(task_ids="model_evaluation", key="accuracy")
    threshold = PIPELINE_CONFIG["accuracy_threshold"]

    reason = (
        f"Model rejected. Accuracy {accuracy:.4f} is below "
        f"threshold {threshold}."
    )
    logging.warning(reason)
    print(f"[REJECTION] {reason}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    client.set_tag(run_id, "model_status",     "REJECTED")
    client.set_tag(run_id, "rejection_reason", reason)

    logging.info("Rejection reason logged to MLflow.")


# =============================================================================
# DAG DEFINITION
# =============================================================================
with DAG(
    dag_id          = "titanic_mlops_pipeline",
    default_args    = default_args,
    description     = "End-to-end Titanic MLOps: Airflow + MLflow",
    schedule_interval = None,          # Manual trigger for experiment comparison
    catchup         = False,
    tags            = ["mlops", "titanic", "mlflow"],
) as dag:

    # ── Anchor ────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # ── Task 2: Ingestion ──────────────────────────────────────
    t_ingest = PythonOperator(
        task_id         = "data_ingestion",
        python_callable = task_data_ingestion,
    )

    # ── Task 3: Validation (retries=2 for retry demo) ─────────
    t_validate = PythonOperator(
        task_id         = "data_validation",
        python_callable = task_data_validation,
        retries         = 2,
        retry_delay     = timedelta(seconds=10),
    )

    # ── Task 4: Parallel processing ───────────────────────────
    t_missing  = PythonOperator(
        task_id         = "handle_missing",
        python_callable = task_handle_missing,
    )
    t_features = PythonOperator(
        task_id         = "feature_engineering",
        python_callable = task_feature_engineering,
    )

    # ── Task 5: Encoding ──────────────────────────────────────
    t_encode = PythonOperator(
        task_id         = "data_encoding",
        python_callable = task_data_encoding,
    )

    # ── Task 6: Training ──────────────────────────────────────
    t_train = PythonOperator(
        task_id         = "model_training",
        python_callable = task_model_training,
    )

    # ── Task 7: Evaluation ────────────────────────────────────
    t_evaluate = PythonOperator(
        task_id         = "model_evaluation",
        python_callable = task_model_evaluation,
    )

    # ── Task 8: Branching ─────────────────────────────────────
    t_branch = BranchPythonOperator(
        task_id         = "branch_on_accuracy",
        python_callable = task_branch_on_accuracy,
    )

    # ── Task 9: Register / Reject ─────────────────────────────
    t_register = PythonOperator(
        task_id         = "register_model",
        python_callable = task_register_model,
    )
    t_reject = PythonOperator(
        task_id         = "reject_model",
        python_callable = task_reject_model,
    )

    # ==========================================================
    # DEPENDENCY GRAPH  (no cycles — strict linear + parallel)
    # ==========================================================
    #
    #  start
    #    │
    #  data_ingestion
    #    │
    #  data_validation          ← retries on first attempt
    #    │
    #  ┌─┴──────────────────┐
    #  handle_missing    feature_engineering   ← PARALLEL
    #  └─────────┬──────────┘
    #           data_encoding
    #              │
    #           model_training
    #              │
    #           model_evaluation
    #              │
    #          branch_on_accuracy
    #           ┌──┴──────────┐
    #     register_model  reject_model
    #           └──────┬──────┘
    #                 end
    #
    # ==========================================================

    start          >> t_ingest
    t_ingest       >> t_validate
    t_validate     >> [t_missing, t_features]   # Fan-out → PARALLEL
    [t_missing, t_features] >> t_encode         # Fan-in
    t_encode       >> t_train
    t_train        >> t_evaluate
    t_evaluate     >> t_branch
    t_branch       >> [t_register, t_reject]
    [t_register, t_reject] >> end

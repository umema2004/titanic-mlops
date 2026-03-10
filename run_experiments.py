#!/usr/bin/env python3
"""
run_experiments.py
==================
Task 10 – Run the DAG 3 times with different hyperparameters via CLI.
Each invocation patches PIPELINE_CONFIG in the DAG and triggers the run.

Usage:
    python run_experiments.py
"""

import subprocess, time, sys, os

AIRFLOW_HOME = os.path.expanduser("~/airflow")
DAG_PATH     = os.path.join(AIRFLOW_HOME, "dags", "mlops_airflow_mlflow_pipeline.py")

# ── 3 experiment configurations ──────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name": "Experiment_1_RF_shallow",
        "model_type":   "RandomForest",
        "n_estimators": 50,
        "max_depth":    3,
        "C":            1.0,
        "max_iter":     200,
    },
    {
        "name": "Experiment_2_RF_deep",
        "model_type":   "RandomForest",
        "n_estimators": 200,
        "max_depth":    10,
        "C":            1.0,
        "max_iter":     200,
    },
    {
        "name": "Experiment_3_LR",
        "model_type":   "LogisticRegression",
        "n_estimators": 100,
        "max_depth":    5,
        "C":            0.5,
        "max_iter":     500,
    },
]


def patch_dag(config: dict):
    """Overwrite PIPELINE_CONFIG block in the DAG file."""
    with open(DAG_PATH, "r") as f:
        content = f.read()

    new_block = f"""PIPELINE_CONFIG = {{
    "model_type":    "{config['model_type']}",
    "n_estimators":  {config['n_estimators']},
    "max_depth":     {config['max_depth']},
    "C":             {config['C']},
    "max_iter":      {config['max_iter']},
    "test_size":     0.2,
    "random_state":  42,
    "accuracy_threshold": 0.80,
}}"""

    import re
    content = re.sub(
        r"PIPELINE_CONFIG = \{.*?\}",
        new_block,
        content,
        flags=re.DOTALL,
    )

    with open(DAG_PATH, "w") as f:
        f.write(content)
    print(f"  ✔ DAG patched for {config['name']}")


def trigger_dag():
    env = {**os.environ, "AIRFLOW_HOME": AIRFLOW_HOME}
    result = subprocess.run(
        ["airflow", "dags", "trigger", "titanic_mlops_pipeline"],
        capture_output=True, text=True, env=env
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print("STDERR:", result.stderr.strip())


def main():
    # Remove retry flag so validation retry fires for each run
    retry_flag = os.path.expanduser("~/airflow/tmp_pipeline/.validation_retried")

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*55}")
        print(f"  Launching {exp['name']}  ({i}/{len(EXPERIMENTS)})")
        print(f"{'='*55}")

        # Remove retry flag so intentional failure happens each run
        if os.path.exists(retry_flag):
            os.remove(retry_flag)

        patch_dag(exp)
        time.sleep(2)   # allow scheduler to pick up DAG change
        trigger_dag()

        print(f"  Waiting 90 s before next run...")
        time.sleep(90)  # wait for run to finish

    print("\n✅  All 3 experiments triggered.")
    print("   Open MLflow UI → http://localhost:5000 to compare runs.")


if __name__ == "__main__":
    main()

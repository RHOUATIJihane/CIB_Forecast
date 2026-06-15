"""DAG Airflow — pipeline CIB (HDFS + Hive + Spark + sklearn).

Prérequis avant le premier run :
  - HDFS : start-dfs.sh
  - Variables d'environnement (voir scripts/airflow_env.sh)

Chaîne (workflow data engineer) :
  init_hive → ingest_bronze → transform_silver → account_policy → features_ml
            → train_models → inference_batch
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.environ.get("CIB_PROJECT_ROOT", "/home/jihane/cib_project/cib_forecast")
PYTHON_BIN = os.environ.get("CIB_PYTHON_BIN", "/home/jihane/cib_project/.venv/bin/python")
SPARK_SUBMIT = os.environ.get("CIB_SPARK_SUBMIT", f"{os.environ.get('HOME', '')}/bigdata/spark/bin/spark-submit")
SPARK_OPTS = os.environ.get("CIB_SPARK_OPTS", "--master local[*]")
SPARK_PYTHON_CONF = os.environ.get(
    "CIB_SPARK_PYTHON_CONF",
    f"--conf spark.pyspark.python={PYTHON_BIN} --conf spark.pyspark.driver.python={PYTHON_BIN}",
)
HIVE_BIN = os.environ.get("CIB_HIVE_BIN", f"{os.environ.get('HOME', '')}/bigdata/hive/bin/hive")

# Chemins Hadoop (WSL pseudo-distribué) — surchargeables via env
HADOOP_HOME = os.environ.get("HADOOP_HOME", f"{os.environ.get('HOME', '')}/bigdata/hadoop")
HIVE_HOME = os.environ.get("HIVE_HOME", f"{os.environ.get('HOME', '')}/bigdata/hive")
SPARK_HOME = os.environ.get("SPARK_HOME", f"{os.environ.get('HOME', '')}/bigdata/spark")

default_args = {
    "owner": "cib",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _prefix_env() -> str:
    """Exports shell pour HDFS, Hive, Spark et config CIB."""
    return (
        f"export CIB_ENV=cib PYTHONPATH={PROJECT_ROOT} "
        f"HADOOP_HOME={HADOOP_HOME} HIVE_HOME={HIVE_HOME} SPARK_HOME={SPARK_HOME} "
        f"PATH={HADOOP_HOME}/bin:{HIVE_HOME}/bin:{SPARK_HOME}/bin:$PATH && "
        f"cd {PROJECT_ROOT} && "
    )


def _bash(script: str) -> str:
    return _prefix_env() + script


def _python(script: str) -> str:
    return _bash(f"{PYTHON_BIN} {script}")


def _spark(script: str) -> str:
    return _bash(f"{SPARK_SUBMIT} {SPARK_OPTS} {SPARK_PYTHON_CONF} {script}")


def _hive_init() -> str:
    return _bash(f'bash "{PROJECT_ROOT}/scripts/init_hive.sh"')


with DAG(
    dag_id="cib_forecast_pipeline",
    description="CIB : DDL → ingestion bronze → ETL silver → policy → features → train → inference",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["cib", "forecast", "hive"],
) as dag:

    init_hive = BashOperator(
        task_id="init_hive",
        bash_command=_hive_init(),
    )

    ingest = BashOperator(
        task_id="ingest_bronze",
        bash_command=_spark("scripts/ingest_bronze.py"),
    )

    scrape_macro = BashOperator(
        task_id="scrape_macro",
        bash_command=_python("scripts/scrape_macro_indicators.py"),
    )

    load_reference = BashOperator(
        task_id="load_reference_tables",
        bash_command=_spark("scripts/load_reference_tables.py"),
    )

    assign_macro = BashOperator(
        task_id="assign_macro",
        bash_command=_spark("assign_macro_job.py"),
    )

    transform = BashOperator(
        task_id="transform_silver",
        bash_command=_spark("transformation_job.py"),
    )

    policy = BashOperator(
        task_id="account_policy",
        bash_command=_spark("policy_job.py"),
    )

    features = BashOperator(
        task_id="features_ml",
        bash_command=_spark("features.py"),
    )

    train = BashOperator(
        task_id="train_models",
        bash_command=_python("train.py"),
    )

    infer = BashOperator(
        task_id="inference_batch",
        bash_command=_python("inference.py"),
    )

    init_hive >> load_reference >> scrape_macro >> ingest >> transform >> assign_macro >> policy >> features >> train >> infer

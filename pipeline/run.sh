#!/usr/bin/env bash
# Orchestration CIB Forecast — workflow data engineer (HDFS + Hive + Spark + Airflow)
#
# Workflow batch :
#   init-hive   →  DDL catalogue Hive (schémas + tables externes)
#   ingest      →  landing zone BRONZE (données brutes sur HDFS)
#   transform   →  zone SILVER (agrégation, qualité)
#   policy      →  règles d'activation des features externes
#   features    →  zone ML / feature store
#   train       →  entraînement modèles
#   inference   →  scoring
#
# Airflow : même enchaînement (voir dags/dag_cib_forecast.py)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CIB_ENV="${CIB_ENV:-cib}"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
export PYSPARK_PYTHON="$PY"
export PYSPARK_DRIVER_PYTHON="$PY"
SPARK_SUBMIT="${SPARK_SUBMIT_BIN:-$HOME/bigdata/spark/bin/spark-submit}"
SPARK_OPTS="${SPARK_SUBMIT_OPTS:---master local[*]}"
SPARK_PYTHON_CONF="--conf spark.pyspark.python=${PY} --conf spark.pyspark.driver.python=${PY}"

spark_submit() {
  "$SPARK_SUBMIT" $SPARK_OPTS $SPARK_PYTHON_CONF "$@"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  init-hive)
    bash "$ROOT/scripts/init_hive.sh" "$@"
    ;;
  ingest|ingestion)
    spark_submit "$ROOT/scripts/ingest_bronze.py" "$@"
    ;;
  load-reference)
    spark_submit "$ROOT/scripts/load_reference_tables.py" "$@"
    ;;
  scrape-macro)
    "$PY" "$ROOT/scripts/scrape_macro_indicators.py" "$@"
    ;;
  assign-macro)
    spark_submit "$ROOT/assign_macro_job.py" "$@"
    ;;
  bootstrap)
    echo "Note: 'bootstrap' → utilisez 'ingest' (ingestion zone bronze)." >&2
    spark_submit "$ROOT/scripts/ingest_bronze.py" "$@"
    ;;
  smoke)
    "$PY" "$ROOT/scripts/local_smoke_test.py" "$@"
    ;;
  derive-policy)
    "$PY" "$ROOT/scripts/derive_policy_from_exports.py" "$@"
    ;;
  transform)
    spark_submit "$ROOT/transformation_job.py" "$@"
    ;;
  policy)
    spark_submit "$ROOT/policy_job.py" "$@"
    ;;
  features)
    spark_submit "$ROOT/features.py" "$@"
    ;;
  train)
    "$PY" "$ROOT/train.py" "$@"
    ;;
  inference)
    "$PY" "$ROOT/inference.py" "$@"
    ;;
  all)
    bash "$ROOT/scripts/init_hive.sh"
    spark_submit "$ROOT/scripts/load_reference_tables.py"
    "$PY" "$ROOT/scripts/scrape_macro_indicators.py"
    spark_submit "$ROOT/scripts/ingest_bronze.py" "$@"
    spark_submit "$ROOT/transformation_job.py"
    spark_submit "$ROOT/assign_macro_job.py"
    spark_submit "$ROOT/policy_job.py"
    spark_submit "$ROOT/features.py"
    "$PY" "$ROOT/train.py"
    "$PY" "$ROOT/inference.py"
    ;;
  *)
    cat <<EOF
Usage: $0 {init-hive|load-reference|scrape-macro|ingest|transform|assign-macro|policy|features|train|inference|all|smoke}

Workflow data engineer :
  init-hive       Catalogue Hive (CREATE TABLE)
  load-reference  Mapping secteur → macro (CSV → bronze)
  scrape-macro    Scrape Yahoo → macro_indicators_weekly (bronze)
  ingest          Ingestion → cib_bronze.* (landing raw)
  transform       ETL → cib_silver.*
  assign-macro    sector → macro_primary par compte
  policy          Règles métier → account_policy
  features     Feature engineering → cib_ml.*
  train        Entraînement sklearn
  inference    Scoring

Airflow : ./scripts/start_airflow.sh webserver + scheduler

Exemple :
  start-dfs.sh
  $0 init-hive && $0 ingest && $0 transform
EOF
    exit 1
    ;;
esac

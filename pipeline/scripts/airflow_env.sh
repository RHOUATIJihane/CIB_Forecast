#!/usr/bin/env bash
# Source ce fichier avant de lancer Airflow :  source scripts/airflow_env.sh
# Configure AIRFLOW_HOME et pointe les DAGs vers ce projet.

export AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow_cib}"
export CIB_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CIB_ENV=cib
export PYTHONPATH="${CIB_PROJECT_ROOT}:${PYTHONPATH:-}"
export CIB_PYTHON_BIN="${CIB_PYTHON_BIN:-/home/jihane/cib_project/.venv/bin/python}"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-$CIB_PYTHON_BIN}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-$CIB_PYTHON_BIN}"
export CIB_SPARK_PYTHON_CONF="--conf spark.pyspark.python=${PYSPARK_PYTHON} --conf spark.pyspark.driver.python=${PYSPARK_DRIVER_PYTHON}"

export HADOOP_HOME="${HADOOP_HOME:-$HOME/bigdata/hadoop}"
export HIVE_HOME="${HIVE_HOME:-$HOME/bigdata/hive}"
export SPARK_HOME="${SPARK_HOME:-$HOME/bigdata/spark}"
export PATH="$HADOOP_HOME/bin:$HIVE_HOME/bin:$SPARK_HOME/bin:$PATH"

export CIB_SPARK_SUBMIT="${CIB_SPARK_SUBMIT:-$SPARK_HOME/bin/spark-submit}"
export CIB_SPARK_OPTS="${CIB_SPARK_OPTS:---master local[*]}"

mkdir -p "$AIRFLOW_HOME/dags"
# Lien symbolique : un seul DAG à maintenir dans le repo
if [[ ! -e "$AIRFLOW_HOME/dags/dag_cib_forecast.py" ]]; then
  ln -sf "$CIB_PROJECT_ROOT/dags/dag_cib_forecast.py" "$AIRFLOW_HOME/dags/dag_cib_forecast.py"
fi

# Pas de DAGs d'exemple (évite tutorial_objectstorage + 60+ DAGs dans l'UI)
if [[ -f "$AIRFLOW_HOME/airflow.cfg" ]] && grep -q '^load_examples = True' "$AIRFLOW_HOME/airflow.cfg" 2>/dev/null; then
  sed -i 's/^load_examples = True/load_examples = False/' "$AIRFLOW_HOME/airflow.cfg"
fi

echo "AIRFLOW_HOME=$AIRFLOW_HOME"
echo "DAGs -> $AIRFLOW_HOME/dags/dag_cib_forecast.py"
echo "Projet -> $CIB_PROJECT_ROOT"

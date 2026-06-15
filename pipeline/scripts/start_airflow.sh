#!/usr/bin/env bash
# Démarre Airflow (scheduler + webserver) pour le PFE CIB.
# Usage :
#   Terminal 1 : ./scripts/start_airflow.sh scheduler
#   Terminal 2 : ./scripts/start_airflow.sh webserver
#   Terminal 3 : start-dfs.sh  (HDFS doit tourner avant un DAG run)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/airflow_env.sh"

CMD="${1:-help}"

case "$CMD" in
  init)
    "$CIB_PYTHON_BIN" -m airflow db migrate
    if [[ -f "$AIRFLOW_HOME/airflow.cfg" ]]; then
      sed -i 's/^load_examples = True/load_examples = False/' "$AIRFLOW_HOME/airflow.cfg"
    fi
  ;;
  scheduler)
    exec "$CIB_PYTHON_BIN" -m airflow scheduler
  ;;
  webserver)
    exec "$CIB_PYTHON_BIN" -m airflow webserver --port 8080
  ;;
  trigger)
    "$CIB_PYTHON_BIN" -m airflow dags trigger cib_forecast_pipeline
  ;;
  test)
    # Test une tâche sans lancer tout le scheduler (debug)
    TASK="${2:-ingest_bronze}"
    "$CIB_PYTHON_BIN" -m airflow tasks test cib_forecast_pipeline "$TASK" "$(date +%Y-%m-%d)"
  ;;
  *)
    cat <<EOF
Usage: $0 {init|scheduler|webserver|trigger|test [task_id]}

Ordre recommandé :
  1. start-dfs.sh                    # HDFS
  2. $0 init                         # une fois : base Airflow
  3. $0 scheduler   (terminal 1)
  4. $0 webserver   (terminal 2)     # http://localhost:8080
  5. $0 trigger                      # lancer le pipeline
  ou : $0 test init_hive             # tester une seule tâche

Login UI par défaut : admin / admin (après airflow users create si besoin)
EOF
    exit 1
  ;;
esac

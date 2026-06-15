#!/usr/bin/env bash
# Crée les bases et tables externes Hive (DDL dans hive/*.hql).
# Prérequis : HDFS démarré, Hive installé (HIVE_HOME ou `hive` dans PATH).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HIVE_HOME="${HIVE_HOME:-$HOME/bigdata/hive}"
HIVE_BIN="${HIVE_HOME}/bin/hive"
SCHEMATOOL="${HIVE_HOME}/bin/schematool"
if [[ ! -x "$HIVE_BIN" ]]; then
  HIVE_BIN="$(command -v hive || true)"
fi
if [[ -z "$HIVE_BIN" || ! -x "$HIVE_BIN" ]]; then
  echo "Erreur : binaire hive introuvable. Définissez HIVE_HOME ou installez Hive." >&2
  exit 1
fi

# Metastore Derby (jdbc:...databaseName=metastore_db) est relatif au cwd → toujours $ROOT
if [[ -x "$SCHEMATOOL" ]]; then
  if ! "$SCHEMATOOL" -dbType derby -info >/dev/null 2>&1; then
    echo "Initialisation du schéma metastore Derby (schematool -initSchema)..."
    rm -rf "$ROOT/metastore_db" "$ROOT/derby.log"
    "$SCHEMATOOL" -dbType derby -initSchema
  fi
else
  echo "Avertissement : schematool introuvable, metastore non vérifié." >&2
fi

echo "Using hive: $HIVE_BIN (cwd=$ROOT)"
for hql in "$ROOT/hive"/bronze_*.hql "$ROOT/hive"/silver_*.hql "$ROOT/hive"/account_policy.hql "$ROOT/hive"/features_cib_forecast.hql; do
  echo "==> $hql"
  "$HIVE_BIN" -f "$hql"
done

echo "Hive DDL applied. Vérification :"
"$HIVE_BIN" -e "SHOW DATABASES LIKE 'cib_*'; SHOW TABLES IN cib_bronze;"

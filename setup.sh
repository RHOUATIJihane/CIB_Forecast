#!/usr/bin/env bash
# Installation rapide de l'environnement CIB Forecast
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/pipeline/.venv"

echo "=== CIB Forecast — Setup ==="

if [ ! -d "$VENV" ]; then
  echo "→ Création du virtualenv..."
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install -q --upgrade pip

echo "→ Installation des dépendances (mode démo)..."
pip install -q -r "$ROOT/requirements-demo.txt"
pip install -q -e "$ROOT/pipeline"

echo ""
echo "✓ Environnement prêt."
echo ""
echo "  source pipeline/.venv/bin/activate"
echo "  cd pipeline && ./run.sh smoke"
echo "  pytest -q"
echo ""

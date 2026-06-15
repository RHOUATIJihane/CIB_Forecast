# CIB Forecast — Prévision cash-flow entreprises (h+1)

**RHOUATI Jihane** — ENSA Al Hoceima / Attijariwafa Bank — PFE 2025/2026

Pipeline de prévision hebdomadaire du cash-flow CIB avec politique d'activation automatique des variables exogènes (macro, calendrier).

---

## Structure du projet

```
CIB_Forecast_Soutenance/
├── README.md                      ← vous êtes ici
├── CIB_Forecast.code-workspace    ← ouvrir ce fichier dans VS Code
├── setup.sh                       ← installation environnement
├── requirements.txt               ← dépendances complètes
├── requirements-demo.txt          ← dépendances démo rapide (smoke test)
│
├── pipeline/                      ← chaîne industrielle (cœur du projet)
│   ├── run.sh                     ← point d'entrée principal
│   ├── config.ini                 ← configuration rec/prd
│   ├── src/                       ← code métier (transformation, ML, policy)
│   ├── hive/                      ← DDL tables Hive (bronze/silver/ml)
│   ├── dags/                      ← orchestration Airflow
│   ├── scripts/                   ← bootstrap, ingest, smoke test
│   ├── tests/                     ← tests pytest
│   └── docs/                      ← architecture (diagrammes)
│
├── experimentation/               ← phase EXP1–EXP7 (données synthétiques)
│   ├── cib_experiments_v6.py
│   └── outputs/                   ← résultats clés (policy, benchmark)
│
├── rapport/
│   └── main.pdf                   ← mémoire PFE
│
└── docs/
    └── GUIDE_SOUTENANCE.md        ← déroulé de présentation
```

---

## Installation (5 min)

```bash
# 1. Ouvrir dans VS Code
code CIB_Forecast.code-workspace

# 2. Installer l'environnement
bash setup.sh

# 3. Activer le venv
source pipeline/.venv/bin/activate

# 4. Lancer la démo
cd pipeline && ./run.sh smoke

# 5. Tests
pytest -q
```

---

## Commandes principales

| Commande | Description |
|----------|-------------|
| `./run.sh smoke` | Démo locale sans HDFS (recommandé soutenance) |
| `./run.sh transform` | Bronze → Silver (PySpark) |
| `./run.sh policy` | Règles d'activation → account_policy |
| `./run.sh features` | Feature store ML |
| `./run.sh train` | Entraînement modèles par compte |
| `./run.sh inference` | Prédictions h+1 |
| `pytest -q` | Tests unitaires |

---

## Stack technique

HDFS · Hive 2.3 · PySpark 3.3 · scikit-learn · LightGBM · Airflow 2.10

---

## Fichiers clés à montrer au jury

| Fichier | Rôle |
|---------|------|
| `pipeline/dags/dag_cib_forecast.py` | Orchestration Airflow |
| `pipeline/src/policy_rules.py` | Politique d'activation par compte |
| `pipeline/src/ml/train_regression.py` | Entraînement ML |
| `pipeline/hive/account_policy.hql` | Table policy Hive |
| `experimentation/outputs/policy_v7_encapsulated.csv` | Résultats EXP |
| `rapport/main.pdf` | Mémoire complet |

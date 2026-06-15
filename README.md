# CIB Forecast 

Projet global de prevision de cashflow CIB organise en deux parties:
- `pipeline/`: industrialisation data + ML (HDFS, Hive, Spark, train, inference, Airflow)
- `notebooks/`: experimentation et analyse exploratoire

## Objectif

Le projet vise a predire les flux de tresorerie hebdomadaires des comptes CIB en combinant:
- des donnees transactionnelles
- des indicateurs macroeconomiques
- des regles metier (policy)
- des modeles de machine learning

## Structure du depot

```
CIB_Forecast_Soutenance/
├── pipeline/
│   ├── run.sh
│   ├── dags/
│   ├── hive/
│   ├── scripts/
│   ├── src/
│   └── README.md
└── notebooks/
    ├── Experimenatation.ipynb
    └── README.md
```

## Demarrage rapide

### 1) Pipeline (production-like)

Consulter la documentation complete:
- `pipeline/README.md`

Execution type:

```bash
cd pipeline
source .venv/bin/activate
./run.sh all
```

Execution etape par etape:

```bash
cd pipeline
./run.sh init-hive
./run.sh load-reference
./run.sh scrape-macro
./run.sh ingest
./run.sh transform
./run.sh assign-macro
./run.sh policy
./run.sh features
./run.sh train
./run.sh inference
```

### 2) Notebooks (experimentation)

Consulter la documentation complete:
- `notebooks/README.md`

Ouverture notebook:

```bash
cd notebooks
# Ouvrir Experimenatation.ipynb dans VS Code/Jupyter
```

## Orchestration Airflow

Le DAG principal est dans:
- `pipeline/dags/dag_cib_forecast.py`

Pour les details de lancement Airflow, voir:
- `pipeline/README.md`

## Notes

- Les chemins, schemas Hive et options Spark sont centralises dans `pipeline/config.ini`.
- Les scripts shell et jobs Python/Spark sont dans `pipeline/`.
- La partie exploration et benchmark est centralisee dans `notebooks/`.

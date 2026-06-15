# Guide de présentation — CIB Forecast

## Message clé (30 s)

> Prévision hebdomadaire du cash-flow par compte entreprise CIB, avec décision
> automatique d'activation des variables externes, industrialisée sur
> HDFS / Hive / PySpark / Airflow.

---

## Déroulé suggéré

### 1. Contexte (2 min) → `rapport/main.pdf` ch. I
- Attijariwafa Bank, pôle Data, périmètre CIB
- Problématique h+1 + activation sélective des exogènes

### 2. Expérimentation EXP1–EXP7 (4 min)
- `experimentation/cib_experiments_v6.py`
- `experimentation/outputs/policy_v7_encapsulated.csv`
- 8 profils A–F, seuil lift 3 %

### 3. Pipeline industriel (5 min)
- `pipeline/dags/dag_cib_forecast.py` — orchestration
- `pipeline/src/policy_rules.py` — règles métier
- `pipeline/hive/*.hql` — catalogue Hive
- `pipeline/docs/architecture.md` — schéma médaillon

### 4. ML (3 min)
- `pipeline/src/ml/train_regression.py`
- Walk-forward, un modèle par compte

### 5. Démo live (3 min)
```bash
source pipeline/.venv/bin/activate
cd pipeline && ./run.sh smoke
pytest -q
```

### 6. Résultats (2 min) → `rapport/main.pdf` ch. IV

---

## Onglets VS Code recommandés

1. `pipeline/dags/dag_cib_forecast.py`
2. `pipeline/src/policy_rules.py`
3. `pipeline/src/ml/train_regression.py`
4. `experimentation/outputs/policy_v7_encapsulated.csv`
5. `rapport/main.pdf`

---

## Questions jury

| Question | Réponse courte |
|----------|----------------|
| Pourquoi données synthétiques ? | Calibrer EXP avant accès données réelles (confidentialité) |
| Pourquoi pas toujours activer exogènes ? | EXP montre dégradation sur profils B/C/E/F |
| Scalabilité ? | PySpark pour ETL ; entraînement batch par compte |

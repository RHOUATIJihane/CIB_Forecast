# Expérimentation EXP1–EXP7

Benchmark sur données synthétiques — phase R&D avant industrialisation.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `cib_experiments_v6.py` | Script complet du protocole EXP |
| `outputs/policy_v7_encapsulated.csv` | Politique finale par catégorie |
| `outputs/category_benchmark_v6.csv` | RMSE baseline vs exogènes |
| `outputs/lift_summary_v6.csv` | Synthèse des lifts |

## Exécution

```bash
source ../pipeline/.venv/bin/activate
python cib_experiments_v6.py
```

## Profils A–F

Seuls **A_regular_stable** et **D_irregular** activent les variables exogènes (seuil lift 3 %).

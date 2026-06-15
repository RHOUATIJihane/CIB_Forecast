# Plan de scraping des indicateurs macro (CIB Forecast)

Ce document décrit **comment** le scraping sera implémenté **après** la table
`sector_macro_mapping` et **avant** de remplacer les macro simulées en prod.

## Objectif

Alimenter `cib_bronze.macro_indicators_weekly` avec de **vraies séries hebdomadaires**
z-scorées pour les 4 variables utilisées par le pipeline :

| Colonne | Usage typique (mapping) |
|---------|-------------------------|
| `oil_price_z` | transport, construction |
| `commodity_index_z` | agriculture |
| `masi_index_z` | investment, mixed, défaut |
| `realestate_index_z` | construction, immobilier |

Le pipeline lit ensuite **toutes** les colonnes dans `features.py`, mais
`train.py` / `inference.py` n'utilisent que celles listées dans
`account_policy.externals_for_model` (macro assignée + calendrier).

## Architecture du job `scrape_macro_indicators.py`

```
Sources web/API
    ↓  (1) fetch daily ou weekly
DataFrame brut par indicateur
    ↓  (2) resample → semaine (lundi)
Série hebdomadaire alignée
    ↓  (3) z-score rolling ou global
macro_indicators_weekly
    ↓  (4) write ORC → cib_bronze
```

## Étape 1 — Sources proposées

| Variable | Source candidate | Fréquence source | Notes |
|----------|------------------|------------------|-------|
| `masi_index_z` | Bourse de Casablanca (MASI) | Quotidienne | Volatile → scrape hebdo minimum |
| `oil_price_z` | EIA / Yahoo Finance (Brent) | Quotidienne | API ou CSV public |
| `commodity_index_z` | Index FAO / World Bank commodities | Hebdo / mensuelle | Interpolation si mensuel |
| `realestate_index_z` | Bank Al-Maghrib / stats publiques | Trimestrielle / mensuelle | Forward-fill entre publications |

**PFE :** commencer par **1 source** (ex. MASI) pour valider le flux, garder
les 3 autres en simulé jusqu'à validation métier.

## Étape 2 — Agrégation hebdomadaire

Règle identique au notebook (`generate_sector_indicators`) :

- Index temporel = **lundi** de chaque semaine (`W-MON`)
- Valeur hebdo = **dernière observation** de la semaine ou **moyenne**
- Alignement sur `calendar_weekly_flags.week`

```python
raw["week"] = raw["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
weekly = raw.groupby("week")["value"].last().reset_index()
```

## Étape 3 — Z-score

Même logique que `synthetic.py` :

```python
z = (series - series.mean()) / series.std()
```

Option prod : fenêtre glissante 104 semaines pour éviter le look-ahead global.

## Étape 4 — Écriture bronze

Format cible (identique à `hive/bronze_macro.hql`) :

```
week | oil_price_z | commodity_index_z | masi_index_z | realestate_index_z
```

- Mode : `overwrite` hebdomadaire (snapshot complet) ou append + dédup
- Décalage +1 semaine déjà géré dans `features_pipeline.join_calendar_and_macro`

## Étape 5 — Intégration DAG Airflow

Ordre proposé :

```
init_hive
  → load_reference_tables      (sector_macro_mapping)
  → scrape_macro_indicators    (NOUVEAU — hebdo)
  → ingest_bronze              (transactions ; macro peut venir du scrape)
  → transform_silver
  → assign_macro
  → account_policy
  → features_ml
  → train_models
  → inference_batch
```

**Note :** tant que le scraping n'est pas prêt, `ingest_bronze.py` continue
de charger les macro **simulées** — le mapping secteur → macro fonctionne déjà.

## Étape 6 — Liaison avec le mapping

Flux par compte :

1. `sector` synthétique (ou `code_activite` en prod)
2. `sector_macro_mapping` → `macro_primary`
3. Policy v3.0 → `use_externals_regression = true/false`
4. Si true → lire les colonnes de `macro_indicators_weekly` correspondantes
5. Entraîner / scorer avec **uniquement** ces colonnes (+ calendrier)

## Risques et garde-fous

| Risque | Mitigation |
|--------|------------|
| Source indisponible | Fallback sur dernière valeur connue ou macro simulée |
| Fréquence mensuelle (immo) | Forward-fill + log `source_lag_weeks` |
| Fuite temporelle | z-score calculé sur passé uniquement ; macro décalée +1 semaine |
| API rate limit | Cache local CSV dans `cib_forecast_data/bronze/raw/` |

## Statut

Implémenté dans :
- `src/macro_scraper/` — fetchers Yahoo + cache, agrégation hebdo, z-score
- `scripts/scrape_macro_indicators.py` — job CLI / Airflow
- `config.ini` — `macro_source=scraped`, symboles Yahoo configurables

### Sources utilisées (PFE)

| Colonne | Symbole Yahoo | Fallback |
|---------|---------------|----------|
| `oil_price_z` | `BZ=F` (Brent) | cache local |
| `commodity_index_z` | `DBC` (ETF commodities) | cache local |
| `masi_index_z` | `MASI.CS` | `^GSPC` (proxy) |
| `realestate_index_z` | `IYR` (ETF immobilier US) | cache local |

Cache : `{local_base}/bronze/raw/macro_cache/`

### Commandes

```bash
# Sans HDFS (dev)
PYTHONPATH=. python scripts/scrape_macro_indicators.py --local-csv --local-only --n-weeks 104

# Pipeline complet (HDFS + Hive)
./run.sh scrape-macro
```

## Prochaine implémentation (quand validé)

1. Créer `scripts/scrape_macro_indicators.py` avec fetchers pluggables
2. Ajouter tâche Airflow `scrape_macro` avant `ingest_bronze`
3. Flag config `macro_source = synthetic|scraped`
4. Tests unitaires sur agrégation hebdo + z-score

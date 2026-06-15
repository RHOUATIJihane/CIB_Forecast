# CIB Cashflow Forecasting — Experiment Suite v6

> Suite d'expériences de prévision de cashflow bancaire sur données synthétiques, avec benchmarking de modèles statistiques et ML, sélection automatique d'indicateurs externes, et segmentation des comptes par profil de régularité.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du code](#2-architecture-du-code)
3. [Génération des données synthétiques (Bloc A)](#3-génération-des-données-synthétiques-bloc-a)
4. [Pipeline de filtrage (Bloc B — Steps 1–8)](#4-pipeline-de-filtrage-bloc-b--steps-18)
5. [Modèles de prévision](#5-modèles-de-prévision)
6. [Sélection des indicateurs externes (§2.4.5)](#6-sélection-des-indicateurs-externes-2-4-5)
7. [Catégorisation des comptes](#7-catégorisation-des-comptes)
8. [Les 7 expériences](#8-les-7-expériences)
9. [Fichiers de sortie](#9-fichiers-de-sortie)
10. [Paramètres clés et configuration](#10-paramètres-clés-et-configuration)
11. [Dépendances](#11-dépendances)
12. [Reproduire les résultats](#12-reproduire-les-résultats)

---

## 1. Vue d'ensemble

Ce notebook implémente un **banc d'essai complet** pour évaluer l'apport d'indicateurs externes (prix du pétrole, indices boursiers, calendrier islamique, etc.) dans la prévision hebdomadaire des flux de trésorerie de comptes bancaires d'entreprises (CIB — Corporate & Institutional Banking).

### Objectifs principaux

| Objectif | Description |
|---|---|
| **Benchmarking modèles** | Comparer 5 modèles de régression (SARIMA, Holt-Winters, Ridge, LGBM, RF) sur données synthétiques contrôlées |
| **Valeur des externes** | Mesurer le "lift" RMSE/MAE/R² apporté par l'ajout d'indicateurs sectoriels et calendaires |
| **Robustesse** | Tester la dégradation du signal (bruit, séries courtes, ruptures structurelles) |
| **Sélection intelligente** | Filtrer automatiquement les externes pertinents par corrélation et importance de features |
| **Segmentation** | Classifier les comptes en profils (regular, sparse, volatile, irregular) pour adapter la stratégie |

### Résultats produits

- `experiment_results_v6.csv` — résultats bruts de toutes les évaluations de régression
- `lift_summary_v6.csv` — tableau pivot baseline vs. with\_externals
- `category_benchmark_v6.csv` — champion model par catégorie de compte
- `policy_draft_v6.csv` — recommandations d'activation des externes par catégorie
- `selection_audit_v6.csv` — audit de sélection des externes par secteur

---

## 2. Architecture du code

```
notebook.py
├── BLOC A — Génération des données synthétiques
│   ├── generate_calendar_flags()         # Flags calendaires hebdomadaires
│   ├── generate_sector_indicators()      # Indicateurs macro par secteur
│   └── generate_account_series_v2()     # Séries de transactions par compte
│
├── BLOC B — Pipeline de filtrage (Steps 1–8)
│   ├── step1_weekly_aggregation()        # Agrégation hebdomadaire (Polars)
│   ├── step2_global_stats()              # Stats globales par compte
│   ├── step3_eligibility_filter()        # Filtre d'éligibilité minimum
│   ├── step4_ts_metrics()                # Métriques time-series (ADF, ACF, PACF)
│   ├── step5_validity_filter()           # Filtre de validité statistique
│   ├── step6_scores()                    # Score composite de prédictibilité
│   ├── step7_quality_filter()            # Tagger les profils (regular/sparse/volatile)
│   └── step8_features()                  # Construction des features ML
│
├── Modèles & évaluation
│   ├── train_eval_sarima()
│   ├── train_eval_holt_winters()
│   ├── train_eval_ml()                   # Ridge, LGBM, RF
│   └── train_eval_any()                  # Dispatcher unifié
│
├── Sélection des externes (§2.4.5)
│   ├── select_externals_by_correlation()
│   ├── select_externals_by_importance()
│   └── select_externals_combined()
│
├── 7 Expériences
│   ├── experiment_1_signal_degradation()
│   ├── experiment_2_structural_breaks()
│   ├── experiment_3_regularity_segmentation()
│   ├── experiment_4_external_quality()
│   ├── experiment_5_wrong_external()
│   ├── experiment_6_irregular_accounts()
│   └── experiment_7_step7_ablation()
│
└── Exports & agrégation
    ├── summarise_lift()
    ├── build_category_benchmark()
    ├── build_selection_audit()
    ├── build_policy_draft()
    └── save_v6_outputs()
```

---

## 3. Génération des données synthétiques (Bloc A)

### 3.1 Secteurs et betas

Six secteurs économiques sont simulés, chacun avec des sensibilités distinctes (`SECTOR_BETAS`) aux 9 indicateurs externes :

| Secteur | Drivers principaux |
|---|---|
| `transport` | Prix du pétrole (β=8000) |
| `agriculture` | Indice des matières premières (β=7000) |
| `investment` | Indice MASI (β=6000) |
| `construction` | Pétrole + Immobilier (β=300 + β=7500) |
| `retail` | Ramadan, Aïd, fin de mois, paie (β=4000–5000) |
| `mixed` | Mix pétrole + MASI (β=1500 chacun) |

### 3.2 Profils de comptes (`ACCOUNT_TYPE_MIX`)

| Profil | Part (%) | φ AR(1) | Bruit | Prob. débit |
|---|---|---|---|---|
| `normal` | 40% | 0.70 | ×1.0 | 15% |
| `seasonal` | 25% | 0.75 | ×1.2 | 12% |
| `trend_up` | 15% | 0.72 | ×0.8 | 18% |
| `volatile` | 10% | 0.75 | ×2.0 | 30% |
| `sparse` | 7% | 0.50 | ×1.5 | 20% |
| `flat` | 3% | 0.30 | ×0.05 | 5% |

### 3.3 Indicateurs externes générés

Quatre séries macro simulées par processus AR(1) avec composante cyclique :

- **`oil_price_z`** — prix du pétrole centré-réduit (μ=85, σ=6, φ=0.92)
- **`commodity_index_z`** — indice matières premières (μ=100, σ=5, φ=0.88)
- **`masi_index_z`** — indice boursier MASI avec trend haussier (μ=12000, φ=0.95)
- **`realestate_index_z`** — indice immobilier (μ=1000, φ=0.97, très persistant)

### 3.4 Flags calendaires

Calculés à fréquence hebdomadaire depuis les dates journalières :

| Flag | Logique |
|---|---|
| `is_ramadan` | Fenêtres 2022–2024 codées en dur |
| `is_eid_alfitr` | J+7 après fin Ramadan |
| `is_month_end_week` | Jour ≥ 26 du mois |
| `is_quarter_end` | Mois {3,6,9,12} et jour ≥ 25 |
| `is_payroll_week` | Jour entre 23 et 27 |
| `is_tax_deadline_week` | Mois {3,6,9,12} et jour ≥ 24 |

### 3.5 Structure des transactions

Pour chaque semaine simulée, le montant net hebdomadaire est décomposé en :
- **N crédits** (Poisson λ=4+1) répartis par Dirichlet
- **N débits** (Binomiale) représentant 5–20% du flux net
- Types d'opérations : VIREMENT (50%), CHEQUE (20%), PRELEVEMENT (20%), ESPECES (10%)

---

## 4. Pipeline de filtrage (Bloc B — Steps 1–8)

### Step 1 — Agrégation hebdomadaire
Utilise **Polars** pour la performance. Regroupe les transactions par compte et semaine, calcule la somme des montants et le nombre de transactions.

### Step 2 — Statistiques globales
Calcule pour chaque compte :
- `n_obs` : nombre de semaines observées
- `total_cashflow`, `mean_cashflow`, `std_cashflow`
- `completeness_ratio` : part de semaines non-nulles
- `cv_cashflow` : coefficient de variation

### Step 3 — Filtre d'éligibilité
Conditions minimales :
- Au moins **24 semaines** d'historique (`n_obs ≥ 24`)
- Variance non nulle (`std_cashflow ≥ 1e-8`)

### Step 4 — Métriques time-series
Pour chaque compte éligible, calcul de :
- **ADF test** (stationnarité) — statistique et p-value
- **ACF** aux lags 1 et 2
- **PACF** au lag 1
- **Trend strength** (R² de régression linéaire)
- **Seasonality strength** (corrélation avec lag 52 semaines)

### Step 5 — Filtre de validité
Conserve uniquement les comptes avec métriques ADF et ACF finies et non-NaN.

### Step 6 — Score composite
Construction d'un score multi-critères pondéré :

```
composite_score =
    predictability_score  × 0.35
  + stationarity_component × 0.20
  + completeness_component × 0.20
  + cashflow_component     × 0.15
  + transaction_component  × 0.10
  - cv_penalty
```

Le `predictability_score` lui-même combine : ACF (×0.5) + trend (×0.25) + saisonnalité (×0.25).

### Step 7 — Profilage (sans exclusion)
> **Modification AXE 3a** : les comptes ne sont plus exclus, mais *taggés* avec un profil.

| Profil | Critères |
|---|---|
| `regular` | ACF1>0.35, ADF p<0.10, completeness>0.60, transactions>100, CV∈[0.30,1.50], predictability>35, score>40 |
| `sparse` | completeness ≤ 0.40 |
| `volatile` | CV > 2.00 |
| `irregular` | Tous les autres |

### Step 8 — Construction des features
> **Modification AXE 3b** : features supplémentaires pour comptes irréguliers.

Features construites par fenêtre glissante :
- Lags AR : lag1 signé, positif, négatif
- Rolling stats : std 4w, max/min 8–12w, EWM (spans 4/12/52)
- Fourier : sin/cos hebdomadaire (harmoniques 1 et 2)
- Indicateurs irréguliers : `is_active_week`, `active_weeks_8w`, `local_completeness_8w`, `max_change_4w`
- Join avec flags calendaires et indicateurs macro (décalés d'1 semaine)

---

## 5. Modèles de prévision

### 5.1 Modèles statistiques

| Modèle | Implémentation | Saisonnalité |
|---|---|---|
| **SARIMA(1,0,1)** | `statsmodels.SARIMAX` | S(1,0,1,52) si historique ≥ 60 sem. |
| **Holt-Winters** | `ExponentialSmoothing` (trend additif) | Correcteur LGBM sur résidus si externes disponibles |

### 5.2 Modèles ML

| Modèle | Implémentation | Hyperparamètres |
|---|---|---|
| **Ridge** | `sklearn.RidgeCV` | α ∈ {0.01, 0.1, 1, 10, 100}, CV=3 + StandardScaler |
| **LGBM** | `LGBMRegressor` | n\_estimators=100, lr=0.05, max\_depth=4 |
| **RF** | `RandomForestRegressor` | n\_estimators=100, max\_depth=5 |

### 5.3 Évaluation — Walk-forward validation

Tous les modèles utilisent une validation **walk-forward** (expanding window) :
- `n_splits = 2` pour SARIMA/HW, `3` pour ML
- `test_size = 4` semaines par split
- Métriques moyennées sur les splits : **RMSE**, **MAE**, **MAPE**, **R²**

### 5.4 Classification (optionnelle)

Activée par `RUN_CLASSIFICATION = True` (désactivée par défaut). Prédit la direction de variation (↑ / stable / ↓) avec un seuil de 5%. Modèles : Logistic Regression CV et LGBM Classifier.

---

## 6. Sélection des indicateurs externes (§2.4.5)

Avant l'entraînement, les 9 indicateurs de l'univers sont filtrés automatiquement selon deux méthodes complémentaires :

### 6.1 Sélection par corrélation
- Test de corrélation de Pearson pour les lags 0 à 4
- Seuil : p < 0.05 et |r| > 0.05
- Au moins un lag significatif → l'indicateur est retenu

### 6.2 Sélection par importance de features
- Entraînement d'un LGBM léger (50 estimators) sur 80% de l'historique
- Feature importance normalisée
- Seuil : importance ≥ 1% (somme des deux lags)

### 6.3 Méthode de combinaison par modèle

| Modèle | Méthode |
|---|---|
| SARIMA | `correlation` uniquement |
| Holt-Winters | `correlation` uniquement |
| Ridge | `union` (corrélation OU importance) |
| LGBM | `importance` uniquement |
| RF | `importance` uniquement |

> **Note** : les flags calendaires (`is_*`) sont **toujours inclus** quelle que soit la sélection.

---

## 7. Catégorisation des comptes

Chaque compte est classé en l'une des 6 catégories suivantes à partir de ses métriques time-series :

| Catégorie | Critères principaux |
|---|---|
| `A_regular_stable` | ACF1>0.60, CV<0.50, saisonnalité<0.30, trend<0.20 |
| `B_seasonal` | Saisonnalité>0.60, ACF1≥0.30, CV<1.0, trend<0.50 |
| `C_trending` | Trend>0.20, saisonnalité<0.60, CV<1.0 |
| `D_volatile` | (ACF1<0.30 ou CV>1.0), saisonnalité<0.60, trend<0.50 |
| `E_short_history` | Historique < 52 semaines |
| `F_mixed` | Tous les autres cas |

---

## 8. Les 7 expériences

### Expérience 1 — Dégradation du signal
**Objectif** : mesurer la robustesse des modèles selon la qualité du signal d'entrée.

| Condition | n_weeks | Bruit (noise_fraction) |
|---|---|---|
| `A_clean` | 104 | 0.25 |
| `B_noisy` | 104 | 0.60 |
| `C_short` | 28 | 0.40 |

- **14 comptes** par secteur × 3 conditions × 6 secteurs = 252 comptes
- Profil de compte : `normal` uniquement

---

### Expérience 2 — Ruptures structurelles
**Objectif** : évaluer l'impact de changements de régime à la semaine 70.

| Condition | Transformation appliquée à partir de la semaine 70 |
|---|---|
| `A_no_break` | Aucune |
| `B_amplified_w70` | Amplification ×2 du signal calendaire sectoriel |
| `C_level_shift_w70` | Décalage de niveau +30% de la moyenne |

- Secteurs propres par secteur pour avoir des betas cohérents

---

### Expérience 3 — Segmentation régularité
**Objectif** : comparer les performances selon la régularité intrinsèque du compte.

- 120 comptes générés aléatoirement (profils : normal 40%, seasonal 25%, trend\_up 25%, volatile 10%)
- Chaque compte est labellisé `regular` ou `irregular` selon : `0.5×ACF1 + 0.5×(1 - min(CV,2)/2) ≥ 0.55`
- La condition (`regular` / `irregular`) devient le nom de l'expérience

---

### Expérience 4 — Qualité des indicateurs externes
**Objectif** : mesurer la dégradation du lift quand les externes sont de mauvaise qualité.

| Condition | Dégradation |
|---|---|
| `A_perfect` | Aucune |
| `B_lag1w` | Décalage temporel de 1 semaine |
| `C_lag3w` | Décalage temporel de 3 semaines |
| `D_noisy30` | Bruit gaussien ajouté (σ = 30% de l'écart-type) |
| `E_missing20` | 20% de valeurs manquantes (imputées par forward-fill) |

> Les flags calendaires (`is_*`) ne sont jamais dégradés.

---

### Expérience 5 — Validation de causalité (wrong external)
**Objectif** : vérifier que le lift ne provient pas d'une coïncidence statistique.

| Condition | Externes fournis |
|---|---|
| `correct_external` | Externes du secteur réel du compte |
| `wrong_external` | Externes d'un autre secteur |
| `random_external` | Série aléatoire gaussienne |

- 10 comptes par secteur
- Si le lift persiste avec `wrong_external` ou `random_external`, il est suspect

---

### Expérience 6 — Comptes irréguliers
**Objectif** : benchmarker les modèles sur des profils atypiques difficiles.

| Profil | Type de compte | Bruit |
|---|---|---|
| `sparse` | `sparse` | 0.70 |
| `volatile` | `volatile` | 0.60 |
| `flat` | `flat` | 0.90 |
| `irregular_normal` | `normal` | 0.55 |

- 4 comptes par cellule (profil × secteur) → 4 × 4 × 6 = 96 comptes

---

### Expérience 7 — Ablation des seuils Step 7
**Objectif** : valider la sensibilité du filtre de profilage (Step 7) à ses hyperparamètres.

| Seuil | ACF1 min | CV min | CV max |
|---|---|---|---|
| `strict` | 0.35 | 0.30 | 1.50 |
| `moderate` | 0.20 | 0.20 | 2.00 |
| `loose` | 0.10 | 0.10 | 3.00 |

- Comptes alternativement `normal` et `volatile`
- Le champ `notes` enregistre si le compte passe ou non chaque seuil

---

## 9. Fichiers de sortie

Tous les fichiers sont écrits dans `/kaggle/working/cib_experiment_outputs/` et copié à la racine `/kaggle/working/` pour faciliter l'accès.

| Fichier | Contenu |
|---|---|
| `experiment_results_v6.csv` | Résultats bruts de toutes les évaluations (une ligne par compte × condition × modèle × feature\_mode) |
| `lift_summary_v6.csv` | Tableau pivot : baseline vs. with\_externals (RMSE, MAE, R², lift %) |
| `lift_with_bins_v6.csv` | Idem avec bins ACF1 / CV / saisonnalité |
| `lift_aggregate_v6.csv` | Agrégat lift par condition × modèle × rôle |
| `category_benchmark_v6.csv` | Meilleur modèle par catégorie de compte (rôle calibration) |
| `category_benchmark_by_condition_v6.csv` | Idem ventilé par condition d'expérience |
| `selection_audit_v6.csv` | Taux de sélection de chaque externe par secteur × modèle |
| `policy_draft_v6.csv` | Recommandations : activer les externes (oui/non) + champion model par catégorie |
| `exp7_step7_ablation_parsed.csv` | Résultats Exp 7 avec parsing du champ `notes` |
| `bronze_transactions_synthetic.csv` | Jeu de données brut de 200 comptes (transactions individuelles) |
| `bronze_calendar_weekly_flags.csv` | Flags calendaires hebdomadaires |
| `bronze_macro_indicators_weekly.csv` | Indicateurs macro hebdomadaires |

---

## 10. Paramètres clés et configuration

```python
# Taille des échantillons (réduire pour tests rapides)
N_PER_SECTOR   = 14    # comptes par secteur dans les expériences 1,2,4,5,7
N_EXP3         = 120   # comptes totaux expérience 3
N_EXP6_CELL    = 4     # comptes par cellule expérience 6

# Activer/désactiver la classification
RUN_CLASSIFICATION = False  # True = plus lent

# Seuil minimum de lift pour recommander les externes
MIN_LIFT_PCT = 3.0  # dans build_policy_draft()

# Paramètre SARIMA — historique minimum pour activer la saisonnalité
SARIMA_TRAIN_MIN_WEEKS = 60

# Sélection des externes
ALPHA_CORR           = 0.05   # p-value seuil corrélation
MAX_LAG_CORR         = 4      # lags testés
IMPORTANCE_THRESHOLD = 0.01   # seuil d'importance normalisée
```

### Reproductibilité
```python
np.random.seed(42)   # génération des données d'expériences
np.random.seed(12345)  # génération du dataset bronze final
```

---

## 11. Dépendances

```
numpy
pandas
polars
scipy
statsmodels          # SARIMAX, ExponentialSmoothing, ADF, ACF
scikit-learn         # Ridge, RF, Logistic, Pipeline, métriques
lightgbm             # LGBMRegressor, LGBMClassifier
matplotlib           # (importé, disponible pour visualisation)
```

---

## 12. Reproduire les résultats

### Exécution complète (Kaggle / machine avec RAM ≥ 8 Go)

```bash
# Lancer directement le notebook
jupyter nbconvert --to script notebook.ipynb --execute
# ou en Python
python notebook.py
```

### Exécution rapide pour tests

Modifier en début de script :

```python
N_PER_SECTOR   = 2    # au lieu de 14
N_EXP3         = 20   # au lieu de 120
N_EXP6_CELL    = 1    # au lieu de 4
RUN_CLASSIFICATION = False
```

Cela réduit le temps d'exécution de ~45 min à ~5 min.

### Lecture des résultats

```python
import pandas as pd

# Vue d'ensemble du lift
lift = pd.read_csv("lift_summary_v6.csv")
print(lift.groupby("model_name")["rmse_lift_pct"].describe())

# Recommandations par catégorie
policy = pd.read_csv("policy_draft_v6.csv")
print(policy[["account_category", "activate_externals_draft", "champion_model_draft", "evidence_lift_mean_pct"]])

# Audit de sélection des externes
audit = pd.read_csv("selection_audit_v6.csv")
print(audit.pivot_table(index="external_name", columns="sector", values="pct_selected"))
```

---


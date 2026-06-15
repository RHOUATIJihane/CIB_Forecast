#  Pipeline CIB Forecast

**Chaîne de prévision de flux de trésorerie** : CSV → HDFS → Hive → PySpark → ML (sklearn/LightGBM) → Airflow

Prévisions de trésorerie client à horizon H+1 utilisant des données macroéconomiques et des indicateurs de qualité de compte.

---

##  Objectif du projet

Ce pipeline prédit le **flux de trésorerie (CIB) par compte client** pour l'horizon suivant, en s'appuyant sur :
- **Données transactionnelles** : comportement historique du client
- **Indicateurs macroéconomiques** : prix du pétrole, indices MASI, immobilier...
- **Règles métier** : activation de features selon le secteur d'activité
- **Modèles ML** : régression et classification (LightGBM, Sklearn)

---

##  Architecture Médaillon

| Couche | Description | Tables | Job |
|--------|-------------|--------|-----|
| **Bronze** | Données brutes (landing zone) | `transactions_raw`, `calendar_weekly_flags`, `macro_indicators_weekly` | `scripts/ingest_bronze.py` |
| **Silver** | Données agrégées & nettoyées | `weekly_cashflow_account`, `account_quality_metrics` | `transformation_job.py` |
| **Policy** | Règles métier & activation | `account_policy` | `policy_job.py` |
| **ML** | Features & prédictions | `features_cib_forecast`, modèles, `predictions_cib_forecast` | `features.py`, `train.py`, `inference.py` |

---

##  Pré-requis

- **Python** ≥ 3.10
- **Spark** 3.3.4 (avec PySpark)
- **HDFS** (mode local possible avec option `--master local[*]`)
- **Hive** (metastore configuré)
- **Airflow** ≥ 2.8.0 (optionnel, pour orchestration)

### Dépendances Python

```
findspark, pyspark==3.3.4, pandas, scikit-learn, lightgbm
(Voir pyproject.toml pour la liste complète)
```

---

##  Démarrage rapide

### 1. Installation initiale

```bash
# Depuis la racine du projet
cd pipeline

# Créer un environnement virtuel
python3.10 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -e .
pip install -e ".[dev]"      # Tests
pip install -e ".[airflow]"  # Airflow (optionnel)
```

### 2. Configuration

Éditer `config.ini` pour adapter :
- Chemins HDFS (`hdfs_base`)
- Emplacements Hive (`hive_warehouse`)
- Répertoire local des artefacts (`local_base`)
- Source macro (`macro_source` : `scraped`, `synthetic`, `hybrid`)

```ini
[cib]
hdfs_base   = hdfs://127.0.0.1:9000/data/cib
local_base  = /home/jihane/cib_project/cib_forecast_data
macro_source = scraped
```

### 3. Test local (sans HDFS)

```bash
# Démo avec données synthétiques
./run.sh smoke

# Tests unitaires
pytest -q
```

---

##  Exécution du Pipeline

### **Option A : Étapes individuelles**

```bash
# 1. Initialiser le catalogue Hive (CREATE TABLE)
./run.sh init-hive

# 2. Charger les tables de référence (secteur → macro)
./run.sh load-reference

# 3. Scraper les indicateurs macro (Yahoo Finance)
./run.sh scrape-macro

# 4. Ingérer les données brutes → bronze
./run.sh ingest

# 5. Transformer → silver (agrégation, qualité)
./run.sh transform

# 6. Assigner macro par secteur → account_macro_assignment
./run.sh assign-macro

# 7. Appliquer règles métier → account_policy
./run.sh policy

# 8. Générer features → features_cib_forecast
./run.sh features

# 9. Entraîner modèles ML
./run.sh train

# 10. Scoring H+1
./run.sh inference
```

### **Option B : Pipeline complet en une seule commande**

```bash
./run.sh all
```

---

##  Orchestration Airflow

### Installation & Configuration

```bash
# 1. Installer Airflow (si pas déjà fait)
pip install -e ".[airflow]"

# 2. Initialiser la base de données Airflow
export AIRFLOW_HOME=$HOME/airflow
airflow db init

# 3. Configurer les variables d'environnement
export CIB_PROJECT_ROOT=/home/jihane/cib_project/cib_forecast
export CIB_PYTHON_BIN=/path/to/.venv/bin/python
export CIB_SPARK_SUBMIT=$HOME/bigdata/spark/bin/spark-submit
export CIB_SPARK_OPTS="--master local[*]"
export HADOOP_HOME=$HOME/bigdata/hadoop
export HIVE_HOME=$HOME/bigdata/hive
export SPARK_HOME=$HOME/bigdata/spark
```

### Démarrer le DAG

**Terminal 1 : Webserver Airflow**
```bash
airflow webserver --port 8080
```

**Terminal 2 : Scheduler Airflow**
```bash
airflow scheduler
```

**Terminal 3 : Interface Web**
```bash
# Accéder à http://localhost:8080
# 1. Chercher le DAG "cib_forecast_pipeline"
# 2. Cliquer sur le toggle pour l'activer (enable)
# 3. Cliquer sur "Trigger DAG" pour lancer une exécution manuelle
```

### Monitoring du DAG

- **Graphe d'exécution** : Vue → Graph → Voir la chaîne de tâches
- **Logs** : Cliquer sur une tâche → "View Logs"
- **Historique** : Tree view pour l'historique des exécutions
- **Configuration** : Code view pour voir la définition du DAG

### Configuration du planificateur

```bash
# Modifier la fréquence d'exécution du DAG
# Éditer dags/dag_cib_forecast.py
schedule="@weekly"      # Hebdomadaire (défaut)
schedule="@daily"       # Quotidien
schedule="0 2 * * 1"    # Lundi 2h du matin (cron)
schedule=None           # Manuel (trigger manuel uniquement)
```

### Alternative : Exécuter le script bash via Airflow

Si Airflow n'est pas disponible, utiliser directement le script :
```bash
source scripts/airflow_env.sh
./run.sh all
```

---

##  Structure du projet

```
pipeline/
├── run.sh                          # Orchestrateur principal (CLI)
├── config.ini                      # Configuration (chemin HDFS, tables, etc.)
├── pyproject.toml                  # Dépendances Python
│
├── 🔧 Jobs Spark & Python
├── transformation_job.py           # Bronze → Silver (PySpark)
├── policy_job.py                   # Silver → Policy rules (PySpark)
├── assign_macro_job.py             # Attribution macro par secteur
├── features.py                     # Feature engineering (PySpark)
├── train.py                        # Entraînement modèles (Python)
├── inference.py                    # Scoring H+1 (Python)
│
├── src/                            # Code métier réutilisable
│   ├── __init__.py
│   ├── transformation.py           # 7 étapes d'agrégation bronze → silver
│   ├── policy.py                   # Règles d'activation features
│   ├── category_policy_v7.py       # Mapping catégories clients
│   ├── account_categorization.py   # Classification clients
│   ├── sector_macro_mapping.py     # Mapping secteur → indicateurs macro
│   ├── features_pipeline.py        # Pipeline features (build, rank, select)
│   ├── common/                     # Utilitaires (config, logging, spark)
│   ├── datagen/                    # Générateur données synthétiques
│   ├── macro_scraper/              # Scraper Yahoo Finance
│   └── ml/                         # Modèles ML (LightGBM, regression, runners)
│
├── scripts/                        # Scripts de bootstrap
│   ├── init_hive.sh                # Exécute DDL Hive
│   ├── load_reference_tables.py    # Charge CSV → bronze (mapping secteur)
│   ├── scrape_macro_indicators.py  # Yahoo Finance → bronze
│   ├── ingest_bronze.py            # Données brutes → HDFS (Spark job)
│   ├── derive_policy_from_exports.py
│   ├── bootstrap_bronze.py
│   ├── local_smoke_test.py         # Test local (données synthétiques)
│   └── *.sh                        # Scripts Airflow, Hive
│
├── hive/                           # DDL Hive (CREATE TABLE, CREATE EXTERNAL TABLE)
│   ├── bronze_transactions.hql
│   ├── bronze_calendar.hql
│   ├── bronze_macro.hql
│   ├── silver_weekly_cashflow.hql
│   ├── account_policy.hql
│   ├── features_cib_forecast.hql
│   └── ...
│
├── dags/                           # Orchestration Airflow
│   └── dag_cib_forecast.py         # DAG principal (même séquence que run.sh)
│
├── data/
│   └── reference/                  # Tables de référence (secteur → macro)
│       └── sector_macro_mapping.csv
│
├── docs/                           # Documentation
│   ├── architecture.md             # Diagrammes, flow, schémas
│   └── scraping_plan.md
│
├── notebooks/
│   ├── Experimentation.ipynb       # Exploration, prototypage
│   └── README.md
│
├── tests/                          # Tests unitaires & intégration
│   └── ...
│
└── cib_forecast.egg-info/          # Métadonnées setuptools
```

---

##  Commandes disponibles

```bash
./run.sh init-hive          # Initialiser schémas Hive
./run.sh load-reference     # Charger tables de référence (CSV)
./run.sh scrape-macro       # Scraper Yahoo Finance → bronze/raw/macro
./run.sh ingest             # Ingérer données brutes (HDFS/Hive)
./run.sh transform          # Bronze → Silver (nettoyage, agrégation)
./run.sh assign-macro       # Attribution macro → account_macro_assignment
./run.sh policy             # Appliquer règles métier → account_policy
./run.sh features           # Feature engineering → features_cib_forecast
./run.sh train              # Entraîner modèles ML
./run.sh inference          # Scoring/prévision H+1
./run.sh all                # Exécuter le pipeline complet
./run.sh smoke              # Test local (données synthétiques)
```

---

##  Tester localement

```bash
# Sans HDFS/Hive requis
./run.sh smoke

# Tests unitaires
pytest -q

# Tests verbose
pytest -v tests/
```

---

##  Configuration avancée

### Variables d'environnement

```bash
export CIB_ENV="cib"                    # Environnement (cib, dev, test, prod)
export PYTHON_BIN="/path/to/python"     # Python explicite
export SPARK_SUBMIT_BIN="/path/to/spark-submit"
export SPARK_SUBMIT_OPTS="--master yarn --deploy-mode cluster"
```

### Spark personnalisé

```bash
# Local (par défaut)
./run.sh features

# Yarn cluster
SPARK_SUBMIT_OPTS="--master yarn --deploy-mode cluster" ./run.sh features

# Yarn client
SPARK_SUBMIT_OPTS="--master yarn --deploy-mode client" ./run.sh features
```

---

##  Documentation additionnelle

- [Architecture détaillée](docs/architecture.md) : diagrammes, schémas, transformations
- [Plan de scraping](docs/scraping_plan.md) : source des données macro
- [Experimentation](notebooks/Experimenatation.ipynb) : exploration, prototypage

---

##  Troubleshooting

### Erreur : `JAVA_HOME not found`
```bash
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
```

### Erreur : `HDFS connection refused`
Vérifier que HDFS est démarré :
```bash
jps  # Chercher NameNode, DataNode
hdfs dfs -ls /
```

### Erreur : `Hive metastore not available`
Vérifier `config.ini` : `hive_warehouse` pointe-t-il vers un chemin accessible ?

### Mode synthétique (pas de HDFS)
```bash
# Éditer config.ini
macro_source = synthetic
./run.sh smoke
```

---

##  Contact & Support

Voir le fichier [docs/architecture.md](docs/architecture.md) pour les contacts de l'équipe et les questions techniques.

---

**Dernière mise à jour** : 2026-06-15

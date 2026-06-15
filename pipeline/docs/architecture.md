# Architecture du projet CIB Forecast

Ce document regroupe les diagrammes à insérer dans le rapport. Tous les
diagrammes sont en **Mermaid** : ils s'affichent directement dans GitHub,
Notion, VSCode, et la plupart des éditeurs Markdown. Pour les exporter en
PNG, utiliser [Mermaid Live Editor](https://mermaid.live/) ou la CLI
`npx -y @mermaid-js/mermaid-cli -i docs/architecture.md -o out.png`.

---

## 1. Vue d'ensemble — Pipeline de bout en bout

```mermaid
flowchart LR
    subgraph SRC["Sources"]
        SYN["Datagen synthétique\n(notebook BLOC A)"]
    end

    subgraph BRZ["Bronze (HDFS / Hive : cib_bronze)"]
        BRZ_TRX["transactions_raw\n(CSV / ORC)"]
        BRZ_CAL["calendar_weekly_flags"]
        BRZ_MAC["macro_indicators_weekly"]
    end

    subgraph SLV["Silver (HDFS / Hive : cib_silver)"]
        SLV_WK["weekly_cashflow_account"]
        SLV_QM["account_quality_metrics"]
        SLV_POL["account_policy"]
    end

    subgraph ML["ML (HDFS / Hive : cib_ml)"]
        FEAT["features_cib_forecast\n(ORC)"]
        MODELS_R[("Modèles RF\nrégression")]
        MODELS_C[("Modèles Logistique\nclassification")]
        PRED["predictions_cib_forecast"]
    end

    subgraph ORCH["Orchestration"]
        AF["Airflow DAG\ncib_forecast_pipeline"]
    end

    SYN -->|bootstrap_bronze.py| BRZ_TRX
    SYN -->|bootstrap_bronze.py| BRZ_CAL
    SYN -->|bootstrap_bronze.py| BRZ_MAC

    BRZ_TRX -->|"transformation_job.py\n(PySpark, steps 1-7)"| SLV_WK
    BRZ_TRX -->|"transformation_job.py"| SLV_QM
    SLV_QM -->|"policy_job.py"| SLV_POL

    SLV_WK -->|"features.py\n(PySpark step 8)"| FEAT
    SLV_POL -->|"features.py"| FEAT
    BRZ_CAL -->|"features.py"| FEAT
    BRZ_MAC -->|"features.py\n(+1 semaine, anti-fuite)"| FEAT

    FEAT -->|"train.py\n(Random Forest)"| MODELS_R
    FEAT -->|"train.py\n(Logistique)"| MODELS_C
    SLV_POL -->|"use_externals_*"| MODELS_R
    SLV_POL -->|"use_externals_*"| MODELS_C

    FEAT --> INF["inference.py"]
    MODELS_R --> INF
    MODELS_C --> INF
    INF --> PRED

    AF -.->|orchestre| BRZ
    AF -.->|orchestre| SLV
    AF -.->|orchestre| ML

    classDef bronze fill:#f9e4c8,stroke:#8a5a00,color:#000
    classDef silver fill:#e0e0e0,stroke:#444,color:#000
    classDef ml fill:#cfe8ff,stroke:#005ec4,color:#000
    classDef orch fill:#fff2b0,stroke:#9a6e00,color:#000
    class BRZ_TRX,BRZ_CAL,BRZ_MAC bronze
    class SLV_WK,SLV_QM,SLV_POL silver
    class FEAT,MODELS_R,MODELS_C,PRED ml
    class AF orch
```

---

## 2. Stack technique — Couches logicielles

```mermaid
flowchart TB
    subgraph OS["Système (WSL2 Ubuntu)"]
        JAVA["OpenJDK 11"]
        SSH["OpenSSH server"]
        PY["Python 3.12 (venv)"]
    end

    subgraph DATA["Stockage et calcul"]
        HDFS["Hadoop 2.7.7 — HDFS pseudo-distribué\n(NameNode + DataNode + SecondaryNameNode)"]
        HIVE["Hive 2.3.9\n(DDL externes ORC)"]
        SPARK["Spark 3.3.4 + PySpark"]
    end

    subgraph APP["Application cib_forecast"]
        SRC["src/ (transformation, policy, features, ml)"]
        JOBS["Jobs racine (transformation_job, policy_job, features, train, inference)"]
        CONF["config.ini (rec / prd, format=csv|hive)"]
    end

    subgraph ORCH["Orchestration et tests"]
        AF["Airflow 2.10.3 (DAG)"]
        RUN["run.sh (CLI locale)"]
        PYT["pytest (16 tests, sans Spark)"]
    end

    JAVA --> HDFS
    JAVA --> SPARK
    JAVA --> HIVE
    SSH -->|"start-dfs.sh\nlance les démons"| HDFS
    PY --> APP
    PY --> AF

    JOBS -->|spark-submit| SPARK
    JOBS -->|lecture/écriture| HDFS
    JOBS -->|"format=hive (prd)"| HIVE
    SRC --> JOBS
    CONF --> JOBS

    AF -->|BashOperator| JOBS
    AF -->|BashOperator| RUN
    RUN --> JOBS
    PYT --> SRC
```

---

## 3. Flux de données détaillé — Étapes du notebook → tables

```mermaid
flowchart LR
    A["Transactions brutes\n(numero_compte, date_operation,\nmontant, sector)"] -->|"Step 1\n agg hebdo"| B["weekly_cashflow_account\n(numero_compte, week, sector,\ntotal_amount, transaction_count)"]
    B -->|"Step 2\n stats par compte"| C["stats globales\n(n_obs, cv, mean, std,\ncompleteness_ratio)"]
    C -->|"Step 3\n eligibility"| D["comptes éligibles\n(n_obs ≥ 24)"]
    D -->|"Step 4\n stats TS"| E["ADF, ACF, PACF,\ntrend, saisonnalité"]
    E -->|"Step 5\n filtre validité"| F["comptes valides\n(NaN exclus)"]
    F -->|"Step 6\n scores"| G["composite_score,\npredictability_score"]
    G -->|"Step 7\n filtre qualité"| H["account_quality_metrics\n(comptes haute qualité)"]
    H -->|"policy_job.py"| I["account_policy\n(use_externals_*,\nregression_model,\nclassification_model)"]

    B -->|"Step 8\n lags + rolling +\njoin calendar + macro+1w"| J["features_cib_forecast"]
    I -->|"par compte"| J
    J -->|"train.py par compte"| K["modèles RF + Logistique\nsauvegardés par horodatage UTC"]
    J -->|"inference.py\n(dernière semaine)"| L["predictions_cib_forecast"]
    K --> L

    classDef bronze fill:#f9e4c8,stroke:#8a5a00,color:#000
    classDef silver fill:#e0e0e0,stroke:#444,color:#000
    classDef ml fill:#cfe8ff,stroke:#005ec4,color:#000
    class A bronze
    class B,C,D,E,F,G,H,I silver
    class J,K,L ml
```

---

## 4. Modules du projet (dépendances internes)

```mermaid
flowchart TB
    subgraph ROOT["Jobs racine (spark-submit / python)"]
        TRX["transformation_job.py"]
        POL["policy_job.py"]
        FEA["features.py"]
        TR["train.py"]
        INF["inference.py"]
    end

    subgraph COMMON["src/common"]
        UTL["utils.py\nSparkSession + config"]
        LDR["table_loader.py"]
        WRT["table_writer.py"]
    end

    subgraph SRC_BL["src/ (logique métier)"]
        DGN["datagen/synthetic.py"]
        TRA["transformation.py\n(steps 1-7 + statsmodels)"]
        PLY["policy.py"]
        FPI["features_pipeline.py"]
    end

    subgraph ML_PKG["src/ml"]
        BF["build_features.py"]
        TRG["train_regression.py\n(Random Forest)"]
        TCL["train_classification.py\n(Logistique)"]
        MET["metrics.py"]
        REG["model_registry.py"]
        TRR["train_runner.py"]
        IRR["inference_runner.py"]
    end

    TRX --> TRA
    TRX --> LDR
    TRX --> WRT
    POL --> PLY
    POL --> LDR
    POL --> WRT
    FEA --> FPI
    FEA --> LDR
    FEA --> WRT
    TR  --> TRR
    INF --> IRR

    LDR --> UTL
    WRT --> UTL

    TRR --> BF
    TRR --> TRG
    TRR --> TCL
    TRR --> MET
    TRR --> REG
    IRR --> BF
    IRR --> REG

    DGN -.->|bootstrap_bronze.py| TRX
```

---

## 5. DAG Airflow

```mermaid
flowchart LR
    A["bootstrap_bronze\n(python)"] --> B["transformation_silver\n(spark-submit)"]
    B --> C["account_policy\n(spark-submit)"]
    C --> D["features_ml\n(spark-submit)"]
    D --> E["train_models\n(python)"]
    E --> F["inference_batch\n(python)"]
```

---

## 6. Politique d'activation des features externes

```mermaid
flowchart TD
    A["Compte (sector, cv_cashflow)"] --> B{"sector ∈\n{retail, agriculture} ?"}
    B -- oui --> R1["use_externals_regression = true"]
    B -- non --> C{"cv_cashflow ∈ [0.5; 1.2] ?"}
    C -- oui --> R1
    C -- non --> R2["use_externals_regression = false"]

    A --> D{"sector = retail ?"}
    D -- oui --> R3["use_externals_classification = true"]
    D -- non --> R4["use_externals_classification = false"]

    classDef yes fill:#c8f7c8,stroke:#2e7d32,color:#000
    classDef no  fill:#ffd6d6,stroke:#c62828,color:#000
    class R1,R3 yes
    class R2,R4 no
```

---

## 7. Validation ML — Walk-forward (3 splits × 4 semaines)

```mermaid
gantt
    title Walk-forward sur 104 semaines (extrait)
    dateFormat  X
    axisFormat  %s
    section Split 1
    Train (88 sem)    :done, 0, 88
    Test (4 sem)      :crit, 88, 92
    section Split 2
    Train (84 sem)    :done, 0, 84
    Test (4 sem)      :crit, 84, 88
    section Split 3
    Train (80 sem)    :done, 0, 80
    Test (4 sem)      :crit, 80, 84
```

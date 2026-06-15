"""Politique d'activation des features externes (régression + classification).

Modes (``policy_mode`` dans ``config.ini``) :

- ``category_v7`` (défaut PFE) : catégories encapsulées (seuils EXP1--EXP6) +
  ``policy_v7_encapsulated.csv``.
- ``category_v6`` : ancienne catégorisation A--F poolée + ``policy_v6_final_*pct.csv``.
- ``rules`` : matrice R0--R4 (``policy_rules.py``), conservée pour comparaison / repli.
- ``lookup`` / ``hybrid`` / ``oracle`` : enrichissements CSV optionnels (legacy).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from src.account_categorization import categorize_account
from src.category_policy_v6 import POLICY_VERSION_V6, load_category_policy_table as load_category_policy_table_v6
from src.category_policy_v7 import POLICY_VERSION_V7, load_category_policy_table as load_category_policy_table_v7
from pyspark.sql.types import BooleanType, DoubleType, StringType, StructField, StructType
from src.common.utils import get_logger
from src.policy_rules import (
    CLASSIFICATION_MODEL,
    COMPLETENESS_EXCLUDE_SPARSE,
    COMPLETENESS_IRREGULAR_MIN,
    COMPLETENESS_REGULAR_MIN,
    COMPOSITE_IRREGULAR_MIN,
    COMPOSITE_VOLATILE_MIN,
    CV_BAND_HIGH,
    CV_BAND_LOW,
    CV_EXCLUDE_FLAT,
    CV_VOLATILE_MIN,
    EXTERNAL_NOISE_MAX_FRAC,
    POLICY_VERSION,
    PREDICTABILITY_REGULAR_MIN,
    REGRESSION_MODEL,
    ACF_IRREGULAR_MAX,
    ACF_REGULAR_MIN,
)

if TYPE_CHECKING:
    from configparser import SectionProxy

    from pyspark.sql import DataFrame, SparkSession

LOG = get_logger(__name__)


def _pandas_to_spark_policy(spark: SparkSession, pdf: "pd.DataFrame") -> DataFrame:
    """CSV temporaire — évite ``createDataFrame`` direct (PySpark 3.3 + Python 3.12)."""
    import os
    import tempfile

    import pandas as pd

    fd, path = tempfile.mkstemp(suffix=".csv", prefix="cib_policy_")
    os.close(fd)
    try:
        pdf.to_csv(path, index=False)
        sdf = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(f"file://{path}")
        )
        sdf = sdf.cache()
        sdf.count()
        return sdf
    finally:
        os.unlink(path)


def _policy_v3_columns(external_noise_frac: float = 0.0):
    """Expressions Spark alignées sur ``evaluate_externals_policy`` (v3.0)."""
    acf = F.col("acf_lag1")
    cv = F.col("cv_cashflow")
    comp = F.col("completeness_ratio")
    pred = F.col("predictability_score")
    composite = F.col("composite_score")
    noise = F.lit(float(external_noise_frac))

    rule_id = (
        F.when(comp <= F.lit(COMPLETENESS_EXCLUDE_SPARSE), F.lit("R0-A"))
        .when(cv < F.lit(CV_EXCLUDE_FLAT), F.lit("R0-B"))
        .when(noise > F.lit(EXTERNAL_NOISE_MAX_FRAC), F.lit("R4"))
        .when(
            (acf >= F.lit(ACF_REGULAR_MIN))
            & (comp > F.lit(COMPLETENESS_REGULAR_MIN))
            & (pred >= F.lit(PREDICTABILITY_REGULAR_MIN)),
            F.lit("R1"),
        )
        .when(
            (acf < F.lit(ACF_IRREGULAR_MAX))
            & (comp > F.lit(COMPLETENESS_IRREGULAR_MIN))
            & (composite >= F.lit(COMPOSITE_IRREGULAR_MIN)),
            F.lit("R2"),
        )
        .when((cv > F.lit(CV_VOLATILE_MIN)) & (composite >= F.lit(COMPOSITE_VOLATILE_MIN)), F.lit("R3"))
        .otherwise(F.lit("baseline"))
    )

    use_reg = rule_id.isin("R1", "R2", "R3")
    use_clf = rule_id == F.lit("R1")

    return use_reg, use_clf, rule_id


def _build_account_policy_category_v6(
    quality_accounts: DataFrame,
    *,
    spark: SparkSession,
    cfg: SectionProxy,
) -> DataFrame:
    """Catégorise chaque compte puis applique la politique notebook par catégorie.

    Implémentation driver pandas (évite bug cloudpickle Spark 3.3 + Python 3.12 sur ``F.when``).
    """
    import pandas as pd

    policy_lookup = load_category_policy_table_v6(cfg).set_index("account_category")
    pdf = quality_accounts.select(
        "numero_compte",
        "sector",
        "cv_cashflow",
        "completeness_ratio",
        "acf_lag1",
        "predictability_score",
        "composite_score",
        "n_obs",
        "trend_strength",
        "seasonality_strength",
    ).toPandas()

    out_rows: list[tuple] = []
    for row in pdf.itertuples(index=False):
        cat = categorize_account(
            float(row.acf_lag1),
            float(row.cv_cashflow),
            float(row.trend_strength),
            float(row.seasonality_strength),
            int(row.n_obs),
        )
        pol = policy_lookup.loc[cat] if cat in policy_lookup.index else None
        use_ext = bool(pol["activate_externals"]) if pol is not None else False
        reg_model = str(pol["regression_model"]) if pol is not None else REGRESSION_MODEL
        out_rows.append(
            (
                row.numero_compte,
                row.sector,
                float(row.cv_cashflow),
                float(row.completeness_ratio),
                float(row.acf_lag1),
                float(row.predictability_score),
                float(row.composite_score),
                cat,
                use_ext,
                use_ext,
                cat,
                reg_model,
                CLASSIFICATION_MODEL,
                POLICY_VERSION_V6,
            )
        )

    result_pdf = pd.DataFrame(
        out_rows,
        columns=[
            "numero_compte",
            "sector",
            "cv_cashflow",
            "completeness_ratio",
            "acf_lag1",
            "predictability_score",
            "composite_score",
            "account_category",
            "use_externals_regression",
            "use_externals_classification",
            "policy_rule_id",
            "regression_model",
            "classification_model",
            "policy_version",
        ],
    )
    policy = _pandas_to_spark_policy(spark, result_pdf)

    LOG.info(
        "[policy] account_policy built (version=%s, categories=%s, accounts=%d)",
        POLICY_VERSION_V6,
        policy_lookup.index.tolist(),
        len(out_rows),
    )
    return policy


def _build_account_policy_category_v7(
    quality_accounts: DataFrame,
    *,
    spark: SparkSession,
    cfg: SectionProxy,
) -> DataFrame:
    """Catégorisation encapsulée v7 + politique ``policy_v7_encapsulated.csv``."""
    import pandas as pd

    policy_lookup = load_category_policy_table_v7(cfg).set_index("account_category")
    pdf = quality_accounts.select(
        "numero_compte",
        "sector",
        "cv_cashflow",
        "completeness_ratio",
        "acf_lag1",
        "predictability_score",
        "composite_score",
        "n_obs",
        "trend_strength",
        "seasonality_strength",
    ).toPandas()

    out_rows: list[tuple] = []
    for row in pdf.itertuples(index=False):
        cat = categorize_account(
            float(row.acf_lag1),
            float(row.cv_cashflow),
            float(row.trend_strength),
            float(row.seasonality_strength),
            int(row.n_obs),
        )
        pol = policy_lookup.loc[cat] if cat in policy_lookup.index else None
        use_ext = bool(pol["activate_externals"]) if pol is not None else False
        reg_model = str(pol["regression_model"]) if pol is not None else REGRESSION_MODEL
        rule_id = str(pol.get("policy_rule_id", cat)) if pol is not None else cat
        out_rows.append(
            (
                row.numero_compte,
                row.sector,
                float(row.cv_cashflow),
                float(row.completeness_ratio),
                float(row.acf_lag1),
                float(row.predictability_score),
                float(row.composite_score),
                cat,
                use_ext,
                use_ext,
                rule_id,
                reg_model,
                CLASSIFICATION_MODEL,
                POLICY_VERSION_V7,
            )
        )

    result_pdf = pd.DataFrame(
        out_rows,
        columns=[
            "numero_compte",
            "sector",
            "cv_cashflow",
            "completeness_ratio",
            "acf_lag1",
            "predictability_score",
            "composite_score",
            "account_category",
            "use_externals_regression",
            "use_externals_classification",
            "policy_rule_id",
            "regression_model",
            "classification_model",
            "policy_version",
        ],
    )
    policy = _pandas_to_spark_policy(spark, result_pdf)
    LOG.info(
        "[policy] account_policy built (version=%s, categories=%s, accounts=%d)",
        POLICY_VERSION_V7,
        policy_lookup.index.tolist(),
        len(out_rows),
    )
    return policy


def _build_account_policy_rules_v3(
    quality_accounts: DataFrame,
    *,
    external_noise: float,
) -> DataFrame:
    use_reg, use_clf, rule_id = _policy_v3_columns(external_noise)

    return quality_accounts.select(
        "numero_compte",
        "sector",
        "cv_cashflow",
        "completeness_ratio",
        "acf_lag1",
        "predictability_score",
        "composite_score",
        F.lit(None).cast("string").alias("account_category"),
        use_reg.cast("boolean").alias("use_externals_regression"),
        use_clf.cast("boolean").alias("use_externals_classification"),
        rule_id.alias("policy_rule_id"),
        F.lit(REGRESSION_MODEL).alias("regression_model"),
        F.lit(CLASSIFICATION_MODEL).alias("classification_model"),
        F.lit(POLICY_VERSION).alias("policy_version"),
    )


def build_account_policy(
    quality_accounts: DataFrame,
    *,
    spark: SparkSession | None = None,
    cfg: SectionProxy | None = None,
) -> DataFrame:
    """Construit ``account_policy`` à partir des métriques qualité silver."""
    if cfg is None:
        raise ValueError("cfg is required for build_account_policy")

    mode = (cfg.get("policy_mode") or "category_v7").strip().lower()
    external_noise = float(cfg.get("policy_external_noise_frac", "0") or 0)

    if mode == "category_v7":
        if spark is None:
            raise ValueError("spark is required when policy_mode=category_v7")
        policy = _build_account_policy_category_v7(quality_accounts, spark=spark, cfg=cfg)
    elif mode == "category_v6":
        if spark is None:
            raise ValueError("spark is required when policy_mode=category_v6")
        policy = _build_account_policy_category_v6(quality_accounts, spark=spark, cfg=cfg)
    elif mode == "rules":
        policy = _build_account_policy_rules_v3(quality_accounts, external_noise=external_noise)
        LOG.info("[policy] account_policy built (version=%s)", POLICY_VERSION)
    else:
        policy = _build_account_policy_rules_v3(quality_accounts, external_noise=external_noise)
        LOG.info("[policy] account_policy built (version=%s, mode=%s baseline rules)", POLICY_VERSION, mode)

    if spark is not None and mode not in {"category_v6", "category_v7"}:
        policy = _apply_policy_enrichments(spark, policy, cfg)
    elif spark is not None and mode in {"lookup", "hybrid", "oracle"}:
        policy = _apply_policy_enrichments(spark, policy, cfg)

    return policy


def _apply_policy_enrichments(
    spark: SparkSession,
    policy: DataFrame,
    cfg: SectionProxy,
) -> DataFrame:
    """Joint lookup sector×CV et/ou overrides par compte si les fichiers existent."""
    mode = (cfg.get("policy_mode") or "category_v6").strip().lower()
    local_base = Path(cfg.get("local_base", "."))
    lookup_path = Path(cfg.get("policy_sector_cv_lookup", "").strip() or local_base / "silver" / "policy_sector_cv_lookup.csv")
    overrides_path = Path(
        cfg.get("policy_account_overrides", "").strip() or local_base / "silver" / "policy_account_overrides.csv"
    )

    if lookup_path.is_file() and mode in {"lookup", "hybrid"}:
        LOG.info("[policy] applying sector×CV lookup from %s", lookup_path)
        lookup = (
            spark.read.option("header", True)
            .csv(str(lookup_path))
            .select(
                F.lower(F.col("sector")).alias("_lk_sector"),
                F.col("cv_bucket").alias("_lk_bucket"),
                F.col("use_externals_regression").cast("boolean").alias("_lk_reg"),
                F.col("use_externals_classification").cast("boolean").alias("_lk_clf"),
            )
        )
        policy = policy.withColumn(
            "_cv_bucket",
            F.when(F.col("cv_cashflow") < CV_BAND_LOW, F.lit("<0.30"))
            .when(
                (F.col("cv_cashflow") >= CV_BAND_LOW) & (F.col("cv_cashflow") < CV_BAND_HIGH),
                F.lit("[0.30,0.50)"),
            )
            .when(
                (F.col("cv_cashflow") >= CV_BAND_HIGH) & (F.col("cv_cashflow") < 0.80),
                F.lit("[0.50,0.80)"),
            )
            .otherwise(F.lit(">=0.80")),
        )
        policy = (
            policy.join(
                lookup,
                (F.lower(F.col("sector")) == F.col("_lk_sector"))
                & (F.col("_cv_bucket") == F.col("_lk_bucket")),
                how="left",
            )
            .withColumn(
                "use_externals_regression",
                F.coalesce(F.col("_lk_reg"), F.col("use_externals_regression")),
            )
            .withColumn(
                "use_externals_classification",
                F.coalesce(F.col("_lk_clf"), F.col("use_externals_classification")),
            )
            .drop("_cv_bucket", "_lk_sector", "_lk_bucket", "_lk_reg", "_lk_clf")
        )

    if overrides_path.is_file() and mode in {"oracle", "hybrid"}:
        LOG.info("[policy] applying per-account overrides from %s", overrides_path)
        overrides = spark.read.option("header", True).csv(str(overrides_path)).select(
            F.col("numero_compte").alias("_ov_account"),
            F.col("use_externals_regression").cast("boolean").alias("_ov_reg"),
            F.col("use_externals_classification").cast("boolean").alias("_ov_clf"),
        )
        policy = (
            policy.join(overrides, on=policy.numero_compte == overrides._ov_account, how="left")
            .withColumn(
                "use_externals_regression",
                F.coalesce(F.col("_ov_reg"), F.col("use_externals_regression")),
            )
            .withColumn(
                "use_externals_classification",
                F.coalesce(F.col("_ov_clf"), F.col("use_externals_classification")),
            )
            .drop("_ov_account", "_ov_reg", "_ov_clf")
        )

    return policy

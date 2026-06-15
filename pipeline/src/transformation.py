"""Pipeline de transformation bronze → silver (steps 1–7 du notebook).

PySpark pour les agrégations / fenêtres, et ``applyInPandas`` pour les
métriques de série temporelle (ADF, ACF, PACF) qui n'ont pas d'équivalent
PySpark natif.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, pacf

from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from src.common.utils import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — agrégation hebdomadaire (lundi de chaque semaine)
# ---------------------------------------------------------------------------

def step1_weekly_aggregation(df_raw: DataFrame) -> DataFrame:
    """Agrégation `numero_compte × week × sector` → total / nombre de transactions."""
    weekly = (
        df_raw.withColumn("week", F.date_trunc("week", F.col("date_operation").cast("timestamp")))
        .withColumn("week", F.to_date("week"))
        .groupBy("numero_compte", "week", "sector")
        .agg(
            F.sum("montant").alias("total_amount"),
            F.count("montant").alias("transaction_count"),
        )
        .orderBy("numero_compte", "week")
    )
    LOG.info("[Step 1] weekly aggregation done")
    return weekly


# ---------------------------------------------------------------------------
# Step 2 — statistiques globales par compte
# ---------------------------------------------------------------------------

def step2_global_stats(weekly: DataFrame) -> DataFrame:
    stats_df = (
        weekly.groupBy("numero_compte")
        .agg(
            F.count("total_amount").alias("n_obs"),
            F.sum("transaction_count").alias("total_transactions"),
            F.sum(F.abs(F.col("total_amount"))).alias("total_cashflow"),
            F.mean(F.abs(F.col("total_amount"))).alias("mean_cashflow"),
            F.stddev("total_amount").alias("std_cashflow"),
            F.sum((F.col("total_amount") != 0).cast("double")).alias("non_zero_periods"),
            F.first("sector").alias("sector"),
        )
        .withColumn("completeness_ratio", F.col("non_zero_periods") / F.col("n_obs"))
        .withColumn("cv_cashflow", F.col("std_cashflow") / F.col("mean_cashflow"))
    )
    return stats_df


# ---------------------------------------------------------------------------
# Step 3 — éligibilité (n_obs >= 24, std > 0)
# ---------------------------------------------------------------------------

def step3_eligibility_filter(stats_df: DataFrame, min_obs: int = 24) -> DataFrame:
    eligible = stats_df.where(
        (F.col("n_obs") >= F.lit(min_obs)) & (F.col("std_cashflow") >= F.lit(1e-8))
    )
    LOG.info("[Step 3] eligibility filter applied (min_obs=%d)", min_obs)
    return eligible


# ---------------------------------------------------------------------------
# Step 4 — métriques de série temporelle par compte (pandas UDF group-apply)
# ---------------------------------------------------------------------------

_TS_METRICS_SCHEMA = StructType([
    StructField("numero_compte", StringType(), nullable=False),
    StructField("adf_statistic", DoubleType()),
    StructField("adf_pvalue", DoubleType()),
    StructField("acf_lag1", DoubleType()),
    StructField("acf_lag2", DoubleType()),
    StructField("pacf_lag1", DoubleType()),
    StructField("trend_strength", DoubleType()),
    StructField("seasonality_strength", DoubleType()),
])


def _compute_ts_metrics_array(series: np.ndarray) -> dict[str, float]:
    """Métriques sur un tableau numpy (ADF, ACF, PACF, trend, saisonnalité)."""
    out: dict[str, float] = {
        "adf_statistic": float("nan"),
        "adf_pvalue": float("nan"),
        "acf_lag1": float("nan"),
        "acf_lag2": float("nan"),
        "pacf_lag1": float("nan"),
        "trend_strength": float("nan"),
        "seasonality_strength": float("nan"),
    }
    series = np.asarray(series, dtype=float)

    try:
        adf = adfuller(series, autolag="AIC")
        out["adf_statistic"] = float(adf[0])
        out["adf_pvalue"] = float(adf[1])
    except Exception:
        pass

    try:
        if len(series) >= 8:
            acf_v = acf(series, nlags=2, fft=True)
            out["acf_lag1"] = float(acf_v[1])
            out["acf_lag2"] = float(acf_v[2])
            out["pacf_lag1"] = float(pacf(series, nlags=1)[1])
    except Exception:
        pass

    try:
        _, _, r, _, _ = stats.linregress(np.arange(len(series)), series)
        out["trend_strength"] = float(r) ** 2
    except Exception:
        pass

    try:
        lag = 52 if len(series) > 52 else len(series) // 2
        if lag > 0:
            corr = np.corrcoef(series[lag:], series[:-lag])[0, 1]
            out["seasonality_strength"] = float(abs(corr)) if np.isfinite(corr) else float("nan")
    except Exception:
        pass
    return out


def _compute_metrics_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """Boucle pandas sur (numero_compte → série) — robuste aux versions Arrow."""
    rows: list[dict[str, float | str]] = []
    pdf = pdf.sort_values(["numero_compte", "week"])
    for account_id, group in pdf.groupby("numero_compte", sort=False):
        series = group["total_amount"].to_numpy(dtype=float)
        m: dict[str, float | str] = dict(_compute_ts_metrics_array(series))
        m["numero_compte"] = str(account_id)
        rows.append(m)
    return pd.DataFrame(rows, columns=[f.name for f in _TS_METRICS_SCHEMA.fields])


def step4_ts_metrics(weekly: DataFrame, eligible: DataFrame) -> DataFrame:
    """Step 4 — métriques TS par compte.

    Collecte côté driver, calcul pandas + statsmodels, puis re-import via
    un CSV intermédiaire (évite les bugs de sérialisation cloudpickle de
    PySpark 3.3 sous Python 3.12 lors de `createDataFrame(pandas)`).
    """
    import tempfile

    spark = weekly.sparkSession
    eligible_ids = eligible.select("numero_compte")
    pdf = (
        weekly.join(eligible_ids, on="numero_compte", how="inner")
        .select("numero_compte", "week", "total_amount")
        .toPandas()
    )
    metrics_pdf = _compute_metrics_pandas(pdf)
    LOG.info("[Step 4] ts metrics computed for %d accounts", len(metrics_pdf))

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        metrics_pdf.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    return spark.read.option("header", "true").schema(_TS_METRICS_SCHEMA).csv(f"file://{tmp_path}")


# ---------------------------------------------------------------------------
# Step 5 — filtre validité (NaN / inf sur métriques clés)
# ---------------------------------------------------------------------------

def step5_validity_filter(metrics: DataFrame) -> DataFrame:
    return metrics.where(
        F.col("adf_pvalue").isNotNull()
        & ~F.isnan("adf_pvalue")
        & F.col("adf_statistic").isNotNull()
        & ~F.isnan("adf_statistic")
        & F.col("acf_lag1").isNotNull()
        & ~F.isnan("acf_lag1")
    )


# ---------------------------------------------------------------------------
# Step 6 — scores composites
# ---------------------------------------------------------------------------

def step6_scores(df: DataFrame) -> DataFrame:
    df = (
        df.withColumn(
            "stationarity_component",
            (F.lit(1.0) - F.greatest(F.lit(0.0), F.least(F.lit(1.0), F.col("adf_pvalue")))) * 100,
        )
        .withColumn("autocorr_component", F.col("acf_lag1") * 100)
        .withColumn("trend_component", F.col("trend_strength") * 50)
        .withColumn("seasonality_component", F.col("seasonality_strength") * 50)
        .withColumn("completeness_component", F.col("completeness_ratio") * 100)
        .withColumn(
            "cashflow_component",
            F.when(F.col("total_cashflow") > 0, F.log10(F.col("total_cashflow")) * 10).otherwise(0.0),
        )
        .withColumn(
            "transaction_component",
            F.when(F.col("total_transactions") > 0, F.log10(F.col("total_transactions")) * 10).otherwise(0.0),
        )
        .withColumn(
            "cv_penalty",
            F.when((F.col("cv_cashflow") >= 0.2) & (F.col("cv_cashflow") <= 2.0), F.lit(0.0))
            .when(F.col("cv_cashflow") < 0.2, (F.lit(0.2) - F.col("cv_cashflow")) * 50)
            .otherwise((F.col("cv_cashflow") - F.lit(2.0)) * 20),
        )
        .withColumn(
            "predictability_score",
            F.col("autocorr_component") * 0.5
            + F.col("trend_component") * 0.25
            + F.col("seasonality_component") * 0.25,
        )
        .withColumn(
            "composite_score",
            F.col("predictability_score") * 0.35
            + F.col("stationarity_component") * 0.20
            + F.col("completeness_component") * 0.20
            + F.col("cashflow_component") * 0.15
            + F.col("transaction_component") * 0.10
            - F.col("cv_penalty"),
        )
    )
    return df


# ---------------------------------------------------------------------------
# Step 7 — filtre qualité (mêmes seuils que le notebook)
# ---------------------------------------------------------------------------

def step7_quality_filter(
    df: DataFrame,
    *,
    acf_min: float = 0.35,
    adf_pvalue_max: float = 0.10,
    completeness_min: float = 0.60,
    transactions_min: int = 100,
    cv_min: float = 0.30,
    cv_max: float = 1.50,
    predictability_min: float = 35.0,
    composite_min: float = 40.0,
) -> DataFrame:
    filtered = df.where(
        (F.col("acf_lag1") > F.lit(acf_min))
        & (F.col("adf_pvalue") < F.lit(adf_pvalue_max))
        & (F.col("completeness_ratio") > F.lit(completeness_min))
        & (F.col("total_transactions") > F.lit(transactions_min))
        & (F.col("cv_cashflow") >= F.lit(cv_min))
        & (F.col("cv_cashflow") <= F.lit(cv_max))
        & (F.col("predictability_score") > F.lit(predictability_min))
        & (F.col("composite_score") > F.lit(composite_min))
    )
    LOG.info("[Step 7] quality filter applied")
    return filtered


# ---------------------------------------------------------------------------
# Pipeline complet : retourne weekly (silver) + quality (table comptes filtrés)
# ---------------------------------------------------------------------------

def run_transformation_pipeline(
    df_transactions: DataFrame,
    *,
    quality_kwargs: dict | None = None,
    min_obs: int = 24,
) -> tuple[DataFrame, DataFrame]:
    """Retourne ``(weekly_silver, quality_accounts)``.

    `weekly_silver` = step1 (à matérialiser)
    `quality_accounts` = step6 enrichi du filtre step7 = méta-table comptes éligibles.
    """
    quality_kwargs = quality_kwargs or {}
    weekly = step1_weekly_aggregation(df_transactions)
    stats_df = step2_global_stats(weekly)
    eligible = step3_eligibility_filter(stats_df, min_obs=min_obs)
    metrics = step4_ts_metrics(weekly, eligible)
    valid_metrics = step5_validity_filter(metrics)
    merged = eligible.join(valid_metrics, on="numero_compte", how="inner")
    scored = step6_scores(merged)
    quality = step7_quality_filter(scored, **quality_kwargs)
    return weekly, quality

"""Step 8 du notebook : construction des features ML (lags, rolling, calendrier, macro).

Sorties typiques : table ``features_cib_forecast`` (ORC, partitionnée par
``dt`` = date d'exécution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

from pyspark.sql import Window
from pyspark.sql import functions as F

from src.common.utils import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

LOG = get_logger(__name__)
TWO_PI = 2.0 * math.pi


def add_temporal_features(weekly: DataFrame) -> DataFrame:
    """Ajoute ``month``, ``week_num`` et 4 harmoniques (sin/cos)."""
    return (
        weekly.withColumn("month", F.month("week"))
        .withColumn("week_num", F.weekofyear("week"))
        .withColumn("week_sin", F.sin(F.lit(TWO_PI) * F.col("week_num") / F.lit(52.0)))
        .withColumn("week_cos", F.cos(F.lit(TWO_PI) * F.col("week_num") / F.lit(52.0)))
        .withColumn(
            "week_sin_2h", F.sin(F.lit(TWO_PI) * F.col("week_num") * F.lit(2.0) / F.lit(52.0))
        )
        .withColumn(
            "week_cos_2h", F.cos(F.lit(TWO_PI) * F.col("week_num") * F.lit(2.0) / F.lit(52.0))
        )
    )


def add_lag_and_rolling_features(df: DataFrame) -> DataFrame:
    """Lags + rolling sur ``total_amount`` (fenêtres par compte)."""
    win_account = Window.partitionBy("numero_compte").orderBy("week")
    df = df.withColumn("_lag1_amount", F.lag("total_amount", 1).over(win_account))
    df = (
        df.withColumn("_lag1_pos", F.when(F.col("_lag1_amount") > 0, 1).otherwise(0))
        .withColumn("_lag1_neg", F.when(F.col("_lag1_amount") < 0, 1).otherwise(0))
        .withColumn(
            "_lag1_sign",
            F.when(F.col("_lag1_amount") > 0, 1)
            .when(F.col("_lag1_amount") < 0, -1)
            .otherwise(0),
        )
    )

    def _roll(window_days: int) -> Window:
        return win_account.rowsBetween(-(window_days - 1), 0)

    df = (
        df.withColumn("negative_count_4w", F.sum("_lag1_neg").over(_roll(4)))
        .withColumn("negative_count_8w", F.sum("_lag1_neg").over(_roll(8)))
        .withColumn("prop_positive_24w", F.avg("_lag1_pos").over(_roll(24)))
        .withColumn("sign_streak_8", F.sum("_lag1_sign").over(_roll(8)))
        .withColumn("rolling_max_8w", F.max("_lag1_amount").over(_roll(8)))
        .withColumn("rolling_min_8w", F.min("_lag1_amount").over(_roll(8)))
        .withColumn("volatility_4w", F.stddev("_lag1_amount").over(_roll(4)))
    )
    return df


def join_calendar_and_macro(
    df: DataFrame,
    calendar: DataFrame,
    macro: DataFrame,
) -> DataFrame:
    """Jointures calendrier (semaine courante) + macro (décalé +1 semaine pour éviter la fuite)."""
    macro_shifted = macro.withColumn("week", F.date_add(F.col("week"), 7))
    return df.join(calendar, on="week", how="left").join(macro_shifted, on="week", how="left")


def build_features(
    weekly: DataFrame,
    quality_accounts: DataFrame,
    calendar: DataFrame,
    macro: DataFrame,
) -> DataFrame:
    """Construit la matrice de features finale pour les comptes haute qualité."""
    df = weekly.join(quality_accounts.select("numero_compte"), on="numero_compte", how="inner")
    df = add_temporal_features(df)
    df = add_lag_and_rolling_features(df)
    df = join_calendar_and_macro(df, calendar, macro)
    df = df.drop("_lag1_amount", "_lag1_pos", "_lag1_neg", "_lag1_sign")
    LOG.info("[features] feature matrix built")
    return df

-- Silver : cashflow hebdomadaire agrégé par compte (step1) + métriques (step6).
CREATE DATABASE IF NOT EXISTS cib_silver;
USE cib_silver;

CREATE EXTERNAL TABLE IF NOT EXISTS weekly_cashflow_account (
  numero_compte      STRING,
  week               DATE,
  sector             STRING,
  total_amount       DOUBLE,
  transaction_count  BIGINT
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/silver/weekly_cashflow_account';

CREATE EXTERNAL TABLE IF NOT EXISTS account_quality_metrics (
  numero_compte           STRING,
  sector                  STRING,
  n_obs                   BIGINT,
  total_transactions      BIGINT,
  total_cashflow          DOUBLE,
  mean_cashflow           DOUBLE,
  std_cashflow            DOUBLE,
  non_zero_periods        DOUBLE,
  completeness_ratio      DOUBLE,
  cv_cashflow             DOUBLE,
  adf_statistic           DOUBLE,
  adf_pvalue              DOUBLE,
  acf_lag1                DOUBLE,
  acf_lag2                DOUBLE,
  pacf_lag1               DOUBLE,
  trend_strength          DOUBLE,
  seasonality_strength    DOUBLE,
  stationarity_component  DOUBLE,
  autocorr_component      DOUBLE,
  trend_component         DOUBLE,
  seasonality_component   DOUBLE,
  completeness_component  DOUBLE,
  cashflow_component      DOUBLE,
  transaction_component   DOUBLE,
  cv_penalty              DOUBLE,
  predictability_score    DOUBLE,
  composite_score         DOUBLE
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/silver/account_quality_metrics';

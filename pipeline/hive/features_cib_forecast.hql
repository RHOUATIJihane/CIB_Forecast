-- ML : matrice de features (silver + calendrier + macro + rolling).
CREATE DATABASE IF NOT EXISTS cib_ml;
USE cib_ml;

CREATE EXTERNAL TABLE IF NOT EXISTS features_cib_forecast (
  numero_compte         STRING,
  week                  DATE,
  sector                STRING,
  total_amount          DOUBLE,
  transaction_count     BIGINT,
  month                 INT,
  week_num              INT,
  week_sin              DOUBLE,
  week_cos              DOUBLE,
  week_sin_2h           DOUBLE,
  week_cos_2h           DOUBLE,
  negative_count_4w     BIGINT,
  negative_count_8w     BIGINT,
  prop_positive_24w     DOUBLE,
  sign_streak_8         BIGINT,
  rolling_max_8w        DOUBLE,
  rolling_min_8w        DOUBLE,
  volatility_4w         DOUBLE,
  is_ramadan            INT,
  is_eid_alfitr         INT,
  is_new_year_week      INT,
  is_month_end_week     INT,
  is_quarter_end        INT,
  is_tax_deadline_week  INT,
  is_payroll_week       INT,
  oil_price_z           DOUBLE,
  commodity_index_z     DOUBLE,
  masi_index_z          DOUBLE,
  realestate_index_z    DOUBLE
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/ml/features_cib_forecast';

CREATE EXTERNAL TABLE IF NOT EXISTS predictions_cib_forecast (
  numero_compte  STRING,
  last_week      DATE,
  task           STRING,
  feature_mode   STRING,
  prediction     DOUBLE
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/ml/predictions_cib_forecast';

-- Silver : politique d'activation des features externes par compte.
CREATE DATABASE IF NOT EXISTS cib_silver;
USE cib_silver;

CREATE EXTERNAL TABLE IF NOT EXISTS account_policy (
  numero_compte                 STRING,
  sector                        STRING,
  cv_cashflow                   DOUBLE,
  completeness_ratio            DOUBLE,
  acf_lag1                      DOUBLE,
  predictability_score          DOUBLE,
  composite_score               DOUBLE,
  account_category              STRING,
  use_externals_regression      BOOLEAN,
  use_externals_classification  BOOLEAN,
  policy_rule_id                STRING,
  macro_primary                 STRING,
  macro_secondary               STRING,
  externals_for_model           STRING,
  regression_model              STRING,
  classification_model          STRING,
  policy_version                STRING
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/silver/account_policy';

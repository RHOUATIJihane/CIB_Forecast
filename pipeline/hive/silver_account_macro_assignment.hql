-- Silver : macro assignée par compte (lookup sector → macro_primary).
CREATE DATABASE IF NOT EXISTS cib_silver;
USE cib_silver;

CREATE EXTERNAL TABLE IF NOT EXISTS account_macro_assignment (
  numero_compte       STRING,
  sector              STRING,
  macro_primary       STRING,
  macro_secondary     STRING,
  include_calendar    BOOLEAN,
  externals_for_model STRING
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/silver/account_macro_assignment';

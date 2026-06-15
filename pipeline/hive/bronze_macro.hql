-- Bronze : indicateurs sectoriels (z-scorés) hebdomadaires.
CREATE DATABASE IF NOT EXISTS cib_bronze;
USE cib_bronze;

CREATE EXTERNAL TABLE IF NOT EXISTS macro_indicators_weekly (
  week                DATE,
  oil_price_z         DOUBLE,
  commodity_index_z   DOUBLE,
  masi_index_z        DOUBLE,
  realestate_index_z  DOUBLE
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/bronze/macro_indicators_weekly';

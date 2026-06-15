-- Bronze : mapping secteur (PFE synthétique) ou prefix NAF (prod) → variable macro.
-- En PFE : ``sector`` du générateur synthétique remplace ``code_activite`` NAF.
CREATE DATABASE IF NOT EXISTS cib_bronze;
USE cib_bronze;

CREATE EXTERNAL TABLE IF NOT EXISTS sector_macro_mapping (
  sector             STRING,
  sector_label       STRING,
  macro_primary      STRING,
  macro_secondary    STRING,
  include_calendar   BOOLEAN,
  priority           INT,
  source             STRING
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/bronze/sector_macro_mapping';

-- Bronze : transactions brutes (synthétiques pour la phase PFE).
CREATE DATABASE IF NOT EXISTS cib_bronze;
USE cib_bronze;

CREATE EXTERNAL TABLE IF NOT EXISTS transactions_raw (
  numero_compte      STRING,
  date_operation     TIMESTAMP,
  montant            DOUBLE,
  type_operation     STRING,
  sector             STRING,
  account_type_true  STRING
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/bronze/transactions_raw';

-- Bronze : flags calendaires hebdomadaires (Maroc).
CREATE DATABASE IF NOT EXISTS cib_bronze;
USE cib_bronze;

CREATE EXTERNAL TABLE IF NOT EXISTS calendar_weekly_flags (
  week                  DATE,
  is_ramadan            INT,
  is_eid_alfitr         INT,
  is_new_year_week      INT,
  is_month_end_week     INT,
  is_quarter_end        INT,
  is_tax_deadline_week  INT,
  is_payroll_week       INT
)
STORED AS ORC
LOCATION 'hdfs://127.0.0.1:9000/data/cib/bronze/calendar_weekly_flags';

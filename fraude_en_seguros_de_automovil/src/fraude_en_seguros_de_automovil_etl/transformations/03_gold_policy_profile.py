"""
Este script construye la tabla de características del perfil del asegurado.
Utiliza el historial SCD2 de la capa Silver para derivar atributos demográficos
estáticos y grupos de riesgo.
"""

###############################################################################
# Imports
###############################################################################
import pyspark.pipelines as dp
from pyspark.sql.functions import col, when, year, current_date, datediff
###############################################################################
# Configuración y Constantes
###############################################################################

silver_policies_source = "silver_policies_history"
gold_profile_table_name = "gold_policy_profile"

gold_profile_comment = """
Esta tabla gestionada actúa como Feature Table en el Feature Store.
Contiene el historial SCD2 de los asegurados y atributos derivados como 
'age_group' y 'vehicle_age_group'.
"""

# Habilitamos Change Data Feed para sincronización eficiente con el Online Store
gold_profile_properties = {"delta.enableChangeDataFeed": "true"}

# Definición del esquema con Primary Key y Timeseries para el Feature Store
gold_profile_schema = """
    policy_id STRING,
    policyholder_age INT,
    gender STRING,
    region_type STRING,
    vehicle_year INT,
    annual_premium_eur DOUBLE,
    has_telematics INT,
    coverage_type STRING,
    vehicle_type STRING,
    policy_start_date DATE,           
    __START_AT TIMESTAMP,             
    __END_AT TIMESTAMP,
    age_group STRING,                 
    vehicle_segment STRING,           
    risk_level_by_premium STRING,     
    driver_experience_years INT,
    is_new_policy_risk INT,           
    CONSTRAINT gold_policy_profile_pk PRIMARY KEY (policy_id, __START_AT TIMESERIES)
"""

@dp.materialized_view(
    name = gold_profile_table_name,
    comment = gold_profile_comment,
    table_properties = gold_profile_properties,
    schema = gold_profile_schema
)
def gold_policy_profile():
    df_policies = spark.read.table(silver_policies_source)

    df_profile = df_policies.select(
        # Seleccionamos las columnas base que confirmamos en el SQL
        col("policy_id"),
        col("policyholder_age"),
        col("gender"),
        col("region_type"),
        col("vehicle_year"),
        col("annual_premium_eur"),
        col("has_telematics"),
        col("coverage_type"),
        col("vehicle_type"),
        col("policy_start_date"),
        col("__START_AT"),
        col("__END_AT"),

        # 1. SEGMENTACIÓN POR EDAD
        when(col("policyholder_age") < 25, "novel")
        .when((col("policyholder_age") >= 25) & (col("policyholder_age") < 65), "standard")
        .otherwise("senior")
        .alias("age_group"),

        # 2. SEGMENTACIÓN DEL VEHÍCULO (Usando vehicle_year)
        when(col("vehicle_year") >= 2022, "new")
        .when((col("vehicle_year") < 2022) & (col("vehicle_year") >= 2015), "mid_age")
        .otherwise("old")
        .alias("vehicle_segment"),

        # 3. EXPERIENCIA (Usando bonus_malus_years directamente como años)
        col("bonus_malus_years").alias("driver_experience_years"),

        # 4. RIESGO DE PÓLIZA RECIENTE (Comparando fecha inicio con hoy)
        when(datediff(current_date(), col("policy_start_date")) < 90, 1).otherwise(0)
        .alias("is_new_policy_risk"),

        # 5. RIESGO POR PRIMA
        when(col("annual_premium_eur") < 500, "low_risk")
        .when((col("annual_premium_eur") >= 500) & (col("annual_premium_eur") < 1200), "medium_risk")
        .otherwise("high_risk")
        .alias("risk_level_by_premium")
    )

    return df_profile
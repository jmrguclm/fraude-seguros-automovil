"""
Este script construye las tablas de características de comportamiento (Behavioral Features)
para el Feature Store. Calcula agregaciones en ventanas deslizantes por 'policy_id' 
(frecuencia de siniestros, importes reclamados y señales de riesgo acumuladas).

Arquitectura: Se implementa un "Training-Serving Split":
1. gold_policy_aggregations: Para entrenamiento (usa datos limpios de Silver).
2. gold_policy_aggregations_inference: Para producción (usa datos inmediatos de Bronze).

Nota técnica: Se utiliza 'rangeBetween' con un límite superior de -1 milisegundo para 
garantizar la exactitud temporal (point-in-time correctness) y evitar el data leakage, 
asegurando que el siniestro actual no se incluya en su propia agregación histórica.
"""

###############################################################################
# 1. Imports
###############################################################################
import pyspark.pipelines as dp
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    avg,
    coalesce,
    col,
    count,
    lit,
    max,
    sum,
    to_timestamp,
    when
)

###############################################################################
# 2. Configuración y Constantes
###############################################################################

# Fuentes de datos
training_source = "silver_enriched_fraud"
inference_claims_source = "bronze_claims"
inference_labels_source = "bronze_labels"

# Parámetros técnicos
EPSILON = 1e-6  # Evita división por cero en el cálculo de ratios
gold_table_properties = {"delta.enableChangeDataFeed": "true"}

# Template de Schema para reutilizar en ambas tablas
gold_aggregations_schema_template = """
    policy_id STRING NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    count_claims_24h BIGINT,
    sum_amount_24h DOUBLE,
    count_claims_7d BIGINT,
    sum_amount_7d DOUBLE,
    avg_amount_7d DOUBLE,
    count_claims_30d BIGINT,
    sum_amount_30d DOUBLE,
    avg_amount_30d DOUBLE,
    max_amount_30d DOUBLE,
    num_fraud_confirmed_30d BIGINT,
    claims_7d_vs_avg_30d_ratio DOUBLE,
    CONSTRAINT {pk_name} PRIMARY KEY (policy_id, timestamp TIMESERIES)
"""

###############################################################################
# 3. Lógica de Agregación Compartida
###############################################################################

def _compute_aggregations(df):
    """
    Función interna que aplica las ventanas deslizantes sobre un DataFrame.
    El DataFrame de entrada debe tener: policy_id, timestamp, ts_ms, 
    claimed_amount_eur e is_fraud.
    """
    
    # Definición de ventanas en milisegundos
    # El rango -1 asegura que NO incluimos el evento actual en el cálculo
    w_24h = Window.partitionBy("policy_id").orderBy("ts_ms").rangeBetween(-86_400_000, -1)
    w_7d  = Window.partitionBy("policy_id").orderBy("ts_ms").rangeBetween(-7 * 86_400_000, -1)
    w_30d = Window.partitionBy("policy_id").orderBy("ts_ms").rangeBetween(-30 * 86_400_000, -1)

    df_agg = df.select(
        col("policy_id"),
        col("timestamp").cast("timestamp"),       
        # Agregaciones 24 horas (Frecuencia y Severidad reciente)
        count("claim_id").over(w_24h).alias("count_claims_24h"),
        coalesce(sum("claimed_amount_eur").over(w_24h), lit(0.0)).alias("sum_amount_24h"),
        
        # Agregaciones 7 días
        count("claim_id").over(w_7d).alias("count_claims_7d"),
        coalesce(sum("claimed_amount_eur").over(w_7d), lit(0.0)).alias("sum_amount_7d"),
        avg("claimed_amount_eur").over(w_7d).alias("avg_amount_7d"),
        
        # Agregaciones 30 días (Historial de medio plazo)
        count("claim_id").over(w_30d).alias("count_claims_30d"),
        coalesce(sum("claimed_amount_eur").over(w_30d), lit(0.0)).alias("sum_amount_30d"),
        avg("claimed_amount_eur").over(w_30d).alias("avg_amount_30d"),
        max("claimed_amount_eur").over(w_30d).alias("max_amount_30d"),
        
        # Señal de riesgo: ¿Cuántos fraudes confirmados ha tenido esta póliza en el último mes?
        coalesce(
            sum(when(col("is_fraud") == 1, 1).otherwise(0)).over(w_30d), lit(0)
        ).alias("num_fraud_confirmed_30d")
    )

    # Ratio de comportamiento: Compara la severidad de la última semana vs el promedio mensual
    # Si no hay historial, se imputa 1.0 (comportamiento normal/conservador)
    return df_agg.withColumn(
        "claims_7d_vs_avg_30d_ratio",
        coalesce(
            col("sum_amount_7d") / (col("avg_amount_30d") + lit(EPSILON)),
            lit(1.0)
        )
    )

###############################################################################
# 4. Tabla de Entrenamiento (Training Feature Table)
###############################################################################

@dp.table(
    name = "gold_policy_aggregations",
    comment = "Características de comportamiento para ENTRENAMIENTO. Usa datos de la capa Silver.",
    table_properties = gold_table_properties,
    schema = gold_aggregations_schema_template.format(pk_name="gold_policy_agg_pk")
)
def gold_policy_aggregations():
    """
    Lee de Silver para garantizar que las etiquetas de fraude son correctas
    y los datos han pasado los filtros de calidad durante el entrenamiento.
    """
    df_silver = spark.read.table(training_source)
    
    # Preparamos el timestamp en milisegundos para las ventanas
    df = df_silver.withColumn("ts_ms", (col("timestamp").cast("double") * 1000).cast("long"))
    
    return _compute_aggregations(df)

###############################################################################
# 5. Tabla de Inferencia (Inference Feature Table)
###############################################################################

@dp.table(
    name = "gold_policy_aggregations_inference",
    comment = "Características de comportamiento para INFERENCIA. Usa datos inmediatos de Bronze.",
    table_properties = gold_table_properties,
    schema = gold_aggregations_schema_template.format(pk_name="gold_policy_agg_inf_pk")
)
def gold_policy_aggregations_inference():
    """
    Lee directamente de Bronze para que el modelo tenga los datos de siniestralidad 
    actualizados al segundo, sin esperar al procesamiento de la capa Silver.
    """
    # Cargamos siniestros y etiquetas de Bronze
    df_claims = spark.read.table(inference_claims_source).withColumn("timestamp", to_timestamp(col("timestamp")))
    df_labels = spark.read.table(inference_labels_source).select("claim_id", "is_fraud")
    
    # Unión externa para obtener etiquetas confirmadas (si existen)
    # Si la etiqueta aún no ha llegado, asumimos 0 (no fraude confirmado aún)
    df_combined = df_claims.join(df_labels, on="claim_id", how="left") \
                           .withColumn("is_fraud", coalesce(col("is_fraud"), lit(0)))
    
    # Preparamos el timestamp en milisegundos
    df = df_combined.withColumn("ts_ms", (col("timestamp").cast("double") * 1000).cast("long"))
    
    return _compute_aggregations(df)
"""
Este script construye la tabla 'Spine' (Columna Vertebral) para el modelo de ML.
Contiene los identificadores primarios, las marcas de tiempo, la variable objetivo 
(is_fraud) y las características en tiempo real disponibles en el parte de accidente.
"""

###############################################################################
# Imports
###############################################################################
import pyspark.pipelines as dp
from pyspark.sql.functions import col, date_format, to_timestamp

###############################################################################
# Configuración y Constantes
###############################################################################

silver_enriched_source = "silver_enriched_fraud"
gold_spine_table_name = "gold_fraud_spine"

gold_spine_comment = """
Esta tabla actúa como el Spine DataFrame para el entrenamiento del modelo.
Contiene las claves (policy_id, claim_id), el target (is_fraud) y los features
del momento del siniestro (accident_type, claimed_amount, etc.).
"""

###############################################################################
# Feature Engineering: Creación del Spine
###############################################################################

@dp.table(
    name = gold_spine_table_name, 
    comment = gold_spine_comment
)
def gold_fraud_spine():
    """
    Construye el spine de ML usando el nombre de columna correcto: label_available_timestamp.
    """
    # Leemos de Silver
    df_events = spark.readStream.table(silver_enriched_source)

    # 1. Aseguramos el tipo del timestamp principal
    # 2. Corregimos el nombre de la columna de fecha de etiqueta
    df_events = df_events.withColumn("timestamp", col("timestamp").cast("timestamp")) \
                         .withColumn("label_available_timestamp", 
                                     to_timestamp(date_format(col("label_available_timestamp"), "yyyy-MM-dd HH:mm:ss.SSS")))

    # Seleccionamos los campos para el Spine usando el nombre sugerido por Spark
    df_spine = df_events.select(
        col("claim_id"),
        col("policy_id"),
        col("timestamp"),

        col("is_fraud"),
        col("label_available_timestamp"),  

        col("accident_type"),
        col("claimed_amount_eur"),
        col("n_parties_involved"),
        col("telematics_anomaly"),
        col("police_report_filed"),
        col("outside_business_hours")
    )

    

    return df_spine

###############################################################################
# Inferencia Spine 
###############################################################################

# Fuente para la inferencia: leemos directamente de bronze para evitar retrasos
inference_events_source = "bronze_claims" 

gold_inference_spine_table_name = "gold_fraud_inference_spine"
gold_inference_spine_comment = """
Esta tabla actúa como el Spine DataFrame para la INFERENCIA en producción.
Lee directamente de bronze_claims para garantizar que cada nuevo siniestro
pueda ser puntuado por el modelo inmediatamente, sin esperar a que lleguen las
etiquetas de fraude o pasen los filtros de calidad de silver.
"""

@dp.table(
    name = gold_inference_spine_table_name, 
    comment = gold_inference_spine_comment
)
def gold_fraud_inference_spine():
    """
    Construye el spine para producción. Excluimos las etiquetas (is_fraud) 
    porque no están disponibles en el momento en que ocurre el accidente.
    """
    return (
        spark.readStream
             .table(inference_events_source)
             .withColumn("timestamp", to_timestamp(col("timestamp")))
             .select(
                 # Claves primarias y tiempo del evento
                 col("claim_id"),
                 col("policy_id"),
                 col("timestamp"),

                 # Características en tiempo real (disponibles en el parte)
                 col("accident_type"),
                 col("claimed_amount_eur"),
                 col("n_parties_involved"),
                 col("telematics_anomaly"),
                 col("police_report_filed"),
                 col("outside_business_hours")
             )
    )
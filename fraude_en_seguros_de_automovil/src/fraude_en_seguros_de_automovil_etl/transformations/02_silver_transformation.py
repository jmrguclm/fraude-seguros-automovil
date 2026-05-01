"""
CAPA SILVER: Refinamiento, Calidad y Enriquecimiento.
Este script implementa:
1. Cuarentena (Dead Letter Queue) para registros inválidos.
2. SCD Tipo 2 (AUTO CDC) para el historial de pólizas.
3. Stream-Stream Join con Watermarks para unir Siniestros y Fraude.
"""

import pyspark.pipelines as dp
from pyspark.sql.functions import col, expr, to_timestamp
from rules import get_rules

###############################################################################
# 1. DOMINIO: PÓLIZAS (Maestro de Asegurados - SCD Tipo 2)
###############################################################################

# Configuración de reglas y expresiones de cuarentena
pol_rules_dict = get_rules("policies")
pol_combined_rules = " AND ".join(pol_rules_dict.values())
pol_quarantine_expr = f"NOT ({pol_combined_rules})"

# Tabla física de Cuarentena para Pólizas
dp.create_streaming_table(
    name = "silver_quarantine_policies",
    comment = "DLQ: Pólizas que no cumplen reglas de integridad o negocio."
)

@dp.table(name = "tmp_eval_policies", temporary = True)
@dp.expect_all(pol_rules_dict)
def eval_policies():
    """Evalúa reglas y añade flag de cuarentena a las pólizas."""
    return (spark.readStream
            .option("skipChangeCommits", "true")
            .table("bronze_policies")
            .withColumn("is_quarantined", expr(pol_quarantine_expr)))

@dp.append_flow(target = "silver_quarantine_policies")
def quarantine_policies():
    """Enruta registros inválidos a la tabla de cuarentena."""
    return spark.readStream.table("tmp_eval_policies").filter("is_quarantined = true").drop("is_quarantined")

@dp.view(name = "vw_clean_policies")
def clean_policies():
    """Vista de paso para registros válidos."""
    return spark.readStream.table("tmp_eval_policies").filter("is_quarantined = false").drop("is_quarantined")

# Tabla de Historial con SCD Tipo 2 (AUTO CDC)
dp.create_streaming_table(
    name = "silver_policies_history",
    comment = "Histórico de versiones de pólizas (SCD2) para point-in-time correctness."
)

dp.create_auto_cdc_flow(
    target = "silver_policies_history",
    source = "vw_clean_policies",
    keys = ["policy_id"],
    sequence_by = col("policy_updated_at"),
    # Quitamos _rescued_data de aquí porque no existe en la fuente
    except_column_list = ["ingestion_timestamp", "source_file"], 
    stored_as_scd_type = "2"
)

###############################################################################
# 2. DOMINIO: SINIESTROS (Claims - Eventos en Streaming)
###############################################################################

cla_rules_dict = get_rules("claims")
cla_combined_rules = " AND ".join(cla_rules_dict.values())
cla_quarantine_expr = f"NOT ({cla_combined_rules})"

dp.create_streaming_table(
    name = "silver_quarantine_claims",
    comment = "DLQ: Partes de accidente con datos erróneos o incompletos."
)

@dp.table(name = "tmp_eval_claims", temporary = True)
@dp.expect_all(cla_rules_dict)
def eval_claims():
    """Tipado de fechas y evaluación de calidad para siniestros."""
    return (spark.readStream.table("bronze_claims")
            .withColumn("incident_timestamp", to_timestamp(col("timestamp"))) 
            .withColumn("is_quarantined", expr(cla_quarantine_expr)))

@dp.append_flow(target = "silver_quarantine_claims")
def quarantine_claims():
    return spark.readStream.table("tmp_eval_claims").filter("is_quarantined = true").drop("is_quarantined")

@dp.view(name = "vw_clean_claims")
def clean_claims():
    return spark.readStream.table("tmp_eval_claims").filter("is_quarantined = false").drop("is_quarantined")

###############################################################################
# 3. DOMINIO: ETIQUETAS (Labels - Feedback de Fraude)
###############################################################################

lbl_rules_dict = get_rules("labels")
lbl_combined_rules = " AND ".join(lbl_rules_dict.values())
lbl_quarantine_expr = f"NOT ({lbl_combined_rules})"

dp.create_streaming_table(
    name = "silver_quarantine_labels",
    comment = "DLQ: Etiquetas de fraude que no pueden vincularse o son inválidas."
)

@dp.table(name = "tmp_eval_labels", temporary = True)
@dp.expect_all(lbl_rules_dict)
def eval_labels():
    return (spark.readStream.table("bronze_labels")
            .withColumn("label_available_timestamp", to_timestamp(col("label_available_date")))
            .withColumn("is_quarantined", expr(lbl_quarantine_expr)))

@dp.append_flow(target = "silver_quarantine_labels")
def quarantine_labels():
    return spark.readStream.table("tmp_eval_labels").filter("is_quarantined = true").drop("is_quarantined")

@dp.view(name = "vw_clean_labels")
def clean_labels():
    return spark.readStream.table("tmp_eval_labels").filter("is_quarantined = false").drop("is_quarantined")

###############################################################################
# 4. UNIFICACIÓN: Stream-Stream Join (Claims + Labels)
###############################################################################

# Un siniestro puede tardar hasta 31 días en ser investigado y etiquetado
claim_watermark_delay = "31 days"
label_watermark_delay = "1 day"



@dp.table(
    name = "silver_enriched_fraud",
    comment = "Núcleo de la capa Silver: Siniestros enriquecidos con su etiqueta de fraude."
)
def silver_events_join():
    """
    Realiza un Left Join entre Siniestros y Etiquetas manejando el retardo 
    de la investigación mediante Watermarks.
    """
    df_cla = spark.readStream.table("vw_clean_claims").withWatermark("incident_timestamp", claim_watermark_delay)
    df_lbl = spark.readStream.table("vw_clean_labels").withWatermark("label_available_timestamp", label_watermark_delay)

    join_condition = [
        col("cla.claim_id") == col("lbl.claim_id"),
        col("lbl.label_available_timestamp") >= col("cla.incident_timestamp"),
        col("lbl.label_available_timestamp") <= col("cla.incident_timestamp") + expr(f"INTERVAL {claim_watermark_delay}")
    ]

    # Seleccionamos columnas de negocio finales para la capa Gold / ML
    return (df_cla.alias("cla")
            .join(df_lbl.alias("lbl"), on = join_condition, how = "leftOuter")
            .select(
                col("cla.*"),
                col("lbl.is_fraud"),
                col("lbl.label_available_timestamp")
            ))
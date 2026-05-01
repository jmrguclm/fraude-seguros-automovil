"""
Arquitectura Medallion - Capa Bronze
------------------------------------
Este script realiza la ingesta declarativa desde la landing_zone hacia tablas 
Delta gestionadas. Implementa:
1. Ingesta Batch para datos maestros (Pólizas).
2. Ingesta Streaming (Auto Loader) para eventos (Siniestros y Etiquetas).
"""

###############################################################################
# 1. Importaciones
###############################################################################
from pathlib import Path
import pyspark.pipelines as dp
from pyspark.sql.functions import col, current_timestamp

###############################################################################
# 2. Configuración de rutas
###############################################################################
base_path = Path("/") / "Volumes" / "workspace" / "fraude-seguros-automovil"
vol_landing_zone = base_path / "landing_zone"

###############################################################################
# 3. Ingesta de Pólizas (Batch)
###############################################################################

policies_table_name = "bronze_policies"
policies_comment = """
Tabla de la capa bronze que almacena el maestro de pólizas y vehículos.
Carga: Batch (Full Refresh).
Origen: context/policies.csv
"""

path_context = vol_landing_zone / "context"

@dp.table(name = policies_table_name, comment = policies_comment)
def bronze_policies():
    """
    Lectura tradicional (Batch) de las pólizas. Al ser datos maestros,
    se sobrescriben en cada ejecución para reflejar el estado actual.
    """
    return (
        spark.read
             .format("csv")
             .option("header", "true")
             .option("inferSchema", "true")
             .load(str(path_context))
             # Metadatos de auditoría
             .withColumn("ingestion_timestamp", current_timestamp())
             .withColumn("source_file", col("_metadata.file_path"))
    )

###############################################################################
# 4. Ingesta de Siniestros / Claims (Streaming)
###############################################################################

claims_table_name = "bronze_claims"
claims_comment = """
Repositorio de eventos crudos de siniestros reportados.
Carga: Streaming (Auto Loader).
Origen: events/claims/
"""

path_events_claims = vol_landing_zone / "events" / "claims"

@dp.table(name = claims_table_name, comment = claims_comment)
def bronze_claims_flow():
    """
    Uso de Auto Loader (cloudFiles) para detectar nuevos ficheros JSON
    de siniestros de forma incremental.
    """
    return (
        spark.readStream
             .format("cloudFiles")
             .option("cloudFiles.format", "json")
             .option("cloudFiles.inferColumnTypes", "true")
             .load(str(path_events_claims))
             # Metadatos de auditoría
             .withColumn("ingestion_timestamp", current_timestamp())
             .withColumn("source_file", col("_metadata.file_path"))
    )

###############################################################################
# 5. Ingesta de Etiquetas / Labels (Streaming)
###############################################################################

labels_table_name = "bronze_labels"
labels_comment = """
Captura el feedback diferido (fraude confirmado) de cada siniestro.
Carga: Streaming (Auto Loader).
Origen: events/labels/
"""

path_events_labels = vol_landing_zone / "events" / "labels"

@dp.table(name = labels_table_name, comment = labels_comment)
def bronze_labels_flow():
    """
    Ingesta incremental de las etiquetas de fraude conforme el perito
    las confirma en el sistema.
    """
    return (
        spark.readStream
             .format("cloudFiles")
             .option("cloudFiles.format", "json")
             .option("cloudFiles.inferColumnTypes", "true")
             .load(str(path_events_labels))
             # Metadatos de auditoría
             .withColumn("ingestion_timestamp", current_timestamp())
             .withColumn("source_file", col("_metadata.file_path"))
    )
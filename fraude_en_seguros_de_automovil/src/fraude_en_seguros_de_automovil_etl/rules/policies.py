"""
Este módulo contiene las expectativas de integridad y negocio para los
datos maestros de pólizas y asegurados.

Las reglas están divididas por dominios (identidad, demografía y contrato)
para facilitar su auditoría en el Catalog Explorer de Unity Catalog.
"""

###############################################################################
# Functions
###############################################################################

def get_identity_rules():
    """
    Reglas para asegurar la presencia de identificadores y claves temporales.
    Fundamentales para la correcta ejecución del flujo AUTO CDC (SCD Tipo 2).
    """
    return [
        {"name": "valid_policy_id", "constraint": "policy_id IS NOT NULL", "tag": "policies"},
        {"name": "valid_updated_at", "constraint": "policy_updated_at IS NOT NULL", "tag": "policies"}
    ]

def get_demographic_rules():
    """
    Validaciones del perfil del asegurado. 
    Ajustado a categorías: urban, suburban, rural.
    """
    return [
        {"name": "valid_age", "constraint": "policyholder_age >= 18", "tag": "policies"},
        {"name": "valid_gender", "constraint": "gender IS NOT NULL", "tag": "policies"}, 
        {"name": "valid_region", "constraint": "TRIM(region_type) IN ('urban', 'suburban', 'rural')", "tag": "policies"}
    ]

def get_contract_rules():
    """
    Reglas de negocio relativas al riesgo y las características del vehículo.
    Asegura la coherencia económica y técnica de la póliza.
    """
    return [
        {"name": "valid_premium", "constraint": "annual_premium_eur > 0", "tag": "policies"},
        {"name": "valid_vehicle_year", "constraint": "vehicle_year > 1900 AND vehicle_year <= 2026", "tag": "policies"},
        {"name": "valid_telematics_flag", "constraint": "has_telematics IN (0, 1)", "tag": "policies"}
    ]

def get_policy_rules():
    """
    Punto de entrada principal para las reglas de calidad de pólizas.
    Agrupa todos los dominios lógicos en una lista única de diccionarios.
    """
    rules = []
    rules.extend(get_identity_rules())
    rules.extend(get_demographic_rules())
    rules.extend(get_contract_rules())
    
    return rules
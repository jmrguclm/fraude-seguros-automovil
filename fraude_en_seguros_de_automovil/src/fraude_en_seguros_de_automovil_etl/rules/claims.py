"""
Este módulo contiene las expectativas de integridad y negocio para el
flujo de siniestros (claims) de automóviles.

Las reglas aseguran la calidad de los datos de los incidentes y las señales
de telemetría antes de realizar el cruce con las etiquetas de fraude.
"""

###############################################################################
# Functions
###############################################################################

def get_identity_and_time_rules():
    """
    Reglas para asegurar que los identificadores primarios, claves foráneas 
    y marcas temporales están presentes para los joins de la capa Silver.
    """
    return [
        {"name": "valid_claim_id", "constraint": "claim_id IS NOT NULL", "tag": "claims"},
        {"name": "valid_policy_id", "constraint": "policy_id IS NOT NULL", "tag": "claims"},
        {"name": "valid_incident_date", "constraint": "timestamp IS NOT NULL", "tag": "claims"}
    ]

def get_incident_details_rules():
    """
    Reglas para validar los valores monetarios y categóricos del siniestro.
    Actualizado con las categorías reales detectadas en el Bronce.
    """
    return [
        {"name": "valid_amount", "constraint": "claimed_amount_eur >= 0", "tag": "claims"},
        {
            "name": "valid_incident_type", 
            "constraint": """accident_type IN (
                'hit_and_run', 'animal_collision', 'single_vehicle', 'theft', 
                'parking', 'side_collision', 'rollover', 'hail_damage', 
                'vandalism', 'rear_end'
            )""", 
            "tag": "claims"
        },
        {"name": "valid_parties", "constraint": "n_parties_involved >= 1", "tag": "claims"}
    ]
    
def get_security_rules():
    """
    Reglas para validar indicadores de seguridad, fraude potencial y 
    consistencia de los informes policiales.
    """
    return [
        {"name": "valid_telematics_anomaly", "constraint": "telematics_anomaly IN (0, 1)", "tag": "claims"},
        {"name": "valid_police_report", "constraint": "police_report_filed IN (0, 1)", "tag": "claims"},
        {"name": "valid_business_hours", "constraint": "outside_business_hours IN (0, 1)", "tag": "claims"}
    ]

def get_claim_rules():
    """
    Punto de entrada principal para las reglas de calidad de siniestros.
    Agrupa todos los dominios específicos en una lista consolidada.
    """
    all_claim_rules = []
    all_claim_rules.extend(get_identity_and_time_rules())
    all_claim_rules.extend(get_incident_details_rules())
    all_claim_rules.extend(get_security_rules())
    
    return all_claim_rules
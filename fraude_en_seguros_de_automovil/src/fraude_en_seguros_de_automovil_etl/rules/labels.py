"""
Este módulo contiene las expectativas de integridad y negocio para el
feedback de confirmación de fraude (etiquetas).

Asegura que las etiquetas puedan unirse correctamente con los siniestros
originales y que los indicadores de fraude sean válidos.
"""

###############################################################################
# Functions
###############################################################################

def get_identity_rules():
    """
    Reglas para asegurar que el identificador primario está presente.
    Sin un ID de siniestro válido, la etiqueta no puede cruzarse con los datos.
    """
    return [
        {
            "name": "valid_lbl_claim_id",
            "constraint": "claim_id IS NOT NULL",
            "tag": "labels"
        }
    ]


def get_feedback_integrity_rules():
    """
    Reglas para validar el contenido del feedback.
    Asegura que el indicador de fraude sea binario y que la fecha exista.
    """
    return [
        {
            "name": "valid_fraud_flag",
            "constraint": "is_fraud IS NOT NULL AND is_fraud IN (0, 1)",
            "tag": "labels"
        },
        {
            "name": "valid_lbl_date",
            "constraint": "label_available_date IS NOT NULL", 
            "tag": "labels"
        }
    ]


def get_label_rules():
    """
    Punto de entrada principal para las reglas de calidad de etiquetas.
    Agrupa todas las reglas específicas en una sola lista.
    """
    all_label_rules = []
    all_label_rules.extend(get_identity_rules())
    all_label_rules.extend(get_feedback_integrity_rules())
    
    return all_label_rules
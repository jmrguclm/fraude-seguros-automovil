"""
Shared utilities for the production inference and label enrichment pipeline.
"""


###############################################################################
# Imports
###############################################################################

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup


###############################################################################
# Table configuration
###############################################################################

spine_table = f"{catalog}.{database}.gold_fraud_inference_spine"
policy_profile_table = f"{catalog}.{database}.gold_policy_profile"
policy_agg_table = f"{catalog}.{database}.gold_policy_aggregations_inference"
inference_enriched_table = f"{catalog}.{database}.gold_fraud_inference_enriched"
fraud_labels_table = f"{catalog}.{database}.bronze_labels"


###############################################################################
# Feature store configuration
###############################################################################

entity_key = "policy_id"
timestamp_key = "timestamp"

# Static or slowly-changing customer profile features.
# Must match exactly the feature_names used in 05_Training_Dataset_Generation
# to guarantee that the enrichment is identical to training.
profile_feature_names = [
    # Demografía del Asegurado 
    "age_group",
    "vehicle_segment",
    "risk_level_by_premium",
    "driver_experience_years", 
    "is_new_policy_risk",
    "gender",
    "region_type",
    "coverage_type",           
    "vehicle_type"             
]

# Behavioral aggregations over rolling windows.
aggregation_feature_names = [
    # Ventana 24 horas
    "count_claims_24h",
    "sum_amount_24h",
    
    # Ventana 7 días
    "count_claims_7d",
    "sum_amount_7d",
    "avg_amount_7d",
    
    # Ventana 30 días
    "count_claims_30d",
    "sum_amount_30d",
    "avg_amount_30d",
    "max_amount_30d",
    "num_fraud_confirmed_30d",
    
    # Ratios e indicadores
    "claims_7d_vs_avg_30d_ratio"
]

profile_lookup = FeatureLookup(
    table_name = policy_profile_table,
    feature_names = profile_feature_names,
    lookup_key = entity_key,
    timestamp_lookup_key = timestamp_key
)

aggregations_lookup = FeatureLookup(
    table_name = policy_agg_table,
    feature_names = aggregation_feature_names,
    lookup_key = entity_key,
    timestamp_lookup_key = timestamp_key
)

feature_lookups = [profile_lookup, aggregations_lookup]

print(f"Profile features ({len(profile_feature_names)}): {profile_feature_names}")
print(f"Aggregation features ({len(aggregation_feature_names)}): {aggregation_feature_names}")
print(f"Total feature columns: {len(profile_feature_names) + len(aggregation_feature_names)}")
print()

print("09_Utils.py script loaded successfully.")

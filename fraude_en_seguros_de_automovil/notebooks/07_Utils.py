"""
Shared utilities for the experimentation in fraud detection project.
"""


###############################################################################
# Imports
###############################################################################

import os
from datetime import datetime, timedelta
from pathlib import Path

from dateutil.relativedelta import relativedelta

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
matplotlib.use("Agg")  # Non-interactive backend: safe on cluster drivers with no display

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.feature import VarianceThresholdSelectorModel
from pyspark.sql import functions as F

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


###############################################################################
# Data and infrastructure
###############################################################################

CATALOG = "workspace"
DATABASE = "fraude-seguros-automovil"
TRAINING_TABLE = f"{CATALOG}.`{DATABASE}`.gold_fraud_training_dataset"

catalog = CATALOG
database = DATABASE
training_table = TRAINING_TABLE

uc_volume_path = Path("/") / "Volumes" / CATALOG / DATABASE / "ml_artifacts"

# Required for logging models on serverless clusters:
# without a volume path, serialization has nowhere to write
os.environ["MLFLOW_DFS_TMP"] = str(uc_volume_path)


###############################################################################
# Column configuration
###############################################################################

LABEL_COLUMN = "is_fraud"
CLASS_WEIGHT_COLUMN = "class_weight"
FEATURES_COLUMN = "features_scaled"
DATE_COLUMN = "timestamp"
LABEL_AVAILABLE_DATE_COLUMN = "label_available_date"
TRANSACTION_ID_COLUMN = "claim_id"

label_column = LABEL_COLUMN
class_weight_column = CLASS_WEIGHT_COLUMN
features_column = FEATURES_COLUMN
date_column = DATE_COLUMN
label_available_date_column = LABEL_AVAILABLE_DATE_COLUMN
transaction_id_column = TRANSACTION_ID_COLUMN


###############################################################################
# Temporal split configuration
###############################################################################

# Test window: everything added after ml.data_previous_max_date of the previous
# cycle; duration varies depending on when retraining is triggered.
# Validation window: VALIDATION_WINDOW_MONTHS months immediately before the test window.
# Training window: TRAINING_WINDOW_MONTHS immediately before the validation window;
# capped to avoid stale fraud patterns degrading performance.
# On the first version of the table, the split falls back to the defaults.
TRAINING_WINDOW_MONTHS = 36
VALIDATION_WINDOW_MONTHS = 12

# Random seed: shared so run tags built in both notebooks are identical
seed = 45127


###############################################################################
# Project metadata
###############################################################################

# Shared across all notebooks and used as tags throughout
current_user = spark.sql("SELECT current_user()").collect()[0][0]
project = "fraude-seguros-automovil_detection"
team = "ml_engineering"
task = "binary_classification"
environment = "development"
algorithm_family = "logistic_regression"
framework = "pyspark.ml"

uc_model_name = f"{catalog}.{database}.fraud_lr_pipeline"

# The experiment is always co-located with the assets of this project
mlflow_experiment_name = "fraud_detection_training"
mlflow_experiment_path = str(
    Path("/") / "Workspace" / "Users" / current_user / ".experiments" / database / mlflow_experiment_name
)


###############################################################################
# Raw data load
###############################################################################

df_raw = spark.table(training_table)

print(f"Total rows: {df_raw.count():,}")
print(f"Total columns: {len(df_raw.columns)}")
print()


###############################################################################
# Temporal split and inverse-frequency class weights
###############################################################################

# Read the table properties of the latest version to determine the split strategy.
# ml.delta_semantic_version drives the branching logic: version 0 uses the fixed
# seminar splits, while any subsequent version uses the rolling window anchored
# to ml.data_previous_max_date persisted by the data generation notebook.
properties_df = spark.sql(f"SHOW TBLPROPERTIES {training_table}")
semantic_version_row = properties_df.filter("key = 'ml.delta_semantic_version'").first()
delta_semantic_version = int(semantic_version_row["value"]) if semantic_version_row else 0

if delta_semantic_version == 0:
    # First version of the table: training from 2020 to 2022, validation 2023, and test 2024 (reserved)
    train_end = datetime(2022, 12, 31)
    validation_end = datetime(2023, 12, 31)
else:
    # Subsequent versions: rolling window anchored to the previous cycle maximum date.
    # ml.data_previous_max_date is the maximum timestamp of the dataset before the last
    # overwrite, persisted in the table properties by the data generation notebook.
    # Everything after that date in the current table becomes the test window.
    # Validation is the VALIDATION_WINDOW_MONTHS immediately before that cutoff.
    # Training is the TRAINING_WINDOW_MONTHS immediately before validation.
    previous_max_date_row = properties_df.filter("key = 'ml.data_previous_max_date'").first()
    validation_end = datetime.strptime(previous_max_date_row["value"], "%Y-%m-%d")
    train_end = validation_end - relativedelta(months = VALIDATION_WINDOW_MONTHS)

train_start = train_end - relativedelta(months = TRAINING_WINDOW_MONTHS)

train_start_date = train_start.strftime("%Y-%m-%d")
train_end_date = train_end.strftime("%Y-%m-%d")
validation_start_date = (train_end + timedelta(days=1)).strftime("%Y-%m-%d")
validation_end_date = validation_end.strftime("%Y-%m-%d")
test_start_date = (validation_end + timedelta(days=1)).strftime("%Y-%m-%d")
test_end_date = df_raw.agg(
    F.max(F.col(date_column)).alias("max_date")
).collect()[0]["max_date"].strftime("%Y-%m-%d")

train_df = df_raw.filter(
    (F.col(date_column) >= train_start_date) & (F.col(date_column) <= train_end_date)
)
validation_df = df_raw.filter(
    (F.col(date_column) >= validation_start_date) & (F.col(date_column) <= validation_end_date)
)
test_df = df_raw.filter(
    (F.col(date_column) >= test_start_date) & (F.col(date_column) <= test_end_date)
)

print(f"Semantic version: {delta_semantic_version}")
print()
print(f"Train period: {train_start_date} → {train_end_date}")
print(f"Validation period: {validation_start_date} → {validation_end_date}")
print(f"Test period: {test_start_date} → {test_end_date}")
print()
print(f"Train rows: {train_df.count():,}")
print(f"Validation rows: {validation_df.count():,}")
print(f"Test rows: {test_df.count():,}")
print()


def apply_class_weights(df):
    """
    Calculates and applies inverse-frequency class weights dynamically
    based on the exact distribution of the provided `DataFrame`.
    """
    n_total = df.count()
    n_fraud = df.filter(F.col(label_column) == 1).count()
    n_legit = n_total - n_fraud

    pct_fraud = 100 * n_fraud / n_total if n_total > 0 else 0.0
    pct_legit = 100 * n_legit / n_total if n_total > 0 else 0.0

    weight_fraud = n_total / (2.0 * n_fraud) if n_fraud > 0 else 1.0
    weight_legit = n_total / (2.0 * n_legit) if n_legit > 0 else 1.0

    print(f"Total rows for training: {n_total:,}")
    print(f"Fraud: {n_fraud:,} ({pct_fraud:.2f}%), weight = {weight_fraud:.2f}")
    print(f"Legit: {n_legit:,} ({pct_legit:.2f}%), weight = {weight_legit:.2f}")
    print()

    weighted_df = df.withColumn(
        class_weight_column,
        F.when(F.col(label_column) == 1.0, weight_fraud).otherwise(weight_legit)
    )

    return weighted_df


###############################################################################
# Column classification
###############################################################################

# Type sets used to route each field to the correct pipeline stage
numeric_types = {"IntegerType", "LongType", "FloatType", "DoubleType", "DecimalType"}
boolean_types = {"BooleanType"}
categorical_types = {"StringType"}

# Binary security flags stored as integers: treated as boolean features
binary_flag_columns = [
    "has_telematics",
    "is_new_policy_risk",
    "police_report_filed",
    "telematics_anomaly",
    "outside_business_hours"
]

# The label, high-cardinality identifiers, and the temporal column
# are never included in the feature vector. Identifiers and the
# timestamp are excluded here rather than dropped in the pipeline
# so that they remain available after transform, enabling joins
# and traceability downstream.
exclude_columns = [
    label_column,  # is_fraud
    "claim_id",
    "policy_id",
    "timestamp"
]

numeric_columns = []
boolean_columns = []
categorical_columns = []

for field in df_raw.schema.fields:
    column_name = field.name
    type_name = type(field.dataType).__name__
    if column_name in exclude_columns:
        continue
    if column_name in binary_flag_columns:
        boolean_columns.append(column_name)
    elif type_name in numeric_types:
        numeric_columns.append(column_name)
    elif type_name in boolean_types:
        boolean_columns.append(column_name)
    elif type_name in categorical_types:
        categorical_columns.append(column_name)

print(f"Numeric ({len(numeric_columns)}): {numeric_columns}")
print(f"Boolean ({len(boolean_columns)}): {boolean_columns}")
print(f"Categorical ({len(categorical_columns)}): {categorical_columns}")
print()


###############################################################################
# Preprocessing configuration
###############################################################################

# Stage 1: imputation

# Count and sum aggregations initialize to 0 on empty windows (no imputation needed).
# Mean, maximum and minimum aggregations arrive as null on empty windows (median imputation needed).
agg_zero_prefixes = (
    "count_",
    "sum_",
    "count_cross_border_",
    "count_tor_vpn_",
    "count_3ds_failed_",
    "num_fraud_",
    "spend_",
    "distinct_"
)
agg_null_prefixes = ("avg_", "max_amount", "min_amount")

agg_zero_columns = [column for column in numeric_columns if column.startswith(agg_zero_prefixes)]
agg_null_columns = [column for column in numeric_columns if column.startswith(agg_null_prefixes)]
profile_numeric_columns = [
    column for column in numeric_columns
    if not column.startswith(agg_zero_prefixes) and not column.startswith(agg_null_prefixes)
]

imputer_input_columns = profile_numeric_columns + agg_null_columns
imputer_output_columns = [f"{column}_imp" for column in imputer_input_columns]

# Stage 2: boolean cast

# COALESCE handles nulls inline: a missing security flag is treated as disabled (0.0)
boolean_cast_expressions = ", ".join([
    f"COALESCE(CAST({c} AS DOUBLE), 0.0) AS {c}_dbl"
    for c in boolean_columns
])
boolean_output_columns = [f"{column}_dbl" for column in boolean_columns]
boolean_statement = f"SELECT *, {boolean_cast_expressions} FROM __THIS__"

# Stage 3: feature engineering

# Only features computable from the current transaction at inference time.
# Features requiring customer history must come from the feature store.
# Stage 3: feature engineering revisado
feature_engineering_statement = (
    "SELECT *, "
    # Usamos COALESCE para que si el resultado es NULL, ponga un 0
    "COALESCE(CAST((count_claims_30d > 1) AS INT), 0) AS is_frequent_claimant, "
    "COALESCE(CAST((is_new_policy_risk = 1 AND count_claims_30d > 0) AS INT), 0) AS is_immediate_fraud_risk, "
    # Para el logaritmo, si es nulo ponemos 0.0 (valor neutro)
    "COALESCE(LOG(claimed_amount_eur + 1), 0.0) AS log_premium, " 
    "COALESCE(CAST((claims_7d_vs_avg_30d_ratio > 1.2) AS INT), 0) AS is_activity_spike "
    "FROM __THIS__"
)


# Actualiza la lista de salida con los nuevos nombres
engineered_columns = [
    "is_frequent_claimant",
    "is_immediate_fraud_risk",
    "log_premium",
    "is_activity_spike"
]

# Stages 4 and 5: categorical encoding

string_indexer_input_columns = categorical_columns
string_indexer_output_columns = [f"{column}_idx" for column in categorical_columns]

ohe_input_columns = string_indexer_output_columns
ohe_output_columns = [f"{column}_ohe" for column in categorical_columns]

# Stage 6: vector assembly

assembler_input_columns = (
    imputer_output_columns
    + agg_zero_columns
    + boolean_output_columns
    + ohe_output_columns
    + engineered_columns
)
assembler_output_column = "features"

# Stages 7 and 8: variance threshold selection and standard scaling

var_selector_input_column = assembler_output_column
var_selector_output_column = "features_var_filtered"
scaler_input_column = var_selector_output_column
scaler_output_column = features_column

print(f"Assembler inputs ({len(assembler_input_columns)}): {assembler_input_columns}")
print()


###############################################################################
# Visualization
###############################################################################

# Threshold sweep grid: shared between the threshold visualization and the
# optimal-threshold search loop so the logged value matches the figure
threshold_sweep_start = 0.01
threshold_sweep_stop = 0.99
threshold_sweep_steps = 99

fig_size_standard = (6, 5)
fig_size_wide = (8, 5)
fig_size_confusion = (11, 4)
fig_size_coef_width = 9
fig_size_coef_row_h = 0.30
fig_size_coef_min_h = 4

color_roc = "#1f77b4"
color_pr = "#ff7f0e"
color_calibration = "#2ca02c"
color_positive_coef = "#d62728"
color_negative_coef = "#1f77b4"
color_random_baseline = "gray"

calibration_n_bins = 10
coefficients_top_n = 30


def save_diagnostic_figure(fig, directory_path, filename):
    """
    Saves a matplotlib figure to the specified directory and closes it to free memory.
    """
    file_path = Path(directory_path) / filename
    fig.savefig(file_path, dpi = 150, bbox_inches = "tight")
    plt.close("all")


def fig_pr_curve(y_true, y_prob, auc_pr, title):
    """
    Plot the PR curve with AUC annotation and a shaded area.
    Preferred over ROC for imbalanced datasets: the random baseline equals
    the positive class rate, not a fixed diagonal.
    """
    precision_values, recall_values, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize = fig_size_standard)
    ax.plot(recall_values, precision_values, lw = 2, color = color_pr, label = f"AUC-PR = {auc_pr:.4f}")
    ax.axhline(baseline, color = color_random_baseline, linestyle = '--', lw = 1, label = f"Random baseline = {baseline:.3f}")
    ax.fill_between(recall_values, precision_values, alpha = 0.08, color = color_pr)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha = 0.3)
    plt.tight_layout()
    return fig


def fig_roc_curve(y_true, y_prob, auc_roc, title):
    """
    Plot the ROC curve with AUC annotation and a shaded area under the curve.
    The random classifier diagonal (AUC = 0.5) is shown as a dashed baseline.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize = fig_size_standard)
    ax.plot(fpr, tpr, lw = 2, color = color_roc, label = f"AUC-ROC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw = 1, alpha = 0.5, label = "Random classifier")
    ax.fill_between(fpr, tpr, alpha = 0.08, color = color_roc)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc = "lower right")
    ax.grid(alpha = 0.3)
    plt.tight_layout()
    return fig


def fig_confusion_matrix(y_true, y_pred, title):
    """
    Plot the confusion matrix as two side-by-side panels: raw counts and
    row-normalized percentages. Row normalization reveals the per-class
    detection rate independently of class frequency.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis = 1, keepdims = True) * 100
    fig, axes = plt.subplots(1, 2, figsize = fig_size_confusion)
    for ax, data, fmt, subtitle in [
        (axes[0], cm, "d", "Counts"),
        (axes[1], cm_pct, ".1f", "Row %")
    ]:
        im = ax.imshow(data, cmap = "Blues")
        plt.colorbar(im, ax = ax)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    format(data[i, j], fmt),
                    ha = "center", va = "center",
                    color = "white" if data[i, j] > data.max() / 2 else "black",
                    fontsize = 12, fontweight = "bold"
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} — {subtitle}")
    plt.tight_layout()
    return fig


def fig_lr_coefficients(coef_array, feature_names, title):
    """
    Plot the top logistic regression coefficients sorted by absolute value.
    Red means positive coefficient (push toward fraud) and
    blue means negative (push toward legit).
    """
    coef = np.array(coef_array)
    n = min(coefficients_top_n, len(coef))
    idx = np.argsort(np.abs(coef))[-n:][::-1]
    fig_height = max(fig_size_coef_min_h, n * fig_size_coef_row_h)
    fig, ax = plt.subplots(figsize = (fig_size_coef_width, fig_height))
    colors = [color_positive_coef if c > 0 else color_negative_coef for c in coef[idx]]
    ax.barh(range(n), coef[idx], color = colors, edgecolor = "white", linewidth = 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize = 8)
    ax.axvline(0, color = "black", lw = 0.8)
    ax.set_xlabel("Coefficient value")
    ax.set_title(title)
    ax.legend(
        handles = [
            Patch(color = color_positive_coef, label = "Indicative of fraud"),
            Patch(color = color_negative_coef, label = "Indicative of legit")
        ],
        loc = "lower right",
        fontsize = 8
    )
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def fig_calibration_curve(y_true, y_prob, title):
    """
    Plot the calibration curve (reliability diagram). A perfectly calibrated
    model follows the diagonal. Deviations above it indicate under-confidence;
    below it, over-confidence.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins = calibration_n_bins)
    fig, ax = plt.subplots(figsize = fig_size_standard)
    ax.plot(mean_pred, frac_pos, "s-", lw = 2, color = color_calibration, label = "Model")
    ax.plot([0, 1], [0, 1], "k--", lw = 1, label = "Perfect calibration")
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha = 0.1, color = color_calibration)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha = 0.3)
    plt.tight_layout()
    return fig


def _compute_threshold_metrics(y_true, y_prob):
    """
    Internal helper to calculate precision, recall, and F1 across all thresholds.
    Ensures mathematical consistency between threshold searching and plotting.
    """
    thresholds = np.linspace(threshold_sweep_start, threshold_sweep_stop, threshold_sweep_steps)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return thresholds, precisions, recalls, f1s


def find_best_threshold(y_true, y_prob):
    """
    Sweeps through decision thresholds to find the one that maximizes F1-score.
    """
    thresholds, _, _, f1s = _compute_threshold_metrics(y_true, y_prob)
    best_idx = np.argmax(f1s)

    return thresholds[best_idx], f1s[best_idx]


def fig_threshold_sweep(y_true, y_prob, title):
    """
    Plot precision, recall, and F1-score across the full range of decision thresholds.
    The vertical dashed line marks the threshold that maximizes F1-score.
    """
    thresholds, precisions, recalls, f1s = _compute_threshold_metrics(y_true, y_prob)
    best_t = thresholds[np.argmax(f1s)]
    fig, ax = plt.subplots(figsize = fig_size_wide)
    ax.plot(thresholds, precisions, label = "Precision", color = color_pr)
    ax.plot(thresholds, recalls, label = "Recall", color = color_roc)
    ax.plot(thresholds, f1s, label = "F1", color = color_calibration, lw = 2)
    ax.axvline(best_t, color = color_random_baseline, linestyle = "--", lw = 1, label = f"Best F1 threshold = {best_t:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha = 0.3)
    plt.tight_layout()
    return fig


###############################################################################
# Evaluation
###############################################################################

raw_prediction_column = "rawPrediction"
prediction_column = "prediction"
probability_column = "probability"
prob_fraud_column = "prob_fraud"

# Hard limit for ".toPandas()" conversions. Bringing massive distributed datasets into
# the cluster driver's local memory for processing can cause severe out-of-memory crashes.
# This acts as a safety threshold.
to_pandas_max_rows = 200_000

eval_auc_roc = BinaryClassificationEvaluator(
    labelCol = label_column,
    rawPredictionCol = raw_prediction_column,
    metricName = "areaUnderROC"
)
eval_auc_pr = BinaryClassificationEvaluator(
    labelCol = label_column,
    rawPredictionCol = raw_prediction_column,
    metricName = "areaUnderPR"
)

eval_f1 = MulticlassClassificationEvaluator(
    labelCol = label_column,
    predictionCol = prediction_column,
    metricName = "f1"
)
eval_precision = MulticlassClassificationEvaluator(
    labelCol = label_column,
    predictionCol = prediction_column,
    metricName = "weightedPrecision"
)
eval_recall = MulticlassClassificationEvaluator(
    labelCol = label_column,
    predictionCol = prediction_column,
    metricName = "weightedRecall"
)
eval_accuracy = MulticlassClassificationEvaluator(
    labelCol = label_column,
    predictionCol = prediction_column,
    metricName = "accuracy"
)


def compute_metrics(predictions):
    """
    Compute all six evaluation metrics.

    Returns a plain dictionary so values can be returned seamlessly
    to the orchestrator notebook without any complex transformations.
    """
    return {
        "auc_roc": eval_auc_roc.evaluate(predictions),
        "auc_pr": eval_auc_pr.evaluate(predictions),
        "f1": eval_f1.evaluate(predictions),
        "precision": eval_precision.evaluate(predictions),
        "recall": eval_recall.evaluate(predictions),
        "accuracy": eval_accuracy.evaluate(predictions)
    }


def to_pandas_predictions(predictions):
    """
    Convert the prediction columns.

    Extracts the fraud probability from the probability vector at index 1
    and adds it as a plain float column. Limits rows to prevent out-of-memory
    errors on large datasets.
    """
    return (
        predictions
        .select(label_column, probability_column, prediction_column)
        .limit(to_pandas_max_rows)
        .toPandas()
        .assign(
            **{
                prob_fraud_column: lambda df: df[probability_column].apply(
                    lambda values: float(values[1])
            )}
        )
    )


def extract_feature_names(pipeline_model, sample_df):
    """
    Extracts the final feature names after `VectorAssembler` and `VarianceThresholdSelector`.
    Requires transforming a single row to evaluate the one-hot encoded vector expansions.
    """
    attrs = (
        pipeline_model
        .transform(sample_df.limit(1))
        .schema["features"]
        .metadata["ml_attr"]["attrs"]
    )

    all_attrs = (
        attrs.get("numeric", [])
        + attrs.get("binary", [])
        + attrs.get("nominal", [])
    )

    expanded_feature_names = [
        attr["name"]
        for attr in sorted(all_attrs, key = lambda x: x["idx"])
    ]

    var_selector_fitted = next(
        stage for stage in pipeline_model.stages
        if isinstance(stage, VarianceThresholdSelectorModel)
    )

    selected_feature_names = [
        expanded_feature_names[i]
        for i in var_selector_fitted.selectedFeatures
    ]

    return expanded_feature_names, selected_feature_names


print("07_Utils.py script loaded successfully.")

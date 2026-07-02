
import json
import os
import numpy as np
import pandas as pd

import src.conformal_prediction as cp
from src.data_loader import load_data
from src.models import BayesRegressor, energinet_model


# Set random seed for reproducibility
RANDOM_SEED = 89
np.random.seed(RANDOM_SEED)

WORKDIR = os.getcwd()
DST_FOLDER = os.path.join(WORKDIR, "predictions", "bart")
RESULTS_FOLDER = os.path.join(WORKDIR, "results", "model_summaries")
os.makedirs(DST_FOLDER, exist_ok=True)

# Set the date thresholds for splitting the data
train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"
zones = ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
targets = ["volbids", "FRR", "CM"]

job_parameters = [[zone, target] for zone in zones for target in targets]
MODEL_VARIANTS = 10

job_idx = int(os.environ.get("LSB_JOBINDEX", 1))
if job_idx < 1 or job_idx > len(job_parameters) * MODEL_VARIANTS:
    raise ValueError(
        f"LSB_JOBINDEX must be between 1 and {len(job_parameters) * MODEL_VARIANTS}, got {job_idx}"
    )

dataset_idx = (job_idx - 1) // MODEL_VARIANTS
model_idx = (job_idx - 1) % MODEL_VARIANTS

if dataset_idx < 0 or dataset_idx >= len(job_parameters):
    raise ValueError(f"Invalid dataset index derived from LSB_JOBINDEX: {dataset_idx}")

zone, target = job_parameters[dataset_idx]


# Split and scale the data
X_train, X_calibration, X_val, y_train, y_calibration, y_val, X_scaler, y_scaler = cp.train_cal_test_date_split(
    *load_data(zone, target), train_end, calibration_end, test_end, scaled=True
)

X_train_full = pd.concat([X_train, X_calibration], axis=0)
y_train_full = pd.concat([y_train, y_calibration], axis=0)

X_CP = pd.concat([X_calibration, X_val], axis=0)
y_CP = pd.concat([y_calibration, y_val], axis=0)

# Import feature selection
fs_path = os.path.join(RESULTS_FOLDER, "bart_feature_selection_summary.json")
if not os.path.exists(fs_path):
    raise FileNotFoundError(f"Feature selection summary not found at {fs_path}")

with open(fs_path, "r") as f:
    feature_selection_results = json.load(f)

key = f"{zone} | {target}"
selected_features_normal = (
    feature_selection_results.get("normal", {}).get(key, {}).get("optimal_features")
)
selected_features_studentT = (
    feature_selection_results.get("studentT", {}).get(key, {}).get("optimal_features")
)

# Fallback: if no selected features are present, use all features
if not selected_features_normal:
    selected_features_normal = list(X_train_full.columns)
if not selected_features_studentT:
    selected_features_studentT = list(X_train_full.columns)

# Energinet model uses first 20 features of the original ordering
energinet_features_full = list(X_train_full.columns[:20])
energinet_features_train = list(X_train.columns[:20])

# StudentT is default in BayesRegressor
if model_idx == 0:  # StudentT BART with selected features for native predictions
    model = BayesRegressor(seed=RANDOM_SEED)
    model.fit(X_train_full[selected_features_studentT], y_train_full)
    preds = model.predict(X_val[selected_features_studentT])
    output_path = os.path.join(DST_FOLDER, f"bart_predictions_native_{zone}_{target}.parquet")
elif model_idx == 1: # Energinet model with all original features for native predictions
    model = energinet_model(seed=RANDOM_SEED)
    model.fit(X_train_full[energinet_features_full], y_train_full)
    preds = model.predict(X_val[energinet_features_full])
    output_path = os.path.join(DST_FOLDER, f"energinet_predictions_native_{zone}_{target}.parquet")
elif model_idx == 2: # StudentT BART with all features for native predictions
    model = BayesRegressor(seed=RANDOM_SEED)
    model.fit(X_train_full, y_train_full)
    preds = model.predict(X_val)
    output_path = os.path.join(DST_FOLDER, f"bart_predictions_native_all_{zone}_{target}.parquet")
elif model_idx == 3: # Normal BART with all features for native predictions
    model = BayesRegressor(distribution="normal", seed=RANDOM_SEED)
    model.fit(X_train_full, y_train_full)
    preds = model.predict(X_val)
    output_path = os.path.join(DST_FOLDER, f"bart_predictions_native_all_Normal_{zone}_{target}.parquet")
elif model_idx == 4: # Normal BART with selected features for native predictions
    model = BayesRegressor(distribution="normal", seed=RANDOM_SEED)
    model.fit(X_train_full[selected_features_normal], y_train_full)
    preds = model.predict(X_val[selected_features_normal])
    output_path = os.path.join(DST_FOLDER, f"bart_predictions_native_Normal_{zone}_{target}.parquet")
elif model_idx == 5: # StudentT BART with selected features for CP predictions
    model = BayesRegressor(seed=RANDOM_SEED)
    model.fit(X_train[selected_features_studentT], y_train)
    preds = model.predict(X_CP[selected_features_studentT])
    output_path = os.path.join(DST_FOLDER, f"bart_predictions_CP_{zone}_{target}.parquet")
elif model_idx == 6: # Energinet model with all original features for CP predictions
    model = energinet_model(seed=RANDOM_SEED)
    model.fit(X_train[energinet_features_train], y_train)
    preds = model.predict(X_CP[energinet_features_train])
    output_path = os.path.join(DST_FOLDER, f"energinet_predictions_CP_{zone}_{target}.parquet")
elif model_idx == 7: # StudentT BART with all features for CP predictions
    model = BayesRegressor(seed=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_CP)
    output_path = os.path.join(DST_FOLDER, f"bart_predictions_CP_all_{zone}_{target}.parquet")
elif model_idx == 8: # Normal BART with all features for CP predictions
    model = BayesRegressor(distribution="normal", seed=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_CP)
    output_path = os.path.join(DST_FOLDER, f"bart_predictions_CP_all_Normal_{zone}_{target}.parquet")
elif model_idx == 9: # Normal BART with selected features for CP predictions
    model = BayesRegressor(distribution="normal", seed=RANDOM_SEED)
    model.fit(X_train[selected_features_normal], y_train)
    preds = model.predict(X_CP[selected_features_normal])
    output_path = os.path.join(DST_FOLDER, f"bart_predictions_CP_Normal_{zone}_{target}.parquet")
else:
    raise ValueError(f"Invalid model index derived from LSB_JOBINDEX: {model_idx}")


if hasattr(preds, "to_parquet"):
    preds.to_parquet(output_path)
else:
    # Try to coerce to DataFrame
    try:
        df_out = pd.DataFrame(preds)
        df_out.to_parquet(output_path)
    except Exception as e:
        raise RuntimeError(f"Unable to persist preds to parquet at {output_path}: {e}")
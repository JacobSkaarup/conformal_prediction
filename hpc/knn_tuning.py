import json
import os
import time

import src.conformal_prediction as cp
from src.data_loader import load_data
from src.model_tuning import knn_tuner

train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"

dst_folder = os.path.join(os.getcwd(), "results", "model_summaries")
os.makedirs(dst_folder, exist_ok=True)

k_grid = [1, 2, 3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 35, 40, 60, 80, 100, 120, 150, 200]
zones = ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
targets = ["volbids", "FRR", "CM"]

total_jobs = len(zones) * len(targets) * 2
job_idx = int(os.environ.get("LSB_JOBINDEX", 1)) - 1

zone = zones[job_idx // (len(targets) * 2)]
target = targets[(job_idx // 2) % len(targets)]
method = "forward" if job_idx % 2 == 0 else "backward"

X_train, X_calibration, X_test, y_train, y_calibration, y_test, X_scaler, y_scaler = cp.train_cal_test_date_split(
    *load_data(zone, target),
    train_end,
    calibration_end,
    test_end,
    scaled=True,
)

tuner = knn_tuner(X_train=X_train, y_train=y_train, val_share=0.2, k_grid=k_grid)

if method == "forward":
    optimal_features, optimal_k, holdout_mse = tuner.forward_selection(tolerance=0.02)
else:
    optimal_features, optimal_k, holdout_mse = tuner.backward_selection(tolerance=0.02)

job_payload = {
    "dataset_key": f"{zone} | {target}",
    "method": method,
    "summary": {
        "optimal_features": optimal_features,
        "optimal_params": {"n_neighbors": int(optimal_k) if optimal_k is not None else None},
        "holdout_mse": float(holdout_mse),
    },
}

job_file = os.path.join(dst_folder, f"knn_summary_job_{job_idx}.json")
with open(job_file, "w") as f:
    json.dump(job_payload, f, indent=4)

if job_idx == total_jobs - 1:
    while True:
        expected_files = [f"knn_summary_job_{i}.json" for i in range(total_jobs)]
        all_files = os.listdir(dst_folder)

        if all(file_name in all_files for file_name in expected_files):
            combined = {}
            for i in range(total_jobs):
                with open(os.path.join(dst_folder, f"knn_summary_job_{i}.json"), "r") as f:
                    payload = json.load(f)

                dataset_key = payload["dataset_key"]
                method_name = payload["method"]
                summary = payload["summary"]

                if dataset_key not in combined:
                    combined[dataset_key] = {}
                combined[dataset_key][method_name] = summary

            # Split combined results into separate forward/backward summaries
            forward_summary = {}
            backward_summary = {}
            for dataset_key, methods in combined.items():
                if "forward" in methods:
                    forward_summary[dataset_key] = methods["forward"]
                if "backward" in methods:
                    backward_summary[dataset_key] = methods["backward"]

            with open(os.path.join(dst_folder, "knn_summary_forward.json"), "w") as f:
                json.dump(forward_summary, f, indent=4)

            with open(os.path.join(dst_folder, "knn_summary_backward.json"), "w") as f:
                json.dump(backward_summary, f, indent=4)

            for file_name in expected_files:
                os.remove(os.path.join(dst_folder, file_name))
            break

        time.sleep(30)

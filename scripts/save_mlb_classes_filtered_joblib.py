#scripts/save_mlb_classes_filtered_joblib.py

import joblib
import numpy as np

# Load the filtered classes safely
classes = np.load("artifacts/mlb_classes_filtered.npy", allow_pickle=True)

# Save using joblib (recommended)
joblib.dump(classes, "artifacts/mlb_classes_filtered.joblib")

print("Saved mlb_classes_filtered.joblib with", len(classes), "classes")

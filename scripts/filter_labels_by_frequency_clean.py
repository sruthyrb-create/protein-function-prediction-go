import numpy as np
import joblib
from scipy import sparse

K = 20  # minimum frequency

# Load original artifacts
Y = sparse.load_npz("artifacts/y_sparse.npz")
mlb_classes = joblib.load("artifacts/mlb_classes.npy")  # REAL GO terms

# Count label frequency
label_counts = np.array(Y.sum(axis=0)).ravel()
mask = label_counts >= K

# Filter
Y_filtered = Y[:, mask]
mlb_classes_filtered = [mlb_classes[i] for i in np.where(mask)[0]]

# Save
sparse.save_npz("artifacts/y_sparse_filtered.npz", Y_filtered)
joblib.dump(mlb_classes_filtered, "artifacts/mlb_classes_filtered.joblib")

print("Filtered labels:", len(mlb_classes_filtered))
print("Example labels:", mlb_classes_filtered[:10])

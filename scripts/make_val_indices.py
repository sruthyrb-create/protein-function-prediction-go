#scripts/make_val_indices.py

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
import os

# Paths
Y_PATH = "artifacts/y_sparse.npz"
OUT_DIR = "outputs/baseline"

os.makedirs(OUT_DIR, exist_ok=True)

# Load full label matrix
Y = sparse.load_npz(Y_PATH)

n = Y.shape[0]
all_indices = np.arange(n)

# MUST match baseline split
train_idx, val_idx = train_test_split(
    all_indices,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

np.save(os.path.join(OUT_DIR, "train_indices.npy"), train_idx)
np.save(os.path.join(OUT_DIR, "val_indices.npy"), val_idx)

print("Train size:", len(train_idx))
print("Val size:", len(val_idx))

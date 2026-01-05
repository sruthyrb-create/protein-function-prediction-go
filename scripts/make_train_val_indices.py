import numpy as np
from sklearn.model_selection import train_test_split

# total proteins = rows in embeddings / y_sparse
N = 82404   # IMPORTANT: this must match esm_embeddings.shape[0]

indices = np.arange(N)

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

np.save("artifacts/train_indices.npy", train_idx)
np.save("artifacts/val_indices.npy", val_idx)

print("Saved indices")
print("Train:", train_idx.shape)
print("Val:", val_idx.shape)


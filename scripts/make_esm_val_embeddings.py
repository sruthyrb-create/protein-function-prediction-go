# scripts/make_esm_val_embeddings.py

import numpy as np

emb = np.load("artifacts/esm_embeddings.npy")
val_idx = np.load("artifacts/val_indices.npy")

np.save("artifacts/esm_embeddings_val.npy", emb[val_idx])
print("Saved:", emb[val_idx].shape)

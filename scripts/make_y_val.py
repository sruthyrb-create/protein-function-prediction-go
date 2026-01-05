import numpy as np
from scipy import sparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--y", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

Y = sparse.load_npz(args.y)
val_idx = np.load("artifacts/val_indices.npy")

Y_val = Y[val_idx]
sparse.save_npz(args.out, Y_val)

print("Y_val shape:", Y_val.shape)

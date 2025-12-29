#scripts/make_y_val.py

import argparse
from scipy import sparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y", required=True, help="Input sparse label matrix")
    parser.add_argument("--out", required=True, help="Output validation label matrix")
    parser.add_argument("--val_size", type=int, default=16481)
    args = parser.parse_args()

    Y = sparse.load_npz(args.y)

    # Use the LAST samples as validation (must match training split)
    Y_val = Y[-args.val_size:]

    sparse.save_npz(args.out, Y_val)

    print("Y_val shape:", Y_val.shape)

if __name__ == "__main__":
    main()

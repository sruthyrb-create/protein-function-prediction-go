# src/baseline_train.py

import os
import joblib
import numpy as np
import pandas as pd
import argparse
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from features import build_feature_matrix


def main():
    import argparse

    # ------------------------
    # Argument parsing
    # ------------------------
    parser = argparse.ArgumentParser(description="Train baseline OVR logistic regression")
    parser.add_argument(
        "--y",
        type=str,
        required=True,
        help="Path to sparse label matrix (.npz)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/baseline",
        help="Output directory"
    )
    args = parser.parse_args()

    # ------------------------
    # Load data
    # ------------------------
    print("Loading data...")
    df = pd.read_csv("artifacts/proteins_sequences.csv")
    Y = sparse.load_npz(args.y)

    X = build_feature_matrix(df["sequence"].tolist())

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # ------------------------
    # Train/validation split
    # ------------------------
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # ------------------------
    # Model
    # ------------------------
    print("Training baseline classifier...")
    clf = OneVsRestClassifier(
        LogisticRegression(
            solver="saga",
            max_iter=100,
            verbose=1,
            n_jobs=-1
        )
    )

    clf.fit(X_train, Y_train)

    # ------------------------
    # Predict probabilities
    # ------------------------
    print("Predicting validation probabilities...")
    Y_val_probs = clf.predict_proba(X_val)

    # ------------------------
    # Save outputs
    # ------------------------
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "val_probs.npy"), Y_val_probs)
    joblib.dump(clf, os.path.join(args.out, "baseline_model.joblib"))

    print("Baseline training complete.")
    print(f"Saved outputs to {args.out}/")


if __name__ == "__main__":
    main()

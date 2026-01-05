import os
import numpy as np
import torch
from scipy import sparse
import argparse

from src.models.mlp_classifier import MLPClassifier



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading embeddings...")
    X = np.load(args.embeddings)  # (N, D)
    print("Loading label dimensions...")

    device = torch.device("cpu")  # FORCE CPU (stable)
    Y = sparse.load_npz("artifacts/y_sparse_filtered.npz")
    output_dim = Y.shape[1]

    model = MLPClassifier(
        input_dim=X.shape[1],
        output_dim=output_dim
    )

    state_dict = torch.load(args.model, map_location=device)

    # If trained without "net." prefix, add it
    if not any(k.startswith("net.") for k in state_dict.keys()):
        state_dict = {f"net.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("Running inference...")
    batch_size = 64
    probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.vstack(probs)
    np.save(args.out, probs)

    print("Saved validation probabilities:", probs.shape)


if __name__ == "__main__":
    main()

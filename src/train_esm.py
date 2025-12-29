import argparse
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse

from src.models.esm_encoder import ESMEncoder
from src.models.esm_classifier import ESMClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out", default="outputs/esm")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("Using device:", device)

    df = pd.read_csv("artifacts/proteins_sequences.csv")
    Y = sparse.load_npz("artifacts/y_sparse_filtered.npz")

    encoder = ESMEncoder().to(device)
    clf = ESMClassifier().to(device)

    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        total = 0
        for i in range(0, len(df), args.batch_size):
            seqs = df.sequence.iloc[i:i+args.batch_size].tolist()
            y = torch.tensor(Y[i:i+args.batch_size].toarray()).float()

            emb = encoder(seqs)
            logits = clf(emb)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1} Loss: {total:.4f}")

    torch.save(clf.state_dict(), f"{args.out}/classifier.pt")
    print("ESM training complete")

if __name__ == "__main__":
    main()

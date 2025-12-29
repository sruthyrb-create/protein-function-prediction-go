print("train_cnn.py loaded")

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.model_selection import train_test_split

from src.models.cnn_model import ProteinCNN
from src.utils.sequence_encode import encode_sequence


# -----------------------------
# Dataset
# -----------------------------
class ProteinDataset(Dataset):
    def __init__(self, sequences, Y_sparse, indices, max_len=1000):
        self.sequences = sequences
        self.Y = Y_sparse
        self.indices = indices
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        seq = self.sequences[i]
        x = encode_sequence(seq, self.max_len)          # (max_len,)
        y = torch.from_numpy(self.Y[i].toarray().squeeze()).float()
        return x, y


# -----------------------------
# Training
# -----------------------------
def main():
    print("Entered main()")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--out", default="outputs/cnn_k20")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cpu")

    print("Using device:", device)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv("artifacts/proteins_sequences.csv")
    sequences = df["sequence"].tolist()
    Y_sparse = sparse.load_npz("artifacts/y_sparse_filtered.npz")

    indices = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_ds = ProteinDataset(sequences, Y_sparse, train_idx, args.max_len)
    val_ds   = ProteinDataset(sequences, Y_sparse, val_idx, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

    # -----------------------------
    # Model
    # -----------------------------
    n_labels = Y_sparse.shape[1]
    model = ProteinCNN(num_labels=n_labels).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} - Train loss: {avg_loss:.4f}")

    # -----------------------------
    # Save model
    # -----------------------------
    torch.save(model.state_dict(), os.path.join(args.out, "cnn_model.pt"))
    print("Training complete. Model saved.")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()

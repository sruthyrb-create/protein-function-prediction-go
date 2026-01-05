import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import sparse
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="outputs/esm_mlp")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cpu")
    print("Using device:", device)

    # -------------------------
    # Load data
    # -------------------------
    X = np.load("artifacts/esm_embeddings.npy")
    Y = sparse.load_npz("artifacts/y_sparse_filtered.npz")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    Y_train = torch.from_numpy(Y_train.toarray()).float()
    Y_val = torch.from_numpy(Y_val.toarray()).float()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), Y_train)
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), Y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # -------------------------
    # Model
    # -------------------------
    model = nn.Sequential(
        nn.Linear(480, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, Y.shape[1])
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {avg_loss:.4f}")

    # -------------------------
    # Save
    # -------------------------
    torch.save(model.state_dict(), os.path.join(args.out, "esm_mlp.pt"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()

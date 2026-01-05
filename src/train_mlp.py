import torch
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from src.models.mlp_classifier import MLPClassifier

def main():
    X = np.load("artifacts/esm_embeddings.npy")
    Y = sparse.load_npz("artifacts/y_sparse_filtered.npz")

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.toarray(), dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MLPClassifier(X.shape[1], Y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(5):
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"Epoch {epoch+1}: loss={total/len(loader):.4f}")

    torch.save(model.state_dict(), "outputs/mlp_esm/model.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()

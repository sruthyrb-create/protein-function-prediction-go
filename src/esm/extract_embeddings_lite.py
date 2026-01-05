import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

MODEL_NAME = "esm2_t12_35M_UR50D"
MAX_LEN = 512
BATCH_SIZE = 1   # DO NOT CHANGE

def truncate(seq, max_len):
    return seq[:max_len]

def main():
    print("Loading ESM-LITE model...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    df = pd.read_csv("artifacts/proteins_sequences.csv")
    sequences = [truncate(s, MAX_LEN) for s in df["sequence"].tolist()]

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(len(sequences))):
            batch = [(str(i), sequences[i])]
            _, _, tokens = batch_converter(batch)

            outputs = model(tokens, repr_layers=[12])
            reps = outputs["representations"][12]

            mask = tokens != alphabet.padding_idx
            pooled = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)

            all_embeddings.append(pooled.cpu().numpy())

            # Flush periodically (VERY IMPORTANT)
            if i % 1000 == 0 and i > 0:
                np.save("artifacts/esm_embeddings_partial.npy",
                        np.vstack(all_embeddings))
                print(f"Saved {i} embeddings")

    X = np.vstack(all_embeddings)
    np.save("artifacts/esm_embeddings.npy", X)

    print("Final embeddings saved:", X.shape)

if __name__ == "__main__":
    main()

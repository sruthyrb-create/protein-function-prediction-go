# src/features.py

import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def aa_composition(sequence: str) -> np.ndarray:
    """
    Convert a protein sequence into a 20-dim amino acid frequency vector.
    """
    vec = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    if not sequence:
        return vec

    for aa in sequence:
        if aa in AA_INDEX:
            vec[AA_INDEX[aa]] += 1.0

    vec /= len(sequence)
    return vec


def build_feature_matrix(sequences):
    """
    Build feature matrix X from a list of protein sequences.
    Shape: (n_proteins, 20)
    """
    X = np.vstack([aa_composition(seq) for seq in sequences])
    return X

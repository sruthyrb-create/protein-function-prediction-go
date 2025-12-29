 # src/data_loader.py
"""
Load sequences (FASTA) and annotation terms (train_terms.tsv),
propagate annotations using GO ancestors, and build a MultiLabelBinarizer.

Produces:
- proteins: list of protein IDs
- sequences: list of sequences aligned with proteins
- labels: list of lists of GO terms (propagated)
- mlb object (fitted MultiLabelBinarizer)
- saves artifacts: artifacts/mlb_classes.npy and artifacts/protein_index_map.joblib
"""

import os
from collections import defaultdict
from Bio import SeqIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

from scipy import sparse
from src.go_utils import load_go_obo, build_parent_map, compute_ancestors, propagate_terms

def read_fasta(fasta_path):
    """
    Return dict protein_id -> amino acid sequence (string).
    Expects FASTA headers with IDs like sp|P9WHI7|RECN_MYCT or simply P9WHI7; we'll extract the accession token.
    """
    prot2seq = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        header = rec.id  # biopython's SeqRecord.id is first token after '>'
        # header might be like 'sp|P9WHI7|RECN_MYCT' -> extract accession as second field if pipe-separated
        if '|' in header:
            parts = header.split('|')
            if len(parts) >= 2:
                acc = parts[1]
            else:
                acc = header
        else:
            acc = header
        seq = str(rec.seq).upper()
        prot2seq[acc] = seq
    return prot2seq

def read_train_terms(train_terms_tsv):
    """
    Read train_terms.tsv expecting columns: protein_id \t GO:xxxxx \t Ontology (MFO/BPO/CCO or similar)
    Returns dict protein_id -> set(terms)
    """
    df = pd.read_csv(train_terms_tsv, sep='\t', header=None, names=['protein','go','ont'], dtype=str)
    ann = defaultdict(set)
    for _, row in df.iterrows():
        pid = row['protein']
        go = row['go']
        if pd.isna(go) or not isinstance(go, str): 
            continue
        ann[pid].add(go.strip())
    return ann

def build_dataset(fasta_path, train_terms_tsv, go_obo_path, save_dir="artifacts"):
    # load GO and ancestors
    G = load_go_obo(go_obo_path)
    parent_map = build_parent_map(G)
    ancestors_map = compute_ancestors(parent_map)

    prot2seq = read_fasta(fasta_path)
    ann = read_train_terms(train_terms_tsv)

    proteins = []
    sequences = []
    labels_list = []
    missing_in_fasta = 0
    for pid, gos in ann.items():
        if pid not in prot2seq:
            missing_in_fasta += 1
            continue
        propagated = propagate_terms(gos, ancestors_map)
        proteins.append(pid)
        sequences.append(prot2seq[pid])
        labels_list.append(sorted(list(propagated)))

    print("Total annotated proteins read:", len(ann))
    print("Proteins with FASTA available and used:", len(proteins))
    if missing_in_fasta:
        print("Proteins present in annotations but missing in FASTA:", missing_in_fasta)

    # build MultiLabelBinarizer (term -> index)
    mlb = MultiLabelBinarizer(sparse_output=False)
    Y = mlb.fit_transform(labels_list)  # shape (n_proteins, n_terms)
    print("Number of unique GO terms (after propagation) in training:", len(mlb.classes_))

    # save artifacts
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(mlb.classes_, os.path.join(save_dir, "mlb_classes.npy"))
    joblib.dump({'proteins': proteins, 'index': {p:i for i,p in enumerate(proteins)}}, os.path.join(save_dir, "protein_index_map.joblib"))
    #np.save(os.path.join(save_dir, "Y.npy"), Y, allow_pickle=False)
    Y_sparse = sparse.csr_matrix(Y)
    sparse.save_npz(os.path.join(save_dir, "y_sparse.npz"), Y_sparse)

    # save sequences mapping for later (maybe large); for now save a small csv
    pd.DataFrame({'protein':proteins, 'sequence':sequences}).to_csv(os.path.join(save_dir, "proteins_sequences.csv"), index=False)

    return proteins, sequences, labels_list, mlb, Y_sparse


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="data/train_sequences.fasta")
    parser.add_argument("--terms", default="data/train_terms.tsv")
    parser.add_argument("--obo", default="data/go-basic.obo")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    proteins, sequences, labels_list, mlb, Y = build_dataset(args.fasta, args.terms, args.obo, save_dir=args.out)
    print("Saved artifacts to", args.out)

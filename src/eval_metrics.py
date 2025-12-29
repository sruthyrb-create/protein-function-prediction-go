#src/eval_metrics.py

"""


Compute information-accretion weighted precision/recall/F1 for GO-term
predictions and search thresholds to maximize weighted F1.

Usage examples (from project root):
# demo with random predictions (no heavy memory usage)
python -m src.eval_metrics --y artifacts/y_sparse.npz --mlb artifacts/mlb_classes.npy --ia data/IA.tsv --obo data/go-basic.obo --demo

# if you have real prediction matrix (n_proteins x n_terms) saved as .npy (memmap-friendly)
python -m src.eval_metrics --y artifacts/y_sparse.npz --mlb artifacts/mlb_classes.npy --ia data/IA.tsv --obo data/go-basic.obo --preds outputs/pred_probs.npy --subontology BP --search

Notes:
- preds file should be same column order as mlb.classes_ (float probabilities in [0,1]).
- y should be a sparse CSR matrix (.npz created with scipy.sparse.save_npz).
"""

import argparse
import numpy as np
from scipy import sparse
import joblib
import os
import obonet
import pandas as pd
from typing import Tuple, List

def load_ia(ia_path: str) -> dict:
    """
    Load IA.tsv or IA-like file. Accepts two-column TSV: GOterm <sep> weight.
    Returns dict {GO: weight}
    """
    ia = {}
    with open(ia_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # try splitting by whitespace or tab
            parts = line.split()
            if len(parts) == 1:
                # maybe file lists one GO and later lines weights; skip
                continue
            go = parts[0].strip()
            try:
                w = float(parts[1])
            except:
                # try tab split
                parts_tab = line.split('\t')
                if len(parts_tab) >= 2:
                    go = parts_tab[0].strip()
                    try:
                        w = float(parts_tab[1].strip())
                    except:
                        continue
                else:
                    continue
            ia[go] = w
    return ia

def load_go_namespace_map(obo_path: str) -> dict:
    """
    Return dict term -> namespace ('biological_process','molecular_function','cellular_component')
    using obonet.read_obo graph node attribute 'namespace'.
    """
    G = obonet.read_obo(obo_path)
    ns_map = {}
    for node, data in G.nodes(data=True):
        ns = data.get('namespace')
        if ns:
            ns_map[node] = ns
    return ns_map

def filter_terms_by_namespace(classes: List[str], ns_map: dict, target_ns: str) -> np.ndarray:
    """
    Given mlb.classes_ (list/array of GO IDs) and ns_map, return boolean mask of classes belonging to target_ns.
    target_ns should be one of: 'biological_process', 'molecular_function', 'cellular_component'
    """
    mask = np.array([ (ns_map.get(c) == target_ns) for c in classes ], dtype=bool)
    return mask

def dataset_weighted_PRF_at_threshold(
    preds: np.ndarray,
    Y_sparse: sparse.csr_matrix,
    ia_array: np.ndarray,
    subontology_mask: np.ndarray,
    threshold: float
) -> Tuple[float,float,float]:
    """
    Compute dataset-level information-accretion weighted precision, recall, F1
    restricted to the columns indicated by subontology_mask.

    preds: (n_proteins, n_terms) numpy array-like of probabilities (can be memmap)
    Y_sparse: (n_proteins, n_terms) scipy csr matrix (binary)
    ia_array: (n_terms,) numpy array of IA weights (zeros where missing)
    subontology_mask: boolean mask (n_terms,) selecting columns in desired subontology
    threshold: scalar cutoff between 0 and 1

    Returns: (precision, recall, f1)
    Implementation does:
      intersection_weight = sum_{proteins i, term f in subontology} IA[f] * 1{pred_ij >= th} * Y[i,f]
      denom_precision = sum_{proteins i, term f in subontology} IA[f] * 1{pred_ij >= th}
      denom_recall = sum_{proteins i, term f in subontology} IA[f] * Y[i,f]
      precision = intersection_weight / denom_precision
      recall = intersection_weight / denom_recall
      f1 = 2 * P * R / (P + R)
    """
    # apply mask to IA
    ia_sub = ia_array[subontology_mask]
    # Get sub-matrix for Y (sparse)
    Y_sub = Y_sparse[:, subontology_mask]  # csr of shape (n_proteins, n_terms_sub)
    # preds_sub: if preds is large memmap, slicing returns view (good)
    preds_sub = preds[:, subontology_mask]

    # binary predicted mask (dense, but manageable per subontology). Use uint8 for memory.
    P_bin = (preds_sub >= threshold).astype(np.uint8)

    # compute predicted counts per term (dense vector)
    pred_counts = P_bin.sum(axis=0)  # shape (n_terms_sub,)
    # compute true counts per term (sparse sum)
    true_counts = np.array(Y_sub.sum(axis=0)).ravel()  # shape (n_terms_sub,)

    # Compute intersection counts per term:
    # Y_sub.multiply(P_bin) does elementwise multiply; returns sparse matrix containing only true positives
    Y_and_P = Y_sub.multiply(P_bin)  # sparse matrix
    inter_counts = np.array(Y_and_P.sum(axis=0)).ravel()

    # Weighted sums (dataset-level)
    numerator = float(np.dot(inter_counts, ia_sub))
    denom_precision = float(np.dot(pred_counts, ia_sub))
    denom_recall = float(np.dot(true_counts, ia_sub))

    precision = numerator / denom_precision if denom_precision > 0 else 0.0
    recall = numerator / denom_recall if denom_recall > 0 else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
    return precision, recall, f1

def search_thresholds_grid(
    preds: np.ndarray,
    Y_sparse: sparse.csr_matrix,
    ia_array: np.ndarray,
    subontology_mask: np.ndarray,
    thresholds: np.ndarray
):
    """
    Search thresholds (1D array) and return DataFrame with threshold, precision, recall, f1
    """
    rows = []
    for t in thresholds:
        P, R, F = dataset_weighted_PRF_at_threshold(preds, Y_sparse, ia_array, subontology_mask, t)
        rows.append({'threshold': float(t), 'precision': float(P), 'recall': float(R), 'f1': float(F)})
        # feedback
    df = pd.DataFrame(rows)
    return df

def load_preds_safe(preds_path: str, n_proteins: int = None, n_terms: int = None):
    """
    Load predictions file safely. Supports:
      - .npy dense arrays (np.load; possibly memmap)
      - .npz (np.load) single array saved as .npz
    Returns numpy array or memmap.
    """
    if preds_path is None:
        return None
    ext = os.path.splitext(preds_path)[1].lower()
    if ext == '.npy':
        # use memmap to avoid loading whole into RAM
        arr = np.load(preds_path, mmap_mode='r')
        return arr
    elif ext == '.npz':
        with np.load(preds_path) as obj:
            # take first array key
            keys = list(obj.keys())
            if len(keys) == 0:
                raise ValueError("Empty npz file")
            return obj[keys[0]]
    else:
        raise ValueError("Unsupported preds extension. Use .npy or .npz.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--y', required=True, help="Path to sparse label matrix (.npz from scipy.sparse.save_npz)")
    parser.add_argument('--mlb', required=True, help="Path to mlb classes (joblib/np saved; e.g. artifacts/mlb_classes.npy)")
    parser.add_argument('--ia', required=True, help="IA weights file (IA.tsv) path")
    parser.add_argument('--obo', required=True, help="Path to go-basic.obo")
    parser.add_argument('--preds', default=None, help="Predicted probabilities (.npy or .npz) aligned to mlb.classes_ order")
    parser.add_argument('--subontology', default='BP', choices=['BP','MF','CC'], help="Which subontology to evaluate")
    parser.add_argument('--search', action='store_true', help="Search thresholds grid (0..1 step 0.01)")
    parser.add_argument('--demo', action='store_true', help="Run demo with random predictions (no preds file needed)")
    parser.add_argument('--out', default='outputs/eval', help="Directory to save threshold results CSV")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load labels (sparse)
    Y_sparse = sparse.load_npz(args.y)
    print("Loaded Y sparse:", Y_sparse.shape)

    # Load mlb classes
    try:
        classes = joblib.load(args.mlb)
    except Exception:
        classes = np.load(args.mlb, allow_pickle=True)
    classes = np.array(classes)  # shape (n_terms,)

    n_proteins, n_terms = Y_sparse.shape
    assert len(classes) == n_terms, "Mismatch: mlb.classes_ length != Y columns"

    # load IA
    ia_map = load_ia(args.ia)
    # build ia_array aligned to classes
    ia_array = np.array([ia_map.get(go, 0.0) for go in classes], dtype=float)

    # load GO obo namespace map
    ns_map = load_go_namespace_map(args.obo)
    # map requested subontology to GO namespace string
    ns_map_name = {'BP': 'biological_process', 'MF': 'molecular_function', 'CC': 'cellular_component'}
    target_ns = ns_map_name[args.subontology]
    sub_mask = filter_terms_by_namespace(classes, ns_map, target_ns)
    if sub_mask.sum() == 0:
        print("Warning: no classes found in chosen subontology. Check GO and mlb order.")
    print(f"Evaluating subontology {args.subontology}: terms={sub_mask.sum()}")

    # load preds or make demo preds
    if args.demo or args.preds is None:
        print("Generating random demo predictions (keeps memory small).")
        # generate reasonable probabilistic matrix but do as memmap-like small array for demo
        # We'll generate floats from beta distribution with small mean; shape n_proteins x n_terms may be big —
        # so generate a small subset for demo: first 200 proteins only (to avoid memory issues)
        demo_n = min(200, n_proteins)
        preds_demo = np.random.beta(a=0.5, b=3.0, size=(demo_n, n_terms)).astype(np.float32)
        # slice corresponding labels
        Y_demo = Y_sparse[:demo_n, :]
        # run search on demo slice
        thresholds = np.linspace(0.0, 1.0, 101)
        df = search_thresholds_grid(preds_demo, Y_demo, ia_array, sub_mask, thresholds)
        out_path = os.path.join(args.out, f"thresholds_demo_{args.subontology}.csv")
        df.to_csv(out_path, index=False)
        best = df.loc[df['f1'].idxmax()]
        print("Demo best threshold:", best.to_dict())
        print("Saved demo thresholds to", out_path)
        return

    preds = load_preds_safe(args.preds)
    # preds could be memmap shape (n_proteins, n_terms)
    if preds.shape != (n_proteins, n_terms):
        raise ValueError(f"Preds shape {preds.shape} does not match labels {(n_proteins, n_terms)}")

    if args.search:
        thresholds = np.linspace(0.0, 1.0, 101)
        df = search_thresholds_grid(preds, Y_sparse, ia_array, sub_mask, thresholds)
        out_path = os.path.join(args.out, f"thresholds_{args.subontology}.csv")
        df.to_csv(out_path, index=False)
        best = df.loc[df['f1'].idxmax()]
        print("Best threshold:", best.to_dict())
        print("Saved thresholds CSV to", out_path)
    else:
        # compute at default 0.5
        P, R, F = dataset_weighted_PRF_at_threshold(preds, Y_sparse, ia_array, sub_mask, 0.5)
        print(f"At threshold 0.5 -> precision={P:.6f}, recall={R:.6f}, f1={F:.6f}")

if __name__ == "__main__":
    main()

import os
import numpy as np
import joblib
import argparse
from scipy import sparse
import obonet

# -----------------------------
# Utilities
# -----------------------------
def load_go_graph(obo_path):
    return obonet.read_obo(obo_path)

def build_parent_map(G):
    parent_map = {}
    for child, parent, data in G.edges(data=True):
        parent_map.setdefault(child, set()).add(parent)
    return parent_map

def propagate_terms(predicted_terms, parent_map):
    expanded = set(predicted_terms)
    stack = list(predicted_terms)
    while stack:
        t = stack.pop()
        for p in parent_map.get(t, []):
            if p not in expanded:
                expanded.add(p)
                stack.append(p)
    return expanded


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs", required=True)
    parser.add_argument("--mlb", required=True)
    parser.add_argument("--obo", required=True)
    parser.add_argument("--protein_map", required=True)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    print("Loading probabilities...")
    probs = np.load(args.probs)

    print("Loading class labels...")
    classes = joblib.load(args.mlb)
    classes = np.array(classes)

    print("Loading protein index map...")
    protein_map = joblib.load(args.protein_map)
    proteins = protein_map["proteins"]

    print("Loading GO graph...")
    G = load_go_graph(args.obo)
    parent_map = build_parent_map(G)

    # Thresholds (from your evaluation)
    BP_TH = 0.12
    MF_TH = 0.14
    CC_TH = 0.18

    # Namespace map
    ns_map = {n: G.nodes[n].get("namespace") for n in G.nodes}

    submission = []

    for i, pid in enumerate(proteins):
        row = probs[i]
        predicted = set()

        for j, p in enumerate(row):
            go = classes[j]
            ns = ns_map.get(go)

            if ns == "biological_process" and p >= BP_TH:
                predicted.add(go)
            elif ns == "molecular_function" and p >= MF_TH:
                predicted.add(go)
            elif ns == "cellular_component" and p >= CC_TH:
                predicted.add(go)

        # Rare label recovery via propagation
        predicted = propagate_terms(predicted, parent_map)

        submission.append((pid, ",".join(sorted(predicted))))

        if i % 5000 == 0:
            print(f"Processed {i}/{len(proteins)} proteins")

    # Save
    with open(args.out, "w") as f:
        f.write("protein_id,go_terms\n")
        for pid, gos in submission:
            f.write(f"{pid},{gos}\n")

    print("Submission saved to:", args.out)


if __name__ == "__main__":
    main()

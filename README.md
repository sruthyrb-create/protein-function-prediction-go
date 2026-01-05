# Protein Function Prediction (GO Terms)

Competition-focused pipeline for large-scale protein function prediction
using Gene Ontology (GO).

## Pipeline
- Label filtering (≥20 frequency)
- Sparse multi-label representation
- Baseline (logistic)
- CNN baseline
- ESM-2 pretrained embeddings
- Ontology-aware evaluation (BP)

## Structure
src/
├── models/
├── utils/
├── eval_metrics.py
├── train_cnn.py
└── train_esm.py

scripts/
├── filter_labels_by_frequency_clean.py
├── make_y_val.py


## Environment
- Python 3.10
- PyTorch
- fair-esm
- scipy, sklearn

## Status
🚧 Active development toward Jan 26 deadline

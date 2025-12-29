import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA)}  # 0 = padding

def encode_sequence(seq, max_len=1000):
    arr = [AA_TO_IDX.get(a, 0) for a in seq[:max_len]]
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.long)

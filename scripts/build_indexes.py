#!/usr/bin/env python3
import gzip
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

FEATURES_DIR = Path("features")
INDEXES_DIR = Path("cache/indexes")
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

effnet_idx_path = INDEXES_DIR / "effnet.faiss"
clap_idx_path = INDEXES_DIR / "clap.faiss"

if effnet_idx_path.exists() and clap_idx_path.exists():
    print("FAISS indexes already exist. Skipping build.")
    sys.exit(0)

files = sorted(FEATURES_DIR.rglob("*.pkl.gz"))
effnet_list, clap_list = [], []
for fpath in tqdm(files, desc="Loading embeddings"):
    try:
        with gzip.open(fpath, "rb") as f:
            d = pickle.load(f)
        effnet_list.append(d["discogs_effnet_embedding"])
        clap_list.append(d["clap_embedding"])
    except Exception:
        continue

effnet = np.array(effnet_list, dtype=np.float32)
clap = np.array(clap_list, dtype=np.float32)
print(f"effnet: {effnet.shape}  clap: {clap.shape}")

faiss.normalize_L2(effnet)
faiss.normalize_L2(clap)

print("Building HNSW32 index for effnet...")
idx_effnet = faiss.index_factory(effnet.shape[1], "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
idx_effnet.add(effnet)

print("Building HNSW32 index for CLAP...")
idx_clap = faiss.index_factory(clap.shape[1], "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
idx_clap.add(clap)

faiss.write_index(idx_effnet, str(effnet_idx_path))
faiss.write_index(idx_clap, str(clap_idx_path))
print(f"Saved indexes to {INDEXES_DIR}")
print(f"  effnet.faiss  {effnet.shape[0]} x {effnet.shape[1]}")
print(f"  clap.faiss    {clap.shape[0]} x {clap.shape[1]}")

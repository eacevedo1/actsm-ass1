#!/usr/bin/env python3
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ui_data import load_collection  # noqa: E402

CACHE_DIR = Path("output/cache")
INDEXES_DIR = CACHE_DIR / "indexes"
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

effnet_idx_path = INDEXES_DIR / "effnet.faiss"
clap_idx_path = INDEXES_DIR / "clap.faiss"

if effnet_idx_path.exists() and clap_idx_path.exists():
    print("FAISS indexes already exist. Skipping build.")
    sys.exit(0)

print("Loading collection...")
_, effnet, clap, _, _ = load_collection()
effnet = effnet.astype(np.float32)
clap = clap.astype(np.float32)
print(f"effnet: {effnet.shape}  clap: {clap.shape}")

# L2-normalise in-place so inner product == cosine similarity
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

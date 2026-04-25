import gzip
import os
import pickle
import sys
from pathlib import Path
from typing import Literal

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.ui_data import load_collection, build_m3u8_content  # noqa: E402
from src.models import load_clap_model  # noqa: E402

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
INDEXES_DIR = ROOT / "cache" / "indexes"
HOST_ROOT = os.environ.get("HOST_ROOT", "")

app = FastAPI(title="MusAV — Unified Playlist API")

df, genre_matrix, genre_labels = load_collection()
df = df.reset_index(drop=True)

# Try to load FAISS indexes; fall back to numpy if not available
faiss_effnet = None
faiss_clap = None
if INDEXES_DIR.exists():
    effnet_idx_path = INDEXES_DIR / "effnet.faiss"
    clap_idx_path = INDEXES_DIR / "clap.faiss"
    if effnet_idx_path.exists() and clap_idx_path.exists():
        faiss_effnet = faiss.read_index(str(effnet_idx_path))
        faiss_clap = faiss.read_index(str(clap_idx_path))
        print("✓ Loaded FAISS indexes")
    else:
        raise RuntimeError("FAISS indexes not found. Run 'make indexes' first.")
label_to_idx = {l: i for i, l in enumerate(genre_labels)}
parent_to_styles: dict[str, list[str]] = {}
for label in genre_labels:
    parent, style = label.split("---", 1)
    parent_to_styles.setdefault(parent, []).append(label)

KEY_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

clap_model = load_clap_model()


def _track_payload(i: int, extra: dict | None = None) -> dict:
    row = df.iloc[i]
    out = {
        "idx": i,
        "track_id": row.track_id,
        "audio_url": f"/audio/{i}",
        "bpm": float(row.bpm),
        "key": row.key_edma,
        "danceability": float(row.danceability_prob),
        "voice_prob": float(row.voice_prob),
        "loudness_lufs": float(row.loudness_lufs),
    }
    if extra:
        out.update(extra)
    return out


@app.get("/api/meta")
def meta():
    return {
        "num_tracks": len(df),
        "parent_genres": sorted(parent_to_styles.keys()),
        "styles_by_parent": {k: sorted(v) for k, v in parent_to_styles.items()},
        "key_roots": KEY_ROOTS,
        "bpm_min": float(df.bpm.min()),
        "bpm_max": float(df.bpm.max()),
    }


@app.get("/api/tracks")
def tracks():
    return [{"idx": i, "track_id": tid} for i, tid in enumerate(df.track_id.tolist())]


class FilterBody(BaseModel):
    bpm_min: float | None = None
    bpm_max: float | None = None
    voice: Literal["any", "vocal", "instrumental"] = "any"
    dance_min: float = 0.0
    dance_max: float = 1.0
    key_root: str | None = None
    key_scale: Literal["any", "major", "minor"] = "any"
    key_min_conf: float = 0.0
    parent_genre: str | None = None
    style: str | None = None
    style_min_act: float = 0.0
    top_k: int = Field(default=10, ge=1, le=100)


@app.post("/api/filter")
def filter_tracks(body: FilterBody):
    n = len(df)
    mask = np.ones(n, dtype=bool)

    bpm_lo = body.bpm_min if body.bpm_min is not None else -np.inf
    bpm_hi = body.bpm_max if body.bpm_max is not None else np.inf
    mask &= (df.bpm.values >= bpm_lo) & (df.bpm.values <= bpm_hi)
    mask &= (df.danceability_prob.values >= body.dance_min) & (
        df.danceability_prob.values <= body.dance_max
    )
    if body.voice == "vocal":
        mask &= df.voice_prob.values >= 0.5
    elif body.voice == "instrumental":
        mask &= df.voice_prob.values < 0.5
    if body.key_root and body.key_root != "any":
        mask &= df.key_edma_root.values == body.key_root
    if body.key_scale != "any":
        mask &= df.key_edma_scale.values == body.key_scale
    mask &= df.key_conf_edma.values >= body.key_min_conf

    score = None
    if body.style and body.style != "any":
        score = genre_matrix[:, label_to_idx[body.style]]
        mask &= score >= body.style_min_act
    elif body.parent_genre and body.parent_genre != "any":
        idxs = [label_to_idx[l] for l in parent_to_styles.get(body.parent_genre, [])]
        if idxs:
            score = genre_matrix[:, idxs].max(axis=1)
            mask &= score >= body.style_min_act

    if score is None:
        score = df.bpm_confidence.values

    positions = np.where(mask)[0]
    if len(positions) == 0:
        return {"results": [], "total": 0}

    order = positions[np.argsort(-score[positions])][: body.top_k]
    results = [_track_payload(int(i), {"score": float(score[int(i)])}) for i in order]
    return {"results": results, "total": int(mask.sum())}


@app.get("/api/similar/{idx}")
def similar(idx: int, k: int = 10):
    if idx < 0 or idx >= len(df):
        raise HTTPException(404, "idx out of range")
    k = min(max(1, k), 50)

    feat_p = Path(df.iloc[idx].feat_path)
    try:
        with gzip.open(feat_p, "rb") as f:
            d = pickle.load(f)
    except Exception:
        raise HTTPException(500, "feature file missing")

    def topk_faiss(index, query_vec, exclude_idx, k_val):
        q = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        distances, indices = index.search(q, k_val + 1)
        indices = indices[0]
        distances = distances[0]
        mask = indices != exclude_idx
        return indices[mask][:k_val], distances[mask][:k_val]

    idx_e, score_e = topk_faiss(faiss_effnet, d["discogs_effnet_embedding"], idx, k)
    idx_c, score_c = topk_faiss(faiss_clap, d["clap_embedding"], idx, k)

    return {
        "query": _track_payload(idx),
        "effnet": [
            _track_payload(int(i), {"score": float(s)}) for i, s in zip(idx_e, score_e)
        ],
        "clap": [
            _track_payload(int(i), {"score": float(s)}) for i, s in zip(idx_c, score_c)
        ],
    }


@app.get("/api/search")
def search(q: str, k: int = 10):
    if not q.strip():
        raise HTTPException(400, "empty query")
    k = min(max(1, k), 50)
    emb = clap_model.get_text_embedding([q], use_tensor=False)[0].astype(np.float32)
    q_vec = emb.reshape(1, -1)
    faiss.normalize_L2(q_vec)
    distances, indices = faiss_clap.search(q_vec, k)
    top_idx = indices[0]
    top_scores = distances[0]

    return {
        "query": q,
        "results": [
            _track_payload(int(t), {"score": float(s)})
            for t, s in zip(top_idx, top_scores)
        ],
    }


class M3U8Body(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$", max_length=64)
    indices: list[int]


@app.post("/api/m3u8")
def export_m3u8(body: M3U8Body):
    if not body.indices:
        raise HTTPException(400, "empty playlist")
    idxs = [i for i in body.indices if 0 <= i < len(df)]
    paths = df.iloc[idxs].audio_path.tolist()
    titles = df.iloc[idxs].track_id.tolist()
    content = build_m3u8_content(paths, titles, host_root=HOST_ROOT)
    return {"content": content, "filename": f"{body.name}.m3u8", "count": len(idxs)}


@app.get("/audio/{idx}")
def audio(idx: int):
    if idx < 0 or idx >= len(df):
        raise HTTPException(404, "out of range")
    p = Path(df.iloc[idx].audio_path)
    if not p.exists():
        raise HTTPException(404, "audio file missing")
    return FileResponse(p, media_type="audio/mpeg")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

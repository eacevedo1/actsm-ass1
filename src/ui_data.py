import gzip
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

FEATURES_DIR = Path("features")
AUDIO_DIR = Path("data")
CACHE_DIR = Path("output/cache")

_GENRE_META_URL = (
    "https://essentia.upf.edu/models/classification-heads/"
    "genre_discogs400/genre_discogs400-discogs-effnet-1.json"
)


def _load_genre_labels() -> list[str]:
    cache = CACHE_DIR / "genre_labels.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with urllib.request.urlopen(_GENRE_META_URL) as r:
        labels = json.load(r)["classes"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(labels))
    return labels


def _audio_path_for(track_feat_path: Path) -> Path:
    rel = track_feat_path.relative_to(FEATURES_DIR)
    return AUDIO_DIR / rel.with_suffix("").with_suffix(".mp3")


def _build_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(FEATURES_DIR.rglob("*.pkl.gz"))

    rows = []
    effnet = np.empty((len(files), 1280), dtype=np.float32)
    clap = np.empty((len(files), 512), dtype=np.float32)
    genre = np.empty((len(files), 400), dtype=np.float32)

    keep = 0
    for fpath in tqdm(files, desc="Building UI cache"):
        try:
            with gzip.open(fpath, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        track_id = fpath.stem.replace(".pkl", "")
        audio_path = _audio_path_for(fpath)
        rows.append({
            "track_id": track_id,
            "audio_path": str(audio_path),
            "bpm": float(d["bpm"]),
            "bpm_confidence": float(d["bpm_confidence"]),
            "key_temperley": f"{d['key_temperley']['key']} {d['key_temperley']['scale']}",
            "key_krumhansl": f"{d['key_krumhansl']['key']} {d['key_krumhansl']['scale']}",
            "key_edma": f"{d['key_edma']['key']} {d['key_edma']['scale']}",
            "key_edma_root": d["key_edma"]["key"],
            "key_edma_scale": d["key_edma"]["scale"],
            "key_conf_edma": float(d["key_edma"]["confidence"]),
            "loudness_lufs": float(d["loudness_integrated_lufs"]),
            "voice_prob": float(d["voice_instrumental"][0]),
            "instrumental_prob": float(d["voice_instrumental"][1]),
            "danceability_prob": float(d["danceability"][1]),
        })
        effnet[keep] = d["discogs_effnet_embedding"]
        clap[keep] = d["clap_embedding"]
        genre[keep] = d["genre_discogs400"]
        keep += 1

    effnet = effnet[:keep]
    clap = clap[:keep]
    genre = genre[:keep]

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE_DIR / "collection.parquet")
    np.save(CACHE_DIR / "effnet.npy", effnet)
    np.save(CACHE_DIR / "clap.npy", clap)
    np.save(CACHE_DIR / "genre.npy", genre)
    return df, effnet, clap, genre


def load_collection(rebuild: bool = False):
    parquet = CACHE_DIR / "collection.parquet"
    if rebuild or not parquet.exists():
        df, effnet, clap, genre = _build_cache()
    else:
        df = pd.read_parquet(parquet)
        effnet = np.load(CACHE_DIR / "effnet.npy")
        clap = np.load(CACHE_DIR / "clap.npy")
        genre = np.load(CACHE_DIR / "genre.npy")
    labels = _load_genre_labels()
    return df, effnet, clap, genre, labels


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0, 1, n)
    return x / n


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int, exclude: int | None = None):
    q = query / (np.linalg.norm(query) + 1e-12)
    m = l2_normalize(matrix)
    scores = m @ q
    if exclude is not None:
        scores[exclude] = -np.inf
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def build_m3u8_content(paths: list[str], titles: list[str] | None = None, docker_root: str = "/code", host_root: str = "") -> str:
    lines = ["#EXTM3U"]
    for i, p in enumerate(paths):
        title = titles[i] if titles else Path(p).stem
        resolved = str(Path(p).resolve())
        if host_root:
            resolved = resolved.replace(docker_root, host_root, 1)
        lines.append(f"#EXTINF:-1,{title}")
        lines.append(resolved)
    return "\n".join(lines) + "\n"


def write_m3u8(paths: list[str], out_path: Path, titles: list[str] | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_m3u8_content(paths, titles))
    return out_path

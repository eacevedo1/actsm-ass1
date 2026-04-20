#!/usr/bin/env python3
import argparse
import gzip
import logging
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("transformers").setLevel(logging.ERROR)

import essentia  # noqa: E402

essentia.log.warningActive = False
essentia.log.infoActive = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import load_audio, convert_to_mono, resample  # noqa: E402
from src.models import (  # noqa: E402
    load_discogs400Effnet_models,
    load_voiceinstrumental_model,
    load_danceability_model,
    load_clap_model,
)
from src.extractor import (  # noqa: E402
    bpm_extractor,
    key_extractor,
    loudnessEBUR_extractor,
    discogs400Effnet_extractor,
    effnet_classifier,
    clap_audio_extractor,
)


_MODELS = None


def _load_models():
    effnet, genre = load_discogs400Effnet_models()
    voice = load_voiceinstrumental_model()
    dance = load_danceability_model()
    clap = load_clap_model()
    return effnet, genre, voice, dance, clap


def _init_worker():
    global _MODELS
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    essentia.log.warningActive = False
    essentia.log.infoActive = False
    _MODELS = _load_models()


def analyze_track(path: Path, models) -> dict:
    effnet, genre, voice, dance, clap = models

    audio_stereo, sr = load_audio(str(path))
    mono = convert_to_mono(audio_stereo)
    mono_16k, _ = resample(mono, sr, 16000)
    mono_48k, _ = resample(mono, sr, 48000)

    bpm, bpm_conf = bpm_extractor(mono_16k)
    key_profiles = {}
    for profile in ("temperley", "krumhansl", "edma"):
        k, s, c = key_extractor(mono_16k, profileType=profile)
        key_profiles[profile] = {"key": k, "scale": s, "confidence": float(c)}
    loudness = loudnessEBUR_extractor(audio_stereo)

    emb_frames, genre_preds = discogs400Effnet_extractor(mono_16k, effnet, genre)
    voice_preds = effnet_classifier(emb_frames, voice)
    dance_preds = effnet_classifier(emb_frames, dance)
    clap_emb = clap_audio_extractor(mono_48k, clap)

    return {
        "bpm": float(bpm),
        "bpm_confidence": float(bpm_conf),
        "key_temperley": key_profiles["temperley"],
        "key_krumhansl": key_profiles["krumhansl"],
        "key_edma": key_profiles["edma"],
        "loudness_integrated_lufs": float(loudness),
        "discogs_effnet_embedding": emb_frames.mean(axis=0).astype(np.float32),
        "genre_discogs400": genre_preds.mean(axis=0).astype(np.float32),
        "voice_instrumental": voice_preds.mean(axis=0).astype(np.float32),
        "danceability": dance_preds.mean(axis=0).astype(np.float32),
        "clap_embedding": clap_emb.astype(np.float32),
    }


def _save(features: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)


def _process_serial(task):
    audio_path, out_path = task
    try:
        features = analyze_track(audio_path, _MODELS)
        _save(features, out_path)
        return audio_path, None
    except Exception as e:
        return audio_path, f"{type(e).__name__}: {e}"


def _scan(audio_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    files = sorted(audio_dir.rglob("*.mp3"))
    tasks = []
    for f in files:
        rel = f.relative_to(audio_dir)
        out_path = output_dir / rel.with_suffix(".pkl.gz")
        if not out_path.exists():
            tasks.append((f, out_path))
    return tasks, len(files)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MP3 collection with Essentia + CLAP."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data"),
        help="Root folder to scan recursively for .mp3 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("features"),
        help="Folder to write per-track .pkl.gz features (mirrors input tree).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes. Each loads models once (memory x N).",
    )
    args = parser.parse_args()

    if not args.audio_dir.exists():
        sys.exit(f"audio-dir not found: {args.audio_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, total = _scan(args.audio_dir, args.output_dir)
    print(f"Found {total} MP3 files. Pending: {len(tasks)}.")
    if not tasks:
        return

    failures = []
    if args.workers <= 1:
        global _MODELS
        print("Loading models...")
        _MODELS = _load_models()
        for task in tqdm(tasks, desc="Analyzing", unit="track"):
            path, err = _process_serial(task)
            if err:
                failures.append((path, err))
                tqdm.write(f"FAIL {path}: {err}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker) as pool:
            for path, err in tqdm(
                pool.imap_unordered(_process_serial, tasks),
                total=len(tasks),
                desc="Analyzing",
                unit="track",
            ):
                if err:
                    failures.append((path, err))
                    tqdm.write(f"FAIL {path}: {err}")

    print(f"Done. Success: {len(tasks) - len(failures)}. Failed: {len(failures)}.")
    if failures:
        log = args.output_dir / "_failures.log"
        with open(log, "w") as f:
            for p, e in failures:
                f.write(f"{p}\t{e}\n")
        print(f"Failure log: {log}")


if __name__ == "__main__":
    main()

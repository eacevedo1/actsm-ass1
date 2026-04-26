# ACTSM Assignment 1 — Audio Content Based Playlists

**Course**: ACTSM, UPF 2026

---

## Folder Structure

```
.
├── api.py                        # FastAPI server (REST + static files)
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt              # Binary wheels
├── requirements-src.txt          # Source-build packages (laion_clap)
│
├── scripts/
│   ├── analyze.py                # Audio analysis pipeline (Essentia + CLAP)
│   └── build_indexes.py          # Builds FAISS indexes from features
│
├── src/
│   ├── extractor.py              # Per-track feature extraction logic
│   ├── loader.py                 # Audio loading helpers
│   ├── models.py                 # Model loading (Essentia + CLAP)
│   └── ui_data.py                # Cache loading, M3U8 export
│
├── static/
│   ├── index.html                # Playlist Builder UI
│   ├── app.js
│   └── style.css
│
├── notebooks/
│   └── report.ipynb              # Collection overview & plots
│
├── models/                       # Downloaded model files (created by make models)
├── data/                         # MusAV MP3 files (created by make data)
├── features/                     # Per-track .pkl.gz features (created by make analyze)
└── cache/                        # Parquet + FAISS indexes (created at runtime)
```

---

## Setup

### 1. Download audio data

```bash
make data
```

Downloads the MusAV collection (~2 092 tracks) from Google Drive into `data/`. Requires `gdown`:

### 2. Download models

```bash
make models
```

Downloads to `models/`:

| File | Description |
|------|-------------|
| `discogs-effnet-bs64-1.pb` | Discogs-EffNet embeddings |
| `genre_discogs400-discogs-effnet-1.pb` | Genre Discogs400 classifier |
| `voice_instrumental-discogs-effnet-1.pb` | Voice/Instrumental classifier |
| `danceability-discogs-effnet-1.pb` | Danceability classifier |
| `music_speech_epoch_15_esc_89.25.pt` | LAION CLAP |

Skips files already present.

### 3. Start the Docker container

```bash
make up
```

Builds the image and starts the container. The repo root is mounted at `/code` inside.

### 4. Analyze the audio collection

```bash
make analyze
```

Scans `data/` for MP3s and writes per-track features to `features/`. Skips tracks already processed.

To run with more workers or a custom path:

```bash
docker compose exec actsm python /code/scripts/analyze.py \
  --audio-dir /code/data \
  --output-dir /code/features \
  --workers 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--audio-dir` | `data` | Root folder scanned for `.mp3` files |
| `--output-dir` | `features` | Output folder for `.pkl.gz` feature files |
| `--workers` | `1` | Parallel worker processes |

### 5. Build FAISS indexes

```bash
make indexes
```

Reads all features and builds cosine-similarity indexes in `cache/indexes/`.

### 6. Launch the app

```bash
make app
```

Starts the FastAPI server. Open [http://localhost:8000](http://localhost:8000) in your browser.

The UI has three tabs:
- **Descriptors** — filter by BPM, key, genre, danceability, voice. Drop a track to auto-fill filters.
- **Similarity** — pick a collection track or drop an audio file to find similar tracks (Discogs-EffNet + CLAP).
- **Text Search** — free-text query via CLAP embeddings.

---

## Notebook

The report notebook runs **inside the Docker container** via Jupyter. Once the container is running you can access it at [http://localhost:8888](http://localhost:8888).

---

## Make Targets

| Target | Description |
|--------|-------------|
| `make build` | Build Docker image |
| `make up` | Build + start container in background |
| `make down` | Stop container |
| `make shell` | Open bash shell in running container |
| `make models` | Download model files to `models/` |
| `make data` | Download MusAV audio to `data/` |
| `make analyze` | Run analysis pipeline (1 worker) |
| `make indexes` | Build FAISS indexes |
| `make app` | Build indexes (if needed) + start API server |

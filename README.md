# ACTSM Assignment 1 — Audio Content Based Playlists

**Course**: ACTSM, UPF 2026 

## Setup

### 1. Download models

```bash
make models
```

Downloads to `models/`:

| File | Source |
|------|--------|
| `discogs-effnet-bs64-1.pb` | Essentia — Discogs-EffNet embeddings |
| `genre_discogs400-discogs-effnet-1.pb` | Essentia — Genre Discogs400 classifier |
| `voice_instrumental-discogs-effnet-1.pb` | Essentia — Voice/Instrumental classifier |
| `danceability-discogs-effnet-1.pb` | Essentia — Danceability classifier |
| `music_speech_epoch_15_esc_89.25.pt` | HuggingFace — LAION CLAP |

Skips files already present.

### 2. Start environment

```bash
make up
```

Builds the Docker image and starts. The repo root is mounted at `/code` inside the container.

Two requirements files are used:
- `requirements.txt` — binary wheels only (`--only-binary=:all:`)
- `requirements-src.txt` — packages requiring source builds (e.g. `laion_clap`)

### 3. Analyze audio collection

```bash
make up      # container must be running
make analyze
```

Runs inside the Docker container. Scans `data/` recursively for MP3 files and writes per-track features to `features/`, mirroring the input folder tree. Skips tracks already processed.

To target a different collection or adjust parallelism, exec into the container:

```bash
docker compose exec actsm python /code/scripts/analyze.py --audio-dir /code/data --output-dir /code/features --workers 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--audio-dir` | `data` | Root folder scanned recursively for `.mp3` files |
| `--output-dir` | `features` | Output folder for `.pkl.gz` feature files |
| `--workers` | `1` | Parallel worker processes (each loads all models into RAM) |

## Make targets

| Target | Description |
|--------|-------------|
| `make build` | Build Docker image |
| `make up` | Build + start container |
| `make down` | Stop container |
| `make shell` | Open bash shell in running container |
| `make models` | Download model files to `models/` |
| `make analyze` | Run analysis pipeline on `data/` with 8 workers |
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
| `discogs_release_embeddings-effnet-bs64-1.pb` | Essentia — discogs-effnet embeddings |
| `genre_discogs400-discogs-effnet-1.pb` | Essentia — Genre Discogs400 classifier |
| `voice_instrumental-audioset-vggish-1.pb` | Essentia — Voice/Instrumental classifier |
| `music_speech_epoch_15_esc_89.25.pt` | HuggingFace — LAION CLAP |

Skips files already present.

### 2. Start environment

```bash
make up
```

Builds the Docker image and starts. The repo root is mounted at `/code` inside the container.

## Make targets

| Target | Description |
|--------|-------------|
| `make build` | Build Docker image |
| `make up` | Build + start container |
| `make down` | Stop container |
| `make shell` | Open bash shell in running container |
| `make models` | Download model files to `models/` |
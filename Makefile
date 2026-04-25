IMAGE ?= actsm-a1
PLATFORM ?= linux/amd64
HOST_ROOT ?= $(shell pwd)
export HOST_ROOT

BUILD_PLATFORM_FLAG := $(if $(PLATFORM),--platform $(PLATFORM),)
RUN_PLATFORM_FLAG := $(if $(PLATFORM),--platform $(PLATFORM),)

MODELS_DIR := models

ESSENTIA_BASE := https://essentia.upf.edu/models
ESSENTIA_MODELS := \
	feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb \
	classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb \
	classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb \
	classification-heads/danceability/danceability-discogs-effnet-1.pb

CLAP_URL := https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt
CLAP_FILE := music_speech_epoch_15_esc_89.25.pt

DATA_DIR := data
DRIVE_FOLDER_ID := 1c5yJMw7znSoohe7ad1SoTKTvJlXS4Rux

.PHONY: build up down shell models analyze app indexes data

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose exec actsm bash || docker compose run --rm actsm bash

analyze:
	docker compose exec actsm python /code/scripts/analyze.py --audio-dir /code/data --output-dir /code/features --workers 1

app: indexes
	docker compose exec actsm uvicorn api:app --host 0.0.0.0 --port 8000 --app-dir /code

models:
	mkdir -p $(MODELS_DIR)
	@$(foreach model,$(ESSENTIA_MODELS), \
		fname=$$(basename $(model)); \
		if [ ! -f "$(MODELS_DIR)/$$fname" ]; then \
			echo "Downloading $$fname..."; \
			curl -L --fail "$(ESSENTIA_BASE)/$(model)" -o "$(MODELS_DIR)/$$fname"; \
		else \
			echo "Skip $$fname (exists)"; \
		fi;)
	@if [ ! -f "$(MODELS_DIR)/$(CLAP_FILE)" ]; then \
		echo "Downloading $(CLAP_FILE)..."; \
		curl -L --fail "$(CLAP_URL)" -o "$(MODELS_DIR)/$(CLAP_FILE)"; \
	else \
		echo "Skip $(CLAP_FILE) (exists)"; \
	fi

indexes:
	docker compose exec actsm python /code/scripts/build_indexes.py

data:
	mkdir -p $(DATA_DIR)
	@command -v gdown >/dev/null 2>&1 || pip install gdown
	gdown --folder $(DRIVE_FOLDER_ID) -O $(DATA_DIR)/ --remaining-ok

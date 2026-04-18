IMAGE ?= actsm-a1
PLATFORM ?= linux/amd64

BUILD_PLATFORM_FLAG := $(if $(PLATFORM),--platform $(PLATFORM),)
RUN_PLATFORM_FLAG := $(if $(PLATFORM),--platform $(PLATFORM),)

MODELS_DIR := models

ESSENTIA_BASE := https://essentia.upf.edu/models
ESSENTIA_MODELS := \
	feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb \
	classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb \
	classification-heads/voice_instrumental/voice_instrumental-audioset-vggish-1.pb

CLAP_URL := https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt
CLAP_FILE := music_speech_epoch_15_esc_89.25.pt

.PHONY: build up down shell models

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose exec actsm bash || docker compose run --rm actsm bash

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

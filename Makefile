IMAGE ?= actsm-a1
PLATFORM ?= linux/amd64

BUILD_PLATFORM_FLAG := $(if $(PLATFORM),--platform $(PLATFORM),)
RUN_PLATFORM_FLAG := $(if $(PLATFORM),--platform $(PLATFORM),)

.PHONY: build up down shell

build:
	docker compose build

up:
	docker compose up --build -d

down:
	docker compose down

shell:
	docker compose exec actsm bash || docker compose run --rm actsm bash

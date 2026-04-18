FROM python:3.10-bookworm

RUN apt-get update \
	&& apt-get install -y --no-install-recommends ffmpeg python3-pip \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-src.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
	&& python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
	&& python -m pip install --no-cache-dir -r requirements-src.txt

WORKDIR /code
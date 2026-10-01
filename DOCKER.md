# Docker Deployment Guide

How to run stt-api in Docker, from the published images or built from this repo.

## Quick Start

### Prebuilt images

CI publishes both images to `ghcr.io/scagood/stt-api`: `latest-cpu` and
`latest-gpu` track `main`, and each release is tagged `<version>-cpu` and
`<version>-gpu`. The CPU image is built for `linux/amd64` and `linux/arm64`, the
GPU image for `linux/amd64`.

```bash
docker run -d --name parakeet-cpu -p 5092:5092 -v parakeet-models:/app/models \
    -e PARAKEET_PRELOAD_MODELS=parakeet-v3 ghcr.io/scagood/stt-api:latest-cpu

docker run -d --name parakeet-gpu -p 5092:5092 --gpus all -v parakeet-models:/app/models \
    -e PARAKEET_PRELOAD_MODELS=parakeet-v3:fp16 ghcr.io/scagood/stt-api:latest-gpu
```

With `PARAKEET_PRELOAD_MODELS` set, the server downloads and loads that model
before it reports ready; without it, the first request for each model waits for
the download.

### CPU Deployment (Recommended for most users)

```bash
# Build and run
docker compose up parakeet-cpu -d

# Or build manually
docker build -f Dockerfile.cpu -t parakeet-tdt:cpu .
docker run -d --name parakeet-cpu -p 5092:5092 -v parakeet-models:/app/models parakeet-tdt:cpu
```

### GPU Deployment (Requires NVIDIA GPU)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Build and run with Docker Compose
docker compose up parakeet-gpu -d

# Or build manually
docker build -f Dockerfile.gpu -t parakeet-tdt:gpu .
docker run -d --name parakeet-gpu -p 5092:5092 --gpus all \
    -v parakeet-models:/app/models parakeet-tdt:gpu
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `http://localhost:5092/health` | Health and configuration details |
| `http://localhost:5092/healthz` | Readiness: 200 once preloaded models are warm (the container healthcheck) |
| `http://localhost:5092/v1/audio/transcriptions` | OpenAI-compatible API |
| `http://localhost:5092/v1/models`, `/v1/aligners` | What the catalog serves |
| `http://localhost:5092/docs` | Swagger documentation |
| `http://localhost:5092/compare` | Compare models and aligners by ear, with `PARAKEET_COMPARE_UI=true` |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `/app/models` | HuggingFace model cache |
| `HF_HUB_CACHE` | `/app/models` | HuggingFace hub cache |
| `PARAKEET_MODEL_CATALOG` | built-in `parakeet_service/models.yaml` | YAML file that replaces the model catalog (e.g. a mounted ConfigMap); validated at startup. See the README's "Your own model catalog". |
| `PARAKEET_PRELOAD_MODELS` | empty | Comma-separated `model` (fp32) or `model:quantization` entries loaded and warmed up before `/healthz` reports ready, e.g. `parakeet-v3` or `parakeet-v3:fp16`. Requests must still name `model=`. `docker-compose.yml` sets `parakeet-v3`. |

The images also set the CPU or GPU defaults (`PARAKEET_USE_GPU`,
`PARAKEET_BATCHED`, ...). Every other variable is in
[the README's configuration section](README.md#configuration). Pass them with
`docker run -e NAME=value`, or for compose, in a `docker-compose.override.yml`
next to `docker-compose.yml`, which compose merges in automatically:

```yaml
services:
  parakeet-cpu:
    environment:
      PARAKEET_PRELOAD_MODELS: "parakeet-v3:int8"
      PARAKEET_MODEL_CACHE_SIZE: "2"
```

### Persistent Model Cache

Models are cached in a Docker volume to avoid re-downloading:

```bash
# List volumes
docker volume ls | grep parakeet

# Inspect volume
docker volume inspect parakeet-models

# Remove volume (forces model re-download)
docker volume rm parakeet-models
```

The volume must be writable even when every model is already in it: each load
links the model's files into a temporary folder there. To run with
`PARAKEET_HF_OFFLINE=true`, seed it first: start once online with
`PARAKEET_PRELOAD_MODELS` listing every model you serve, and send one word
request naming each aligner your clients use, since aligners aren't preloaded.

## Testing

```bash
# Check health
curl http://localhost:5092/health

# Transcribe audio (OpenAI-compatible)
curl -X POST http://localhost:5092/v1/audio/transcriptions \
    -F "file=@audio.mp3" \
    -F "model=parakeet-v3"
```

## Troubleshooting

**Container won't start:**
- Check logs: `docker logs parakeet-cpu` (or `parakeet-gpu`)
- On first start, every model in `PARAKEET_PRELOAD_MODELS` is downloaded and
  loaded before `/healthz` is ready. The compose healthchecks allow 180 s (CPU)
  and 240 s (GPU) for this; raise `start_period` on a slow connection.

**GPU not detected:**
- Verify NVIDIA Container Toolkit: `nvidia-smi` should work inside container
- Run: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

**Out of memory:**
- With `parakeet-v3` loaded, the server uses about 2.5 GB of RAM at fp32 and
  about 1 GB at int8. Each aligner adds about 0.5 GB at int8, and every other
  model loaded adds its own share; cap them with `PARAKEET_MODEL_CACHE_SIZE`.
  (Measured on CPU with a short clip; long audio needs more.)
- GPU memory hasn't been measured for the current `parakeet-v3` export. On a
  GPU, `fp16` roughly halves it.

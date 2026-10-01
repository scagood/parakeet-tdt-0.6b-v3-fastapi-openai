# Docker Deployment Guide

This document covers Docker deployment options for Parakeet TDT transcription service.

## Quick Start

### Prebuilt images

CI publishes both images to `ghcr.io/scagood/stt-api`: `latest-cpu` and
`latest-gpu` track `main`, and each release is tagged `<version>-cpu` and
`<version>-gpu`. The CPU image is built for `linux/amd64` and `linux/arm64`, the
GPU image for `linux/amd64`.

```bash
docker run -d --name parakeet-cpu -p 5092:5092 -v parakeet-models:/app/models \
    -e PARAKEET_PRELOAD_MODELS=parakeet-v3 ghcr.io/scagood/stt-api:latest-cpu
```

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
`PARAKEET_BATCHED`, ...). For every other variable, see
[OPTIMIZATION.md](OPTIMIZATION.md#env-knobs) and the README.

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

## Files Created

| File | Description |
|------|-------------|
| `Dockerfile.cpu` | CPU-only image (Python 3.14 slim) |
| `Dockerfile.gpu` | GPU image (Python 3.14 slim; CUDA/cuDNN from the `onnxruntime-gpu` wheels) |
| `docker-compose.yml` | Orchestration for both variants |
| `.dockerignore` | Excludes unnecessary files from build |

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
- CPU image requires ~2GB RAM
- GPU image requires ~4GB VRAM

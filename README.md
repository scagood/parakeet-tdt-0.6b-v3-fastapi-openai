# Parakeet TDT Transcription with ONNX Runtime

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Parakeet TDT** is a high-performance implementation of NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model using [ONNX Runtime](https://onnxruntime.ai/), designed for ultra-fast inference on CPU.

This implementation achieves exceptional real-time speeds, outperforming standard [openai/whisper](https://github.com/openai/whisper) and competing directly with GPU-accelerated [faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementations while running entirely on consumer CPUs. The efficiency is achieved through the architectural advantages of the Token-and-Duration Transducer (TDT) model combined with 8-bit quantization.

## 🚀 Optimized FastAPI service (v2)

A refactored async service lives under [`parakeet_service/`](parakeet_service/)
and is started via [`server.py`](server.py). It keeps the OpenAI-compatible
contract of the previous Flask service but adds:

- In-process audio decode (single `ffmpeg` per request, none per chunk)
- **Silero-VAD auto-chunking** that splits long files on pause midpoints
- **Parallel `InferencePool`** that fans-out single-item ORT calls across
  multiple threads — both for concurrent requests and for the chunks of
  one long request

Compared to the legacy Flask+Waitress service on a 12700KF CPU:

| Workload                | Legacy            | Optimized          | Δ        |
|-------------------------|-------------------|--------------------|----------|
| 300 s file (single)     | 17.96 s / 15.7×   | **10.41 s / 27.2×**| **+73%** |
| 16× 10 s concurrent     | 34.6× throughput  | **39.3× throughput**| +13%    |

The service defaults to CUDA with GPU micro-batching. The numbers below were
measured on the FP32 profile, which was the default when they were taken; the
GPU default is now `parakeet-v3-fp16`, which halves VRAM at the same output.

| Workload                | CPU optimized      | GPU profile (FP32) | Δ        |
|-------------------------|--------------------|--------------------|----------|
| 300 s file (single)     | 10.41 s / 27.2×    | **1.37 s / 205.9×**| **+7.6×** |
| 16× 10 s concurrent     | 39.3× throughput   | **200.3× throughput**| **+5.1×** |

See [OPTIMIZATION.md](OPTIMIZATION.md) for the full benchmark, design
rationale, and tunable env knobs.

```bash
python server.py                  # serve on :5092

# CPU override (selects parakeet-v3-fp32, the CPU default)
PARAKEET_USE_GPU=false \
PARAKEET_BATCHED=0 \
python server.py
```

## ⚡ Lower-latency WAV uploads

The server now includes a faster request path for short PCM WAV uploads with no API changes required:

- WAV duration is read directly from the WAV header instead of spawning `ffprobe`
- Already-normalized **16 kHz mono PCM WAV** uploads skip FFmpeg conversion entirely
- Short unchunked PCM WAV uploads can be decoded and resampled **in process** before being passed straight to ONNX Runtime
- FFmpeg remains the fallback for unsupported, compressed, non-WAV, or chunked inputs

On a 20-file English/Spanish Chatterbox WAV benchmark corpus, this reduced endpoint RTF from **0.0459** to **0.0379** and improved effective throughput from **21.80x** to **26.40x** real time, while keeping **20/20** correlation passes.

## 🌍 Multilingual Support

**Parakeet TDT 0.6B v3** features robust multilingual capabilities with **automatic language detection**. The model can automatically identify and transcribe speech in any of the **25 supported languages** without requiring manual language specification:

English, Spanish, French, Russian, German, Italian, Polish, Ukrainian, Romanian, Dutch, Hungarian, Greek, Swedish, Czech, Bulgarian, Portuguese, Slovak, Croatian, Danish, Finnish, Lithuanian, Slovenian, Latvian, Estonian, Maltese

Simply send audio in any of these languages, and the model will automatically detect and transcribe it with high accuracy, including proper punctuation and capitalization.

## Benchmark

### LibriSpeech test-clean (Verified Ground Truth) ⭐

Benchmarked on **LibriSpeech test-clean** dataset with professionally verified human transcriptions. This provides reliable, reproducible accuracy metrics.

**Test Environment:** CPU-only inference, 50 samples (~350 seconds of audio)

| Model | Precision | Accuracy | WER | CER | Speedup (RTF) |
|-------|-----------|----------|-----|-----|---------------|
| **Parakeet TDT 0.6B v3** | INT8 | **97.84%** | 2.16% | 0.56% | **18.41x** (0.054) |
| **Parakeet TDT 0.6B v3** | FP16 | **97.84%** | 2.16% | 0.56% | **18.82x** (0.053) |
| **Parakeet TDT 0.6B v3** | FP32 | **97.84%** | 2.16% | 0.56% | **19.42x** (0.052) |
| Whisper Large v3* | FP16 | ~95-96% | ~4-5% | ~2-3% | varies |

> *Whisper Large v3 benchmarks from published literature on LibriSpeech test-clean. Actual results vary by implementation and hardware.

**Key Findings:**
- All Parakeet precision variants achieve **identical accuracy** (97.84%)
- INT8 quantization has **zero accuracy loss** vs FP32
- Real-time factor (RTF) of ~0.05 means 20x faster than real-time
- Competitive with Whisper Large v3 accuracy with significantly faster CPU inference

---

### Parakeet TDT vs Faster Whisper

We compare the performance of **Parakeet TDT (CPU)** against **faster-whisper (GPU & CPU)**.

The metric used is **Speedup Factor** (Audio Duration / Processing Time). Higher is better.

| Implementation | Hardware | Model | Precision | Speedup |
| --- | --- | --- | --- | --- |
| **Parakeet TDT** (Ours) | **CPU** (i7-12700KF) | **TDT 0.6B v3** | **int8** | **~29.7x** |
| **Parakeet TDT** (Ours) | **CPU** (i7-4790) | **TDT 0.6B v3** | **int8** | **~17.0x** |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | int8 | 13.2x |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | fp16 | 12.4x |
| faster-whisper | CPU (i7-12700K) | Small | int8 | 7.6x |
| faster-whisper | CPU (i7-12700K) | Small | fp32 | 4.9x |

*   **Parakeet TDT**: Benchmarked on Intel Core i7-12700K with ONNX Runtime INT8.
*   **faster-whisper**: Benchmarks from [official faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper).

### Detailed Parakeet Performance

| Metrics | Result |
| --- | --- |
| **Average Speedup** | **29.7x** |
| **Real Time Factor (RTF)** | **0.033** |
| **Max Speedup** | **~30x** |

### Extended Multilingual Benchmark (YouTube Samples)

Additional benchmark on real-world YouTube content across multiple languages:

| Language | Model Variant | Latency (s) | Speedup (RTF) | WER | CER |
| --- | --- | ---: | ---: | ---: | ---: |
| English | INT8 (`parakeet-tdt-0.6b-v3`) | 70.60 | 20.32x (0.049) | 5.13% | 2.35% |
| English | FP16 (`grikdotnet/parakeet-tdt-0.6b-fp16`) | 135.43 | 10.59x (0.094) | 5.48% | 2.83% |
| English | FP32 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 112.80 | 12.72x (0.079) | 5.53% | 2.85% |
| English | Whisper-Large-v3 (DeepInfra) | 53.45 | 26.84x (0.037) | 4.25% | 3.91% |
| Spanish | INT8 (`parakeet-tdt-0.6b-v3`) | 29.92 | 18.64x (0.054) | 19.45% | 13.79% |
| Spanish | FP16 (`grikdotnet/parakeet-tdt-0.6b-fp16`) | 48.52 | 11.49x (0.087) | 15.31% | 11.33% |
| Spanish | FP32 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 38.99 | 14.30x (0.070) | 15.31% | 11.33% |
| Spanish | Whisper-Large-v3 (DeepInfra) | 15.79 | 35.30x (0.028) | 20.70% | 18.05% |

> ⚠️ **Note:** YouTube subtitle references may contain errors. For verified accuracy, see LibriSpeech benchmark above.

## Requirements

*   [Docker](https://docs.docker.com/get-docker/) (Recommended)
*   Or: Python 3.14 and [FFmpeg](https://ffmpeg.org/)

### CPU Optimization
ONNX Runtime's CPU execution provider automatically dispatches AVX2/FMA kernels from the standard wheel when the host CPU supports them. The server now detects AVX2 at startup, reports the result in `/health`, and configures ONNX Runtime threading to use the available physical CPU cores while preventing NumPy/BLAS thread pools from competing with inference.

For hybrid CPUs (like Intel 12th-14th Gen), performance is still improved by pinning the process to Performance cores (P-cores). You can also override the auto-tuned defaults:

* `PARAKEET_ORT_INTRA_THREADS`: ONNX Runtime intra-op worker threads. Defaults to the lower of detected physical CPUs and available logical CPUs in the container/affinity mask, clamped to the cgroup CPU quota when one is set. Minimum: `1`.
* `PARAKEET_ORT_INTER_THREADS`: ONNX Runtime inter-op threads. Defaults to `1`, which is best for single-model inference. Minimum: `1`.
* `PARAKEET_INFER_WORKERS`: concurrent single-item inference calls on CPU (`PARAKEET_BATCHED=0`). Defaults to the available logical CPUs (after the cgroup quota is applied) divided by `PARAKEET_ORT_INTRA_THREADS`, capped at `4`, so workers × intra-op threads fits the CPU budget. Minimum: `1`.

### Running under an orchestrator

Two defaults matter when replicas start and stop frequently:

* **CPU limits are quotas, not cpusets.** A Kubernetes `resources.limits.cpu` is invisible to `sched_getaffinity()` and `psutil`, which keep reporting the node's full core count. Thread pools are now sized from the cgroup quota when one is present, so a 4-core pod no longer starts dozens of ORT threads. The quota is read from the process's own cgroup (via `/proc/self/cgroup`) and its ancestors, so it is found under systemd `CPUQuota=` and `--cgroupns=host` as well as in a private cgroup namespace. `/health` reports `cgroup_quota` next to the detected core counts so you can confirm what was applied.
* **Cold start.** `PARAKEET_WARMUP` (on by default) pushes one synthetic chunk through the model before `/healthz` reports ready, moving ONNX Runtime's first-inference kernel and arena setup into startup instead of onto the first real request. A warm-up that fails, or exceeds `PARAKEET_WARMUP_TIMEOUT_SEC` (default `120`), fails startup rather than reporting a replica ready that cannot run inference — raise the timeout on a slow host, or set `PARAKEET_WARMUP=false` to skip it. Container healthcheck `start_period` values in the Dockerfiles and `docker-compose.yml` allow for model load plus the default timeout; raise them by the same amount if you raise the timeout. Set `PARAKEET_HF_OFFLINE=true` when the model cache is pre-seeded — it skips the Hugging Face revision check that otherwise runs on every start, which adds up when many replicas start at once.

## Installation

### 🐳 Docker (Recommended)

The easiest way to get started. No dependencies to install!

**CPU Deployment:**
```bash
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
docker compose up parakeet-cpu -d
```

**GPU Deployment** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
```bash
docker compose up parakeet-gpu -d
```

The server will be available at `http://localhost:5092`. See [DOCKER.md](DOCKER.md) for more options.

---

### Conda (Alternative)

For development or customization:

```bash
conda create -n parakeet-onnx python=3.14
conda activate parakeet-onnx
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
pip install -r requirements.txt
```

## Usage

### Start the Server

Parakeet TDT provides an OpenAI-compatible API server.

```bash
conda activate parakeet-onnx
python server.py
```
*   **Port**: 5092
*   **Docs**: [http://127.0.0.1:5092/docs](http://127.0.0.1:5092/docs)

### Client Example (Python)

You can use the standard `openai` Python library to interact with the server.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="sk-no-key-required"
)

audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="parakeet-v3-fp32",  # omit to use the server default; see Model Selection
  file=audio_file,
  response_format="text"
)

print(transcript)
```

### Model Selection

Six variants are served. `GET /v1/models` returns these names, each with its
aliases in an `aliases` field; `GET /v1/models/{id}` accepts either form.

| Model Name | Precision | ONNX weights | Languages | Former names, still accepted |
|------------|-----------|--------------|-----------|------------------------------|
| `parakeet-v3-fp32` | FP32 | `istupakov/parakeet-tdt-0.6b-v3-onnx` | 25 | `parakeet-v3`, `istupakov/parakeet-tdt-0.6b-v3-onnx` |
| `parakeet-v3-fp16` | FP16 | `grikdotnet/parakeet-tdt-0.6b-fp16` | 25 | `grikdotnet/parakeet-tdt-0.6b-fp16` |
| `parakeet-v3-int8` | INT8 | `nemo-parakeet-tdt-0.6b-v3` | 25 | `parakeet-tdt-0.6b-v3` |
| `parakeet-v2-fp32` | FP32 | `istupakov/parakeet-tdt-0.6b-v2-onnx` | English only | `parakeet-v2`, `istupakov/parakeet-tdt-0.6b-v2-onnx` |
| `parakeet-v2-fp16` | FP16 | `ysdede/parakeet-tdt-0.6b-v2-onnx` | English only | — |
| `parakeet-v2-int8` | INT8 | `nemo-parakeet-tdt-0.6b-v2` | English only | `parakeet-tdt-0.6b-v2` |

**Defaults.** FP16 halves VRAM at identical output on GPU, so a GPU deployment
defaults to `parakeet-v3-fp16`. On CPU, ONNX Runtime upcasts FP16 (slower), so
`PARAKEET_USE_GPU=false` defaults to `parakeet-v3-fp32`. With
`PARAKEET_USE_GPU=auto` the choice is made at startup by probing whether CUDA
actually loads on the host. `PARAKEET_DEFAULT_MODEL` overrides all three, and
`GET /health` reports the model that was resolved.

INT8 is the fastest on CPU but measurably drops words after silences, and the
multilingual benchmark above shows it ~4 WER points worse than FP32 on Spanish.
Pick it deliberately rather than by default.

The default model is loaded before the service reports ready; the others are
lazy-loaded on first use and cached afterwards.

**To select a model via API:**
```python
transcript = client.audio.transcriptions.create(
  model="parakeet-v3-fp16",  # Select the FP16 variant
  file=audio_file,
  response_format="text"
)
```

### Response formats

`response_format` accepts `json` (default), `text`, `srt`, `vtt` and
`verbose_json`. `verbose_json` returns segments, and word timestamps as well
when `timestamp_granularities[]=word` is sent.

### Batch transcription

`POST /v1/audio/transcriptions/batch` takes several `files=` parts in one
request and returns `{"results": [{"filename", "text", "duration"}, ...],
"batch_size": N}`. It shares the model form field with the single-file
endpoint. Requests are bounded by `PARAKEET_MAX_BATCH_FILES` (16) and
`PARAKEET_MAX_BATCH_BYTES` (512 MiB); see the env knob table in
[OPTIMIZATION.md](OPTIMIZATION.md#env-knobs) for the per-request limits that
apply to both endpoints.

### Interactive API docs

The server exposes Swagger UI for trying requests from the browser, including
picking a model variant per request.
Access it at: **[http://127.0.0.1:5092/docs](http://127.0.0.1:5092/docs)**

There is no longer a drag-and-drop upload page: it belonged to the removed
Flask service and was never served by `server.py`.

## 🔌 Open WebUI Integration

**This project provides out-of-the-box compatibility with [Open WebUI](https://openwebui.com/)**, serving as a drop-in replacement for OpenAI's speech-to-text API. Experience lightning-fast, local transcription across 25 languages with automatic language detection!

### Setup Instructions

1.  **Start the Parakeet Server** (if not already running):
    ```bash
    conda activate parakeet-onnx
    python server.py
    ```
    The server will be available at `http://127.0.0.1:5092`

2.  **Configure Open WebUI**:
    - Navigate to **Open WebUI Settings -> Audio**
    - Set **STT Engine** to `OpenAI`
    - Set **OpenAI Base URL** to `http://127.0.0.1:5092/v1`
    - Set **OpenAI API Key** to `sk-no-key-required`
    - Set **STT Model** to `parakeet-v3-fp32` (or leave it as the server default)
    - Click **Save**

3.  **Start Using Voice!**
    - All voice interactions in Open WebUI will now be transcribed locally
    - Enjoy real-time transcription speeds (up to 30x faster than real-time on modern CPUs)
    - Automatic language detection across all 25 supported languages
    - Complete privacy - all processing happens locally on your machine

## Model details

When running the application, the ONNX models are downloaded and cached under the `models/` directory (`PARAKEET_MODELS_DIR`). The models served are **Parakeet TDT 0.6B v3** converted to ONNX at FP32, FP16 and INT8, plus the English-only v2 equivalents. The v3 variants cover 25 European languages; which one is used by default is described under [Model Selection](#model-selection).

## 🙏 Acknowledgments

This project stands on the shoulders of giants and wouldn't be possible without:

- **[Shadowfita](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi)** - For the original FastAPI implementation that served as the foundation for this project
- **[NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** - For developing and open-sourcing the exceptional Parakeet TDT model family
- **[groxaxo](https://github.com/groxaxo)** - The mastermind behind this project, bringing together ONNX optimization, multilingual support, and seamless OpenAI API compatibility

Thank you to all contributors and the open-source community for making high-performance, local speech recognition accessible to everyone!

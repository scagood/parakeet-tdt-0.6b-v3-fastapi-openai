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
measured at FP32, the default precision; on GPU, `quantization=fp16` halves
VRAM at the same output.

| Workload                | CPU optimized      | GPU profile (FP32) | Δ        |
|-------------------------|--------------------|--------------------|----------|
| 300 s file (single)     | 10.41 s / 27.2×    | **1.37 s / 205.9×**| **+7.6×** |
| 16× 10 s concurrent     | 39.3× throughput   | **200.3× throughput**| **+5.1×** |

See [OPTIMIZATION.md](OPTIMIZATION.md) for the full benchmark, design
rationale, and tunable env knobs.

```bash
python server.py                  # serve on :5092

# CPU override
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
* **Cold start.** List the models a replica should serve warm in `PARAKEET_PRELOAD_MODELS` (comma-separated `model` or `model:quantization`, e.g. `parakeet-v3`); nothing is preloaded by default. `PARAKEET_WARMUP` (on by default) then pushes one synthetic chunk through each preloaded model before `/healthz` reports ready, moving ONNX Runtime's first-inference kernel and arena setup into startup instead of onto the first real request. A warm-up that fails, or exceeds `PARAKEET_WARMUP_TIMEOUT_SEC` (default `120`), fails startup rather than reporting a replica ready that cannot run inference — raise the timeout on a slow host, or set `PARAKEET_WARMUP=false` to skip it. Container healthcheck `start_period` values in the Dockerfiles and `docker-compose.yml` allow for model load plus the default timeout; raise them by the same amount if you raise the timeout. Set `PARAKEET_HF_OFFLINE=true` when the model cache is pre-seeded — it skips the Hugging Face revision check that otherwise runs on every start, which adds up when many replicas start at once.

## Installation

### 🐳 Docker (Recommended)

The easiest way to get started. No dependencies to install!

**CPU Deployment:**
```bash
git clone https://github.com/scagood/stt-api
cd stt-api
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
git clone https://github.com/scagood/stt-api
cd stt-api
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
It is a client dependency, not a server one, so install it separately with
`pip install openai`.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="sk-no-key-required"
)

audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="parakeet-v3",  # required; see Model Selection
  file=audio_file,
  response_format="text"
)

print(transcript)
```

### Model Selection

Every request must name its `model`; there is no server default, and a request
without one is rejected with 422. Precision is a separate, optional
`quantization` field: `fp32` (the default, on any hardware), `fp16` or `int8`.
It can also follow the name after a colon: `model=parakeet-v3:fp16` is the same
as `model=parakeet-v3` with `quantization=fp16`; if a request sends both, they
must agree.
`GET /v1/models` lists every model with the languages it transcribes and the
quantizations it offers.

| Model | Languages | fp32 | fp16 | int8 |
|-------|-----------|------|------|------|
| `parakeet-v3` | 25 | `Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx` | `Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx` (fp16 encoder, fp32 decoder) | `Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx` |
| `parakeet-v2` | English only | `istupakov/parakeet-tdt-0.6b-v2-onnx` | `ysdede/parakeet-tdt-0.6b-v2-onnx` | `istupakov/parakeet-tdt-0.6b-v2-onnx` |

Whisper is served as `whisper-tiny`, `whisper-base`, `whisper-small`,
`whisper-medium`, `whisper-large-v3` and `whisper-large-v3-turbo` (99
languages), plus English-only `whisper-tiny.en`, `whisper-base.en`,
`whisper-small.en` and `whisper-medium.en`, each in all three quantizations.

**Choosing a precision.** FP16 halves VRAM on GPU, so ask for `fp16` there.
On CPU, ONNX Runtime upcasts FP16 (slower), so keep `fp32`.

`parakeet-v3` runs Olicorne's re-export of NVIDIA's `.nemo` checkpoint in all
three precisions. On a 648 s English audiobook chapter it scored 1.13% (fp32),
1.07% (fp16) and 1.20% (int8) WER against 1.20%, 1.20% and 1.83% for the
istupakov/grikdotnet exports it replaced, with int8 ~25% faster. It was chosen
on CPU and has **not been tested on a GPU** or outside English; the
`parakeet-v3` entry in `parakeet_service/models.yaml` lists what to revert to if
CUDA gives trouble.

INT8 is the fastest on CPU. istupakov's `parakeet-v3` int8 measurably dropped
words after silences, and the multilingual benchmark above shows it ~4 WER
points worse than FP32 on Spanish; Olicorne's, which replaced it, matched FP32
on the English chapter above but has no multilingual numbers yet. Pick INT8
deliberately rather than by default.

Models named in `PARAKEET_PRELOAD_MODELS` (as `model` for fp32, or
`model:quantization`) are loaded (and warmed up) before the service reports
ready; the others are lazy-loaded on first use and cached afterwards. A
preloaded model is never used for a request that names another. A model that
cannot be loaded (its download fails, it is missing from the cache under
`PARAKEET_HF_OFFLINE=true`, ONNX Runtime refuses it) answers 503 with a
`detail` naming the model and the cause; the failure is not cached, so the
next request tries again.

**To select a model and precision via API:**
```python
transcript = client.audio.transcriptions.create(
  model="parakeet-v3",
  file=audio_file,
  response_format="text",
  extra_body={"quantization": "fp16"},  # omit for fp32
)
```

#### Your own model catalog

The models above are defined in [`parakeet_service/models.yaml`](parakeet_service/models.yaml):
per model its family, languages and chunk lengths, and per quantization a
Hugging Face repo, a pinned commit and, where they differ from the fp32
defaults, the files to load. Its `aligners` section lists the
[word aligners](#word-timestamps) the same way: per aligner its export layout,
the languages it aligns, the steps that spell a transcript for it, its CTC
tokens and default quantization, and per quantization a repo, a pinned commit
and the files that differ (`aligners: {}` serves none). To serve a
different set without rebuilding the image, point `PARAKEET_MODEL_CATALOG` at
another file of the same shape. It **replaces** the built-in catalog, so copy
the built-in file and edit it. The file is checked at startup and the service
refuses to start on a mistake, naming it; it is read only then, so restart
after changing it.

On Kubernetes, keep it in a ConfigMap:

```bash
kubectl create configmap parakeet-models --from-file=models.yaml=parakeet_service/models.yaml
```

```yaml
# in the Deployment's pod spec
containers:
  - name: parakeet
    env:
      - name: PARAKEET_MODEL_CATALOG
        value: /config/models.yaml
    volumeMounts:
      - name: model-catalog
        mountPath: /config
volumes:
  - name: model-catalog
    configMap:
      name: parakeet-models
```

### Response formats

`response_format` accepts `json` (default), `text`, `srt`, `vtt` and
`verbose_json`. `verbose_json` returns segments, and word timestamps as well
when `timestamp_granularities[]=word` is sent. `timestamp_granularities[]`
takes `word` and `segment` (segments come either way); anything else is a 400.

#### Word timestamps

Word times can come from a forced aligner rather than from Parakeet. Parakeet
decides the words, then a character-level CTC model finds where each one
starts and ends, WhisperX-style but on ONNX Runtime with no PyTorch. Parakeet's
own word times sit on 80 ms frames and their ends are estimated; aligned times
sit on 20 ms frames and the ends come from the audio.

A request names the aligner, as it names the model: send the form field
`aligner`, and optionally `aligner_quantization` (else the aligner's default),
or both at once as `aligner=mms-300m-forced-aligner:fp32`.
There is no default aligner: a request that names none gets Parakeet's times.
`GET /v1/aligners` lists them, like `GET /v1/models`:

| Aligner | Languages | Error, start / end (English TTS) | License |
|---|---|---|---|
| [`mms-300m-forced-aligner`](https://huggingface.co/onnx-community/mms-300m-1130-forced-aligner-ONNX) | `en` | 37 / 106 ms | **CC-BY-NC-4.0: non-commercial only** |
| [`wav2vec2-large-xlsr-53-english`](https://huggingface.co/Xenova/wav2vec2-large-xlsr-53-english) | `en` | 48 / 104 ms | Apache-2.0 (the model it exports) |
| [`omnilingual-ctc-300m`](https://huggingface.co/OpenVoiceOS/omnilingual-asr-ctc-300m-onnx) | `en` and Parakeet v3's other 24 | 45 / 117 ms | Apache-2.0 |
| [`wav2vec2-base-960h`](https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX) | `en` | 57 / 131 ms | Apache-2.0 |

For English, use `mms-300m-forced-aligner`: it is the most accurate, and on
real audiobook narration it sounds clearly the best. Its licence is
non-commercial; for commercial use, `wav2vec2-large-xlsr-53-english` is the
next best. `wav2vec2-base-960h` is the smallest and fastest, but the least
accurate. Each is offered at `int8` (the default) and `fp32`, which is about as
accurate and 4x the download.

```python
transcript = client.audio.transcriptions.create(
  model="parakeet-v3",
  file=audio_file,
  response_format="verbose_json",
  timestamp_granularities=["word"],
  language="en",
  extra_body={"aligner": "mms-300m-forced-aligner"},  # non-commercial licence
)
```

An unknown aligner or quantization, or an aligner that does not align the
request's language, is a 400 naming what is available. Which aligners exist,
and the languages each aligns, is set in the
[model catalog](#your-own-model-catalog).

* **Language.** `language` is a bare ISO 639-1 code (`en`, `fr`), or empty or
  `auto` for the default; anything else (`en-US`, `EN`, `English`) is a 400 on
  every request. For word timestamps it must be one the aligner aligns. A
  request without `language` (or with `auto`) is aligned as
  `PARAKEET_ALIGN_DEFAULT_LANGUAGE`, English unless you change it: nothing
  detects the language. An English-only model's words (`parakeet-v2`,
  `whisper-*.en`) are aligned as English whatever `language` says. A chunk
  whose words are mostly in another alphabet than the aligner's (Cyrillic,
  Greek, ...) keeps Parakeet's times, but other Latin-script languages sent
  without `language` are aligned as English — send `language`. Whisper has no
  word times of its own, so it returns words only when the request names an
  aligner, and a multilingual Whisper model only when the request also names
  the language. If a chunk can't be aligned at all, `words` is null; a word the
  aligner can't place sits between its aligned neighbours.
* **Numbers and symbols** are aligned as spoken by the English aligners
  (`mms-300m-forced-aligner`, `wav2vec2-large-xlsr-53-english`,
  `wav2vec2-base-960h`). `omnilingual-ctc-300m` drops numbers in every
  language, English included: a number keeps Parakeet's times, and the words
  around it are aligned as usual. As said in English: `42` as "forty two", `2026` as
  "twenty twenty six", `$5 million` and `$5m` as "five million dollars", `-5`,
  `50%`, `21st`, and units like `20lb`, `5kg`, `70mph`, `20°C`; accents are
  folded (`café`). A number is read together with the words that change how it
  is said, the same way [spoken numbers](#spoken-numbers) reads it (on or off):
  `5 May` as "the fifth of May", `July 4` as "July fourth", `90 mph` as "ninety
  miles per hour", `715 a.m.` as "seven fifteen a.m.", `100-200` as "one
  hundred to two hundred". Only timing depends on this — the text is never
  changed — so money heard as weight (`25 lb` for "twenty five pounds") still
  lines up. A count in year range (`1500`) is read as a year, so if it was said
  "one thousand five hundred" it starts a little late. A lone letter is looked
  for as its name, which is how it sounds (`R&D` as "ar and dee", `Vitamin C`
  as "vitamin see", the "p" of `£11.40p`).
* **Transcript mistakes.** On clean speech, a word Parakeet missed or got wrong
  does not drag its neighbours' times (as in MMS forced alignment, the gaps
  between words can absorb speech the transcript lacks); in heavy noise it
  occasionally still does. An invented word needs somewhere to go: in a pause
  it takes the pause, but a long one invented in the middle of continuous
  speech pushes its neighbours aside.

The aligner only runs when the request names one and words are returned (and
for [spoken numbers](#spoken-numbers)), one request at a time on its own thread
pool so it never holds up other requests' audio decoding. Each aligner
downloads on the first request that names it (int8: ~320 MB, or ~95 MB for
`wav2vec2-base-960h`; fp32 is 4x) and runs on CPU, adding roughly 3 s per 30 s
of audio on a 4-core machine (2 s for `wav2vec2-base-960h`). If a download
fails, word times fall back to Parakeet's and the load is retried every 5
minutes; `/health` reports each aligner's state per
quantization under `aligner`.

| Variable | Default | |
|---|---|---|
| `PARAKEET_ALIGN_DEFAULT_LANGUAGE` | `en` | language assumed when a request sends none, for alignment and spoken numbers: a bare ISO 639-1 code, or empty to use them only when `language` is sent |
| `PARAKEET_ALIGN_THREADS` | `min(4, physical cores)` | CPU threads for the aligner |

#### Comparing models and aligners by ear

With `PARAKEET_COMPARE_UI=true`, `GET /compare` serves a page for choosing
between them by listening. Pick an audio file and cut it to a clip (From and
To, in seconds), then add rows: a model at a quantization, with or without an
aligner (`parakeet-v2:int8`; `parakeet-v2:int8` + `mms-300m-forced-aligner:int8`;
...). **Compare** sends the clip through `/v1/audio/transcriptions` once per
row, one row at a time, and lines up each row's words under the clip's
waveform. Click a word to hear exactly the span that row gave it; the same word
is then picked out in every row, close up and in a table of starts and ends.
The arrow keys step through words and switch rows, Enter replays, and playback
can be slowed to ½×. A file of exact word times, as synthetic speech has (JSON:
a verbose_json response or its `words`; or SRT/VTT with one cue per word), adds
a dashed reference row and each row's average error against it.

The browser decodes the file and sends only the clip, as a 16 kHz WAV, so the
page and the server hear the same samples. The page is off by default: each
row is a full transcription, and loads any model or aligner it names, which
then stays loaded (see `PARAKEET_MODEL_CACHE_SIZE`).

| Variable | Default | |
|---|---|---|
| `PARAKEET_COMPARE_UI` | `false` | serve the `/compare` page |

#### Spoken numbers

Parakeet writes numbers its own way, and not consistently: "twenty-five pounds"
may come back as `£25` or `25 lb`, "five dollars" as `$5`, "ten thirty" as
`1030`, and "nine one one" as `911`. With the form field `spoken_numbers=true`
(OpenAI SDK: `extra_body={"spoken_numbers": True}`), or
`PARAKEET_SPOKEN_NUMBERS=true` for requests that don't send it, English
transcripts (text, segments, words, every response format and the batch
endpoint) say numbers, money and units in words, the way they were said:

| Parakeet wrote | Transcript says |
|---|---|
| `$5`, `cost$25` | five dollars, cost twenty-five dollars |
| `£25`, `25 lb` | twenty-five pounds |
| `$5 million`, `$5m` | five million dollars |
| `20°C`, `50%`, `21st` | twenty degrees Celsius, fifty percent, twenty-first |
| `5 May`, `90 mph` | the fifth of May, ninety miles per hour |
| `MP3`, `COVID-19`, `5m`, `12C` | unchanged: names, or ambiguous |

Different speech often comes out as the same text — `£2.10` is "two pounds
ten", "two pounds and ten pence" or "two ten"; `1500` is "fifteen hundred" or
"one thousand five hundred"; `911` is "nine one one" or "nine eleven" — so each
number's possible readings are scored against its stretch of the audio by the
request's [aligner](#word-timestamps) (`aligner=`, as for word times), and the
one that was said is kept. The choice was tuned and measured with
`wav2vec2-base-960h`; the other aligners have not been measured for it yet.
Likely mishearings of an amount are scored too (`£1.10` for "two pounds ten",
`€3` for "thirty euros"; not of a time, date, ordinal or code), and one
replaces Parakeet's number only when the audio prefers it by a clear margin.
Word times follow the spoken words.

* **Cost.** Almost every number has several readings or a likely mishearing
  (97% of the chunks with a number in our test corpus), so the aligner model
  runs for most chunks that contain a number: roughly 2 s per 30 s chunk on a
  4-core CPU, plus 0.5–0.9 s to choose on a dense 60–75 s chunk (a number every
  few seconds; prices cost the most). Those requests queue on the aligner's
  single worker with word requests, so audio with numbers in it is heard at
  about 13x real time however many requests are waiting. Requests with no
  number, or whose numbers have nothing to decide (`6pm`, `6 pm`), are not held
  up. With no aligner named, or the model unavailable, each number's
  first reading is used — a fixed default, usually the most common one ("two
  pounds ten", "ten to fifteen"), though a code is read as an amount (`911` as
  "nine hundred eleven"), and so are a time and a year before 1100 (`1030` as
  "one thousand thirty", `1066` as "one thousand sixty-six") — and Parakeet's
  number is kept.
* **Language.** English only, decided as for word timestamps: a request without
  `language` is taken as `PARAKEET_ALIGN_DEFAULT_LANGUAGE`, English unless you
  change it. A chunk mostly in another alphabet is left as written, but
  Latin-script languages sent without `language` are rewritten as if English —
  set the default empty if you serve them.

| Variable | Default | |
|---|---|---|
| `PARAKEET_SPOKEN_NUMBERS` | `false` | spoken numbers for requests that don't send `spoken_numbers`; `true` says them in every English transcript unless the request sends `spoken_numbers=false` |

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

To compare models and aligners on your own audio by ear, see
[the compare page](#comparing-models-and-aligners-by-ear).

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
    - Set **STT Model** to `parakeet-v3` (required: the server has no default model)
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

# Optimization report (1.x)

How the 1.x FastAPI service replaced the original Flask one, and what was
measured along the way. It is kept as history: every figure here was measured
on the `parakeet-v3` exports 1.x served (istupakov fp32 and int8, grikdotnet
fp16). 2.0 serves Olicorne's export instead, which has **not been tested on a
GPU**. For current settings, see [the README](README.md#configuration).

## TL;DR

Replaced the legacy Flask + Waitress + `ffmpeg-silencedetect` design with a
new FastAPI service inspired by `parakeet-flash`. On a single i7-12700KF
(8P + 4E, 20 threads), CPU-only ONNX INT8:

| Workload                | Baseline (Flask/Waitress) | Optimized (FastAPI/InferencePool) | Δ          |
|-------------------------|---------------------------|-----------------------------------|------------|
| 10 s file (single)      | 0.690 s  /  14.6× RTFx    | **0.661 s  /  15.2× RTFx**        | +4%        |
| 60 s file (single)      | 3.11 s   /  18.2× RTFx    | **3.02 s   /  18.7× RTFx**        | +3%        |
| **300 s file (single)** | 17.96 s  /  15.7× RTFx    | **10.41 s  /  27.2× RTFx**        | **+73%**   |
| 16× 10 s concurrent     | wall 4.64 s / **34.6×** thrpt | wall 4.10 s / **39.3×** thrpt | +13%       |

> RTFx = audio_seconds / wall_seconds (higher is better).

The biggest CPU win is on long files: parallel Silero‑VAD chunking + a fan-out
inference pool turn a 5-minute clip from 18 s of inference into 10 s.

With `onnxruntime-gpu==1.26.0` and CUDA provider binding validated from the
live ORT sessions, the best stable RTX 3090 profile was FP32 + GPU
micro-batching.

| Workload                | CPU optimized      | GPU profile        | Δ        |
|-------------------------|--------------------|--------------------|----------|
| 10 s file (single)      | 0.661 s / 15.2×    | **0.058 s / 174.4×**| +11.5×  |
| 60 s file (single)      | 3.02 s / 18.7×     | **0.229 s / 246.9×**| +13.2×  |
| **300 s file (single)** | 10.41 s / 27.2×    | **1.37 s / 205.9×** | **+7.6×** |
| 16× 10 s concurrent     | 39.3× throughput   | **200.3× throughput** | **+5.1×** |

That profile is now the default, so a plain `python server.py` on a GPU host
runs it. The CPU runs used:

```bash
PARAKEET_USE_GPU=false \
PARAKEET_PRELOAD_MODELS=parakeet-v3:int8 \
PARAKEET_ORT_INTRA_THREADS=12 \
python server.py
```

## Method

The rule was to establish a measured baseline before changing code, and to
identify bottlenecks with evidence:

1. **Baseline**: the Flask `app.py` (removed in 2.0) running on Waitress,
   default settings, warmed up.
2. **Profile**: looked at where time was going during a 300 s clip.
   The legacy server spawns `ffmpeg -loglevel … -filter silencedetect` to
   find chunks, then runs **sequential** per-chunk inference. Inference
   threads in Waitress are 8 but they share the ORT intra-op thread pool,
   so concurrent requests battle for CPU.
3. **Hypotheses tested**:
   - Drop the per-chunk ffmpeg subprocess by decoding once in-process.
   - Replace `silencedetect` with Silero-VAD-based packing into ~60 s
     chunks on pause midpoints.
   - Cross-request micro-batching (`recognize([w1..wN])`).
   - Parallel single-item inference pool (`InferencePool`).
   - Pin to P-cores only via `taskset`.
   - GPU provider setup and model/worker/batch sweeps on RTX 3090.

## Findings

### What worked

- **In-process audio decode** (`parakeet_service/audio.py`): single
  `ffmpeg -i pipe:0 -ac 1 -ar 16000 -f s16le pipe:1` for non-WAV inputs,
  and stdlib `wave` + numpy for WAVs already at 16 kHz. Removes per-chunk
  subprocess fork/exec.
- **Silero-VAD auto-chunking** (`parakeet_service/chunker.py`): pack speech
  segments into each model's target chunk length (60 s for Parakeet v3),
  cutting on pause midpoints, with min/max guards. Falls back to energy-RMS
  when silero-vad is unavailable.
  - Bypasses chunking entirely for clips no longer than the model's
    `chunk_max_sec` (75 s for Parakeet v3), so 10 s and 60 s files are
    processed in a single ORT call.
- **Parallel `InferencePool`** (`parakeet_service/batchworker.py`):
  4 worker threads, each calling `model.recognize(single_wav)`. Used for
  both concurrent requests *and* fan-out of multiple chunks from one long
  request, via `asyncio.gather`. This is what produced the 73% jump on
  300 s files.
- **FastAPI + uvicorn**: removes Flask's per-thread blocking model and
  makes the audio pipeline async-friendly without changing the
  OpenAI-compatible response shape.
- **CUDA preload + provider validation** (`parakeet_service/model.py`):
  `onnxruntime-gpu` can list `CUDAExecutionProvider` even when the provider
  later fails to load cuDNN. The loader now calls `ort.preload_dlls()` and,
  when `PARAKEET_USE_GPU=true`, raises if the live encoder/decoder sessions
  do not actually bind to CUDA/TensorRT first.
- **GPU micro-batching**: on RTX 3090 the fastest stable profile was FP32
  model `istupakov/parakeet-tdt-0.6b-v3-onnx` (the 1.x export) with `PARAKEET_BATCHED=1`,
  `PARAKEET_MAX_BATCH_SIZE=4`, and `PARAKEET_BATCH_WINDOW_MS=4`.

### What did NOT work (and why)

- **Cross-request micro-batching on CPU INT8** (`BatchWorker`):
  initial implementation collected jobs in an 8 ms window then called
  `model.recognize([w1..w8])`. Result: concurrent throughput **dropped
  from 34.6× to 20.6×**. Reason: on CPU INT8, batched `recognize` scales
  near-linearly in wall time per item (you pay padding to the longest
  clip times the batch size). The optimization is GPU-shaped, not
  CPU-shaped. We **kept** `BatchWorker` behind the `PARAKEET_BATCHED=1`
  env flag for users on `onnxruntime-gpu`, where this design is expected
  to win, but the default is `InferencePool`.
- **P-core pinning** (`taskset -c 0-15` via `pin_pcores.sh`): mildly
  *hurt* on this workload — concurrent throughput dropped from
  39.3× to 33.4×. Restricting affinity also blocks ORT's lightweight
  ops and audio I/O from spilling onto the 4 E-cores. Kept the script
  available for users who want predictability but it is not the default.
- **Resampling WAVs in numpy** (`audio.py`): the in-process path only
  covers 16 kHz input, so every 44.1/48 kHz upload pays an ffmpeg
  fork/exec. Tried replacing that with a polyphase windowed-sinc
  resampler in numpy. It is correct — 47 dB SNR against `swr` — but
  slower than the subprocess it replaces on anything but very short
  clips: 60 s of 44.1 kHz stereo took 627 ms against ffmpeg's 213 ms,
  and 300 s took 3.6 s against 892 ms. A plain `np.interp` resample does
  beat ffmpeg (75 ms on that 60 s file) but aliases badly, 21-24 dB SNR,
  which is not a trade worth making in front of an ASR model. What did
  hold up is the conversion work either side of resampling: downmixing
  and integer-width conversion in numpy beat ffmpeg at every size tested
  — 2.9x on 300 s of 16 kHz stereo, 5.6x on 300 s of 24-bit mono, and
  38-580x on 5 s clips, where ffmpeg's ~55 ms fork/exec dominates — and
  match its output to within one int16 LSB. So `_decode_pcm_wav` takes
  mono and stereo at any PCM sample width when the file is already at
  16 kHz, and hands everything else to ffmpeg. Measured on a 4-core
  Xeon 2.10 GHz with ffmpeg 6.1.1 and numpy 2.4.6, not the 8-core box
  the RTFx numbers above come from.
- **INT8 on CUDA**: the 1.x INT8 export (istupakov) is a poor CUDA target. It
  bound to CUDA after preload, but measured only about 8.6× RTFx on the
  300 s file and 8.7× concurrent throughput. Use it on CPU, not GPU.
- **FP32 pool with 4 workers under sustained concurrent load**: one-shot
  concurrency looked very fast, but the repeated finalist run hit CUDA OOM
  and produced zero successful concurrent requests. Avoid that profile.

## Architecture

```
stt-api/
├── server.py                    # uvicorn entry point (port 5092)
├── pin_pcores.sh                # Optional P-core taskset wrapper
├── benchmark.py                 # Older standalone benchmark (see below)
└── parakeet_service/
    ├── config.py                # Env knobs, CPU detection, catalog loading
    ├── models.yaml              # Model and aligner catalog
    ├── audio.py                 # In-process decode (wave / single ffmpeg)
    ├── chunker.py               # Silero-VAD auto-chunking
    ├── model.py                 # ORT session options, providers, cache
    ├── batchworker.py           # InferencePool (CPU) + BatchWorker (GPU)
    ├── aligner.py               # CTC forced alignment for word timestamps
    ├── spoken.py                # Spoken-form numbers, money and units
    ├── number_parse.py          # Number parsing for spoken.py and routes.py
    ├── routes.py                # OpenAI-compatible endpoints
    ├── compare.html             # The /compare page
    └── main.py                  # FastAPI lifespan
```

## Reproducing the benchmark

The bench corpus (three real recordings of 10 s, 60 s and 300 s) and its
`bench.py`, which measured sequential mean/p50/p95 plus concurrent wall-clock
throughput, were never committed, so these runs can't be reproduced from this
repo. `benchmark.py` at the root is a separate, older script: it takes no
options, and its server URL and audio folder (`/home/op/mp3`) are hard-coded.

## GPU sweep highlights

All rows below were validated from live ORT session providers with CUDA first.
The final rows use `--sequential-n 3`; earlier exploratory sweeps used one
sequential run per duration.

| Profile | 10 s | 60 s | 300 s | 16×10 s throughput | Notes |
|---------|------|------|-------|--------------------|-------|
| FP32 batch b4/w4ms (final) | 174.4× | 246.9× | **205.9×** | **200.3×** | best stable overall |
| FP16 batch b4/w2ms (final) | **221.4×** | **509.5×** | 143.0× | 91.0× | best short/medium single request |
| FP16 pool w4 (exploratory) | 217.0× | 337.3× | 134.7× | 109.7× | simple low-risk GPU profile |
| FP32 pool w4 (exploratory) | 197.1× | 228.6× | 100.3× | 244.3× | OOMed under final concurrent repeat |
| INT8 pool w1 on CUDA | 8.8× | 8.9× | 8.6× | 8.7× | avoid on GPU |

## Accuracy and speed benchmarks

For the current export's accuracy, see [the README](README.md#choosing-a-model).

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
- All three precisions scored the same (97.84%) on these 50 short samples. On
  longer audio the int8 export dropped words after silences, and it was ~4 WER
  points worse on Spanish (below)
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

*   **Parakeet TDT**: Benchmarked on the CPUs listed, with ONNX Runtime INT8.
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
| English | INT8 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 70.60 | 20.32x (0.049) | 5.13% | 2.35% |
| English | FP16 (`grikdotnet/parakeet-tdt-0.6b-fp16`) | 135.43 | 10.59x (0.094) | 5.48% | 2.83% |
| English | FP32 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 112.80 | 12.72x (0.079) | 5.53% | 2.85% |
| English | Whisper-Large-v3 (DeepInfra) | 53.45 | 26.84x (0.037) | 4.25% | 3.91% |
| Spanish | INT8 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 29.92 | 18.64x (0.054) | 19.45% | 13.79% |
| Spanish | FP16 (`grikdotnet/parakeet-tdt-0.6b-fp16`) | 48.52 | 11.49x (0.087) | 15.31% | 11.33% |
| Spanish | FP32 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 38.99 | 14.30x (0.070) | 15.31% | 11.33% |
| Spanish | Whisper-Large-v3 (DeepInfra) | 15.79 | 35.30x (0.028) | 20.70% | 18.05% |

> ⚠️ **Note:** YouTube subtitle references may contain errors. For verified accuracy, see LibriSpeech benchmark above.

## Future work

- **Multi-GPU serving**: the host has 3× RTX 3090 + 1× RTX 3060. The current
  service intentionally binds one ORT model to one CUDA device. Running one
  uvicorn process per GPU behind a local load balancer should scale aggregate
  throughput further.
- **Streaming endpoint**: the `parakeet-flash` reference project ships
  WebSocket streaming with VAD; adding it here is a natural follow-up.

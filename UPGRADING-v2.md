# Upgrading from 1.5.0 to 2.0.0

2.0.0 is everything merged since v1.5.0. Most of it is opt-in, but the way a
request picks its model has changed, and no compatibility shim covers the old
way. **Clients that don't send `model`, or that send a precision-suffixed name
such as `parakeet-v3-fp16`, get an error after the upgrade.** Read the API
section before you upgrade a shared server.

The project is also renamed to `stt-api`, and 2.0.0 is published only as
`ghcr.io/scagood/stt-api`. The old image name stops updating, so pulling it
won't upgrade you. See [The project is now `stt-api`](#the-project-is-now-stt-api).

- [For API users](#for-api-users)
- [For server admins](#for-server-admins)

## For API users

### Always send `model`, and name the precision with a colon or separately

`model` is now required. A request without it gets a **422**. It names a model,
and optionally its precision after a colon: `model=parakeet-v3:fp16`. Or pick
the precision with the separate `quantization` form field (`fp32`, `fp16` or
`int8`); the two mean the same, and if you send both they must agree. With no
precision you get `fp32` on every server, GPU or CPU. An unknown model or
quantization gets a **400** that lists the valid choices. A hyphen doesn't
work: `parakeet-v3-fp16` is an unknown model.

| You sent in 1.5.0 | Send in 2.0.0 |
|---|---|
| no `model` to a GPU server | `model=parakeet-v3:fp16` (what 1.5.0 gave you) |
| no `model` to a CPU server | `model=parakeet-v3` |
| `parakeet-v3`, `parakeet-v3-fp32`, `istupakov/parakeet-tdt-0.6b-v3-onnx` | `model=parakeet-v3` |
| `parakeet-v3-fp16`, `grikdotnet/parakeet-tdt-0.6b-fp16` | `model=parakeet-v3:fp16` |
| `parakeet-v3-int8`, `parakeet-tdt-0.6b-v3` | `model=parakeet-v3:int8` |
| `parakeet-v2`, `parakeet-v2-fp32`, `istupakov/parakeet-tdt-0.6b-v2-onnx` | `model=parakeet-v2` |
| `parakeet-v2-fp16` | `model=parakeet-v2:fp16` |
| `parakeet-v2-int8`, `parakeet-tdt-0.6b-v2` | `model=parakeet-v2:int8` |

Each `model=name:precision` in the table can instead be sent as `model=name`
with `quantization=precision`.

`parakeet-tdt-0.6b-v3` was an alias for the **int8** weights, not fp32. If you
used it without meaning to pick int8, send `model=parakeet-v3` and nothing else.

With the colon form, OpenAI SDK clients only change the model string.
`quantization` isn't an OpenAI parameter, so the SDK would send it through
`extra_body`:

```python
# 1.5.0
client.audio.transcriptions.create(model="parakeet-v3-fp16", file=f)

# 2.0.0
client.audio.transcriptions.create(model="parakeet-v3:fp16", file=f)
# or
client.audio.transcriptions.create(model="parakeet-v3", file=f, extra_body={"quantization": "fp16"})
```

```bash
curl http://localhost:5092/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=parakeet-v3:fp16
```

The batch endpoint, `POST /v1/audio/transcriptions/batch`, takes the same
`model` and `quantization` fields.

**Switching before the server upgrades.** `model=parakeet-v3` and
`model=parakeet-v2` already mean fp32 on 1.5.0, and 1.5.0 ignores a
`quantization` field. Clients that want fp32 can switch now and keep working
across the upgrade. Clients that want fp16 or int8 have no request that means
the same thing on both versions (1.5.0 answers `parakeet-v3:fp16` with a 400),
so switch those when the server upgrades.

### What else you might notice

- **`parakeet-v3` transcripts change slightly.** All three precisions now load
  a different export of the same NVIDIA checkpoint
  (`Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx`). On a 648 s English chapter
  it scored 1.13 / 1.07 / 1.20% WER (fp32 / fp16 / int8), against 1.20 / 1.20 /
  1.83% for 1.5.0. If you compare against stored transcripts or golden files,
  expect small differences.
- **Long `parakeet-v2` audio no longer loses speech.** v2 is now chunked at
  25–30 s, not 60–75 s. At the longer length it silently dropped whole
  sentences, so long v2 transcripts will gain words.
- **Numbers keep their word break.** Text read `in2005.` or `was£1.10`. It now
  reads `in 2005.` and `was £1.10`, and word timestamps list the number as its
  own word.
- **A model that can't be loaded answers 503**, not a bare 500. Its `detail`
  names the model and the cause (a failed download, a file missing from an
  offline cache, ONNX Runtime refusing it). The failure isn't cached, so a
  retry tries the load again.
- **`GET /v1/models`** lists each model once (`parakeet-v3`, `parakeet-v2`,
  `whisper-*`). Each card has a new `quantizations` list and no longer has
  `aliases`. `owned_by` is now the model's author (`nvidia`, `openai`).
- **`GET /health`** no longer has `default_model`, because there is no default.
  It has a new `aligner` object, keyed `aligner:quantization`. `models` lists
  the new names, and `loaded` lists `model:quantization` keys such as
  `parakeet-v3:fp32`.
- **`language` must be a bare ISO 639-1 code** (`en`, `fr`), or empty or
  `auto`. 1.5.0 ignored the field; 2.0.0 answers `en-US`, `en_GB`, `EN` or
  `English` with a **400** on every request, so fix clients that send them.
  `verbose_json` now echoes it (1.5.0 always said `"auto"`). It still doesn't
  steer transcription: it only affects word alignment and spoken numbers
  (below).

### New, all opt-in

- **Whisper models.** `whisper-tiny`, `-base`, `-small`, `-medium`,
  `-large-v3` and `-large-v3-turbo`, plus English-only `whisper-tiny.en`,
  `-base.en`, `-small.en` and `-medium.en`, each in all three quantizations.
  They return text and segment times. Word times are only returned when the
  request names an `aligner` and the language is known (an `.en` model, or
  `language`).
- **`aligner`** retimes word timestamps from the audio with a forced aligner:
  20 ms precision, and word ends measured from the audio instead of estimated.
  Name one the way you name a model; there is no default, so without `aligner`
  words keep the model's own times. `GET /v1/aligners` lists them:

  | `aligner` | Languages |
  |---|---|
  | `mms-300m-forced-aligner` | English, the most accurate; **non-commercial licence (CC-BY-NC-4.0)** |
  | `wav2vec2-large-xlsr-53-english` | English, the next best, for commercial use |
  | `omnilingual-ctc-300m` | Parakeet v3's 25 languages; numbers keep the model's times |
  | `wav2vec2-base-960h` | English, the smallest and fastest, the least accurate |

  The precision is `int8` (the default) or `fp32`, after a colon
  (`aligner=mms-300m-forced-aligner:fp32`) or in `aligner_quantization`, as for
  `model`. When you ask for word timestamps, `language` must be one the aligner
  aligns; a request without it is taken as the server's
  `PARAKEET_ALIGN_DEFAULT_LANGUAGE`, English unless your admin changed it.
  English-only models (`parakeet-v2`, `whisper-*.en`) are aligned as English
  whatever `language` says. An unknown aligner or quantization, or a language
  the aligner doesn't align, gets a **400** that lists the valid choices. For
  Whisper, if a chunk can't be aligned at all, `words` is null, never guessed
  times; a word the aligner can't place sits between its aligned neighbours.
  It adds about 2–3 s of CPU per 30 s of audio. Send it with
  `response_format=verbose_json` and `timestamp_granularities[]=word`:

  ```python
  client.audio.transcriptions.create(
      model="parakeet-v3",
      file=f,
      response_format="verbose_json",
      timestamp_granularities=["word"],
      language="en",
      extra_body={"aligner": "mms-300m-forced-aligner"},  # non-commercial licence
  )
  ```
- **`spoken_numbers=true`** writes numbers, money and units in English
  transcripts as words, the way they were said: `$5` becomes "five dollars",
  `£25` or `25 lb` becomes "twenty-five pounds". Where Parakeet's number could
  have been said several ways ("£2.10": "two pounds ten", "two ten", ...), the
  request's `aligner` hears which one was; without an aligner, the most common
  reading is used. That choice was tuned with `wav2vec2-base-960h`; the other
  aligners haven't been measured for it yet.

See [Word timestamps](README.md#word-timestamps) and
[Spoken numbers](README.md#spoken-numbers) for how each works and what it
costs. Your server admin can switch spoken numbers on for every request. If
they do, send `spoken_numbers=false` to opt out.

## For server admins

### The project is now `stt-api`

The repository moved to `scagood/stt-api`, and the images with it. 2.0.0 and
later are published only as `ghcr.io/scagood/stt-api`, with the same tags
(`2.0.0-cpu`, `latest-gpu`, ...). The old
`ghcr.io/scagood/parakeet-tdt-0.6b-v3-fastapi-openai` images stay pullable but
are no longer updated, `latest-*` included, so update the image in your compose
file or manifest. GitHub
redirects the old git URL, but point existing clones at the new one:

```bash
git remote set-url origin https://github.com/scagood/stt-api.git
```

### Python and dependencies

Bare-metal and conda installs need **Python 3.14**. The Docker images already
use it. Reinstall from `requirements.txt`: it adds `pyyaml` and
`huggingface-hub`, and drops `flask`, `waitress`, `openai` and
`typing_extensions`. If a client script on the same host imported `openai`,
install it yourself.

The legacy Flask service (`app.py`) and its upload page are gone, along with
`PARAKEET_WAITRESS_THREADS`. Run `python server.py`. The Docker images already
ran it.

### Environment variables

| 1.5.0 | 2.0.0 |
|---|---|
| `PARAKEET_DEFAULT_MODEL` | **Removed and ignored.** There is no default model. Delete it, and tell the clients who relied on it (see [For API users](#for-api-users)). |
| `PARAKEET_CHUNK_TARGET_SEC`, `PARAKEET_CHUNK_MAX_SEC` | **Removed and ignored.** Chunk lengths are now set per model in the catalog (`chunk_target_sec`, `chunk_max_sec`). `PARAKEET_CHUNK_MIN_SEC` stays. |
| `PARAKEET_WAITRESS_THREADS` | Removed with the Flask service. |
| — | `PARAKEET_PRELOAD_MODELS`: models to load and warm up before ready (next section). |
| — | `PARAKEET_MODEL_CATALOG`: a YAML file that replaces the built-in model catalog. |
| — | `PARAKEET_MODEL_CACHE_SIZE`: keep at most N loaded models, and separately at most N loaded aligners, evicting the least recently used. `0` (the default) means no limit. |
| — | `PARAKEET_SPOKEN_NUMBERS`: the answer for requests that don't send `spoken_numbers`. Defaults to `false`. There is no server-wide switch for alignment: a request names its `aligner`. |
| — | `PARAKEET_ALIGN_DEFAULT_LANGUAGE` (`en`), `PARAKEET_ALIGN_THREADS` (`min(4, physical cores)`): see [Word timestamps](README.md#word-timestamps). |
| — | `PARAKEET_COMPARE_UI`: serve `GET /compare`, a page for comparing models and aligners by ear. Defaults to `false`. Each row it runs is a full transcription and loads whatever model or aligner it names, so leave it off on shared hosts. See [the compare page](README.md#comparing-models-and-aligners-by-ear). |

### Warm start is now opt-in

1.5.0 always loaded and warmed its default model before `/healthz` reported
ready. 2.0.0 loads nothing at startup unless you list models in
`PARAKEET_PRELOAD_MODELS`. Without it, `/healthz` is ready almost at once and
the first request for each model pays for the download and the load.

List what your clients actually request, as `model` (fp32) or
`model:quantization`:

```bash
PARAKEET_PRELOAD_MODELS=parakeet-v3          # CPU
PARAKEET_PRELOAD_MODELS=parakeet-v3:fp16     # GPU, if clients ask for fp16
```

A preloaded model is never used as a fallback. Preloading `parakeet-v3:fp16`
does nothing for a request that sends `model=parakeet-v3` with no
quantization, because that request asks for fp32. `docker-compose.yml`
preloads `parakeet-v3` (fp32) for both profiles.

### GPU hosts: fp32 is the default now

A GPU server on 1.5.0 answered model-less requests with fp16. On 2.0.0,
requests that don't send `quantization` get fp32, which uses about twice the
VRAM. To keep fp16, have clients send `model=parakeet-v3:fp16` (or
`quantization=fp16`), and preload `parakeet-v3:fp16`.

The new `parakeet-v3` export was only measured on CPU and **has not been tested
on a GPU**. If CUDA misbehaves (a load error, int8 running slowly, fp16 output
drifting from fp32), the comment above the `parakeet-v3` entry in
[`parakeet_service/models.yaml`](parakeet_service/models.yaml) lists the 1.5.0
repos and revisions to go back to. Put them in your own catalog (next section).

### Models, downloads and the cache

- **Every `parakeet-v3` precision downloads again** on first load, from
  `Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx`. The built-in catalog no
  longer uses `models--istupakov--parakeet-tdt-0.6b-v3-onnx` or
  `models--grikdotnet--parakeet-tdt-0.6b-fp16`. Once 2.0.0 is serving, you can
  delete them from the model cache (`/app/models` in the images), unless your
  own catalog points back at them. `parakeet-v2` still uses the same repos.
- **Every file is pinned to a commit** in the catalog. Upstream changes reach
  you only when a revision in the catalog is bumped.
- **Offline hosts** (`PARAKEET_HF_OFFLINE=true`) won't find the new files in a
  cache seeded by 1.5.0, so model loads fail (a 503 naming the model). Reseed the cache first: run 2.0.0
  once with the host online and `PARAKEET_PRELOAD_MODELS` listing everything
  you serve, then turn offline mode back on. Aligners aren't preloaded: while
  online, also send one word request naming each `aligner` (and
  `aligner_quantization`) your clients use. `wav2vec2-base-960h` is about
  95 MB at int8; the others are about 320 MB at int8 and 1.2–1.3 GB at fp32.
- **The model cache must be writable** even when it is fully seeded. Each load
  links the model's files into a temporary `.load-*` folder inside it.
- The service sets `HF_HUB_DISABLE_SHARED_BLOBS=1` so that each model's ONNX
  file stays next to its external data. Don't override it: onnxruntime 1.30
  refuses to load the split files.

### Bound what clients can load

Any client can now ask for any model in the catalog: 12 models × 3
quantizations. Each one is downloaded on first request and, by default, kept in
memory forever. On a shared host, set `PARAKEET_MODEL_CACHE_SIZE`, or serve a
smaller catalog:

```bash
cp parakeet_service/models.yaml /etc/parakeet/models.yaml   # edit: delete what you don't serve
PARAKEET_MODEL_CATALOG=/etc/parakeet/models.yaml
```

Aligners are bounded the same way: 4 aligners × 2 quantizations, each loaded on
first request. `PARAKEET_MODEL_CACHE_SIZE` caps them as it caps models (N of
each), but like models they are unbounded until you set it. To serve fewer,
delete them from the file's `aligners` section, and delete
`mms-300m-forced-aligner` if your use is commercial. The section is
required: `aligners: {}` serves none.

The service checks the file at startup and won't start if it's invalid; the
error names the problem. It's read only at startup, so restart after you
change it. For a Kubernetes ConfigMap example, see
[Your own model catalog](README.md#your-own-model-catalog).

### The word aligner

An aligner runs only for requests that name one. It always runs on the CPU,
even on GPU hosts, and handles one request at a time on its own thread, so it
never holds up audio decoding. If its download fails, words keep Parakeet's
times and the load is retried every 5 minutes. `/health` reports each
aligner's state per quantization under `aligner`.

### Monitoring and logs

- `/health` has no `default_model` field. It has a new `aligner` field, and
  `cpu.align_threads`. `GET /v1/aligners` lists the aligners, as
  `GET /v1/models` lists the models.
- The per-request log line names the variant and adds a stitch time:
  `transcribe model=parakeet-v3:fp32 ... stitch=12ms total=...`. Update any
  log parsers that read `model=`.
- A model that fails to load logs `Model 'parakeet-v3:fp32' could not be
  loaded: ...` with a traceback, once per request that asks for it, and that
  request gets a 503. Alert on it: on an offline host it means the cache is
  missing that model.

### Rolling back

Go back to the 1.5.0 image and restore any variables you removed. On 1.5.0,
`model=parakeet-v3` still means fp32, and the `quantization` field is silently
ignored, so a client that sends `quantization=fp16` gets fp32 until the server
is upgraded again. A client that sends `model=parakeet-v3:fp16` gets a 400
(unknown model): move it back to `model=parakeet-v3-fp16`. 1.5.0 has no
aligner and ignores `aligner`, so word times are Parakeet's again.

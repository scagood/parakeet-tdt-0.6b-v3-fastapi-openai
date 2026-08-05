# Changelog

## [1.3.0](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/compare/v1.2.0...v1.3.0) (2026-08-05)


### 🌟 Features

* **service:** OpenAI-compatible /v1/models endpoints ([d609b35](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/d609b350b5ee199b2b85ed09ce36a2f55b2adb17))

## [1.2.0](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/compare/v1.1.0...v1.2.0) (2026-08-05)


### 🌟 Features

* **service:** resolve default model at startup with CUDA probe ([53d2b4b](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/53d2b4bb9d97a8cb94209fa8901c65577f9b9e29))
* **service:** short model names, aliases, and v2/fp16 catalog ([a650eed](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/a650eed247402a9a3b3b9fa63b3d05a205e4bfe5))


### 🩹 Fixes

* sanitize export buttons, add auto-scroll toggle, add /docs endpoint ([02c5b9e](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/02c5b9ea1badbb5ce0f9d6833bfb1ec51424acde))
* **service:** run on Python 3.13+ where audioop is removed ([09759c2](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/09759c22c351db00a4c509313a3401b535919d65))
* **service:** stop word timestamps ballooning across silences ([9cc4ddb](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/9cc4ddbd03f7d583305e539f6e30ff92e17a1578))


### 📚 Documentation

* Add Open WebUI Integration guide ([c01cf71](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/c01cf719350393e3396e445682e995f53e947991))
* Polished install steps and added Parakeet model name support ([8327f66](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/8327f66b2b6ff52d85b5e18253bdc8377506e8d3))
* Update benchmarks to max 30x speedup ([f8d38fd](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/f8d38fd919180659ead08ec91000002abc266519))
* Update README with faster-whisper style comparison ([aebb1a5](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/aebb1a58aa8aced7561fb98b2ad00b7d7f0b72ce))


### 🧹 Chores

* also test 3.13 and 3.14 ([948d441](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/948d4418dd35a3b0d1e81de1c71578a622cfb9cb))


### 🤖 Automation

* add release-please and renovate ([61170bd](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/61170bd0db6c1bb9f91240a31bb1b334810e01a0))
* auto build images ([dd1ddcc](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/dd1ddcc6871ac1b5c628fadfc1c9502f89d7632d))
* install fastapi for route-level unit tests ([ce29273](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/ce2927326d2ff2df57ef3d0bce4d8ffea5e457ea))
* nightly GHCR untagged-image cleanup + editorconfig ([b6639db](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/b6639dbb895901de32b41721cbfaec97ca3dcce5))
* python3.12 test ([73803ce](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/73803ce77566ad0fc9d36eeba0d4e3017f19d85f))
* stub onnx stack in tests when not installed ([a65aaa9](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/a65aaa9f68908d61acfe2bdc35ba988c31b3725b))
* support arm64 too ([f204b71](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/f204b71c274b4c3bc2c9e135549b645c497277a1))
* update to latest action stages ([53b6324](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/53b63248bc4df10f50f85ba3a53eaba8b1df5765))

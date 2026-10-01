# Changelog

## [2.0.0](https://github.com/scagood/stt-api/compare/v1.5.0...v2.0.0) (2026-10-01)


### ⚠ BREAKING CHANGES

* **words:** `align_words` and PARAKEET_ALIGN_WORDS are removed; send `aligner` instead. /health reports aligner state as "name:quantization". A custom catalog's `aligners` entries need the new shape.
* **api:** `model` is required and names a model, not a precision: parakeet-v3-fp32/-fp16/-int8 and the other -quant names are gone (send `quantization` instead, default fp32 also on GPU), as are the aliases (parakeet-tdt-0.6b-v3, HF repo ids, whisper-large, whisper-turbo, ...) and PARAKEET_DEFAULT_MODEL. Nothing is preloaded unless PARAKEET_PRELOAD_MODELS is set. Model cards drop `aliases`, gain `quantizations`, and report `owned_by` as the model's author (nvidia, openai); /health drops `default_model`.
* **models:** PARAKEET_CHUNK_TARGET_SEC and PARAKEET_CHUNK_MAX_SEC are removed; chunk lengths are per model. PARAKEET_CHUNK_MIN_SEC remains.
* Python 3.14 is now the minimum supported interpreter, and the legacy Flask service (`app.py`) and its browser upload page at `/` are removed. The OpenAI-compatible API under `parakeet_service/` is unchanged.

### 🌟 Features

* **api:** name a model and pick its precision separately; fix the Whisper catalog ([da78ff9](https://github.com/scagood/stt-api/commit/da78ff96e0a1cd904e89c8e000fc90b54e63d394))
* **audio:** decode stereo and 24-bit WAVs in process ([#22](https://github.com/scagood/stt-api/issues/22)) ([35d843c](https://github.com/scagood/stt-api/commit/35d843c872fdd01d14fe13a10113067826117d23)), closes [#15](https://github.com/scagood/stt-api/issues/15)
* **compare:** a page to compare models and aligners by ear ([#48](https://github.com/scagood/stt-api/issues/48)) ([1a88458](https://github.com/scagood/stt-api/commit/1a88458247c8f2f1ecb637b433946998c919674d))
* **models:** add opt-in Whisper model support ([#25](https://github.com/scagood/stt-api/issues/25)) ([2e6aaf5](https://github.com/scagood/stt-api/commit/2e6aaf5b78d4a3f56a5c8547e14c56b4a85553b2))
* **models:** opt-in LRU cap on the loaded-model cache ([#37](https://github.com/scagood/stt-api/issues/37)) ([dd71261](https://github.com/scagood/stt-api/commit/dd71261052d1dbeca52bd07a0d2c045cf31457a5))
* **models:** serve parakeet-v3 from Olicorne's re-export ([#42](https://github.com/scagood/stt-api/issues/42)) ([a819f87](https://github.com/scagood/stt-api/commit/a819f8776095879f07d70afefd861e8b56441f7f))
* **models:** YAML model catalog with pinned, explicit files ([#41](https://github.com/scagood/stt-api/issues/41)) ([f2439eb](https://github.com/scagood/stt-api/commit/f2439eb47b38cd256c22597e7107f54616d9fef1))
* require Python 3.14 and drop the legacy Flask service ([#11](https://github.com/scagood/stt-api/issues/11)) ([250dfeb](https://github.com/scagood/stt-api/commit/250dfeb8c55e6401fdb54163915a83b3f9182638))
* **transcripts:** opt-in spoken-form numbers, money and units ([#30](https://github.com/scagood/stt-api/issues/30)) ([d1245e4](https://github.com/scagood/stt-api/commit/d1245e4b35db0286370a724a6182dc846f83fc5d))
* **words:** align English word timestamps with wav2vec2 ([#26](https://github.com/scagood/stt-api/issues/26)) ([f2bc41d](https://github.com/scagood/stt-api/commit/f2bc41d3d8daa291c281a1fe56bd2aaa63c185b3))
* **words:** make word alignment opt-in per request ([#33](https://github.com/scagood/stt-api/issues/33)) ([9dbf5eb](https://github.com/scagood/stt-api/commit/9dbf5ebb7db157586072661b0870fe585c6981f8))
* **words:** named word aligners from the model catalog, in Parakeet v3's 25 languages ([#45](https://github.com/scagood/stt-api/issues/45)) ([07193b4](https://github.com/scagood/stt-api/commit/07193b43f78d194fbead9aa7c3d6068fb6915fa2))
* **words:** Whisper word timestamps via forced alignment + real language list ([#34](https://github.com/scagood/stt-api/issues/34)) ([612aec4](https://github.com/scagood/stt-api/commit/612aec4f847f83441fff19c053efb06d6a8ab46e))


### 🩹 Fixes

* **docker:** compose healthchecks wait for /healthz, not /health ([510625e](https://github.com/scagood/stt-api/commit/510625e215b94e31f1c8ed1f9657d09548c4d652))
* **models:** answer 503 naming the model when it cannot be loaded ([#50](https://github.com/scagood/stt-api/issues/50)) ([8c2a679](https://github.com/scagood/stt-api/commit/8c2a679e2c051c8648e02eb7f0dcb2c872d52758))
* **models:** chunk Parakeet v2 at 25/30 s so long audio stops dropping speech ([#39](https://github.com/scagood/stt-api/issues/39)) ([4037c62](https://github.com/scagood/stt-api/commit/4037c62e17457522aaf8ec1c1ec9a857ebb01a09))
* **models:** keep ONNX external data beside its model in the HF cache ([73a44e0](https://github.com/scagood/stt-api/commit/73a44e0e4356b13fa9cf1002c87aa9fa3263e72c)), closes [#35](https://github.com/scagood/stt-api/issues/35)


### 📚 Documentation

* add the 1.5.0 to 2.0.0 upgrade guide ([63b735e](https://github.com/scagood/stt-api/commit/63b735e4e34964e8c905ebceb7158ec5ae1b3e66))
* correct the model names, defaults and missing endpoints ([#21](https://github.com/scagood/stt-api/issues/21)) ([c90be47](https://github.com/scagood/stt-api/commit/c90be47db0066449f568bfe89a142a978923fe4e)), closes [#16](https://github.com/scagood/stt-api/issues/16)
* correct what the docs say is true today ([a459ae5](https://github.com/scagood/stt-api/commit/a459ae5440a1f2929f58add8295a75f0fd4a113f))
* restructure the README around using the API; one config reference ([42d6208](https://github.com/scagood/stt-api/commit/42d6208feb8a4d880c68dfc8a5c2f1cb2e341a19))


### 📦 Dependencies

* **pkg:** drop the unused openai and typing_extensions pins ([#18](https://github.com/scagood/stt-api/issues/18)) ([0e25b2c](https://github.com/scagood/stt-api/commit/0e25b2c192a4806bae5d845ab0d20ce3d5e0a3a9)), closes [#12](https://github.com/scagood/stt-api/issues/12)


### 🧹 Chores

* delete the three unreferenced root diagnostic scripts ([#19](https://github.com/scagood/stt-api/issues/19)) ([e2ee3c0](https://github.com/scagood/stt-api/commit/e2ee3c01b98f82bff1345acb9cca0baa28502efb)), closes [#13](https://github.com/scagood/stt-api/issues/13)
* remove the unreferenced parakeet.png ([#20](https://github.com/scagood/stt-api/issues/20)) ([48df6f3](https://github.com/scagood/stt-api/commit/48df6f3b713d487236399309aea9d27c2fb3adab)), closes [#14](https://github.com/scagood/stt-api/issues/14)
* rename the project to stt-api ([422e145](https://github.com/scagood/stt-api/commit/422e1458a1df8a0377b4a74327e76bbb87491cec))


### 🤖 Automation

* **docker:** smoke-test the CPU image by loading parakeet-v3 int8 ([#52](https://github.com/scagood/stt-api/issues/52)) ([ae10c97](https://github.com/scagood/stt-api/commit/ae10c973bfb146e74d96f58f3b206dd358fcd419))

## [1.5.0](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/compare/v1.4.0...v1.5.0) (2026-09-17)


### 🌟 Features

* **service:** size thread pools from cgroup quota, warm up before ready ([#7](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/issues/7)) ([432f728](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/432f728b333c9c53fd0bfc630e049a029aaee005))


### 🩹 Fixes

* **docker:** stop the CPU image pulling CUDA wheels, drop emulated arm64 from PRs ([#10](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/issues/10)) ([db30a89](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/db30a89e7e19266e038ab76e30e4c5ac907ca560))


### 🧹 Chores

* **renovate:** track the Dockerfile.cpu pin and group it with onnxruntime-gpu ([#9](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/issues/9)) ([494e08b](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/494e08b8451b74b570840823318b784ea48b599e))

## [1.4.0](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/compare/v1.3.0...v1.4.0) (2026-08-05)


### 🌟 Features

* **service:** explicit -int8 model ids and aliases in /v1/models ([4360ba9](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/4360ba94f5abbd88bfc525c7e6ed12d727f98725))


### 🩹 Fixes

* **docker:** drop stale PARAKEET_DEFAULT_MODEL pins ([d1aa26f](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/d1aa26fc439abbad9ee0cae1959a5ac1ed8a68c2))

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

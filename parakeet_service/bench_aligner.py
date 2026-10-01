"""Aligner throughput on this machine, for choosing how to run it.

    python -m parakeet_service.bench_aligner
    python -m parakeet_service.bench_aligner wav2vec2-base-960h:int8 --threads 8,16 --at-once 1,2,4

An aligned request spends nearly all its alignment time in the aligner's ONNX
pass (the Viterbi after it is ~3%), and one pass stops getting faster past a
few threads. This times that pass on windows the size aligner._emission runs,
sweeping intra-op threads against windows run at once on one shared session.
The model's work doesn't depend on what is said, so the audio is noise. It
keeps every core busy for several minutes: run it on an idle machine.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort

from . import aligner
from .config import ALIGNER_CONFIGS, ALIGN_THREADS, CPU_INFO
from .model import _build_sess_options


def main() -> None:
    cores = CPU_INFO["logical"]
    parser = argparse.ArgumentParser(prog="python -m parakeet_service.bench_aligner")
    parser.add_argument("aligner", nargs="?", default="mms-300m-forced-aligner:int8", help="name:quantization")
    parser.add_argument("--threads", default=",".join(map(str, sorted({4, 8, 16, cores}))))
    parser.add_argument("--at-once", default="1,2,4,8")
    args = parser.parse_args()
    name, _, quant = args.aligner.partition(":")
    spec = ALIGNER_CONFIGS[name]
    variant = spec["quantizations"][quant or spec["default_quantization"]]

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(variant["repo"], variant["files"]["model.onnx"], revision=variant["revision"])
    size = aligner._WINDOW + 2 * aligner._CONTEXT
    windows = [np.random.default_rng(i).standard_normal((1, size), dtype=np.float32) for i in range(16)]
    window_sec = aligner._WINDOW / aligner.TARGET_SR

    def per_window(threads: int, at_once: int) -> float:
        session = ort.InferenceSession(
            path, sess_options=_build_sess_options(threads, spinning=False), providers=["CPUExecutionProvider"]
        )
        feed = session.get_inputs()[0].name
        session.run(None, {feed: windows[0]})  # warm-up
        started = time.perf_counter()
        with ThreadPoolExecutor(at_once) as pool:
            list(pool.map(lambda window: session.run(None, {feed: window}), windows))
        return (time.perf_counter() - started) / len(windows)

    at_once = [int(n) for n in args.at_once.split(",")]
    print(f"{args.aligner}: {CPU_INFO['physical']} physical / {cores} logical CPUs, "
          f"onnxruntime {ort.__version__}, PARAKEET_ALIGN_THREADS now {ALIGN_THREADS}")
    print(f"seconds per {window_sec:.0f} s of audio (lower is better)")
    print("threads" + "".join(f"{f'{n} at once':>12}" for n in at_once))
    for threads in (int(n) for n in args.threads.split(",")):
        print(f"{threads:7d}", end="", flush=True)
        for n in at_once:
            print(f"{per_window(threads, n):12.2f}", end="", flush=True)
        print()


if __name__ == "__main__":
    main()

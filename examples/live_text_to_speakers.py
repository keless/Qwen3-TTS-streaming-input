# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simulates word-by-word streaming text input (see LiveTextSource) and plays
the generated audio to the default output device in real time, instead of
writing a wav file. Requires `pip install sounddevice` (not a project
dependency -- only needed for this script).

Generation and playback run on separate threads, connected by a queue:
- The producer thread does nothing but call stream_generate_custom_voice_live_text
  and enqueue chunks -- it never touches the audio device, so device-latency
  overhead in sd.OutputStream.write() (measured at ~0.3s per call on this
  hardware, independent of chunk size) can't slow generation down.
- The consumer thread does nothing but drain the queue and feed the device,
  batching all currently-available chunks into a single write() call each
  time it wakes up, to amortize that per-call overhead instead of paying it
  once per small chunk.

Also enables enable_fast_codebook_gen(True) on the talker, which bypasses
HF generate()'s per-call Python overhead for subtalker codebook prediction --
a genuine, verified ~20% speedup on non-CUDA (CPU/MPS eager) hardware.

Even with both fixes, generation on non-CUDA hardware (no flash-attention,
no CUDA graphs, no torch.compile -- this repo's actual streaming
optimizations are CUDA-only) sits close to real-time rather than
comfortably above it, so occasional gaps are still possible, especially
for longer utterances or under the "slow producer" case. For guaranteed
gap-free listening, use examples/streaming_text_input_poc.py instead
(writes a wav after generating, rather than trying to play in real time).
"""
import argparse
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from huggingface_hub import snapshot_download

from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.live_text_source import LiveTextSource

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "Vivian"
LANGUAGE = "English"
DEFAULT_TEXT = "Hello there, this is a test of streaming text input while the audio keeps playing."
PREBUFFER_SECONDS = 0.5


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16, "flash_attention_2"
    if torch.backends.mps.is_available():
        return "mps", torch.float32, "eager"
    return "cpu", torch.float32, "eager"


def resolve_model_path(model_name):
    # Skips a network round-trip when the model is already cached locally
    # (transformers' mistral-regex patch check hits the HF API for non-local
    # paths, which fails hard behind a blocking proxy).
    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def feed_words(source: LiveTextSource, sentence: str, delay_s: float):
    for word in sentence.split():
        time.sleep(delay_s)
        source.push(word + " ")
    source.close()


def generate_chunks(tts, source, on_frame, chunk_queue):
    """Producer: only generates and enqueues. Never touches the audio device."""
    try:
        for chunk, sr in tts.stream_generate_custom_voice_live_text(
            text_source=source,
            speaker=SPEAKER,
            language=LANGUAGE,
            on_frame=on_frame,
        ):
            chunk_queue.put((chunk, sr))
    finally:
        chunk_queue.put(None)  # sentinel: generation finished (or raised)


def play_chunks(chunk_queue, stats):
    """Consumer: only drains the queue and feeds the device, batching
    whatever's currently available into one write() call at a time."""
    prebuffered = []
    prebuffered_seconds = 0.0
    stream = None
    sr = None
    t0 = time.time()

    def drain_available(first_item):
        # Collect first_item plus anything else already queued, non-blocking,
        # so multiple chunks can be written in one call. Stops (and reports)
        # as soon as the end-of-stream sentinel is seen, rather than
        # swallowing it into the batch -- if it's silently dropped here, the
        # outer loop's next chunk_queue.get() blocks forever with no more
        # sentinels coming.
        batch = [first_item]
        saw_sentinel = False
        while True:
            try:
                nxt = chunk_queue.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                saw_sentinel = True
                break
            batch.append(nxt)
        return batch, saw_sentinel

    finished = False
    while not finished:
        item = chunk_queue.get()
        if item is None:
            break
        batch, finished = drain_available(item)
        chunks = [c for c, _ in batch]
        sr = batch[-1][1]
        combined = np.concatenate(chunks).astype(np.float32)
        stats["total_samples"] += len(combined)

        if stream is None:
            prebuffered.append(combined)
            prebuffered_seconds += len(combined) / sr
            if prebuffered_seconds < PREBUFFER_SECONDS:
                continue
            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            stream.start()
            stats["first_chunk_time"] = time.time() - t0
            stream.write(np.concatenate(prebuffered))
        else:
            stream.write(combined)

    if stream is None and prebuffered:
        # Whole utterance was shorter than PREBUFFER_SECONDS -- play what we have.
        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
        stream.start()
        stats["first_chunk_time"] = time.time() - t0
        stream.write(np.concatenate(prebuffered))

    if stream is not None:
        stream.stop()
        stream.close()
    stats["sr"] = sr
    stats["elapsed"] = time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Sentence to stream word-by-word.")
    parser.add_argument("--speaker", default=SPEAKER, help="CustomVoice speaker name.")
    parser.add_argument(
        "--delay", type=float, default=0.05,
        help="Seconds between simulated word arrivals. 0.05 ~= normal pace "
        "(no pad-fallback stalls); try 0.3 to hear the degraded-pacing case.",
    )
    args = parser.parse_args()

    device, dtype, attn_implementation = pick_device_and_dtype()
    tts = Qwen3TTSModel.from_pretrained(
        resolve_model_path(MODEL_NAME), device_map=device, dtype=dtype, attn_implementation=attn_implementation,
    )
    tts.model.talker.enable_fast_codebook_gen(True)

    source = LiveTextSource(idle_timeout_s=10.0)
    feeder = threading.Thread(target=feed_words, args=(source, args.text, args.delay), daemon=True)
    feeder.start()

    counts = {"pad": 0, "real": 0}

    def on_frame(step_idx, used_pad):
        counts["pad" if used_pad else "real"] += 1

    chunk_queue = queue.Queue()
    stats = {"total_samples": 0, "first_chunk_time": None, "sr": None, "elapsed": 0.0}

    producer = threading.Thread(
        target=generate_chunks, args=(tts, source, on_frame, chunk_queue), daemon=True
    )
    producer.start()
    play_chunks(chunk_queue, stats)
    producer.join(timeout=5.0)
    feeder.join(timeout=1.0)

    duration = stats["total_samples"] / stats["sr"] if stats["sr"] else 0.0
    first_chunk_time = stats["first_chunk_time"] if stats["first_chunk_time"] is not None else 0.0
    print(
        f"first_chunk={first_chunk_time:.2f}s total={stats['elapsed']:.2f}s audio={duration:.2f}s "
        f"real_frames={counts['real']} pad_frames={counts['pad']}"
    )


if __name__ == "__main__":
    main()

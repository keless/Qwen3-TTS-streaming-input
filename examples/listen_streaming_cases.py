"""
Listen to different streaming cases and compare audio quality / latency.

This script generates audio using several different modes and saves each
result to a separate WAV file so you can listen side-by-side:

  1. standard          -- non-streaming baseline (hear everything at once)
  2. streaming_base    -- streaming without torch.compile / CUDA graphs
  3. streaming_opt     -- streaming with torch.compile + fast codebook
  4. streaming_twophase -- two-phase streaming (aggressive first chunk,
                           then stable settings)
  5. custom_voice      -- CustomVoice model streaming
  6. custom_voice_opt  -- CustomVoice with optimizations

Usage:
    python examples/listen_streaming_cases.py

Prerequisites:
    - A GPU (CUDA)
    - Reference audio file: kuklina-1.wav (or set REF_AUDIO_PATH below)
"""

import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# ── Configuration ────────────────────────────────────────────────────────────

# Path to a reference audio file for voice cloning.
# Place a short (~5-30 s) WAV file here and update the path if needed.
REF_AUDIO_PATH = "kuklina-1.wav"

# Transcript of the reference audio (required for ICL mode).
REF_TEXT = (
    "This is the reference transcript.  Provide the actual text spoken in the "
    "reference audio file so the model can match pronunciation and prosody."
)

# Text to synthesize in each case.
TEST_TEXT = (
    "Hello! This is a test of the streaming voice synthesis system.  "
    "Please listen to each output file and compare the quality, naturalness, "
    "and consistency across the different streaming modes."
)

# Output directory (relative to cwd).
OUT_DIR = "output_streaming_cases"

# ── Helpers ──────────────────────────────────────────────────────────────────

def log_time(start: float, operation: str) -> float:
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


def save_audio(audio: np.ndarray, sr: int, filename: str) -> None:
    path = f"{OUT_DIR}/{filename}"
    sf.write(path, audio, sr)
    dur = len(audio) / sr if sr > 0 else 0
    print(f"  Saved: {path}  ({dur:.2f}s, {sr} Hz)")


def run_standard(model, text, language, prompt) -> tuple[np.ndarray, int]:
    """Non-streaming baseline generation."""
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=prompt,
    )
    return wavs[0], sr


def run_streaming(
    model,
    text: str,
    language: str,
    prompt,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    overlap_samples: int = 0,
    first_chunk_emit_every: int = 0,
    first_chunk_decode_window: int = 48,
    first_chunk_frames: int = 48,
) -> tuple[np.ndarray, int, float, int]:
    """
    Streaming generation.

    Returns (final_audio, sample_rate, first_chunk_latency_s, chunk_count).
    """
    overall_start = time.time()
    chunks = []
    first_chunk_time = None
    chunk_count = 0
    sr = 24000

    kwargs = {
        "text": text,
        "language": language,
        "voice_clone_prompt": prompt,
        "emit_every_frames": emit_every_frames,
        "decode_window_frames": decode_window_frames,
        "overlap_samples": overlap_samples,
    }
    # Two-phase parameters (only relevant when first_chunk_emit_every > 0)
    kwargs["first_chunk_emit_every"] = first_chunk_emit_every
    kwargs["first_chunk_decode_window"] = first_chunk_decode_window
    kwargs["first_chunk_frames"] = first_chunk_frames

    for chunk, chunk_sr in model.stream_generate_voice_clone(**kwargs):
        chunk_count += 1
        chunks.append(chunk)
        sr = chunk_sr
        if first_chunk_time is None:
            first_chunk_time = time.time() - overall_start

    final_audio = np.concatenate(chunks) if chunks else np.array([])
    latency = first_chunk_time or 0.0
    return final_audio, sr, latency, chunk_count


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    log_time(start, "Model loaded")

    # Build voice clone prompt
    print(f"\nReference audio : {REF_AUDIO_PATH}")
    print(f"Reference text  : {REF_TEXT}")

    start = time.time()
    prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO_PATH,
        ref_text=REF_TEXT,
    )
    log_time(start, "Voice clone prompt created")

    language = "English"

    results = {}

    # ── Case 1: Standard (non-streaming) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Case 1: Standard (non-streaming) generation")
    print("=" * 60)
    start = time.time()
    audio, sr = run_standard(model, TEST_TEXT, language, prompt)
    elapsed = time.time() - start
    results["standard"] = (audio, sr, 0, 1)
    save_audio(audio, sr, "01_standard.wav")
    print(f"  Total: {elapsed:.2f}s, Audio: {len(audio)/sr:.2f}s")

    # ── Case 2: Streaming WITHOUT optimizations ──────────────────────────
    print("\n" + "=" * 60)
    print("Case 2: Streaming WITHOUT optimizations")
    print("=" * 60)
    start = time.time()
    audio, sr, latency, chunks = run_streaming(
        model, TEST_TEXT, language, prompt,
        emit_every_frames=8,
        decode_window_frames=80,
    )
    elapsed = time.time() - start
    results["streaming_base"] = (audio, sr, latency, chunks)
    save_audio(audio, sr, "02_streaming_base.wav")
    print(f"  Total: {elapsed:.2f}s, First chunk: {latency:.2f}s, Chunks: {chunks}")

    # ── Case 3: Streaming WITH optimizations ─────────────────────────────
    print("\n" + "=" * 60)
    print("Case 3: Streaming WITH torch.compile + fast codebook")
    print("=" * 60)
    print("\nEnabling optimizations (first run compiles)...")
    model.enable_streaming_optimizations(
        decode_window_frames=80,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="reduce-overhead",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    # Warmup
    for i, warmup_text in enumerate([
        "Quick warmup one.",
        "Second warmup sentence for stability.",
        "Third warmup to finalize compilation.",
    ], 1):
        run_streaming(model, warmup_text, language, prompt,
                      emit_every_frames=8, decode_window_frames=80)
        print(f"  Warmup {i} done")

    start = time.time()
    audio, sr, latency, chunks = run_streaming(
        model, TEST_TEXT, language, prompt,
        emit_every_frames=8,
        decode_window_frames=80,
    )
    elapsed = time.time() - start
    results["streaming_opt"] = (audio, sr, latency, chunks)
    save_audio(audio, sr, "03_streaming_opt.wav")
    print(f"  Total: {elapsed:.2f}s, First chunk: {latency:.2f}s, Chunks: {chunks}")

    # ── Case 4: Two-phase streaming ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Case 4: Two-phase streaming (aggressive first chunk)")
    print("=" * 60)
    # Two-phase: emit every 2 frames for first 24 frames, then switch to 8.
    start = time.time()
    audio, sr, latency, chunks = run_streaming(
        model, TEST_TEXT, language, prompt,
        emit_every_frames=8,
        decode_window_frames=80,
        first_chunk_emit_every=2,
        first_chunk_decode_window=48,
        first_chunk_frames=24,
    )
    elapsed = time.time() - start
    results["streaming_twophase"] = (audio, sr, latency, chunks)
    save_audio(audio, sr, "04_streaming_twophase.wav")
    print(f"  Total: {elapsed:.2f}s, First chunk: {latency:.2f}s, Chunks: {chunks}")

    # ── Case 5: CustomVoice streaming ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Case 5: CustomVoice streaming (no optimizations)")
    print("=" * 60)
    try:
        cv_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        log_time(start, "CustomVoice model loaded")

        speakers = cv_model.get_supported_speakers()
        if speakers:
            speaker = speakers[0]
            print(f"  Using speaker: {speaker}")
        else:
            print("  No speakers found, skipping CustomVoice case.")
            speaker = None

        if speaker:
            cv_test_text = "Hello, this is a test of the CustomVoice streaming model."
            start = time.time()
            audio, sr, latency, chunks = run_streaming(
                cv_model, cv_test_text, "English", None,
                emit_every_frames=8,
                decode_window_frames=80,
            )
            elapsed = time.time() - start
            results["custom_voice"] = (audio, sr, latency, chunks)
            save_audio(audio, sr, "05_custom_voice.wav")
            print(f"  Total: {elapsed:.2f}s, First chunk: {latency:.2f}s, Chunks: {chunks}")
    except Exception as e:
        print(f"  Skipped CustomVoice: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Case':<25} {'1st Chunk':>10} {'Audio':>8} {'Chunks':>7}")
    print("-" * 55)

    for label in ["standard", "streaming_base", "streaming_opt", "streaming_twophase"]:
        if label not in results:
            continue
        audio, sr, latency, chunks = results[label]
        dur = len(audio) / sr if sr > 0 else 0
        print(f"  {label:<22} {latency:>9.2f}s {dur:>7.2f}s {chunks:>7}")

    print(f"\nOutput files are in: {OUT_DIR}/")
    print("Listen to each file and compare quality, naturalness, and first-chunk latency.")
    print("Files are numbered 01-04 for the Base model cases, and 05 for CustomVoice if loaded.")


if __name__ == "__main__":
    main()

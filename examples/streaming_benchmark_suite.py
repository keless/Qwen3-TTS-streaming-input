#!/usr/bin/env python3
"""
Streaming TTS Benchmark Suite

Comprehensive benchmarking tool for Qwen3-TTS streaming generation.
Compares multiple configurations across text lengths and optimization levels.

Usage:
    # Basic run with default settings
    python examples/streaming_benchmark_suite.py

    # With custom model and reference audio
    python examples/streaming_benchmark_suite.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --ref-audio ref.wav --ref-text "The reference transcript."

    # Custom streaming parameters
    python examples/streaming_benchmark_suite.py \
        --emit-every 4 --decode-window 48 \
        --text-lengths short medium long

    # JSON output for automated analysis
    python examples/streaming_benchmark_suite.py --json-out results.json

    # Custom voice model
    python examples/streaming_benchmark_suite.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --speaker Alice

    # Voice design model
    python examples/streaming_benchmark_suite.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
        --speaker Alice --instruct "Speak with a warm, friendly tone"

    # Skip certain test categories
    python examples/streaming_benchmark_suite.py --skip-optimized --skip-twophase
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRun:
    """Single benchmark run result."""
    label: str
    method: str  # "standard" | "streaming_baseline" | "streaming_optimized" | "streaming_twophase"
    total_time: float
    audio_duration: float
    sample_rate: int
    first_chunk_time: Optional[float]  # None for standard
    chunk_count: int
    avg_chunk_duration: float
    rtf: float  # real-time factor
    audio: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _elapsed(start: float) -> float:
    return time.time() - start


def _rtf(total_time: float, audio_duration: float) -> float:
    if audio_duration <= 0:
        return 0.0
    return total_time / audio_duration


def _format_rtf(rtf: float) -> str:
    if rtf <= 0:
        return "N/A"
    return f"{rtf:.2f}x"


def _format_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _print_header(title: str, width: int = 80) -> None:
    print(f"\n{'=' * width}")
    print(f" {title}")
    print(f"{'=' * width}")


def _print_subheader(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Core benchmarking functions
# ---------------------------------------------------------------------------

def run_streaming_test(
    model: Qwen3TTSModel,
    text: str,
    language: str,
    voice_clone_prompt: Optional[Dict[str, Any]] = None,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    first_chunk_emit_every: int = 0,
    first_chunk_frames: int = 48,
    label: str = "streaming",
    method: str = "streaming_baseline",
    **stream_kwargs,
) -> BenchmarkRun:
    """Run a single streaming generation and collect timing metrics."""
    start = _now()
    chunks = []
    chunk_sizes = []
    first_chunk_time = None
    chunk_count = 0
    sample_rate = 24000

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
        overlap_samples=0,
        first_chunk_emit_every=first_chunk_emit_every,
        first_chunk_frames=first_chunk_frames,
        **stream_kwargs,
    ):
        chunk_count += 1
        chunks.append(chunk)
        chunk_sizes.append(len(chunk))
        sample_rate = chunk_sr
        if first_chunk_time is None:
            first_chunk_time = _now() - start

    total_time = _now() - start
    final_audio = np.concatenate(chunks) if chunks else np.array([])
    audio_duration = len(final_audio) / sample_rate if sample_rate > 0 else 0
    avg_chunk_duration = (np.mean(chunk_sizes) / sample_rate) if chunk_sizes else 0

    return BenchmarkRun(
        label=label,
        method=method,
        total_time=total_time,
        audio_duration=audio_duration,
        sample_rate=sample_rate,
        first_chunk_time=first_chunk_time,
        chunk_count=chunk_count,
        avg_chunk_duration=avg_chunk_duration,
        rtf=_rtf(total_time, audio_duration),
        audio=final_audio,
    )


def run_standard_test(
    model: Qwen3TTSModel,
    text: str,
    language: str,
    voice_clone_prompt: Optional[Dict[str, Any]] = None,
    label: str = "standard",
    **gen_kwargs,
) -> BenchmarkRun:
    """Run a single non-streaming generation and collect timing metrics."""
    start = _now()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        **gen_kwargs,
    )
    total_time = _now() - start
    audio = wavs[0] if wavs else np.array([])
    audio_duration = len(audio) / sr if sr > 0 else 0

    return BenchmarkRun(
        label=label,
        method="standard",
        total_time=total_time,
        audio_duration=audio_duration,
        sample_rate=sr,
        first_chunk_time=None,
        chunk_count=1,
        avg_chunk_duration=audio_duration,
        rtf=_rtf(total_time, audio_duration),
        audio=audio,
    )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

def build_voice_clone_prompt(
    model: Qwen3TTSModel,
    ref_audio: str,
    ref_text: str,
) -> Dict[str, Any]:
    """Create a voice clone prompt item."""
    items = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    return model._prompt_items_to_voice_clone_prompt(items)


def build_custom_voice_prompt(
    model: Qwen3TTSModel,
    speaker: str,
    instruct: Optional[str] = None,
) -> Dict[str, Any]:
    """Build kwargs for custom voice / voice design model."""
    kwargs = {"speaker": speaker}
    if instruct is not None:
        kwargs["instruct"] = instruct
    return kwargs


def run_benchmarks(
    model: Qwen3TTSModel,
    model_type: str,  # "base" | "custom_voice" | "voice_design"
    # Inputs
    ref_audio: Optional[str],
    ref_text: Optional[str],
    speaker: Optional[str],
    instruct: Optional[str],
    language: str,
    # Streaming params
    emit_every_frames: int,
    decode_window_frames: int,
    first_chunk_emit_every: int,
    first_chunk_frames: int,
    # Text lengths
    text_lengths: List[str],
    # Flags
    skip_optimized: bool,
    skip_twophase: bool,
    # Output
    output_dir: str,
) -> List[BenchmarkRun]:
    """Run the full benchmark suite and return all results."""

    # --- Build prompts / kwargs ---
    if model_type == "base":
        assert ref_audio is not None, "--ref-audio is required for base models"
        assert ref_text is not None, "--ref-text is required for base models"
        voice_clone_prompt = build_voice_clone_prompt(model, ref_audio, ref_text)
    else:
        voice_clone_prompt = None

    # --- Texts ---
    texts_by_length = {
        "short": "This is a short sentence for testing.",
        "medium": (
            "This is a medium-length paragraph that contains multiple sentences. "
            "It tests how the streaming system handles a moderate amount of text. "
            "The benchmark measures latency, throughput, and real-time factor."
        ),
        "long": (
            "This is a longer passage designed to stress-test the streaming TTS pipeline. "
            "It contains many sentences and tests sustained generation performance. "
            "The system should maintain consistent chunk timing and real-time factor "
            "throughout the entire generation. Audio quality should remain stable "
            "from the first chunk to the last. This passage also includes punctuation "
            "and varied sentence structures to exercise the prosody modeling capabilities "
            "of the model. Proper handling of pauses, intonation, and rhythm is important "
            "for natural-sounding speech synthesis."
        ),
        "tiny": "Hi there.",
    }

    texts = {}
    for length in text_lengths:
        if length in texts_by_length:
            texts[length] = texts_by_length[length]
        else:
            texts[length] = length  # treat as literal text

    results: List[BenchmarkRun] = []

    # ===================================================================
    # Phase 1: Standard (non-streaming) baseline
    # ===================================================================
    _print_header("PHASE 1: Standard (Non-Streaming) Baseline")

    for length, text in texts.items():
        _print_subheader(f"Standard -- {length} ({len(text)} chars)")
        run = run_standard_test(
            model,
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            label=f"standard_{length}",
        )
        results.append(run)
        print(
            f"  Total: {_format_time(run.total_time)}, "
            f"Audio: {_format_time(run.audio_duration)}, "
            f"RTF: {_format_rtf(run.rtf)}"
        )
        sf.write(f"{output_dir}/standard_{length}.wav", run.audio, run.sample_rate)

    # ===================================================================
    # Phase 2: Streaming baseline (no optimizations)
    # ===================================================================
    _print_header("PHASE 2: Streaming Baseline (No Optimizations)")

    for length, text in texts.items():
        _print_subheader(f"Streaming baseline -- {length} ({len(text)} chars)")
        run = run_streaming_test(
            model,
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            label=f"streaming_baseline_{length}",
            method="streaming_baseline",
        )
        results.append(run)
        print(
            f"  1st chunk: {_format_time(run.first_chunk_time or 0)}, "
            f"Total: {_format_time(run.total_time)}, "
            f"Chunks: {run.chunk_count}, "
            f"RTF: {_format_rtf(run.rtf)}"
        )
        sf.write(
            f"{output_dir}/streaming_baseline_{length}.wav",
            run.audio,
            run.sample_rate,
        )

    # ===================================================================
    # Phase 3: Streaming with optimizations
    # ===================================================================
    if not skip_optimized:
        _print_header("PHASE 3: Streaming with Optimizations")

        print("\n  Enabling streaming optimizations...")
        model.enable_streaming_optimizations(
            decode_window_frames=decode_window_frames,
            use_compile=True,
            use_cuda_graphs=False,
            compile_mode="reduce-overhead",
            use_fast_codebook=True,
            compile_codebook_predictor=True,
            compile_talker=True,
        )

        # Warmup runs
        warmup_texts = [
            "Test one two three four five.",
            "Hello, how are you? This is the second warmup run.",
            "Third warmup run for full compilation of all model components.",
        ]
        print("\n  Warmup runs (compilation happens here)...")
        for i, wtext in enumerate(warmup_texts, 1):
            wr = run_streaming_test(
                model,
                text=wtext,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                emit_every_frames=emit_every_frames,
                decode_window_frames=decode_window_frames,
                label=f"warmup_{i}",
                method="streaming_optimized",
            )
            print(
                f"    Warmup {i}: Total: {_format_time(wr.total_time)}, "
                f"RTF: {_format_rtf(wr.rtf)}"
            )

        # Actual benchmark runs
        for length, text in texts.items():
            _print_subheader(f"Optimized -- {length} ({len(text)} chars)")
            run = run_streaming_test(
                model,
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                emit_every_frames=emit_every_frames,
                decode_window_frames=decode_window_frames,
                label=f"streaming_optimized_{length}",
                method="streaming_optimized",
            )
            results.append(run)
            print(
                f"  1st chunk: {_format_time(run.first_chunk_time or 0)}, "
                f"Total: {_format_time(run.total_time)}, "
                f"Chunks: {run.chunk_count}, "
                f"RTF: {_format_rtf(run.rtf)}"
            )
            sf.write(
                f"{output_dir}/streaming_optimized_{length}.wav",
                run.audio,
                run.sample_rate,
            )

    # ===================================================================
    # Phase 4: Two-phase streaming (aggressive first chunk)
    # ===================================================================
    if not skip_twophase:
        _print_header("PHASE 4: Two-Phase Streaming (Aggressive First Chunk)")

        for length, text in texts.items():
            _print_subheader(f"Two-phase -- {length} ({len(text)} chars)")
            run = run_streaming_test(
                model,
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                emit_every_frames=emit_every_frames,
                decode_window_frames=decode_window_frames,
                first_chunk_emit_every=4,
                first_chunk_frames=first_chunk_frames,
                label=f"streaming_twophase_{length}",
                method="streaming_twophase",
            )
            results.append(run)
            print(
                f"  1st chunk: {_format_time(run.first_chunk_time or 0)}, "
                f"Total: {_format_time(run.total_time)}, "
                f"Chunks: {run.chunk_count}, "
                f"RTF: {_format_rtf(run.rtf)}"
            )
            sf.write(
                f"{output_dir}/streaming_twophase_{length}.wav",
                run.audio,
                run.sample_rate,
            )

    return results


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(results: List[BenchmarkRun]) -> None:
    """Print a formatted summary table of all benchmark results."""
    _print_header("BENCHMARK SUMMARY", 90)

    # Group by method
    methods = {}
    for r in results:
        methods.setdefault(r.method, []).append(r)

    # Collect all text lengths present
    all_lengths = set()
    for r in results:
        for suffix in ["_tiny", "_short", "_medium", "_long"]:
            if r.label.endswith(suffix):
                all_lengths.add(suffix.lstrip("_"))
                break
        else:
            if r.label.startswith("standard_") or r.label.startswith("streaming_"):
                parts = r.label.split("_", 1)
                if len(parts) > 1:
                    all_lengths.add(parts[1])
    all_lengths = sorted(all_lengths)

    # Header
    method_headers = ["standard", "streaming_baseline", "streaming_optimized", "streaming_twophase"]
    present_methods = [m for m in method_headers if m in methods]

    # Column widths
    label_w = 40
    time_w = 14
    rtf_w = 10
    chunk_w = 8
    header = (
        f"{'Configuration':<{label_w}}"
        f"{'1st Chunk':>{time_w}}"
        f"{'Total':>{time_w}}"
        f"{'Audio':>{time_w}}"
        f"{'RTF':>{rtf_w}}"
        f"{'Chunks':>{chunk_w}}"
    )
    for m in present_methods:
        header += f"{'RTF':>{rtf_w}}"

    print(f"\n{header}")
    print("-" * len(header))

    for length in all_lengths:
        row = f"{length:<{label_w}}"
        for m in present_methods:
            runs = [r for r in methods[m] if length in r.label]
            if runs:
                r = runs[0]
                first = _format_time(r.first_chunk_time) if r.first_chunk_time is not None else "N/A"
                row += (
                    f"{first:>{time_w}}"
                    f"{_format_time(r.total_time):>{time_w}}"
                    f"{_format_time(r.audio_duration):>{time_w}}"
                    f"{_format_rtf(r.rtf):>{rtf_w}}"
                    f"{r.chunk_count:>{chunk_w}}"
                )
            else:
                row += f"{'N/A':>{rtf_w}}"
        print(row)

    # RTF speedup section
    if "streaming_baseline" in methods:
        baseline = methods["streaming_baseline"][0]
        baseline_rtf = baseline.rtf

        _print_subheader("RTF Speedup vs Streaming Baseline")

        for m in ["streaming_optimized", "streaming_twophase"]:
            if m not in methods:
                continue
            runs = methods[m]
            if not runs:
                continue
            avg_rtf = np.mean([r.rtf for r in runs])
            speedup = baseline_rtf / avg_rtf if avg_rtf > 0 else 0
            print(f"  {m:<30} RTF: {avg_rtf:.3f}  Speedup: {speedup:.2f}x")

    # First chunk latency comparison
    streaming_methods = [m for m in present_methods if m != "standard"]
    if len(streaming_methods) >= 2:
        _print_subheader("First Chunk Latency Comparison")
        for length in all_lengths:
            print(f"\n  Text length: {length}")
            for m in streaming_methods:
                runs = [r for r in methods[m] if length in r.label]
                if runs:
                    r = runs[0]
                    print(f"    {m:<30}  {_format_time(r.first_chunk_time or 0)}")

    # Per-method statistics
    _print_subheader("Per-Method Statistics")
    for m in present_methods:
        runs = methods[m]
        if not runs:
            continue
        rtf_vals = [r.rtf for r in runs if r.rtf > 0]
        if not rtf_vals:
            continue
        avg_rtf = np.mean(rtf_vals)
        std_rtf = np.std(rtf_vals)
        first_chunks = [r.first_chunk_time for r in runs if r.first_chunk_time is not None]
        if first_chunks:
            avg_first = np.mean(first_chunks)
            print(
                f"  {m:<30}  Avg RTF: {avg_rtf:.3f} (+/- {std_rtf:.3f})  "
                f"Avg 1st Chunk: {_format_time(avg_first)}"
            )
        else:
            print(
                f"  {m:<30}  Avg RTF: {avg_rtf:.3f} (+/- {std_rtf:.3f})  "
                f"(no streaming)"
            )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def results_to_json(results: List[BenchmarkRun]) -> List[Dict[str, Any]]:
    """Convert results to a JSON-serializable list of dicts."""
    out = []
    for r in results:
        d = {k: v for k, v in asdict(r).items() if k != "audio"}
        d["rtf"] = round(d["rtf"], 4)
        d["total_time"] = round(d["total_time"], 4)
        d["audio_duration"] = round(d["audio_duration"], 4)
        d["first_chunk_time"] = round(d["first_chunk_time"], 4) if d["first_chunk_time"] is not None else None
        d["avg_chunk_duration"] = round(d["avg_chunk_duration"], 6)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming TTS Benchmark Suite for Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace model name or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help="Device map for model loading (default: cuda:0)",
    )

    # Voice clone inputs
    parser.add_argument(
        "--ref-audio",
        default="kuklina-1.wav",
        help="Reference audio file for voice cloning (default: kuklina-1.wav)",
    )
    parser.add_argument(
        "--ref-text",
        default=(
            "This is the reference transcript. It contains a few sentences "
            "to help the model learn the speaker's voice characteristics."
        ),
        help="Reference transcript text (default: inline sample)",
    )

    # Custom voice / voice design
    parser.add_argument(
        "--speaker",
        default=None,
        help="Speaker name for CustomVoice or VoiceDesign models",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Style instruction for VoiceDesign models (optional)",
    )

    # Language
    parser.add_argument(
        "--language",
        default="Auto",
        help="Language for synthesis (default: Auto)",
    )

    # Streaming parameters
    parser.add_argument(
        "--emit-every",
        type=int,
        default=8,
        help="Emit audio chunk every N codec frames (default: 8)",
    )
    parser.add_argument(
        "--decode-window",
        type=int,
        default=80,
        help="Decode window in frames (default: 80)",
    )
    parser.add_argument(
        "--first-chunk-frames",
        type=int,
        default=48,
        help="Switch to stable settings after N frames in two-phase mode (default: 48)",
    )

    # Text lengths to benchmark
    parser.add_argument(
        "--text-lengths",
        nargs="+",
        default=["short", "medium", "long"],
        choices=["tiny", "short", "medium", "long"],
        help="Text lengths to benchmark (default: short medium long)",
    )

    # Flags
    parser.add_argument(
        "--skip-optimized",
        action="store_true",
        help="Skip the optimized streaming phase",
    )
    parser.add_argument(
        "--skip-twophase",
        action="store_true",
        help="Skip the two-phase streaming phase",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output WAV files (default: current directory)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Path to write JSON results (optional)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set float32 precision
    torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print("Qwen3-TTS Streaming Benchmark Suite")
    print("=" * 80)
    print(f"  Model:       {args.model}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Device:      {args.device_map}")
    print(f"  Emit every:  {args.emit_every} frames")
    print(f"  Decode win:  {args.decode_window} frames")
    print(f"  Text lengths: {args.text_lengths}")
    print(f"  Skip optimized: {args.skip_optimized}")
    print(f"  Skip twophase:  {args.skip_twophase}")
    print("=" * 80)

    # --- Load model ---
    _print_header("Loading Model")
    start = _now()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    load_time = _elapsed(start)
    print(f"  Model loaded in {_format_time(load_time)}")
    print(f"  Model type: {model.model.tts_model_type}")
    print(f"  Device: {model.device}")

    # --- Determine model type and build prompt ---
    model_type = model.model.tts_model_type

    if model_type == "base":
        voice_clone_prompt = build_voice_clone_prompt(model, args.ref_audio, args.ref_text)
        print(f"  Voice clone prompt created from: {args.ref_audio}")
    else:
        voice_clone_prompt = None
        assert args.speaker is not None, f"--speaker is required for {model_type} models"
        print(f"  Speaker: {args.speaker}")
        if args.instruct:
            print(f"  Instruct: {args.instruct}")

    # --- Run benchmarks ---
    results = run_benchmarks(
        model=model,
        model_type=model_type,
        ref_audio=args.ref_audio if model_type == "base" else None,
        ref_text=args.ref_text if model_type == "base" else None,
        speaker=args.speaker,
        instruct=args.instruct,
        language=args.language,
        emit_every_frames=args.emit_every,
        decode_window_frames=args.decode_window,
        first_chunk_emit_every=4,
        first_chunk_frames=args.first_chunk_frames,
        text_lengths=args.text_lengths,
        skip_optimized=args.skip_optimized,
        skip_twophase=args.skip_twophase,
        output_dir=args.output_dir,
    )

    # --- Summary ---
    print_summary(results)

    # --- JSON output ---
    if args.json_out:
        json_data = results_to_json(results)
        with open(args.json_out, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nJSON results written to: {args.json_out}")

    # --- Total time ---
    print(f"\nTotal benchmark time: {_format_time(_elapsed(start))}")


if __name__ == "__main__":
    main()

"""
Live text source streaming TTS demo.

Reads text from stdin line by line and generates streaming audio in real time.
Each line is synthesized as it arrives, with audio chunks concatenated into a single output file.

This demonstrates the two-phase streaming pipeline with live input:
1. Load model and build voice clone prompt
2. Read text from stdin (or a file) line by line
3. Stream-generate audio for each line as it arrives
4. Concatenate all chunks and save to output file

Usage:
    # Pipe text from a file
    cat text.txt | python examples/test_live_text_source.py

    # Interactive: type text, press Enter for each line
    python examples/test_live_text_source.py

    # Pipe from another command
    echo "Hello world. This is a test." | python examples/test_live_text_source.py

    # With optimizations enabled
    python examples/test_live_text_source.py --optimized

    # Specify output file
    python examples/test_live_text_source.py -o output.wav

    # Use custom streaming parameters
    python examples/test_live_text_source.py --emit-every 4 --decode-window 80

    # Voice clone with custom reference
    python examples/test_live_text_source.py \
        --ref-audio reference.wav \
        --ref-text "The reference transcript goes here."
"""

import argparse
import sys
import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


def main():
    parser = argparse.ArgumentParser(
        description="Live text source streaming TTS. Reads text from stdin and generates audio in real time."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace model name or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file for voice cloning (required for base model)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference transcript for voice cloning (required when ref-audio is used with ICL mode)",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use speaker embedding only (no ICL, ignores ref-text)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Speaker name for CustomVoice model",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for CustomVoice/VoiceDesign model",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="live_output.wav",
        help="Output audio file path (default: live_output.wav)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g. 'English', 'Russian', 'Chinese'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Enable streaming optimizations (torch.compile + CUDA graphs)",
    )
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
        help="Decode window size in frames (default: 80)",
    )
    parser.add_argument(
        "--first-chunk-emit",
        type=int,
        default=4,
        help="Emit interval for first chunk phase (default: 4, 0=disabled)",
    )
    parser.add_argument(
        "--first-chunk-frames",
        type=int,
        default=48,
        help="Switch to stable settings after N frames (default: 48)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap samples for crossfade between chunks (default: 0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10000,
        help="Maximum codec frames per line (default: 10000)",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Read text from a file instead of stdin",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup runs when using --optimized",
    )

    args = parser.parse_args()

    total_start = time.time()

    # ============== Load model ==============
    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    log_time(start, "Model loaded")

    # ============== Enable optimizations ==============
    if args.optimized:
        print("\nEnabling streaming optimizations...")
        model.enable_streaming_optimizations(
            decode_window_frames=args.decode_window,
            use_compile=True,
            use_cuda_graphs=False,
            compile_mode="reduce-overhead",
            use_fast_codebook=False,
            compile_codebook_predictor=True,
            compile_talker=True,
        )

        if not args.no_warmup:
            warmup_texts = [
                "Тест один два три четыре пять.",
                "Привет, как дела? Это второй прогрев системы.",
                "Третий тестовый запуск для полного прогрева.",
            ]
            print("\nWarmup runs (compilation happens here)...")
            for i, warmup_text in enumerate(warmup_texts, 1):
                warmup_start = time.time()
                warmup_chunks = []
                for chunk, sr in model.stream_generate_voice_clone(
                    text=warmup_text,
                    language=args.language,
                    voice_clone_prompt={},  # Dummy, not used during warmup timing
                    emit_every_frames=args.emit_every,
                    decode_window_frames=args.decode_window,
                    overlap_samples=args.overlap,
                    first_chunk_emit_every=args.first_chunk_emit,
                    first_chunk_frames=args.first_chunk_frames,
                    max_frames=min(200, args.max_frames),
                ):
                    warmup_chunks.append(chunk)
                warmup_time = time.time() - warmup_start
                print(f"  Warmup {i}: {warmup_time:.2f}s")
            print("Warmup complete.")

    # ============== Build voice clone prompt or get speaker ==============
    voice_clone_prompt = None
    if args.ref_audio is not None:
        print(f"\nBuilding voice clone prompt from: {args.ref_audio}")
        start = time.time()
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only,
        )
        log_time(start, "Voice clone prompt created")
    elif args.speaker is not None:
        print(f"\nUsing CustomVoice speaker: {args.speaker}")
    else:
        print(f"\nModel type: {model.model.tts_model_type}")
        if model.model.tts_model_type == "base":
            print("WARNING: No --ref-audio provided. Voice cloning requires a reference audio file.")
            print("Use --ref-audio path/to/reference.wav to clone a voice.")

    # ============== Read text from source ==============
    print("\n" + "=" * 60)
    print("Reading text input...")
    print("=" * 60)

    lines = []
    if args.text_file is not None:
        print(f"Reading from file: {args.text_file}")
        with open(args.text_file, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
    else:
        print("Reading from stdin (Ctrl+D to finish):")
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line.strip():
                lines.append(line)

    if not lines:
        print("No text provided. Exiting.")
        return

    print(f"Read {len(lines)} line(s) of text:")
    for i, line in enumerate(lines):
        print(f"  [{i+1}] {line}")
    print()

    # ============== Generate audio ==============
    print("=" * 60)
    print("Generating audio (streaming)...")
    print("=" * 60)

    all_chunks = []
    total_first_chunk_time = 0.0
    total_streaming_time = 0.0
    chunk_count = 0
    sample_rate = 24000
    line_times = []

    for line_idx, line_text in enumerate(lines, 1):
        line_start = time.time()
        line_chunks = []
        line_first_chunk_time = None

        print(f"\n--- Line {line_idx}: \"{line_text[:60]}...\" ---")

        try:
            if model.model.tts_model_type == "base":
                gen_iter = model.stream_generate_voice_clone(
                    text=line_text,
                    language=args.language,
                    voice_clone_prompt=voice_clone_prompt,
                    emit_every_frames=args.emit_every,
                    decode_window_frames=args.decode_window,
                    overlap_samples=args.overlap,
                    max_frames=args.max_frames,
                    first_chunk_emit_every=args.first_chunk_emit,
                    first_chunk_decode_window=args.decode_window,
                    first_chunk_frames=args.first_chunk_frames,
                )
            elif model.model.tts_model_type == "custom_voice":
                gen_iter = model.stream_generate_custom_voice(
                    text=line_text,
                    speaker=args.speaker,
                    language=args.language,
                    instruct=args.instruct,
                    emit_every_frames=args.emit_every,
                    decode_window_frames=args.decode_window,
                    overlap_samples=args.overlap,
                    max_frames=args.max_frames,
                    first_chunk_emit_every=args.first_chunk_emit,
                    first_chunk_decode_window=args.decode_window,
                    first_chunk_frames=args.first_chunk_frames,
                )
            else:
                print(f"ERROR: Unsupported model type: {model.model.tts_model_type}")
                return

            for chunk, chunk_sr in gen_iter:
                chunk_count += 1
                line_chunks.append(chunk)
                sample_rate = chunk_sr
                if line_first_chunk_time is None:
                    line_first_chunk_time = time.time() - line_start
                    print(f"  First chunk: {line_first_chunk_time:.2f}s ({len(chunk)} samples)")

        except Exception as e:
            print(f"  ERROR generating line {line_idx}: {e}")
            continue

        line_time = time.time() - line_start
        line_times.append(line_time)
        total_first_chunk_time += line_first_chunk_time
        total_streaming_time += line_time
        all_chunks.extend(line_chunks)

        audio_duration = sum(len(c) for c in line_chunks) / sample_rate if sample_rate > 0 else 0
        print(f"  Line {line_idx} done: {line_time:.2f}s, {len(line_chunks)} chunks, {audio_duration:.2f}s audio")

    # ============== Concatenate and save ==============
    print("\n" + "=" * 60)
    print("Saving output...")
    print("=" * 60)

    if all_chunks:
        final_audio = np.concatenate(all_chunks)
        sf.write(args.output, final_audio, sample_rate)
        total_audio_duration = len(final_audio) / sample_rate
        print(f"Saved {len(all_chunks)} chunks to: {args.output}")
        print(f"Total audio duration: {total_audio_duration:.2f}s")
    else:
        print("No audio generated.")
        return

    # ============== Summary ==============
    elapsed = time.time() - total_start
    avg_first_chunk = total_first_chunk_time / len(lines) if lines else 0
    avg_line_time = sum(line_times) / len(line_times) if line_times else 0
    rtf = total_streaming_time / total_audio_duration if total_audio_duration > 0 else 0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Lines processed: {len(lines)}")
    print(f"Total chunks: {chunk_count}")
    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Average first chunk latency: {avg_first_chunk:.2f}s")
    print(f"Average time per line: {avg_line_time:.2f}s")
    print(f"Total streaming time: {total_streaming_time:.2f}s")
    print(f"RTF (real-time factor): {rtf:.2f}")
    print(f"Total script time: {elapsed:.2f}s")
    print(f"Output file: {args.output}")

    if args.optimized:
        print("\nOptimizations enabled: torch.compile + CUDA graphs")
    else:
        print("\nOptimizations disabled. Use --optimized for faster generation.")


if __name__ == "__main__":
    main()

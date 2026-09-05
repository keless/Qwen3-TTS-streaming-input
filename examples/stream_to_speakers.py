"""
Stream TTS audio directly to speakers in real-time.

Plays synthesized speech through your system audio as each chunk is generated,
minimizing first-chunk latency. Supports two model types:

  1. Base (voice cloning):   --ref-audio <file> --ref-text <text>
  2. CustomVoice:            --speaker <name>

Usage examples:

  # Voice cloning with streaming optimizations
  python examples/stream_to_speakers.py \
      Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      --ref-audio ref.wav --ref-text "Reference transcript here." \
      --text "Text to synthesize." \
      --optimize

  # CustomVoice with optimized streaming
  python examples/stream_to_speakers.py \
      Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
      --speaker Vivian \
      --text "Hello, this is a test of real-time playback." \
      --optimize

  # Two-phase streaming for lower first-chunk latency
  python examples/stream_to_speakers.py \
      Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      --ref-audio ref.wav --ref-text "Reference transcript here." \
      --text "Text to synthesize." \
      --optimize --two-phase

Requirements:
  - GPU with CUDA
  - sounddevice (for playback)
  - soundfile, numpy, torch (standard dependencies)
"""

import argparse
import sys
import time

import numpy as np
import torch
import sounddevice as sd
from qwen_tts import Qwen3TTSModel


def play_streaming(model, text, language, play_fn, two_phase, optimize,
                   first_chunk_frames, emit_every, decode_window,
                   optimizations):
    """Run streaming generation and play each chunk as it arrives."""

    # ---- Timing ----
    start = time.time()
    first_chunk_time = None
    chunk_count = 0
    total_samples = 0
    sample_rate = 24000  # default; updated from first chunk

    # ---- Prepare playback stream ----
    # Use a blocking playback stream so audio plays smoothly while we wait
    # for the next chunk. The stream buffers internally.
    stream = sd.PlaybackStream(
        samplerate=sample_rate,
        channels=1,
        blocksize=4096,
    )
    stream.start()

    # Reconfigure sample_rate once we know it from the first chunk
    first_chunk = True

    try:
        for chunk, sr in play_fn(
            text=text,
            language=language,
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
            overlap_samples=0,
            first_chunk_emit_every=(first_chunk_frames if two_phase else 0),
            first_chunk_decode_window=first_chunk_frames,
            first_chunk_frames=first_chunk_frames,
            use_optimized_decode=optimize,
        ):
            chunk_count += 1
            total_samples += len(chunk)
            sample_rate = sr

            if first_chunk:
                first_chunk_time = time.time() - start
                first_chunk = False
                # Reconfigure stream to match actual sample rate
                stream.samplerate = sr

            # Normalize to [-1, 1] and play
            chunk_np = np.asarray(chunk, dtype=np.float32)
            if np.issubdtype(chunk_np.dtype, np.floating):
                m = np.max(np.abs(chunk_np)) if chunk_np.size else 0.0
                if m > 1.0 + 1e-6:
                    chunk_np = chunk_np / m
            chunk_np = np.clip(chunk_np, -1.0, 1.0)

            stream.write(chunk_np)

    except GeneratorExit:
        pass
    finally:
        stream.flush()
        stream.close()

    elapsed = time.time() - start
    audio_duration = total_samples / sample_rate if sample_rate > 0 else 0

    print(f"\nPlayback complete.")
    print(f"  Chunks: {chunk_count}")
    print(f"  Total audio: {audio_duration:.2f}s")
    print(f"  First chunk latency: {first_chunk_time:.2f}s" if first_chunk_time else f"  First chunk latency: N/A")
    print(f"  Total time: {elapsed:.2f}s")
    if audio_duration > 0:
        rtf = elapsed / audio_duration
        print(f"  RTF: {rtf:.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stream TTS audio directly to speakers in real-time.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "checkpoint",
        help="Model checkpoint path or HuggingFace repo id "
             "(e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    parser.add_argument(
        "--text", "-t",
        required=True,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--language", "-l",
        default="Auto",
        help="Language (default: Auto).",
    )

    # Model loading
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for model (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        help="Enable FlashAttention-2 (default: enabled).",
    )

    # Voice clone (Base model)
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Path to reference audio file for voice cloning.",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of the reference audio (required for Base model "
             "unless --xvector-only is set).",
    )
    parser.add_argument(
        "--xvector-only",
        action="store_true",
        help="Use only speaker embedding for voice cloning (ignores ref_text).",
    )

    # CustomVoice
    parser.add_argument(
        "--speaker",
        default=None,
        help="Speaker name (for CustomVoice models).",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Style instruction (for CustomVoice models).",
    )

    # Playback
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="sounddevice playback device index (default: default device).",
    )
    parser.add_argument(
        "--volume-scale",
        type=float,
        default=1.0,
        help="Playback volume multiplier (default: 1.0).",
    )

    # Streaming parameters
    parser.add_argument(
        "--emit-every",
        type=int,
        default=8,
        help="Emit audio chunk every N codec frames (default: 8). "
             "Lower = lower latency, potentially more artifacts.",
    )
    parser.add_argument(
        "--decode-window",
        type=int,
        default=80,
        help="Decode window size in frames (default: 80). "
             "Higher = better quality, more latency per chunk.",
    )
    parser.add_argument(
        "--first-chunk-frames",
        type=int,
        default=48,
        help="Switch to stable settings after this many frames (default: 48).",
    )
    parser.add_argument(
        "--two-phase",
        action="store_true",
        help="Enable two-phase streaming: aggressive first chunk for lower "
             "first-chunk latency, then stable settings for quality.",
    )

    # Optimization
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable streaming optimizations (torch.compile + CUDA graphs).",
    )

    args = parser.parse_args(argv)

    # Resolve dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype.lower(), torch.bfloat16)
    attn_impl = "flash_attention_2" if args.flash_attn else None

    # Resolve playback device
    device_info = None
    if args.device_index is not None:
        device_info = args.device_index
    else:
        # Show available devices
        devices = sd.query_devices()
        default_host = sd.default.device[0]
        if default_host is not None:
            device_info = default_host
        print(f"Using default playback device: {device_info}")
        print(f"  {sd.query_device(device_info)['name']}")

    # ---- Load model ----
    print(f"Loading model: {args.checkpoint}")
    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    print(f"Model loaded in {time.time() - start:.2f}s")

    model_type = getattr(model.model, "tts_model_type", None)
    print(f"Model type: {model_type}")

    # ---- Validate model type vs arguments ----
    if model_type == "base":
        if args.speaker is not None:
            print("Error: --speaker is for CustomVoice models, not Base.")
            sys.exit(1)
        if args.ref_audio is None:
            print("Error: --ref-audio is required for Base (voice cloning) models.")
            sys.exit(1)

    elif model_type == "custom_voice":
        if args.ref_audio is not None:
            print("Error: --ref-audio is for Base models, not CustomVoice.")
            sys.exit(1)
        if args.speaker is None:
            print("Error: --speaker is required for CustomVoice models.")
            print(f"Available speakers: {model.get_supported_speakers()}")
            sys.exit(1)

    elif model_type == "voice_design":
        print("Warning: VoiceDesign models do not support streaming playback. "
              "Only Base (voice cloning) and CustomVoice models are supported.")
        sys.exit(1)
    else:
        print(f"Error: Unknown model type: {model_type}")
        sys.exit(1)

    # ---- Build the streaming generator function ----
    if model_type == "base":
        # Build voice clone prompt once, then stream
        print(f"\nBuilding voice clone prompt from: {args.ref_audio}")
        prompt = model.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.xvector_only,
        )
        voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt)

        def play_fn(**kwargs):
            return model.stream_generate_voice_clone(
                voice_clone_prompt=voice_clone_prompt_dict,
                **kwargs,
            )

    elif model_type == "custom_voice":
        def play_fn(**kwargs):
            return model.stream_generate_custom_voice(
                speaker=args.speaker,
                instruct=args.instruct,
                **kwargs,
            )

    # ---- Enable optimizations ----
    if args.optimize:
        print("\nEnabling streaming optimizations (torch.compile + CUDA graphs)...")
        model.enable_streaming_optimizations(
            decode_window_frames=args.decode_window,
            use_compile=True,
            use_cuda_graphs=False,  # reduce-overhead mode includes CUDA graphs
            compile_mode="reduce-overhead",
            use_fast_codebook=False,
            compile_codebook_predictor=True,
            compile_talker=True,
        )
        print("Optimizations enabled. Running warmup...")

        # Warmup run: compilation happens on first streaming call, which would
        # add noticeable latency to the first audio chunk. Run a short warmup
        # with a trivial text so the real generation starts immediately.
        warmup_text = "Test."
        for _ in play_fn(
            text=warmup_text,
            language=args.language,
            emit_every_frames=args.emit_every,
            decode_window_frames=args.decode_window,
            overlap_samples=0,
            first_chunk_emit_every=0,
            first_chunk_decode_window=args.decode_window,
            first_chunk_frames=args.decode_window,
            use_optimized_decode=True,
        ):
            pass
        print("Warmup complete.")

    # ---- Play ----
    print(f"\n{'='*60}")
    print(f"Streaming to speakers")
    print(f"  Text: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print(f"  Language: {args.language}")
    print(f"  Two-phase: {args.two_phase}")
    print(f"  Emit every: {args.emit_every} frames")
    print(f"  Decode window: {args.decode_window} frames")
    print(f"{'='*60}\n")

    play_streaming(
        model,
        text=args.text,
        language=args.language,
        play_fn=play_fn,
        two_phase=args.two_phase,
        optimize=args.optimize,
        first_chunk_frames=args.first_chunk_frames,
        emit_every=args.emit_every,
        decode_window=args.decode_window,
        optimizations=args.optimize,
    )


if __name__ == "__main__":
    main()

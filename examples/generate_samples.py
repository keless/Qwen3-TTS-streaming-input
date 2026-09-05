"""
Generate TTS samples with Qwen3-TTS.

Generates audio samples using the three model types:
  - Base: voice cloning from reference audio
  - CustomVoice: predefined speakers
  - VoiceDesign: natural-language style instructions

Usage examples:
    # Voice cloning (base model), non-streaming
    python examples/generate_samples.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --mode voice_clone \
        --ref-audio ref.wav \
        --ref-text "Reference transcript here." \
        --text "Text to synthesize." \
        --language Russian \
        --output output.wav

    # Voice cloning with streaming and optimizations
    python examples/generate_samples.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --mode voice_clone \
        --ref-audio ref.wav \
        --ref-text "Reference transcript here." \
        --text "Text to synthesize." \
        --language Russian \
        --output output.wav \
        --streaming \
        --optimize

    # CustomVoice model
    python examples/generate_samples.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --mode custom_voice \
        --speaker Alice \
        --text "Text to synthesize." \
        --language en \
        --output output.wav

    # VoiceDesign model
    python examples/generate_samples.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
        --mode voice_design \
        --text "Text to synthesize." \
        --instruct "Speak in a warm, friendly female voice." \
        --language en \
        --output output.wav

    # Generate multiple samples at once
    python examples/generate_samples.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --mode voice_clone \
        --ref-audio ref.wav \
        --ref-text "Reference transcript here." \
        --text-file texts.txt \
        --output-dir ./samples \
        --streaming \
        --optimize
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def load_model(model_path: str, device_map: str = "cuda:0") -> Qwen3TTSModel:
    """Load a Qwen3-TTS model."""
    print(f"Loading model from {model_path} ...")
    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")
    return model


def build_voice_clone_prompt(
    model: Qwen3TTSModel,
    ref_audio: str,
    ref_text: str,
) -> dict:
    """Build a voice clone prompt from reference audio and text."""
    print(f"  Building voice clone prompt from {ref_audio} ...")
    start = time.time()
    items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    prompt = model._prompt_items_to_voice_clone_prompt(items)
    elapsed = time.time() - start
    print(f"  Prompt built in {elapsed:.1f}s")
    return prompt


def generate_non_streaming(
    model: Qwen3TTSModel,
    mode: str,
    text: str,
    language: str,
    voice_clone_prompt: dict | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    output_path: str | None = None,
    **gen_kwargs,
) -> tuple[np.ndarray, int]:
    """Generate audio using non-streaming (full) generation."""
    print("  Generating (non-streaming) ...")
    start = time.time()

    if mode == "voice_clone":
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            **gen_kwargs,
        )
    elif mode == "custom_voice":
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            **gen_kwargs,
        )
    elif mode == "voice_design":
        wavs, sr = model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
            **gen_kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    elapsed = time.time() - start
    audio = wavs[0] if wavs else np.array([])
    duration = len(audio) / sr if sr > 0 else 0
    rtf = elapsed / duration if duration > 0 else 0

    print(f"  Done in {elapsed:.1f}s (audio {duration:.1f}s, RTF {rtf:.2f})")

    if output_path is not None:
        sf.write(output_path, audio, sr)
        print(f"  Saved to {output_path}")

    return audio, sr


def generate_streaming(
    model: Qwen3TTSModel,
    mode: str,
    text: str,
    language: str,
    voice_clone_prompt: dict | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    output_path: str | None = None,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    overlap_samples: int = 0,
    first_chunk_emit_every: int = 0,
    first_chunk_decode_window: int = 48,
    first_chunk_frames: int = 48,
    **gen_kwargs,
) -> tuple[np.ndarray, int]:
    """Generate audio using streaming generation, collecting all chunks."""
    print(f"  Generating (streaming, emit_every={emit_every_frames}) ...")
    start = time.time()
    chunks = []
    chunk_sr = 24000
    first_chunk_time = None
    chunk_count = 0

    if mode == "voice_clone":
        generator = model.stream_generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            first_chunk_emit_every=first_chunk_emit_every,
            first_chunk_decode_window=first_chunk_decode_window,
            first_chunk_frames=first_chunk_frames,
            **gen_kwargs,
        )
    elif mode == "custom_voice":
        generator = model.stream_generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            first_chunk_emit_every=first_chunk_emit_every,
            first_chunk_decode_window=first_chunk_decode_window,
            first_chunk_frames=first_chunk_frames,
            **gen_kwargs,
        )
    else:
        raise ValueError(f"Streaming not supported for mode: {mode}")

    for chunk, sr in generator:
        chunk_count += 1
        chunks.append(chunk)
        chunk_sr = sr
        if first_chunk_time is None:
            first_chunk_time = time.time() - start

    total_time = time.time() - start
    audio = np.concatenate(chunks) if chunks else np.array([])
    duration = len(audio) / chunk_sr if chunk_sr > 0 else 0
    rtf = total_time / duration if duration > 0 else 0

    print(f"  Done in {total_time:.1f}s "
          f"(first chunk {first_chunk_time:.2f}s, "
          f"{chunk_count} chunks, "
          f"audio {duration:.1f}s, RTF {rtf:.2f})")

    if output_path is not None:
        sf.write(output_path, audio, chunk_sr)
        print(f"  Saved to {output_path}")

    return audio, chunk_sr


def generate_samples(
    model: Qwen3TTSModel,
    mode: str,
    text: str,
    language: str,
    voice_clone_prompt: dict | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    streaming: bool = False,
    optimize: bool = False,
    output_path: str | None = None,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    overlap_samples: int = 0,
    first_chunk_emit_every: int = 0,
    first_chunk_decode_window: int = 48,
    first_chunk_frames: int = 48,
    **gen_kwargs,
) -> tuple[np.ndarray, int]:
    """
    Generate a single audio sample.

    If optimize=True, enables streaming optimizations (torch.compile, CUDA graphs)
    before generation. For streaming mode, this must be done before the first
    streaming call since compilation is one-time.
    """
    if optimize:
        print("  Enabling streaming optimizations ...")
        model.enable_streaming_optimizations(
            decode_window_frames=decode_window_frames,
            use_compile=True,
            use_cuda_graphs=False,  # reduce-overhead already includes CUDA graphs
            compile_mode="reduce-overhead",
            use_fast_codebook=False,
            compile_codebook_predictor=True,
            compile_talker=True,
        )

    if streaming:
        return generate_streaming(
            model, mode, text, language,
            voice_clone_prompt=voice_clone_prompt,
            speaker=speaker,
            instruct=instruct,
            output_path=output_path,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            first_chunk_emit_every=first_chunk_emit_every,
            first_chunk_decode_window=first_chunk_decode_window,
            first_chunk_frames=first_chunk_frames,
            **gen_kwargs,
        )
    else:
        return generate_non_streaming(
            model, mode, text, language,
            voice_clone_prompt=voice_clone_prompt,
            speaker=speaker,
            instruct=instruct,
            output_path=output_path,
            **gen_kwargs,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS samples with Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model path (HuggingFace repo or local directory). "
             "Default: Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to load model on. Default: cuda:0",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--mode",
        choices=["voice_clone", "custom_voice", "voice_design"],
        help="Generation mode. Mutually exclusive with --mode-shorthand.",
    )
    mode_group.add_argument(
        "--mode-shorthand",
        choices=["vc", "cv", "vd"],
        help="Shorthand for mode: vc=voice_clone, cv=custom_voice, vd=voice_design",
    )

    # Voice clone inputs
    parser.add_argument(
        "--ref-audio",
        help="Path to reference audio for voice cloning.",
    )
    parser.add_argument(
        "--ref-text",
        help="Transcript of the reference audio.",
    )

    # CustomVoice inputs
    parser.add_argument(
        "--speaker",
        help="Speaker name for CustomVoice model.",
    )

    # VoiceDesign inputs
    parser.add_argument(
        "--instruct",
        help="Style instruction for VoiceDesign model.",
    )

    # Text input
    parser.add_argument(
        "--text",
        help="Text to synthesize (single string).",
    )
    parser.add_argument(
        "--text-file",
        help="Path to a text file with one text per line. "
             "Generates one sample per line.",
    )

    # Language
    parser.add_argument(
        "--language",
        default="Auto",
        help="Language for synthesis. Default: Auto",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output WAV file path (single sample mode).",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for multi-sample mode. "
             "Files named as <index>_<text_hash>.wav.",
    )

    # Generation options
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming generation (yields chunks as they are produced).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable streaming optimizations (torch.compile, CUDA graphs).",
    )

    # Streaming parameters
    parser.add_argument(
        "--emit-every",
        type=int,
        default=8,
        help="Emit audio every N codec frames. Default: 8",
    )
    parser.add_argument(
        "--decode-window",
        type=int,
        default=80,
        help="Decoder context window in frames. Default: 80",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Crossfade overlap samples between chunks. Default: 0",
    )
    parser.add_argument(
        "--first-chunk-emit",
        type=int,
        default=0,
        help="Phase 1 emit interval for faster first chunk (0=disabled). Default: 0",
    )
    parser.add_argument(
        "--first-chunk-window",
        type=int,
        default=48,
        help="Phase 1 decode window size. Default: 48",
    )
    parser.add_argument(
        "--first-chunk-frames",
        type=int,
        default=48,
        help="Switch to phase 2 after N frames. Default: 48",
    )

    # Generation params
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty.",
    )

    args = parser.parse_args()

    # Resolve mode shorthand
    mode = args.mode
    if mode is None and args.mode_shorthand is not None:
        mapping = {"vc": "voice_clone", "cv": "custom_voice", "vd": "voice_design"}
        mode = mapping[args.mode_shorthand]

    # Resolve text
    texts = []
    if args.text is not None:
        texts.append(args.text)
    elif args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Provide either --text or --text-file.")

    if not texts:
        parser.error("No text provided.")

    # Resolve output paths
    output_paths = []
    for i, text in enumerate(texts):
        if len(texts) == 1:
            if args.output is not None:
                output_paths.append(args.output)
            else:
                # Default filename
                text_short = text[:50].replace(" ", "_").replace("/", "_")
                output_paths.append(f"{text_short}.wav")
        else:
            if args.output_dir is None:
                args.output_dir = "samples"
            os.makedirs(args.output_dir, exist_ok=True)
            text_short = text[:30].replace(" ", "_").replace("/", "_")
            name = f"{i}_{text_short}.wav"
            output_paths.append(os.path.join(args.output_dir, name))

    # Load model
    model = load_model(args.model, device_map=args.device)

    # Build voice clone prompt if needed
    voice_clone_prompt = None
    if mode == "voice_clone":
        if args.ref_audio is None:
            parser.error("--ref-audio is required for voice_clone mode.")
        voice_clone_prompt = build_voice_clone_prompt(
            model, args.ref_audio, args.ref_text or ""
        )

    # Validate mode-specific inputs
    if mode == "custom_voice" and args.speaker is None:
        parser.error("--speaker is required for custom_voice mode.")
    if mode == "voice_design" and args.instruct is None:
        parser.error("--instruct is required for voice_design mode.")

    # Collect generation kwargs
    gen_kwargs = {}
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty

    # Print configuration summary
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  Model:        {args.model}")
    print(f"  Mode:         {mode}")
    print(f"  Language:     {args.language}")
    print(f"  Streaming:    {args.streaming}")
    print(f"  Optimized:    {args.optimize}")
    print(f"  Samples:      {len(texts)}")
    if mode == "custom_voice":
        print(f"  Speaker:      {args.speaker}")
    if mode == "voice_design":
        print(f"  Instruct:     {args.instruct}")
    print(f"  Emit every:   {args.emit_every}")
    print(f"  Decode window:{args.decode_window}")
    if args.first_chunk_emit > 0:
        print(f"  Two-phase:    Yes (emit={args.first_chunk_emit}, "
              f"window={args.first_chunk_decode_window}, "
              f"frames={args.first_chunk_frames})")
    print("=" * 60)

    # Generate samples
    total_start = time.time()
    results = []

    for i, text in enumerate(texts):
        print(f"\n--- Sample {i + 1}/{len(texts)} ---")
        print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        try:
            audio, sr = generate_samples(
                model=model,
                mode=mode,
                text=text,
                language=args.language,
                voice_clone_prompt=voice_clone_prompt,
                speaker=args.speaker,
                instruct=args.instruct,
                streaming=args.streaming,
                optimize=args.optimize,
                output_path=output_paths[i],
                emit_every_frames=args.emit_every,
                decode_window_frames=args.decode_window,
                overlap_samples=args.overlap,
                first_chunk_emit_every=args.first_chunk_emit,
                first_chunk_decode_window=args.first_chunk_window,
                first_chunk_frames=args.first_chunk_frames,
                **gen_kwargs,
            )
            results.append({"audio": audio, "sr": sr, "path": output_paths[i]})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"error": str(e), "path": output_paths[i]})

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done. Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    # List generated files
    for r in results:
        status = "OK" if "audio" in r else f"ERROR: {r['error']}"
        print(f"  [{status}] {r['path']}")


if __name__ == "__main__":
    main()

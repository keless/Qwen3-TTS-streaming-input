"""
Streaming text input demo for Qwen3-TTS.

This script demonstrates the difference between two text processing modes:

  non_streaming_mode=True   (default in generate_voice_clone):
    The entire text is pre-built into the talker input embeddings before
    generation begins. Trailing text hidden states are set to a pad embedding.

  non_streaming_mode=False  (streaming text input):
    Text is processed incrementally during generation. The trailing text
    hidden states are computed from the actual text tokens and passed
    through step-by-step as audio is generated.

Both modes generate audio in chunks via the streaming API. The key
difference is how the text context is prepared and maintained internally.

Streaming text input is useful for real-time applications where text
arrives incrementally (e.g., a chatbot that streams text to the TTS
as the LLM generates it).

Usage:
    cd Qwen3-TTS
    python examples/streaming_text_input_poc.py
"""

import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


def run_streaming_generation(
    model,
    text: str,
    language: str,
    voice_clone_prompt: dict,
    non_streaming_mode: bool,
    label: str,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
):
    """Run streaming generation with the given mode and return stats."""
    start = time.time()
    chunks = []
    first_chunk_time = None
    chunk_count = 0
    sample_rate = 24000

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=non_streaming_mode,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
        overlap_samples=0,
    ):
        chunk_count += 1
        chunks.append(chunk)
        sample_rate = chunk_sr
        if first_chunk_time is None:
            first_chunk_time = time.time() - start

    total_time = time.time() - start
    final_audio = np.concatenate(chunks) if chunks else np.array([])
    audio_duration = len(final_audio) / sample_rate if sample_rate > 0 else 0

    print(f"\n--- {label} ---")
    print(f"  non_streaming_mode: {non_streaming_mode}")
    print(f"  First chunk: {first_chunk_time:.2f}s")
    print(f"  Total time:  {total_time:.2f}s")
    print(f"  Audio:       {audio_duration:.2f}s ({chunk_count} chunks)")

    return {
        "label": label,
        "non_streaming_mode": non_streaming_mode,
        "first_chunk_time": first_chunk_time,
        "total_time": total_time,
        "audio": final_audio,
        "sample_rate": sample_rate,
        "audio_duration": audio_duration,
        "chunk_count": chunk_count,
    }


def main():
    total_start = time.time()

    print("=" * 60)
    print("Streaming Text Input Demo")
    print("=" * 60)

    # ---- Load model ----
    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    log_time(start, "Model loaded")

    # ---- Voice clone prompt ----
    ref_audio_path = "kuklina-1.wav"
    ref_text = (
        "This is a reference voice sample used for cloning. "
        "The model will use this to extract speaker characteristics."
    )

    start = time.time()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    voice_clone_prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)
    log_time(start, "Voice clone prompt created")

    # ---- Test text ----
    test_text = (
        "This is a demonstration of streaming text input in Qwen3-TTS. "
        "The model can process text either all at once or incrementally "
        "as it arrives. Both approaches produce the same audio output, "
        "but streaming text input is designed for real-time applications "
        "where text comes from a streaming source like a chatbot."
    )

    # ---- Run both modes ----
    results = []

    # Mode 1: Non-streaming text input (text fully pre-built)
    results.append(
        run_streaming_generation(
            model,
            text=test_text,
            language="English",
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True,
            label="Mode 1: Non-streaming text input",
        )
    )
    sf.write("output_non_streaming_text.wav", results[-1]["audio"], results[-1]["sample_rate"])

    # Mode 2: Streaming text input (text processed incrementally)
    results.append(
        run_streaming_generation(
            model,
            text=test_text,
            language="English",
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=False,
            label="Mode 2: Streaming text input",
        )
    )
    sf.write("output_streaming_text.wav", results[-1]["audio"], results[-1]["sample_rate"])

    # ---- Compare audio ----
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    r1 = results[0]  # non-streaming text
    r2 = results[1]  # streaming text

    audio1 = r1["audio"]
    audio2 = r2["audio"]

    # Check if audio outputs are similar
    if len(audio1) == len(audio2):
        diff = np.abs(audio1 - audio2).max()
        print(f"  Audio length match:  {len(audio1)} == {len(audio2)} samples")
        print(f"  Max absolute diff:   {diff:.6f}")
        if diff < 1e-4:
            print("  Audio outputs are IDENTICAL (within numerical precision)")
        else:
            print("  Audio outputs differ slightly (expected due to sampling)")
    else:
        print(f"  Audio length mismatch: {len(audio1)} vs {len(audio2)} samples")
        print("  (This is expected when generation takes different paths)")

    print(f"\n  Mode 1 total time:    {r1['total_time']:.2f}s")
    print(f"  Mode 2 total time:    {r2['total_time']:.2f}s")
    time_diff = r2["total_time"] - r1["total_time"]
    print(f"  Time difference:      {abs(time_diff):.2f}s ({'Mode 2 faster' if time_diff < 0 else 'Mode 1 faster'})")

    # ---- How it works ----
    print("\n" + "=" * 60)
    print("HOW IT WORKS")
    print("=" * 60)
    print("""
The key difference is in _build_talker_inputs() inside the model:

  non_streaming_mode=True:
    - Text embedding is padded with codec pad tokens to match codec length
    - Trailing text hidden = tts_pad_embed (no text information passed through)
    - Entire text context is available before generation starts

  non_streaming_mode=False:
    - Text embedding is aligned with codec embedding at each step
    - Trailing text hidden = actual text tokens + EOS (passed through step-by-step)
    - Text context is processed incrementally during generation

Both modes use the same streaming audio API and produce audio in chunks.
The streaming text mode is designed for real-time applications where
text arrives incrementally (e.g., streaming from a chatbot LLM).

Key files:
  - qwen_tts/core/models/modeling_qwen3_tts.py
      generate_icl_prompt()  - builds text+codec embedding alignment
      _build_talker_inputs() - sets trailing_text_hidden based on mode
      stream_generate_pcm()  - the streaming audio generation loop
""")

    print(f"\n[{time.time() - total_start:.2f}s] TOTAL")


if __name__ == "__main__":
    main()

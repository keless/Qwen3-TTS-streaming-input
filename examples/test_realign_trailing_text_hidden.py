"""
Test trailing text hidden state computation and realignment during streaming generation.

This script demonstrates how "trailing text hidden" states are:
1. Computed from input text tokens in _build_talker_inputs()
2. Passed through each streaming forward pass
3. Realigned with codec generation steps in Qwen3TTSTalkerForConditionalGeneration.forward()

Key concept: The trailing text hidden states provide text context at each codec frame step.
During streaming, the text hidden at position `generation_step` is added to the codec
embedding, effectively realigning text token positions with audio frame positions.

Usage:
    cd Qwen3-TTS
    python examples/test_realign_trailing_text_hidden.py
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


def inspect_trailing_text_hidden(model, text, language, voice_clone_prompt_dict):
    """
    Build talker inputs and inspect trailing_text_hidden properties.

    This mirrors the internal computation in _build_talker_inputs() and
    demonstrates how trailing text hidden states are structured for streaming.
    """
    print("\n" + "=" * 60)
    print("Trailing Text Hidden State Inspection")
    print("=" * 60)

    # Reconstruct what _build_talker_inputs does internally
    input_texts = [f"assistant\n{text}assistant\n"]
    input_ids = []
    for t in input_texts:
        inp = model.processor(text=t, return_tensors="pt", padding=True)
        input_id = inp["input_ids"].to(model.device)
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        input_ids.append(input_id)

    # Build ref_ids
    ref_text = voice_clone_prompt_dict.get("ref_text", None)
    ref_ids = None
    if ref_text:
        ref_tok = model.processor(text=f"assistant\n{ref_text}assistant\n", return_tensors="pt", padding=True)
        ref_id = ref_tok["input_ids"].to(model.device)
        ref_id = ref_id.unsqueeze(0) if ref_id.dim() == 1 else ref_id
        ref_ids = [ref_id]

    # Build talker inputs using the internal method
    (
        talker_input_embeds,
        talker_attention_mask,
        trailing_text_hiddens,
        tts_pad_embed,
    ) = model.model._build_talker_inputs(
        input_ids=input_ids,
        instruct_ids=None,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=[language],
        speakers=None,
        non_streaming_mode=False,
    )

    print(f"\nInput text: '{text}' ({len(text)} chars)")
    print(f"Input text tokens: {input_ids[0].shape[1]} tokens")

    print(f"\n--- Talker Input Embeddings ---")
    print(f"  Shape: {talker_input_embeds.shape}")
    print(f"  dtype: {talker_input_embeds.dtype}")

    print(f"\n--- Attention Mask ---")
    print(f"  Shape: {talker_attention_mask.shape}")
    print(f"  Active tokens: {talker_attention_mask[0].sum().item()}")

    print(f"\n--- Trailing Text Hidden ---")
    print(f"  Shape: {trailing_text_hiddens.shape}")
    print(f"  dtype: {trailing_text_hiddens.dtype}")
    print(f"  Device: {trailing_text_hiddens.device}")

    # The trailing text hidden has shape [batch, num_text_tokens, hidden_dim]
    # Each position corresponds to a text token that will be added at a generation step
    num_text_tokens = trailing_text_hiddens.shape[1]
    hidden_dim = trailing_text_hiddens.shape[2]
    print(f"  Number of text token positions: {num_text_tokens}")
    print(f"  Hidden dimension: {hidden_dim}")

    # Show per-token statistics
    token_norms = trailing_text_hiddens[0].norm(dim=-1).cpu().numpy()
    print(f"\n  Per-token hidden norm statistics:")
    print(f"    Mean: {token_norms.mean():.4f}")
    print(f"    Std:  {token_norms.std():.4f}")
    print(f"    Min:  {token_norms.min():.4f}")
    print(f"    Max:  {token_norms.max():.4f}")

    # Show the EOS token embedding (last position)
    eos_norm = trailing_text_hiddens[0, -1:].norm(dim=-1).item()
    print(f"\n  EOS token hidden norm: {eos_norm:.4f}")

    # Show tts_pad_embed
    print(f"\n--- TTS Pad Embed ---")
    print(f"  Shape: {tts_pad_embed.shape}")
    print(f"  Norm: {tts_pad_embed.norm().item():.4f}")

    # Explain the realignment concept
    print(f"\n--- Realign Concept ---")
    print(f"  During streaming, at each codec generation step k:")
    print(f"    if k < trailing_text_hidden.shape[1]:")
    print(f"        inputs_embeds += trailing_text_hidden[:, k]")
    print(f"    else:")
    print(f"        inputs_embeds += tts_pad_embed")
    print(f"")
    print(f"  This means the text hidden at position k is added to the codec")
    print(f"  embedding at generation step k, realigning text token positions")
    print(f"  with audio frame positions.")
    print(f"")
    print(f"  With 12Hz codec (12 frames/sec) and {num_text_tokens} text tokens,")
    print(f"  the trailing text hidden can cover up to {num_text_tokens / 12:.1f} seconds")
    print(f"  of audio before falling back to pad embeddings.")

    return {
        "talker_input_embeds": talker_input_embeds,
        "talker_attention_mask": talker_attention_mask,
        "trailing_text_hiddens": trailing_text_hiddens,
        "tts_pad_embed": tts_pad_embed,
        "num_text_tokens": num_text_tokens,
        "hidden_dim": hidden_dim,
    }


def test_trailing_text_in_streaming(model, text, language, voice_clone_prompt_dict, output_path):
    """
    Run streaming generation and trace how trailing_text_hidden is used.

    This demonstrates the realignment by showing which generation steps
    use trailing text hidden vs pad embeddings.
    """
    print("\n" + "=" * 60)
    print("Streaming Generation with Trailing Text Hidden Trace")
    print("=" * 60)

    # Rebuild talker inputs
    input_texts = [f"assistant\n{text}assistant\n"]
    input_ids = []
    for t in input_texts:
        inp = model.processor(text=t, return_tensors="pt", padding=True)
        input_id = inp["input_ids"].to(model.device)
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        input_ids.append(input_id)

    ref_text = voice_clone_prompt_dict.get("ref_text", None)
    ref_ids = None
    if ref_text:
        ref_tok = model.processor(text=f"assistant\n{ref_text}assistant\n", return_tensors="pt", padding=True)
        ref_id = ref_tok["input_ids"].to(model.device)
        ref_id = ref_id.unsqueeze(0) if ref_id.dim() == 1 else ref_id
        ref_ids = [ref_id]

    (
        talker_input_embeds,
        talker_attention_mask,
        trailing_text_hiddens,
        tts_pad_embed,
    ) = model.model._build_talker_inputs(
        input_ids=input_ids,
        instruct_ids=None,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=[language],
        speakers=None,
        non_streaming_mode=False,
    )

    num_text_tokens = trailing_text_hiddens.shape[1]
    samples_per_frame = model.model.speech_tokenizer.get_decode_upsample_rate()

    # Run streaming and count frames
    chunks = []
    chunk_sr = 0
    total_frames = 0
    first_chunk_time = None
    start = time.time()

    for chunk, sr in model.model.stream_generate_pcm(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=[language],
        non_streaming_mode=False,
        emit_every_frames=8,
        decode_window_frames=80,
        overlap_samples=0,
        max_frames=10000,
    ):
        chunk_sr = sr
        chunks.append(chunk)
        total_frames += len(chunk) // samples_per_frame
        if first_chunk_time is None:
            first_chunk_time = time.time() - start

    total_time = time.time() - start
    final_audio = np.concatenate(chunks) if chunks else np.array([])
    audio_duration = len(final_audio) / chunk_sr if chunk_sr > 0 else 0

    # Estimate how many generation steps use trailing text vs pad
    steps_using_trailing = min(total_frames, num_text_tokens)
    steps_using_pad = max(0, total_frames - num_text_tokens)

    print(f"\n  Text: '{text}' ({len(text)} chars)")
    print(f"  Trailing text hidden positions: {num_text_tokens}")
    print(f"  Total codec frames generated: {total_frames}")
    print(f"  Frames covered by trailing text: {steps_using_trailing}")
    print(f"  Frames using pad embedding: {steps_using_pad}")
    print(f"  First chunk latency: {first_chunk_time:.2f}s")
    print(f"  Total generation time: {total_time:.2f}s")
    print(f"  Audio duration: {audio_duration:.2f}s")

    # Show the ratio
    if total_frames > 0:
        trailing_ratio = steps_using_trailing / total_frames
        print(f"  Trailing text coverage: {trailing_ratio:.1%} of audio frames")

    sf.write(output_path, final_audio, chunk_sr)
    print(f"  Saved to: {output_path}")

    return {
        "total_frames": total_frames,
        "steps_using_trailing": steps_using_trailing,
        "steps_using_pad": steps_using_pad,
        "num_text_tokens": num_text_tokens,
        "first_chunk_time": first_chunk_time,
        "total_time": total_time,
        "audio_duration": audio_duration,
    }


def main():
    total_start = time.time()

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

    # Reference audio setup
    ref_audio_path = "kuklina-1.wav"
    ref_text = (
        "Это брат Кэти, моей одноклассницы. А что у тебя с рукой? И почему ты голая? У него ведь куча наград по "
        "боевые искусствам. Кэти рассказывала, правда, Лео? Понимаешь кого ты побила, Лая? "
        "Только потрогай эти мышцы... Не знала, что у тебя такой классный котик. Рожденная луной. "
        "Лай всегда откопает что-нибудь этакое. Да, жаль только, что занимает почти всё её время. "
        "Не понимаю, почему эта рухлядь не может подождать, пока ты проведешь время с сестрой."
    )

    start = time.time()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)
    log_time(start, "Voice clone prompt created")

    # Test with different text lengths to show how trailing text coverage varies
    test_texts = [
        ("Короткий.", "Short text"),
        ("Это текст средней длины для теста стриминга.", "Medium text"),
        ("Всем привет! Это тестовый текст для озвучки! Стриминг звучит нормально только через несколько секунд. Мы проверяем как работает система генерации аудио в реальном времени с различными длинами входного текста.", "Long text"),
    ]

    results = []

    for text, label in test_texts:
        print(f"\n{'#' * 60}")
        print(f"# Test: {label}")
        print(f"{'#' * 60}")

        # Inspect trailing text hidden states
        inspection = inspect_trailing_text_hidden(
            model, text, "Russian", voice_clone_prompt_dict
        )

        # Run streaming generation
        output_path = f"output_trailing_{label.lower().replace(' ', '_')}.wav"
        streaming_result = test_trailing_text_in_streaming(
            model, text, "Russian", voice_clone_prompt_dict, output_path
        )

        results.append({
            "label": label,
            "text_length": len(text),
            "num_text_tokens": inspection["num_text_tokens"],
            "hidden_dim": inspection["hidden_dim"],
            **streaming_result,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Trailing Text Hidden Coverage")
    print("=" * 80)

    print(f"\n{'Test':<20} {'Text Len':>8} {'Text Toks':>9} {'Frames':>7} {'Covered':>8} {'Pad':>6} {'1st Chunk':>10} {'Total':>8}")
    print("-" * 80)

    for r in results:
        coverage_pct = (r["steps_using_trailing"] / r["total_frames"] * 100) if r["total_frames"] > 0 else 0
        print(
            f"{r['label']:<20} {r['text_length']:>8} {r['num_text_tokens']:>9} "
            f"{r['total_frames']:>7} {coverage_pct:>7.0f}% {r['steps_using_pad']:>6} "
            f"{r['first_chunk_time']:>9.2f}s {r['total_time']:>7.2f}s"
        )

    print(f"\n{'#' * 60}")
    print(f"# Key Takeaways")
    print(f"{'#' * 60}")
    print("""
1. Trailing text hidden states are computed ONCE from input text tokens
   during _build_talker_inputs() before streaming begins.

2. They have shape [batch, num_text_tokens, hidden_dim] where each position
   corresponds to a text token's hidden state.

3. During streaming, at generation step k:
   - If k < num_text_tokens: add trailing_text_hidden[:, k] to codec embedding
   - If k >= num_text_tokens: add tts_pad_embed (zero/fallback)

4. This provides text context to the codec predictor for the first N frames
   (where N = num_text_tokens). After that, the model relies on its
   autoregressive context without explicit text embeddings.

5. Shorter text = earlier transition to pad embeddings.
   Longer text = more frames covered by trailing text hidden states.

6. The "realign" in the name refers to aligning text token positions
   with codec frame positions during generation.
""")

    print(f"\n[{time.time() - total_start:.2f}s] TOTAL SCRIPT TIME")


if __name__ == "__main__":
    main()

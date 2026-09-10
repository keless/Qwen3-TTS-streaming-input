# Qwen3-TTS Streaming

Real-time streaming audio generation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Features

From [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming):
- `stream_generate_voice_clone()` - streaming with voice cloning
- `stream_generate_pcm()` - real-time PCM audio streaming
- `torch.compile` + CUDA graphs optimization
- Crossfade overlap for seamless chunk transitions

From [kunzite-app/Qwen3-TTS-streaming](https://github.com/kunzite-app/Qwen3-TTS-streaming)
- **Two-phase streaming** - faster first-chunk latency

Added in this fork:
- **Streaming text input** — text can be fed to the talker incrementally, as it arrives from an upstream source (e.g. an LLM generating a response token-by-token), instead of requiring the full sentence up front. This means time to audio can be minimized down to the limit of how fast your first text tokens can be generated, and multiple text blocks can be streamed without losing prosody context between them.

## Streaming Text Input

Beyond streaming audio *out*, this fork streams text *in*: text can be fed to the talker incrementally instead of requiring the full sentence up front. Audio for earlier words starts playing while later words are still arriving, using the same causal KV-cache the whole way through — no re-prefill, no discontinuity when new text lands.

### How it works

- **`LiveTextSource`** (`qwen_tts/inference/live_text_source.py`) — a thread-safe text ingestion buffer. A producer calls `push(fragment)` with text of any granularity (LLM token deltas, whole words, whole sentences) and `close()` when there's no more; the generation loop calls `poll()` once per decode step to drain newly whitespace-boundary-safe-committed words, or `wait_for_more()` to block until there's something new.
- **`stream_generate_pcm_live_text`** / **`stream_generate_custom_voice_live_text`** — the streaming-audio generation loop, adapted to pull from a `LiveTextSource` instead of a fully-known input (single-sample, CustomVoice path).
- **`tts_server.py`** — a small FastAPI server exposing this as a live HTTP session: `POST /speak_live` starts a session, `POST /text` / `POST /chunks` feed text incrementally, `POST /end_stream` signals completion and waits for generation + speaker playback to finish.

### Note: Pause, don't pad

The model was only ever trained to pad-condition (`tts_pad_embed`) at the legitimate end of an utterance, waiting for its own end-of-speech token. Naively reusing padding for a different case — filling in whenever the generation loop catches up to the text source before more text has arrived — pushes the model out-of-distribution: audio drifts into slow, dragged-out speech, and it can spontaneously emit an end token mid-sentence.

The solution is to **pause generation instead of padding through the gap**: when the loop is caught up on committed text and the source isn't finished yet, it simply doesn't call `forward()` for that tick. The talker has no notion of wall-clock time — position only depends on how many `forward()` calls have happened — so pausing and resuming later is fully transparent to it. It will keep the same KV-cache and generation step, with no discontinuity.

### Usage

```python
import threading
from qwen_tts.inference.live_text_source import LiveTextSource

source = LiveTextSource()

# Producer: push text as it arrives (e.g. from a streaming LLM response)
def producer():
    for word in "Hello there, this is streaming text input.".split():
        source.push(word + " ")
    source.close()

threading.Thread(target=producer, daemon=True).start()

for chunk, sr in model.stream_generate_custom_voice_live_text(
    text_source=source,
    speaker="Vivian",
    language="English",
):
    play_audio(chunk, sr)
```

See `examples/live_text_to_speakers.py` for a full runnable example (decoupled generation/playback threads, pre-buffering, and speaker output), or run `tts_server.py` for the HTTP session version.

### Known limitation: non-CUDA hardware

This fork's streaming *audio* optimizations (`torch.compile`, CUDA graphs, flash-attention) are CUDA-only. On CPU/MPS (e.g. Apple Silicon), generation sits close to real-time rather than comfortably above it, so playback can still show occasional gaps — pre-buffering and `enable_fast_codebook_gen(True)` help (see `examples/live_text_to_speakers.py`) but don't fully close the gap on longer utterances.

## Two-Phase Streaming

Standard streaming with Qwen's TTS library waits for `emit_every_frames` (e.g., 12) before emitting the first audio. Two-phase uses aggressive settings for the first chunk to improve latency, then switches to stable settings.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 (First N frames)      │  PHASE 2 (Rest of audio)      │
│  - emit_every = 5 (fast)       │  - emit_every = 12 (stable)   │
│  - decode_window = 48          │  - decode_window = 80         │
│  - optimized = OFF             │  - optimized = ON             │
│  → FAST first chunk            │  → QUALITY for rest           │
└─────────────────────────────────────────────────────────────────┘
```

## Two-Phase Usage

```python
for chunk, sr in model.stream_generate_voice_clone(
    text="Hello!",
    language="en",
    voice_clone_prompt=prompt,
    # Phase 2 settings
    emit_every_frames=12,
    decode_window_frames=80,
    # Phase 1 settings (two-phase)
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    play_audio(chunk, sr)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 8 | Emit audio every N frames |
| `decode_window_frames` | 80 | Decoder context window |
| `overlap_samples` | 512 | Crossfade overlap between chunks |
| `first_chunk_emit_every` | 0 | Phase 1 emit interval (0 = disabled) |
| `first_chunk_decode_window` | 48 | Phase 1 decode window |
| `first_chunk_frames` | 48 | Switch to phase 2 after N frames |

## Installation

```bash
sudo apt install sox
pip install torch torchaudio flash-attn
pip install -e .
```

---

Based on:
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)

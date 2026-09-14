import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.live_text_source import LiveTextSource


MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

SPEAKER = "Sohee"
LANGUAGE = "English"
INSTRUCT = "Natural conversational speech."

# How long to wait for the *first* text after /speak_live before giving up on
# the session entirely (e.g. an LLM producer that takes a while to produce
# its first token). Deliberately much larger than LiveTextSource's own
# default idle_timeout_s=10.0 (which governs mid-utterance "producer seems
# done" pause detection, a different concern) -- see the priming-loop fix in
# stream_generate_pcm_live_text that makes this budget actually usable rather
# than being cut short by that shorter idle guess.
PRIMING_TIMEOUT_S = 60.0


print("Loading Qwen3-TTS...")

# Resolve to the local cache snapshot when available so `from_pretrained` never
# needs a network round-trip (transformers' mistral-regex patch check hits the
# HF API for non-local paths, which fails hard behind a blocking proxy).
try:
    model_path = snapshot_download(MODEL_NAME, local_files_only=True)
except Exception:
    model_path = MODEL_NAME

if torch.backends.mps.is_available():
    device_map = "mps"
elif torch.cuda.is_available():
    device_map = "auto"
else:
    device_map = "cpu"

model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map=device_map,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

# Make sure this is on if you're in MacOS, or you'll get slow/choppy output; 
# wouldnt hurt on CUDA either.
model.model.talker.enable_fast_codebook_gen(True)

print("Qwen3-TTS ready.")

app = FastAPI()

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    text: str


class SpeechLiveRequest(BaseModel):
    chunks: Optional[list[str]] = None
    text: Optional[str] = None


class TextRequest(BaseModel):
    text: str


class ChunksRequest(BaseModel):
    chunks: list[str]


# ---------------------------------------------------------------------------
# Session management (single active session at a time)
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    source: LiveTextSource
    audio_queue: queue.Queue
    playback_done: threading.Event
    generation_done: threading.Event
    thread: Optional[threading.Thread] = None
    sr: Optional[int] = None
    total_samples: int = 0
    total_text_length: int = 0
    real_count: int = 0
    pad_count: int = 0
    error: Optional[BaseException] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)


# Global session state.
_active_session: Optional[SessionState] = None
_session_lock = asyncio.Lock()  # protects _active_session creation


# ---------------------------------------------------------------------------
# Playback thread (runs per-session)
# ---------------------------------------------------------------------------

# Generation on hardware that sits close to real-time and starting playback on 
# the very first chunk leaves zero margin -- any momentary generation slowdown 
# immediately underruns the output device, heard as a click/gap. Buffering a 
# short cushion first absorbs that jitter instead.
PREBUFFER_SECONDS = 0.4

# Opening a new OutputStream on the default device shortly after a previous
# one closed can transiently fail on macOS CoreAudio (PaMacCore/AUHAL) with
# "Invalid Property Value" ([-9986]) while the OS finishes tearing down the
# old one -- observed in practice between back-to-back live-text sessions. 
# So retry after a short delay or it will fail to open.
STREAM_OPEN_RETRIES = 3
STREAM_OPEN_RETRY_DELAY_S = 0.2


def _open_output_stream(samplerate: int) -> sd.OutputStream:
    for attempt in range(1, STREAM_OPEN_RETRIES + 1):
        try:
            stream = sd.OutputStream(
                samplerate=samplerate, channels=1, dtype="float32", blocksize=0,
            )
            stream.start()
            return stream
        except sd.PortAudioError as exc:
            if attempt == STREAM_OPEN_RETRIES:
                raise
            print(
                f"[TTS/live] OutputStream open failed (attempt {attempt}/"
                f"{STREAM_OPEN_RETRIES}): {exc!r} -- retrying"
            )
            time.sleep(STREAM_OPEN_RETRY_DELAY_S)


def _playback_worker(audio_queue: queue.Queue, done_event: threading.Event):
    """Drain audio chunks from the queue and play them via sounddevice."""
    stream = None
    prebuffered: list = []
    prebuffered_seconds = 0.0
    last_sr = None
    try:
        while True:
            item = audio_queue.get()
            if item is None:
                break
            chunk, chunk_sr = item
            last_sr = chunk_sr
            if stream is None:
                prebuffered.append(chunk)
                prebuffered_seconds += len(chunk) / chunk_sr
                if prebuffered_seconds < PREBUFFER_SECONDS:
                    continue
                stream = _open_output_stream(chunk_sr)
                stream.write(np.concatenate(prebuffered))
            else:
                stream.write(chunk)
        if stream is None and prebuffered:
            # Whole utterance was shorter than PREBUFFER_SECONDS -- play what we have.
            stream = _open_output_stream(last_sr)
            stream.write(np.concatenate(prebuffered))
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        done_event.set()


# ---------------------------------------------------------------------------
# Generation thread (runs per-session, consumes from LiveTextSource)
# ---------------------------------------------------------------------------

def _generation_worker(
    source: LiveTextSource,
    audio_queue: queue.Queue,
    gen_done: threading.Event,
    state: SessionState,
):
    """Generate audio from the live text source and push chunks to the queue."""
    try:
        def on_frame(step_idx: int, used_pad: bool):
            with state._lock:
                if used_pad:
                    state.pad_count += 1
                else:
                    state.real_count += 1

        for i, (chunk, chunk_sr) in enumerate(
            model.stream_generate_custom_voice_live_text(
                text_source=source,
                speaker=SPEAKER,
                language=LANGUAGE,
                instruct=INSTRUCT,
                on_frame=on_frame,
                priming_timeout_s=PRIMING_TIMEOUT_S,
                use_optimized_decode=True,
                first_chunk_emit_every=8,
                first_chunk_decode_window=48,
                first_chunk_frames=48,
            )
        ):
            chunk = chunk.astype(np.float32, copy=False)
            with state._lock:
                state.total_samples += len(chunk)
                state.sr = chunk_sr
            print(
                f"[TTS/live] chunk {i}: "
                f"samples={len(chunk)}, "
                f"queued={audio_queue.qsize()}, "
                f"real={state.real_count} pad={state.pad_count}"
            )
            audio_queue.put((chunk, chunk_sr))

    except Exception as exc:
        with state._lock:
            state.error = exc
        print(f"[TTS/live] generation error: {exc!r}")
    finally:
        gen_done.set()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Original full-text endpoint (unchanged) ---

@app.post("/speak")
async def speak(request: SpeechRequest):
    async with tts_lock:
        print(f"[TTS] {request.text}")

        audio_queue = queue.Queue()
        playback_done = threading.Event()
        sr = None
        total_samples = 0

        def playback_worker():
            nonlocal sr
            stream = None
            try:
                while True:
                    item = audio_queue.get()
                    if item is None:
                        return
                    chunk, chunk_sr = item
                    sr = chunk_sr
                    print("[TTS] Starting audio playback")
                    stream = _open_output_stream(sr)
                    stream.write(chunk)
                    break
                while True:
                    item = audio_queue.get()
                    if item is None:
                        break
                    chunk, chunk_sr = item
                    stream.write(chunk)
            finally:
                if stream is not None:
                    stream.stop()
                    stream.close()
                playback_done.set()

        playback_thread = threading.Thread(target=playback_worker, daemon=True)
        playback_thread.start()

        try:
            last_time = time.monotonic()
            for i, (chunk, chunk_sr) in enumerate(
                model.stream_generate_custom_voice(
                    text=request.text,
                    language=LANGUAGE,
                    speaker=SPEAKER,
                    instruct=INSTRUCT,
                    emit_every_frames=1,
                    decode_window_frames=80,
                    overlap_samples=0,
                    first_chunk_emit_every=1,
                    first_chunk_decode_window=48,
                    first_chunk_frames=48,
                )
            ):
                now = time.monotonic()
                elapsed = now - last_time
                last_time = now
                chunk = chunk.astype(np.float32, copy=False)
                audio_duration = len(chunk) / chunk_sr
                total_samples += len(chunk)
                sr = chunk_sr
                print(
                    f"[TTS] chunk {i}: "
                    f"generated in {elapsed:.3f}s, "
                    f"audio={audio_duration:.3f}s, "
                    f"samples={len(chunk)}, "
                    f"queued={audio_queue.qsize()}"
                )
                audio_queue.put((chunk, chunk_sr))
        finally:
            audio_queue.put(None)
            playback_done.wait()

        print(
            f"[TTS] Generated {total_samples} samples "
            f"({total_samples / sr:.2f}s)"
        )

    return {"status": "ok", "sample_rate": sr}


# --- Live-text streaming endpoints ---

@app.post("/speak_live")
async def speak_live_start():
    """
    Start a new live-text session.

    Creates a LiveTextSource, starts the generation thread, and returns
    immediately. Text is then fed incrementally via /text or /chunks,
    and the session is ended via /end_stream.

    If a session is already active, returns 409 Conflict.
    """
    global _active_session

    async with _session_lock:
        if _active_session is not None:
            raise HTTPException(
                status_code=409,
                detail="A live-text session is already active. Call /end_stream first.",
            )
        state = SessionState(
            source=LiveTextSource(),
            audio_queue=queue.Queue(),
            playback_done=threading.Event(),
            generation_done=threading.Event(),
        )
        _active_session = state

    # Start generation thread (it blocks on source.poll() until text arrives).
    gen_thread = threading.Thread(
        target=_generation_worker,
        args=(state.source, state.audio_queue, state.generation_done, state),
        daemon=True,
    )
    gen_thread.start()
    state.thread = gen_thread

    # Start playback thread (it blocks on audio_queue until chunks arrive).
    play_thread = threading.Thread(
        target=_playback_worker,
        args=(state.audio_queue, state.playback_done),
        daemon=True,
    )
    play_thread.start()

    print("[TTS/live] Session started. Send text via /text or /chunks, then /end_stream.")
    return {"status": "started"}


@app.post("/text")
async def append_text(request: TextRequest):
    """
    Append a text string to the active session's LiveTextSource.

    Can be called repeatedly before /end_stream. Each call pushes the text
    immediately so the generation loop can consume it as it arrives.
    """
    global _active_session

    if _active_session is None:
        raise HTTPException(status_code=400, detail="No active session. Call /speak_live first.")

    with _active_session._lock:
        if _active_session.source.is_closed():
            raise HTTPException(
                status_code=409,
                detail="Session already ended. Call /speak_live for a new session.",
            )
        if _active_session.generation_done.is_set():
            raise HTTPException(
                status_code=409,
                detail=(
                    "Generation already ended for this session "
                    f"(error={_active_session.error!r}). Call /end_stream and "
                    "start a new session with /speak_live."
                ),
            )
        _active_session.source.push(request.text)
        _active_session.total_text_length += len(request.text)
        total_length = _active_session.total_text_length

    print(f"[TTS/live] +text: {request.text!r}")
    return {"status": "ok", "total_length": total_length}


@app.post("/chunks")
async def append_chunks(request: ChunksRequest):
    """
    Append multiple text chunks to the active session's LiveTextSource.

    Can be called repeatedly before /end_stream.
    """
    global _active_session

    if _active_session is None:
        raise HTTPException(status_code=400, detail="No active session. Call /speak_live first.")

    with _active_session._lock:
        if _active_session.source.is_closed():
            raise HTTPException(
                status_code=409,
                detail="Session already ended. Call /speak_live for a new session.",
            )
        if _active_session.generation_done.is_set():
            raise HTTPException(
                status_code=409,
                detail=(
                    "Generation already ended for this session "
                    f"(error={_active_session.error!r}). Call /end_stream and "
                    "start a new session with /speak_live."
                ),
            )
        for chunk in request.chunks:
            _active_session.source.push(chunk)
            _active_session.total_text_length += len(chunk)
        total_length = _active_session.total_text_length

    print(f"[TTS/live] +chunks: {len(request.chunks)} chunk(s)")
    return {"status": "ok", "total_length": total_length}


@app.post("/end_stream")
async def end_stream():
    """
    Signal end of text input and wait for generation + playback to finish.

    Marks the LiveTextSource as done, waits for the generation thread to
    complete (including any text already pushed), then waits for playback
    to finish.

    Returns session stats (sample rate, frame counts, total duration).
    """
    global _active_session

    async with _session_lock:
        if _active_session is None:
            raise HTTPException(status_code=400, detail="No active session. Call /speak_live first.")
        state = _active_session
        # Not cleared here -- a new /speak_live must keep 409ing until this 
        # session's playback thread has actually released the audio device below.

    print("[TTS/live] End of text signaled.")
    state.source.close()

    # Wait for generation to finish (it will drain remaining text). Offloaded
    # via to_thread -- so we dont block which would make /health, etc., calls 
    # time out while we're still ending.
    await asyncio.to_thread(state.generation_done.wait)
    print("[TTS/live] Generation complete.")

    state.audio_queue.put(None)

    # Wait for all queued audio to actually play (see to_thread note above).
    await asyncio.to_thread(state.playback_done.wait)
    print("[TTS/live] Playback complete.")

    # Only now has this session fully released the audio device -- safe to
    # let a new /speak_live claim _active_session.
    async with _session_lock:
        if _active_session is state:
            _active_session = None

    with state._lock:
        sr = state.sr
        total = state.total_samples
        real = state.real_count
        pad = state.pad_count
        err = state.error

    if err is not None:
        print(f"[TTS/live] Session ended with error: {err!r}")
        return {
            "status": "error",
            "detail": str(err),
            "real_frames": real,
            "pad_frames": pad,
        }

    duration = total / sr if sr else 0.0
    print(
        f"[TTS/live] Session done: {total} samples ({duration:.2f}s) "
        f"real={real} pad={pad}"
    )
    return {
        "status": "ok",
        "sample_rate": sr,
        "total_samples": total,
        "duration_s": round(duration, 2),
        "real_frames": real,
        "pad_frames": pad,
    }


@app.get("/status")
async def session_status():
    """Return the current session state (non-blocking)."""
    async with _session_lock:
        if _active_session is None:
            return {"status": "idle"}
        s = _active_session
    return {
        "status": "active",
        "source_done": s.source.is_closed(),
        "generation_done": s.generation_done.is_set(),
        "playback_done": s.playback_done.is_set(),
        "total_length": s.total_text_length,
        "total_samples": s.total_samples,
        "real_frames": s.real_count,
        "pad_frames": s.pad_count,
        "sample_rate": s.sr,
    }


# ---------------------------------------------------------------------------
# Legacy single-session lock (for /speak endpoint only)
# ---------------------------------------------------------------------------
tts_lock = asyncio.Lock()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
    )

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
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.live_text_source import LiveTextSource


MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

SPEAKER = "Sohee"
LANGUAGE = "English"
INSTRUCT = "Natural conversational speech."


print("Loading Qwen3-TTS...")

model = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

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

def _playback_worker(audio_queue: queue.Queue, done_event: threading.Event):
    """Drain audio chunks from the queue and play them via sounddevice."""
    stream = None
    try:
        while True:
            item = audio_queue.get()
            if item is None:
                return
            chunk, chunk_sr = item
            if stream is None:
                stream = sd.OutputStream(
                    samplerate=chunk_sr,
                    channels=1,
                    dtype="float32",
                    blocksize=0,
                )
                stream.start()
            stream.write(chunk)
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
                    stream = sd.OutputStream(
                        samplerate=sr, channels=1, dtype="float32", blocksize=0,
                    )
                    stream.start()
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
        if _active_session.source.is_done():
            raise HTTPException(
                status_code=409,
                detail="Session already ended. Call /speak_live for a new session.",
            )
        _active_session.source.append(request.text)

    print(f"[TTS/live] +text: {request.text!r}")
    return {"status": "ok", "total_length": _active_session.source.get_total_length()}


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
        if _active_session.source.is_done():
            raise HTTPException(
                status_code=409,
                detail="Session already ended. Call /speak_live for a new session.",
            )
        for chunk in request.chunks:
            _active_session.source.append(chunk)

    print(f"[TTS/live] +chunks: {len(request.chunks)} chunk(s)")
    return {"status": "ok", "total_length": _active_session.source.get_total_length()}


@app.post("/end_stream")
async def end_stream():
    """
    Signal end of text input and wait for generation + playback to finish.

    Marks the LiveTextSource as done, waits for the generation thread to
    complete (including any text already pushed), drains remaining audio
    chunks to the playback queue, then waits for playback to finish.

    Returns session stats (sample rate, frame counts, total duration).
    """
    global _active_session

    async with _session_lock:
        if _active_session is None:
            raise HTTPException(status_code=400, detail="No active session. Call /speak_live first.")
        state = _active_session
        _active_session = None

    # Signal that no more text is coming.
    print("[TTS/live] End of text signaled.")
    state.source.done()

    # Wait for generation to finish (it will drain remaining text).
    state.generation_done.wait()
    print("[TTS/live] Generation complete.")

    # Drain any remaining audio chunks into the playback queue so they play out.
    while True:
        try:
            item = state.audio_queue.get_nowait()
            if item is not None:
                state.audio_queue.put(item)  # re-queue for playback thread
            else:
                break
        except queue.Empty:
            break

    # Send the playback sentinel.
    state.audio_queue.put(None)

    # Wait for all queued audio to actually play.
    state.playback_done.wait()
    print("[TTS/live] Playback complete.")

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
        "source_done": s.source.is_done(),
        "generation_done": s.generation_done.is_set(),
        "playback_done": s.playback_done.is_set(),
        "total_length": s.source.get_total_length(),
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

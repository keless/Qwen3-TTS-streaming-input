# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Live text source for incremental text input during streaming TTS generation.

This module provides a thread-safe text buffer that allows text to be appended
incrementally while a consumer reads chunks as they become available. It is
designed for real-time scenarios where text arrives from a streaming source
(e.g., an LLM response, WebSocket, or user input) and needs to be fed to the
TTS generator without waiting for the full text to be available.

Example usage::

    from qwen_tts.inference.live_text_source import LiveTextSource

    source = LiveTextSource()

    # Producer thread: appends text as it arrives
    def producer():
        source.append("Hello, ")
        source.append("world!")
        source.done()

    # Consumer: reads chunks as they become available
    while not source.is_done():
        chunk = source.get_next_chunk(timeout=1.0)
        if chunk is None:
            break
        print(f"Got chunk: {chunk}")
"""

import threading
from collections import deque
from typing import Optional


class LiveTextSource:
    """Thread-safe text buffer for incremental text input during streaming.

    This class provides a producer-consumer interface for text that arrives
    incrementally. Producers append text via ``append()``, and consumers read
    chunks via ``get_next_chunk()`` which blocks until text is available
    or the source is marked as done.

    The internal buffer uses a deque for O(1) appends and pops. A condition
    variable coordinates blocking between producers and consumers.

    Attributes:
        total_length: Total number of characters appended so far (read-only).
    """

    def __init__(self) -> None:
        self._buffer: deque[str] = deque()
        self._done = False
        self._total_length: int = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def append(self, text: str) -> None:
        """Append a text chunk to the buffer.

        This method is thread-safe and can be called from any thread.
        If the source is already marked as done, the text is silently
        discarded.

        Args:
            text: The text chunk to append. Empty strings are ignored.
        """
        if not text:
            return

        with self._not_empty:
            self._buffer.append(text)
            self._total_length += len(text)
            self._not_empty.notify_all()

    def done(self) -> None:
        """Mark the source as done.

        After calling this method, ``get_next_chunk()`` will return ``None``
        once the buffer is empty. Further calls to ``append()`` are
        silently discarded.

        This wakes up any thread currently blocked in ``get_next_chunk()``.
        """
        with self._not_empty:
            self._done = True
            self._not_empty.notify_all()

    def get_next_chunk(self, timeout: Optional[float] = None) -> Optional[str]:
        """Block until a text chunk is available, then return it.

        This method blocks the calling thread until either:
        - At least one text chunk is available in the buffer (returns the
          oldest chunk).
        - The source is marked as done and the buffer is empty (returns
          ``None``).
        - The optional timeout expires (returns ``None``).

        Args:
            timeout: Maximum seconds to wait. ``None`` means wait
                indefinitely.

        Returns:
            The next text chunk as a string, or ``None`` if the source is
            done (or timed out) with nothing available.
        """
        with self._not_empty:
            while len(self._buffer) == 0:
                if self._done:
                    return None
                if not self._not_empty.wait(timeout=timeout):
                    # Timeout expired
                    if len(self._buffer) == 0:
                        return None
                    # New data arrived during wait, fall through
            chunk = self._buffer.popleft()
            return chunk

    def is_done(self) -> bool:
        """Check whether the source has been marked as done.

        Returns:
            ``True`` if ``done()`` has been called, ``False`` otherwise.
        """
        with self._lock:
            return self._done

    def peek(self) -> str:
        """Return the next available text chunk without removing it.

        If the buffer is empty, returns an empty string.

        Returns:
            The next text chunk, or ``""`` if nothing is available.
        """
        with self._lock:
            if self._buffer:
                return self._buffer[0]
            return ""

    def get_total_length(self) -> int:
        """Return the total number of characters appended so far.

        This is a monotonic counter of all text passed to ``append()``
        (excluding empty strings). It does not decrease when chunks are
        consumed.

        Returns:
            Total character count.
        """
        with self._lock:
            return self._total_length

    def poll(self) -> tuple[list[str], bool]:
        """Return all buffered chunks and done-status without blocking.

        This method is non-blocking and returns immediately. It is designed
        for the TTS generation loop which polls for new text on every
        generation step.

        Returns:
            A tuple of (chunks, finished) where:
            - chunks: list of text strings buffered since last poll (may be empty)
            - finished: True if done() was called and buffer is now empty
        """
        with self._lock:
            chunks = list(self._buffer)
            self._buffer.clear()
            finished = self._done and len(chunks) == 0
            return chunks, finished

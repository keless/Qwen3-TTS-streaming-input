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
"""
Thread-safe, granularity-agnostic text ingestion buffer for streaming
generation. Producers call push()/close() with text fragments of any size
(LLM token deltas, whole words, whole sentences); consumers call poll() once
per generation step to drain newly boundary-safe-committed text, or
wait_for_more() to block until there's something new instead of polling in
a sleep loop.

"Finished" (poll()'s second return value) has two different sources with
different permanence:
  - Explicit close() -- the producer confirmed there's no more text. Permanent.
  - Idle timeout -- we *guessed* the producer is done because nothing arrived
    for idle_timeout_s. This is only a guess and can be wrong (a slow LLM, a
    thinking pause), so it's revocable: a push() after an idle-timeout-based
    finish is accepted normally (not dropped) and un-finishes the source,
    since fresh input is direct evidence the guess was wrong. A push() after
    an explicit close() is always dropped -- close() is a confirmed fact, not
    a guess, and doesn't get revoked.

This falls out for free from computing "is finished" fresh on every call
rather than caching it: `_closed` is the only permanent bit; the idle check
is always relative to `_last_arrival`, which push() naturally refreshes.

See docs/superpowers/specs/2026-08-31-streaming-text-input-design.md.
"""
import threading
import time
from typing import List, Optional, Tuple


class LiveTextSource:
    def __init__(self, idle_timeout_s: float = 10.0):
        self._idle_timeout_s = idle_timeout_s
        self._cv = threading.Condition()
        self._buffer = ""
        self._closed = False
        self._last_arrival = time.monotonic()

    def push(self, fragment: str) -> None:
        with self._cv:
            if self._closed:
                return
            self._buffer += fragment
            self._last_arrival = time.monotonic()
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def is_closed(self) -> bool:
        """Whether close() has been called (explicit end-of-input, not an
        idle-timeout guess). Useful for callers that need a synchronous
        "has the producer confirmed it's done" check outside the
        poll()/wait_for_more() generation-loop protocol -- e.g. an HTTP
        server rejecting further writes to an already-ended session.
        """
        with self._cv:
            return self._closed

    def poll(self) -> Tuple[List[str], bool]:
        with self._cv:
            return self._poll_locked()

    def wait_for_more(self, timeout: Optional[float] = None) -> None:
        """Block until poll() would return something new (a boundary-safe
        commit, or finished), or `timeout` seconds elapse, whichever comes
        first. Returns immediately if something is already available.
        Does not itself consume/return anything -- call poll() after this
        returns.

        The idle-timeout can't be woken by notify() (nothing happens, so
        nothing calls it) -- the one bounded wait below is sized to the
        exact remaining idle budget, not an arbitrary poll interval, so
        this still wakes up immediately on push()/close() and otherwise
        wakes up at precisely the moment the idle-timeout would fire.
        """
        with self._cv:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._has_pending_output_locked():
                wait_s = self._idle_timeout_s - (time.monotonic() - self._last_arrival)
                wait_s = max(0.0, wait_s)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    wait_s = min(wait_s, remaining)
                self._cv.wait(timeout=wait_s)

    def _is_finished_locked(self) -> bool:
        if self._closed:
            return True
        return (time.monotonic() - self._last_arrival) >= self._idle_timeout_s

    def _has_pending_output_locked(self) -> bool:
        if self._is_finished_locked():
            return True
        return self._last_whitespace_end(self._buffer) != -1

    def _poll_locked(self) -> Tuple[List[str], bool]:
        if self._is_finished_locked():
            committed = self._buffer.split()
            self._buffer = ""
            return committed, True

        boundary = self._last_whitespace_end(self._buffer)
        if boundary == -1:
            return [], False

        safe_part, remaining = self._buffer[:boundary], self._buffer[boundary:]
        self._buffer = remaining
        return safe_part.split(), False

    @staticmethod
    def _last_whitespace_end(text: str) -> int:
        for i in range(len(text) - 1, -1, -1):
            if text[i].isspace():
                return i + 1
        return -1

"""Rolling perception history, agent rendezvous, and the audit trail.

Three collaborating stores, each bounded in memory and append-only on disk:

    SensorDB          the rolling ten-minute perception window
    AgentDecisionLog  the frame-id rendezvous between the agent and the arbiter
    FinalDecisionLog  the audit trail of decisions shown to the driver

No database server is involved. History is an in-memory deque evicted by age,
and the disk format is JSON Lines, so a crash truncates at most the final
record and never corrupts what came before. All three flush on every write: a
safety system that fails must not take the evidence of why it failed with it.

Every method that touches shared state is a coroutine, so the single event loop
serialises access and no thread locking is required.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

# Retention of the rolling perception window, in milliseconds. Ten minutes is
# the trajectory horizon the reasoning agent interprets; older frames cannot
# influence a decision, so they are evicted rather than stored.
WINDOW_MS = 600_000


class SensorDB:
    """The rolling perception window, mirrored to a JSONL log.

    The sensor task inserts one record per frame and the context task snapshots
    the window for the reasoning agent. Retention is capped by wall-clock age
    rather than record count, so the window spans the same ten minutes whether
    the camera runs at 15 or 30 FPS. The log file itself is never truncated and
    keeps the full session for offline analysis.
    """

    def __init__(self, path: str = "sensor_log.jsonl") -> None:
        """Open the sensor log for appending and prepare the rolling buffer.

        Parameters
        ----------
        path : str
            JSONL file to append frame records to, opened in append mode so a
            new session extends the log instead of destroying it.

        Returns
        -------
        None
            The constructor only opens the log and prepares internal state.
        """
        self._records: deque[Dict[str, Any]] = deque()
        self._lock = asyncio.Lock()
        self._fh = Path(path).open("a", encoding="utf-8")

    async def insert(self, record: Dict[str, Any]) -> None:
        """Append one frame record and evict everything older than the window.

        The record is copied before storage so a later mutation by the caller
        cannot rewrite history the agent has already been shown. A record
        missing its identifying fields is a programming error rather than a
        runtime condition, so it raises instead of being silently dropped.

        Parameters
        ----------
        record : dict
            One frame's metrics; must carry frame_id, which the arbiter uses to
            correlate a verdict with the frame that produced it, and
            timestamp_ms, which eviction sorts on.

        Returns
        -------
        None
            The record is stored and written to disk; nothing is returned.

        Raises
        ------
        ValueError
            If frame_id or timestamp_ms is absent.
        """
        item = dict(record)
        if "frame_id" not in item or "timestamp_ms" not in item:
            raise ValueError("frame_id and timestamp_ms required")

        async with self._lock:
            self._records.append(item)
            self._fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._fh.flush()

            # The deque is ordered by arrival, so the oldest record is always at
            # the front and one pass is enough to evict everything expired.
            cutoff = int(time.time() * 1000) - WINDOW_MS
            while self._records and int(self._records[0]["timestamp_ms"]) < cutoff:
                self._records.popleft()

    async def window(self) -> List[Dict[str, Any]]:
        """Return a snapshot copy of the records currently in the window.

        Each record is copied so the caller can hold and aggregate the snapshot
        while the sensor task keeps writing. That copy is what makes the context
        task's aggregation safe to perform outside the lock.

        Returns
        -------
        list of dict
            The retained frame records, oldest first; empty before the first
            insert.
        """
        async with self._lock:
            return [dict(x) for x in self._records]

    def close(self) -> None:
        """Close the sensor log during shutdown.

        Deliberately synchronous, because it runs after the event loop has
        stopped. Any failure here is suppressed so it cannot mask whatever is
        already unwinding the process; every record was flushed as it was
        written, so a failed close loses nothing.

        Returns
        -------
        None
            The function performs an action without returning a value.
        """
        try:
            self._fh.close()
        except Exception:
            pass


class AgentDecisionLog:
    """Frame-id rendezvous between the agent task and the arbiter.

    The agent task records a verdict when inference finishes; the arbiter waits
    for the verdict belonging to the specific frame it dispatched. The two are
    decoupled, since inference takes seconds while frames arrive every 33 ms, so
    waiting on "the next verdict" would not be correct. A condition variable
    rather than a plain lock is what allows a waiter to sleep until a particular
    key appears. Every verdict is also appended to a JSONL log, giving a
    replayable record of what the model said about which frame and when.
    """

    def __init__(self, path: str = "agent_rendezvous_log.jsonl") -> None:
        """Open the rendezvous log and prepare the wait and notify machinery.

        Parameters
        ----------
        path : str
            JSONL file to append verdict records to.

        Returns
        -------
        None
            The constructor only opens the log and prepares internal state.
        """
        self._entries: Dict[int, Dict[str, Any]] = {}
        self._condition = asyncio.Condition()
        self._fh = Path(path).open("a", encoding="utf-8")

    async def add(self, frame_id: int, status: str, decision: Dict[str, Any]) -> None:
        """Record a verdict for one frame and wake everything waiting on it.

        All waiters are woken rather than one, because they are keyed by frame
        id: waking a single waiter could rouse a coroutine waiting on a
        different frame while the intended one sleeps on. At this rate the cost
        is negligible and no wakeup can be lost.

        Parameters
        ----------
        frame_id : int
            The frame this verdict belongs to.
        status : str
            Outcome tag for the audit trail, such as ok, timeout or invalid.
        decision : dict
            The agent's decision, copied before storage.

        Returns
        -------
        None
            The verdict is stored, logged and announced; nothing is returned.
        """
        entry = {
            "frame_id": int(frame_id),
            "status": str(status),
            "decision": dict(decision),
            "decided_at_ms": int(time.time() * 1000),
        }
        async with self._condition:
            self._entries[int(frame_id)] = entry
            self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._condition.notify_all()

    async def wait(self, frame_id: int, timeout_s: float) -> Optional[Dict[str, Any]]:
        """Await the verdict for one frame, giving up when the deadline passes.

        The deadline is computed once from the monotonic clock and each re-wait
        receives only the time remaining, so a stream of notifications about
        other frames cannot extend the wait indefinitely, and a system clock
        adjustment cannot corrupt it. Returning nothing on timeout is a safety
        property rather than a convenience: the caller falls back to the
        deterministic decision, so a slow or wedged model degrades the system to
        its deterministic layer instead of stalling the pipeline.

        Parameters
        ----------
        frame_id : int
            The frame whose verdict is awaited.
        timeout_s : float
            The longest the arbiter is willing to wait, in seconds.

        Returns
        -------
        dict or None
            A copy of the verdict record, or None if the deadline passed first.
        """
        deadline = time.monotonic() + timeout_s
        async with self._condition:
            while int(frame_id) not in self._entries:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None
            return dict(self._entries[int(frame_id)])

    def close(self) -> None:
        """Close the rendezvous log during shutdown.

        Returns
        -------
        None
            The function performs an action without returning a value.
        """
        try:
            self._fh.close()
        except Exception:
            pass


class FinalDecisionLog:
    """Append-only audit trail of the decisions actually shown to the driver.

    One row is written per change of published decision rather than per frame,
    so the file reads as a timeline of what the driver experienced instead of
    thirty near-identical rows a second. Each row carries the originating frame
    id, which lets any alert be traced back through the rendezvous log to the
    perception record that caused it.
    """

    def __init__(self, path: str = "final_decision_log.jsonl") -> None:
        """Open the audit log for appending.

        Parameters
        ----------
        path : str
            JSONL file to append published decisions to.

        Returns
        -------
        None
            The constructor only opens the log and prepares internal state.
        """
        self._fh = Path(path).open("a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def add(
        self,
        decision: Dict[str, Any],
        *,
        frame_id: Optional[int],
        event: str,
    ) -> None:
        """Append one published decision to the audit trail.

        Parameters
        ----------
        decision : dict
            The decision exactly as published, copied before writing.
        frame_id : int or None
            The frame this decision derives from, or None for decisions with no
            single originating frame, such as a temporal guard release.
        event : str
            Audit tag describing why the row was written, which is what makes
            the trail readable as a sequence of causes rather than states.

        Returns
        -------
        None
            The row is written and flushed; nothing is returned.
        """
        row = {
            "timestamp_ms": int(time.time() * 1000),
            "frame_id": frame_id,
            "event": event,
            "decision": dict(decision),
        }
        async with self._lock:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Close the audit log during shutdown.

        Returns
        -------
        None
            The function performs an action without returning a value.
        """
        try:
            self._fh.close()
        except Exception:
            pass

"""AuraDrive — concurrent Cold → Agent → Arbiter task definitions.

Responsibility: the five event-loop coroutines and their shared plumbing
(queues, reflex latch, publisher with sticky-agent and temporal guard).

T1: Camera, perception, SensorDB and direct reflex emergency.
T2: Latest-only deterministic cold decision (graded AHP, reference_engine).
T3: Periodic ten-minute history snapshot.
T4: Native Ollama agent request (always-on, single-flight).
T5: Cold baseline, exact frame-id rendezvous and cold-versus-agent arbitration.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

import decision as dec
from audio import AlertSounder
from reference_engine import get_cold_decision
from context_summary import build_context_summary
from db import AgentDecisionLog, FinalDecisionLog, SensorDB
from evaluator import TemporalGuard, evaluate

CAMERA_INDEX = int(os.getenv("AURADRIVE_CAMERA_INDEX", "0"))
SHOW_WINDOW = True
WINDOW_TITLE = "AuraDrive"
AGENT_TIMEOUT = float(os.getenv("AURADRIVE_AGENT_WAIT_TIMEOUT", "70"))
CONTEXT_REFRESH = float(os.getenv("AURADRIVE_CONTEXT_REFRESH", "0.5"))


async def latest_put(queue: asyncio.Queue, item: Any) -> None:
    """Drain-and-replace for noncritical streams; the freshest item wins."""
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


@dataclass
class ReflexLatch:
    """Publish one emergency per continuous >=2 s closure.

    Re-arming requires the eyes to be confirmed open for OPEN_CONFIRM_FRAMES
    consecutive frames, so a single-frame landmark flicker during one long
    closure cannot reset the latch and fire a duplicate EMERGENCY.
    """
    active: bool = False
    _open_frames: int = 0
    OPEN_CONFIRM_FRAMES: int = 5  # ~150 ms at 30 FPS

    def observe(self, frame: Dict[str, Any]) -> Tuple[bool, bool]:
        """Advance the latch by one frame and report whether to fire.

        The latch is what makes a microsleep a single event rather than a
        cascade: a two-second closure spans roughly sixty frames, and without it
        each one would raise its own EMERGENCY. Re-arming requires the eyes to
        be confirmed open across several consecutive frames, so a momentary
        landmark failure part-way through one long closure cannot reset the
        latch and produce a duplicate alert.

        Parameters
        ----------
        frame : Dict[str, Any]
            The metrics for this frame, read for eye state and closure duration.

        Returns
        -------
        tuple
            Whether the latch is currently held, and whether this frame is the
            one that should publish the EMERGENCY.
        """
        state = str(frame.get("Driver_State", "EYES_OPEN"))
        try:
            closed_ms = float(frame.get("Eyes Closed Duration", 0) or 0)
        except (TypeError, ValueError):
            closed_ms = 0.0

        if state != "EYES_CLOSED":
            self._open_frames += 1
            if self._open_frames >= self.OPEN_CONFIRM_FRAMES:
                self.active = False
            return self.active, False

        self._open_frames = 0
        if closed_ms >= dec.MICROSLEEP_THRESHOLD_MS:
            publish_now = not self.active
            self.active = True
            return True, publish_now
        return self.active, False


@dataclass
class SharedState:
    history: List[Dict[str, Any]] = field(default_factory=list)
    history_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    decision: Optional[Dict[str, Any]] = None
    decision_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    agent_busy: bool = False
    agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def set_history(self, items: List[Dict[str, Any]]) -> None:
        """Store the latest snapshot of the retained window.

        Parameters
        ----------
        items : List[Dict[str, Any]]
            Frame records copied from the rolling window.

        Returns
        -------
        None
            The function performs an action without returning a value.
        """
        async with self.history_lock:
            self.history = list(items)

    async def get_history(self) -> List[Dict[str, Any]]:
        """Return the current history snapshot for building agent evidence.

        Returns
        -------
        List[Dict[str, Any]]
            A copy of the snapshot, so the caller can aggregate it while the
            context task keeps refreshing.
        """
        async with self.history_lock:
            return list(self.history)

    async def set_decision(self, decision: Dict[str, Any]) -> None:
        """Record the decision currently published to the driver.

        Parameters
        ----------
        decision : Dict[str, Any]
            The decision just published, stored so other tasks can read the
            live state without reaching into the publisher.

        Returns
        -------
        None
            The function performs an action without returning a value.
        """
        async with self.decision_lock:
            self.decision = dict(decision)

    async def get_decision(self) -> Optional[Dict[str, Any]]:
        """Return the decision currently published, if there is one.

        Returns
        -------
        Dict[str, Any] or None
            A copy of the live decision, or None before the first publication.
        """
        async with self.decision_lock:
            return dict(self.decision) if self.decision else None

    async def reserve_agent(self) -> bool:
        """Try to claim the inference slot, without waiting if it is taken.

        Only one inference runs at a time, since a second concurrent call would
        compete for the same model and slow both. Returning immediately rather
        than blocking is the point: the caller skips the agent for this frame
        and continues on the deterministic baseline instead of stalling.

        Returns
        -------
        bool
            True if the slot was claimed and the caller may dispatch.
        """
        async with self.agent_lock:
            if self.agent_busy:
                return False
            self.agent_busy = True
            return True

    async def release_agent(self) -> None:
        """Release the inference slot so the next frame may dispatch.

        Returns
        -------
        None
            The function performs an action without returning a value.
        """
        async with self.agent_lock:
            self.agent_busy = False


class AgentProcess:
    """Long-lived newline-delimited JSON subprocess wrapper for agent.py."""

    def __init__(self) -> None:
        """Prepare the handle to the agent subprocess without starting it.

        The process is started lazily on first use, and its path is resolved
        relative to this module so the layout can move without reconfiguration.

        Returns
        -------
        None
            The constructor only prepares internal state.
        """
        self.proc: Optional[subprocess.Popen[str]] = None
        self.path = Path(__file__).with_name("agent.py")

    def _start(self) -> subprocess.Popen[str]:
        """Return the running agent process, starting or restarting it if needed.

        One long-lived process is what keeps the model warm across a session; a
        fresh process per frame would pay the model load every time. Checking
        liveness on each call means a crashed agent is transparently replaced on
        the next assessment rather than disabling reasoning for the session.

        Returns
        -------
        subprocess.Popen
            A live agent process ready to accept an evidence object.
        """
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", str(self.path)],
                cwd=str(self.path.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1,
            )
        return self.proc

    def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send one evidence object to the agent and read back its decision.

        This is a blocking exchange over pipes, which is why the caller runs it
        on a worker thread rather than on the event loop. Every failure is
        converted into an error result instead of an exception, and the process
        handle is dropped so the next call starts a fresh one. That is what lets
        the arbiter treat a crashed model as a timeout and fall back to the
        deterministic decision.

        Parameters
        ----------
        payload : Dict[str, Any]
            The evidence object for one frame.

        Returns
        -------
        Dict[str, Any]
            The agent's decision, or an error result naming the failure.
        """
        try:
            process = self._start()
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line.strip():
                raise RuntimeError("empty_agent_response")
            value = json.loads(line)
            return value if isinstance(value, dict) else {"status": "error", "detail": "agent_response_not_object"}
        except Exception as exc:  # noqa: BLE001
            self.proc = None
            return {"status": "error", "detail": f"agent_process:{type(exc).__name__}:{exc}"}

    def stop(self) -> None:
        """Shut the agent process down at the end of a session.

        Closing standard input asks the agent to finish its loop and exit on its
        own, which lets it flush its log; termination is the fallback if it does
        not exit promptly.

        Returns
        -------
        None
            The function performs an action without returning a value.
        """
        if self.proc and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                self.proc.terminate()


class Publisher:
    """The only point that writes UI state, temporal state and audit events."""

    def __init__(self, shared: SharedState, audit: FinalDecisionLog) -> None:
        """Build the single point through which every decision reaches the driver.

        Concentrating actuation here is what keeps the audio, the banner, the
        temporal guard and the audit trail consistent with one another: there is
        no second path by which a decision could reach the driver unlogged or
        unsmoothed.

        Parameters
        ----------
        shared : SharedState
            Shared runtime state, updated with each published decision.
        audit : FinalDecisionLog
            The audit trail that records each change of published decision.

        Returns
        -------
        None
            The constructor only prepares internal state.
        """
        self.shared = shared
        self.audit = audit
        self.sounder = AlertSounder()
        self.temporal = TemporalGuard()
        self.last_signature: tuple[str, str] | None = None
        self.last_at = 0.0
        # Sticky-agent: the most recent arbitrated agent verdict stands as the
        # decision between (slow) agent refreshes, so always-on agency is
        # actually visible. Cold/reflex may still escalate above it instantly;
        # it expires after STICKY_TTL so a stalled agent never pins a stale
        # verdict forever.
        self.sticky_agent: Dict[str, Any] | None = None
        self.sticky_at = 0.0
        self.sticky_ttl = float(os.getenv("AURADRIVE_STICKY_TTL", "40"))

    async def publish(
        self,
        candidate: Dict[str, Any],
        *,
        frame_id: int | None,
        event: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Publish one decision: smooth it, actuate it, and audit the change.

        The order matters. The temporal guard runs first so what is actuated is
        what is recorded, and the audit row is written only when the decision
        actually changes, which keeps the trail a readable timeline instead of
        thirty identical rows a second.

        Parameters
        ----------
        candidate : Dict[str, Any]
            The decision proposed for publication.
        frame_id : int or None
            The originating frame, or None where no single frame applies.
        event : str
            Audit tag naming which path produced this decision.
        force : bool
            Write an audit row even if the decision has not changed.

        Returns
        -------
        Dict[str, Any]
            The decision actually published after temporal smoothing.
        """
        final = self.temporal.apply(candidate)
        await self.shared.set_decision(final)
        self.sounder.notify(final)  # audio actuation: chime + spoken message (non-blocking)
        signature = (str(final.get("command")), str(final.get("reason")))
        now = time.monotonic()
        # Log on change (or when forced) only. Re-logging unchanged held state every
        # second inflated the audit — e.g. 6 microsleep events read as ~58 EMERGENCY
        # rows. shared.decision is still updated every frame above, so the UI is live.
        if force or signature != self.last_signature:
            await self.audit.add(final, frame_id=frame_id, event=event)
            self.last_signature = signature
            self.last_at = now
            sys.stderr.write(
                f"[T5] {event} frame={frame_id} cmd={final['command']} src={final['decision_source']}\n"
            )
        return final

    def note_agent(self, decision: Dict[str, Any]) -> None:
        """Record the latest arbitrated agent verdict as the standing decision."""
        self.sticky_agent = dict(decision)
        self.sticky_at = time.monotonic()

    def _fuse_sticky(self, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Cold/reflex may escalate above the standing agent verdict instantly;
        otherwise the agent's verdict stands until it expires (STICKY_TTL)."""
        sticky = self.sticky_agent
        if sticky is None:
            return baseline
        if time.monotonic() - self.sticky_at > self.sticky_ttl:
            self.sticky_agent = None
            return baseline
        if dec.rank(baseline.get("command")) > dec.rank(sticky.get("command")):
            return baseline
        return sticky

    async def publish_baseline(self, baseline: Dict[str, Any], *, frame_id: int | None) -> Dict[str, Any]:
        """Publish the per-frame decision: the cold baseline, unless a fresher,
        higher-or-equal agent verdict is still standing (sticky-agent)."""
        effective = self._fuse_sticky(baseline)
        is_sticky = effective is not baseline
        return await self.publish(
            effective,
            frame_id=frame_id,
            event="T5_agent_sticky" if is_sticky else "T5_cold_baseline",
        )


def cold_baseline(cold: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical deterministic fallback: Cold is the sole safety baseline."""
    return dec.make_decision(
        command=str(cold.get("command", "EMERGENCY_ALERT")),
        decision_source="cold_baseline",
        reason=str(cold.get("reason", "cold")),
        message=str(cold.get("message", "")),
        cognitive_state=str(cold.get("cognitive_state", "Unknown")),
        failure_risk_10min=str(cold.get("failure_risk_10min", "LOW")),
        evidence={"cold": cold},
    )


def _banner_text(message: str) -> str:
    """Prepare LLM text for cv2.putText (ASCII-only font): map typographic
    punctuation to ASCII equivalents (an em-dash must not become '?')."""
    for src, dst in (("—", " - "), ("–", "-"), ("’", "'"), ("‘", "'"),
                     ("“", '"'), ("”", '"'), ("…", "...")):
        message = message.replace(src, dst)
    message = " ".join(message.split())
    return message.encode("ascii", errors="replace").decode()


def _wrap_lines(text: str, first_limit: int, second_limit: int) -> tuple[str, str]:
    """Split on a word boundary into up to two banner lines; only if the text
    exceeds BOTH lines is it ellipsized (word-boundary, no dangling words)."""
    if len(text) <= first_limit:
        return text, ""
    words = text.split(" ")
    line1, rest = "", ""
    for i, word in enumerate(words):
        candidate = f"{line1} {word}".strip()
        if len(candidate) > first_limit:
            rest = " ".join(words[i:])
            break
        line1 = candidate
    if len(rest) > second_limit:
        kept = rest[:second_limit].split(" ")[:-1]
        while kept and len(kept[-1].strip(",;:-.")) <= 2:
            kept.pop()
        rest = " ".join(kept).rstrip(",;:-") + "..."
    return line1, rest


def render_no_face(frame: Any, seconds: float) -> None:
    """Top-of-frame banner for sustained face-tracking loss (FR 4.2/4.3).
    Deliberately NOT an alert color — tracking loss is a sensing condition,
    not a fatigue severity."""
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 44), (70, 60, 50), -1)
    cv2.putText(frame, f"NO FACE DETECTED ({seconds:.0f}s) - searching for driver",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def render_decision(frame: Any, decision: Dict[str, Any]) -> None:
    """Draw the graded alert banner for the current decision onto the frame.

    Rendering helper. The banner is colour-coded by severity so the alert level
    is readable at a glance, and nothing is drawn when no action is called for,
    which keeps the display quiet during normal driving.

    Parameters
    ----------
    frame : Any
        The camera image to draw on, modified in place.
    decision : Dict[str, Any]
        The published decision supplying the severity and message.

    Returns
    -------
    None
        The function draws on the frame and returns nothing.
    """
    if decision.get("command") == "NO_ACTION":
        return
    color = {"LOW": (0, 200, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}.get(
        decision.get("severity"), (200, 200, 200)
    )
    height, width = frame.shape[:2]
    y = max(0, height - 108)
    cv2.rectangle(frame, (0, y), (width, height), color, -1)
    text = _banner_text(str(decision.get("message") or decision.get("reason", "")))
    prefix = f"{decision.get('command')}: "
    # Two message lines so the full 200-char intervention fits on screen.
    line1, line2 = _wrap_lines(text, first_limit=max(20, 95 - len(prefix)), second_limit=100)
    cv2.putText(frame, prefix + line1, (15, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    if line2:
        cv2.putText(frame, line2, (15, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(frame, str(decision.get("cognitive_state", "Unknown"))[:60], (15, y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


async def task_sensor(
    sensor: Any,
    camera: Any,
    db: SensorDB,
    sensor_q: asyncio.Queue,
    shared: SharedState,
    publisher: Publisher,
) -> None:
    """T1: capture, perceive, latch the reflex, store, and render.

    The fastest task and the only one that touches the camera. It runs the
    perception layer, checks the reflex latch, writes the frame to the rolling
    window and draws the display, all at camera rate. A reflex EMERGENCY is
    published from here directly, bypassing the cold engine, the agent and the
    arbiter, because the most dangerous state must not wait on any other layer.

    Parameters
    ----------
    sensor : Any
        The perception layer that turns a frame into metrics.
    camera : Any
        The open capture device.
    db : SensorDB
        The rolling window each frame is written to.
    sensor_q : asyncio.Queue
        Outbound queue carrying the freshest frame to the cold engine.
    shared : SharedState
        Shared runtime state.
    publisher : Publisher
        Used to actuate a reflex EMERGENCY without leaving this task.

    Returns
    -------
    None
        Runs until the operator quits, which cancels the task.
    """
    frame_id = 0
    started = time.monotonic()
    reflex = ReflexLatch()
    no_face_frames = 0
    NO_FACE_CONFIRM = 15  # ~0.5 s at 30 FPS: ignores single-frame tracking flicker

    while True:
        ok, frame = camera.read()
        if not ok:
            await asyncio.sleep(0.02)
            continue

        metrics = sensor.process_frame(frame)
        frame_id += 1
        metrics.update(
            {
                "frame_id": frame_id,
                "timestamp_ms": int(time.time() * 1000),
                "time_of_day": time.strftime("%H:%M"),
                "trip_duration_min": int((time.monotonic() - started) / 60),
            }
        )

        reflex_active, reflex_published = reflex.observe(metrics)
        metrics["_reflex_active"] = reflex_active
        metrics["_reflex_published"] = reflex_published
        await db.insert(metrics)

        if reflex_published:
            closed = float(metrics.get("Eyes Closed Duration", 0) or 0)
            emergency = dec.make_decision(
                command="EMERGENCY_ALERT",
                decision_source="reflex",
                reason="reflex_microsleep",
                message=f"Emergency: eyes closed {int(closed)} ms. Pull over safely immediately.",
                cognitive_state="Microsleep",
                failure_risk_10min="HIGH",
                evidence={"frame_id": frame_id, "reflex_event": True},
            )
            await publisher.publish(emergency, frame_id=frame_id, event="T1_reflex", force=True)

        # All frames are retained in SensorDB; T2 receives only the latest
        # noncritical work item when it cannot keep up.
        await latest_put(sensor_q, metrics)

        no_face_frames = no_face_frames + 1 if metrics.get("no_face") else 0
        no_face_active = no_face_frames >= NO_FACE_CONFIRM
        publisher.sounder.notify_no_face(no_face_active)

        decision = await shared.get_decision()
        if SHOW_WINDOW:
            if decision:
                render_decision(frame, decision)
            if no_face_active:
                render_no_face(frame, no_face_frames / 30.0)
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                raise asyncio.CancelledError
        await asyncio.sleep(0)


async def task_cold(sensor_q: asyncio.Queue, cold_q: asyncio.Queue) -> None:
    """T2: turn the freshest frame into the deterministic baseline.

    Consumes only the newest frame rather than a backlog, so the baseline always
    describes the present. Frames already handled by the reflex latch are passed
    over, since re-deciding one would duplicate the emergency and pay for an
    inference the reflex path deliberately skipped.

    Parameters
    ----------
    sensor_q : asyncio.Queue
        Inbound frames from perception.
    cold_q : asyncio.Queue
        Outbound baseline decisions for the arbiter.

    Returns
    -------
    None
        Runs until cancelled.
    """
    while True:
        frame = await sensor_q.get()

        # T1 already published a single emergency for this continuous event.
        # T2 must not duplicate it or invoke the LLM for it.
        if bool(frame.get("_reflex_active")) or bool(frame.get("_reflex_published")):
            continue

        cold = get_cold_decision(frame)
        baseline = cold_baseline(cold)

        # Any non-reflex score must be capped at URGENT. This is defensive:
        # cold_decision.py follows the same invariant independently.
        if baseline["command"] == "EMERGENCY_ALERT":
            baseline = dec.make_decision(
                command="URGENT_ALERT",
                decision_source="safety_override",
                reason="non_reflex_cold_emergency_capped",
                message="Strong deterministic fatigue signals detected. Pull over at the next safe location.",
                cognitive_state=str(baseline.get("cognitive_state", "Fighting Sleep")),
                failure_risk_10min="HIGH",
                evidence={"original_cold": cold},
            )

        await latest_put(cold_q, {"frame": frame, "cold": cold, "baseline": baseline})


async def task_context(db: SensorDB, shared: SharedState) -> None:
    """T3: refresh the history snapshot the agent reasons over.

    Aggregating the window is comparatively expensive, so it happens on its own
    slow cadence rather than per frame. Doing it here means an agent dispatch
    can pick up a ready snapshot instead of building one inside the request
    path, which keeps that path short.

    Parameters
    ----------
    db : SensorDB
        The rolling window being snapshotted.
    shared : SharedState
        Where the snapshot is published for the arbiter to use.

    Returns
    -------
    None
        Runs until cancelled.
    """
    while True:
        await shared.set_history(await db.window())
        await asyncio.sleep(CONTEXT_REFRESH)


async def task_agent(
    process: AgentProcess,
    agent_q: asyncio.Queue,
    agent_log: AgentDecisionLog,
    shared: SharedState,
) -> None:
    """T4: run the model on dispatched frames, one inference at a time.

    The blocking call is moved to a worker thread so the event loop keeps
    serving perception while the model runs, which is what allows an inference
    measured in seconds to coexist with a camera loop measured in milliseconds.
    The inference slot is released in a finally block, so a failed assessment
    cannot leave the agent permanently marked busy and silently disable
    reasoning for the rest of the session.

    Parameters
    ----------
    process : AgentProcess
        Handle to the agent subprocess.
    agent_q : asyncio.Queue
        Inbound dispatch requests from the arbiter.
    agent_log : AgentDecisionLog
        The rendezvous the arbiter is waiting on.
    shared : SharedState
        Shared runtime state, used here to release the inference slot.

    Returns
    -------
    None
        Runs until cancelled.
    """
    while True:
        request = await agent_q.get()
        frame_id = int(request["frame_id"])
        try:
            result = await asyncio.to_thread(process.call, request["payload"])
            await agent_log.add(frame_id, str(result.get("status", "error")), result)
        finally:
            await shared.release_agent()


async def resolve_agent(
    frame_id: int,
    baseline: Dict[str, Any],
    agent_log: AgentDecisionLog,
    publisher: Publisher,
) -> None:
    """Await one frame's verdict, arbitrate it, and publish the outcome.

    Runs as its own task so the arbiter can keep publishing baselines while this
    one waits. If the verdict does not arrive within the timeout the
    deterministic baseline simply stands, which is the graceful degradation the
    design depends on: a slow or absent model costs contextual refinement, never
    protection.

    Parameters
    ----------
    frame_id : int
        The frame whose verdict is awaited.
    baseline : Dict[str, Any]
        The deterministic decision for that frame, and the floor for
        arbitration.
    agent_log : AgentDecisionLog
        The rendezvous carrying the verdict.
    publisher : Publisher
        Used to publish the arbitrated result.

    Returns
    -------
    None
        The function performs an action without returning a value.
    """
    entry = await agent_log.wait(frame_id, AGENT_TIMEOUT)
    if entry is None:
        # The cold baseline was already published. This event makes the
        # fallback explicit in the audit trail without altering safety.
        fallback = dict(baseline)
        fallback["decision_source"] = "cold_fallback"
        fallback["reason"] = "agent_timeout"
        await publisher.publish(fallback, frame_id=frame_id, event="T5_agent_timeout")
        return
    if entry.get("status") != "ok":
        fallback = dict(baseline)
        fallback["decision_source"] = "cold_fallback"
        fallback["reason"] = "agent_unavailable"
        fallback["evidence"] = {**dict(baseline.get("evidence") or {}), "agent_error": entry.get("decision", {}).get("detail", "unknown")}
        await publisher.publish(fallback, frame_id=frame_id, event="T5_agent_fallback")
        return

    agent_decision = dec.from_agent(entry["decision"])
    arbitration = evaluate(baseline, agent_decision)
    published = await publisher.publish(arbitration.final, frame_id=frame_id, event=f"T5_agent_{arbitration.reason}")
    # The agent's verdict becomes the standing decision until the next refresh,
    # so always-on agency is visible instead of being overwritten by the next
    # cold baseline. Timeouts/errors above deliberately do NOT update it — the
    # last good verdict rides until STICKY_TTL.
    publisher.note_agent(published)


async def task_arbiter(
    cold_q: asyncio.Queue,
    agent_q: asyncio.Queue,
    agent_log: AgentDecisionLog,
    shared: SharedState,
    publisher: Publisher,
) -> None:
    """T5: publish the baseline at once, then refine it when the agent replies.

    The ordering is the safety property. The deterministic baseline is published
    on the frame it was computed from, so protection is never delayed by
    reasoning; the agent is dispatched only if its slot is free, and its verdict
    is arbitrated against that same baseline when it arrives. Between refreshes
    the most recent agent verdict continues to stand, subject to its own expiry,
    so a slow model still contributes rather than flickering in and out.

    Parameters
    ----------
    cold_q : asyncio.Queue
        Inbound deterministic baselines.
    agent_q : asyncio.Queue
        Outbound dispatch requests for the agent.
    agent_log : AgentDecisionLog
        The rendezvous verdicts arrive on.
    shared : SharedState
        Shared runtime state, holding the history snapshot and inference slot.
    publisher : Publisher
        The single point through which decisions reach the driver.

    Returns
    -------
    None
        Runs until cancelled.
    """
    while True:
        packet = await cold_q.get()
        frame = packet["frame"]
        cold = packet["cold"]
        baseline = packet["baseline"]
        frame_id = int(frame["frame_id"])

        # Real-time protection is never delayed by local-model latency. The cold
        # baseline actuates immediately — unless a fresher agent verdict is still
        # standing (sticky-agent), in which case it holds until cold escalates
        # above it or it expires.
        await publisher.publish_baseline(baseline, frame_id=frame_id)

        # Always-on hot path: the cold and agent paths both run on every frame and
        # the evaluator arbitrates. The agent is consulted even when cold sees
        # nothing (NO_ACTION), so the LLM can catch subtle, trend-based fatigue the
        # deterministic thresholds miss. Single-flight reservation means it runs
        # back-to-back on the freshest free frame, not literally per-frame.
        if baseline["command"] == "EMERGENCY_ALERT":
            # Reflex owns EMERGENCY; consulting the LLM here would add nothing.
            # Normal reflex emergencies never reach T5 — this is the defensive branch.
            await publisher.publish(baseline, frame_id=frame_id, event="T5_cold_emergency", force=True)
            continue
        if not await shared.reserve_agent():
            continue

        history = await shared.get_history()
        summary = build_context_summary(frame, history, cold, baseline["command"])
        payload = {
            "frame_id": frame_id,
            "system_log": summary.system_log,
            "facts_text": summary.facts_text,
            "frame_narrative": summary.frame_narrative,
            # Python-only verification data; user_message() never embeds it.
            "facts": summary.facts,
            "safety_floor_command": baseline["command"],
        }
        await latest_put(agent_q, {"frame_id": frame_id, "payload": payload})
        asyncio.create_task(resolve_agent(frame_id, baseline, agent_log, publisher), name=f"resolve_agent_{frame_id}")

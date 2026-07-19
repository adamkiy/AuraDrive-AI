#!/usr/bin/env python3
"""
AuraDrive — asyncio entry point.

Responsibility: wire the five concurrent tasks defined in tasks.py into one
event loop and manage startup/shutdown. Contains no decision logic — every
behavioural rule lives in tasks.py / reference_engine.py / evaluator.py / agent.py.

  T1 task_sensor   — camera + perception + SensorDB + reflex latch + UI
  T2 task_cold     — latest-only graded-AHP cold decision (reference_engine)
  T3 task_context  — periodic 10-minute history snapshot for the LLM
  T4 task_agent    — native-Ollama agent request (always-on, single-flight)
  T5 task_arbiter  — cold baseline + frame-id rendezvous + cold-vs-agent arbitration

Run while STATIONARY only. Press 'q' in the video window to quit.
"""
from __future__ import annotations

import asyncio

import cv2

from sensor import EyeBlinkSensor
from db import SensorDB, AgentDecisionLog, FinalDecisionLog
from tasks import (
    CAMERA_INDEX,
    SharedState,
    AgentProcess,
    Publisher,
    task_sensor,
    task_cold,
    task_context,
    task_agent,
    task_arbiter,
)


WARMUP_PAYLOAD = {
    "frame_id": 0,
    "safety_floor_command": "NO_ACTION",
    "system_log": "SYSTEM LOG — startup warm-up.",
    "facts_text": "DETERMINISTIC FACTS — startup warm-up. Cold command: NO_ACTION. Safety floor: NO_ACTION.",
    "frame_narrative": "CURRENT FRAME NARRATIVE — startup warm-up. Driver state EYES_OPEN.",
    "facts": {"microsleep_event_count": 0},
}


async def _warmup_agent(process: AgentProcess, shared: SharedState) -> None:
    """Load the local model before the first real dispatch.

    A cold model load costs far more than a warm inference, and without this the
    penalty would land on the first frame that actually needs an assessment. The
    warm-up holds the inference lock so it cannot race a real call, and it runs
    as its own task so perception starts immediately alongside it. Failures are
    ignored: warming is an optimisation, and a system that cannot reach the
    model must still start and run on its deterministic layers.

    Parameters
    ----------
    process : AgentProcess
        Handle to the agent subprocess that will serve real assessments.
    shared : SharedState
        Shared runtime state, used here for the inference lock.

    Returns
    -------
    None
        The function performs an action without returning a value.
    """
    if await shared.reserve_agent():
        try:
            await asyncio.to_thread(process.call, WARMUP_PAYLOAD)
        except Exception:  # noqa: BLE001
            pass
        finally:
            await shared.release_agent()


async def run() -> None:
    """Build every component, start the five tasks, and own the shutdown.

    The tasks are peers rather than a chain: each owns one stage and they are
    joined only by the capacity-one queues created here, which is what lets a
    slow inference proceed without holding up perception. The teardown block
    matters as much as the startup, because a monitoring session ends by
    releasing the camera, stopping audio mid-phrase and closing every log
    cleanly, whether it ended by request or by failure.

    Returns
    -------
    None
        Runs until the operator quits or a task cancels; returns nothing.

    Raises
    ------
    RuntimeError
        If the camera cannot be opened, since there is no useful degraded mode
        for a driver monitor with no view of the driver.
    """
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    sensor    = EyeBlinkSensor(debug=True)
    db        = SensorDB()
    agent_log = AgentDecisionLog()
    audit     = FinalDecisionLog()
    shared    = SharedState()
    process   = AgentProcess()
    publisher = Publisher(shared, audit)

    # Latest-only streams (drain-and-replace via tasks.latest_put); the freshest
    # item always wins, so maxsize=1 is intentional.
    sensor_q: asyncio.Queue = asyncio.Queue(maxsize=1)
    cold_q:   asyncio.Queue = asyncio.Queue(maxsize=1)
    agent_q:  asyncio.Queue = asyncio.Queue(maxsize=1)

    tasks = [
        asyncio.create_task(
            task_sensor(sensor, camera, db, sensor_q, shared, publisher), name="T1_sensor"),
        asyncio.create_task(task_cold(sensor_q, cold_q), name="T2_cold"),
        asyncio.create_task(task_context(db, shared), name="T3_context"),
        asyncio.create_task(task_agent(process, agent_q, agent_log, shared), name="T4_agent"),
        asyncio.create_task(
            task_arbiter(cold_q, agent_q, agent_log, shared, publisher), name="T5_arbiter"),
        asyncio.create_task(_warmup_agent(process, shared), name="warmup"),
    ]

    print("[MAIN] AuraDrive started. Press 'q' in the window to quit.")
    try:
        # task_sensor raises CancelledError on 'q'; that propagates here.
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("[MAIN] shutdown requested.")
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        process.stop()
        publisher.sounder.stop()
        db.close()
        agent_log.close()
        audit.close()
        camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("[MAIN] AuraDrive session ended.")


def main() -> None:
    """Start the event loop and translate an operator interrupt into a clean stop.

    This is the synchronous entry point the launcher calls. Catching the
    keyboard interrupt here keeps a deliberate stop from printing a traceback,
    so the console output distinguishes an operator ending a session from the
    system actually failing.

    Returns
    -------
    None
        The function performs an action without returning a value.
    """
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[MAIN] interrupted.")


if __name__ == "__main__":
    main()

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
    """Load the local model before the first real dispatch so frame 1 doesn't eat the
    cold-start latency. Holds the single-flight lock so it never races a real agent
    call; runs concurrently with perception (the camera loop is never blocked)."""
    if await shared.reserve_agent():
        try:
            await asyncio.to_thread(process.call, WARMUP_PAYLOAD)
        except Exception:  # noqa: BLE001
            pass
        finally:
            await shared.release_agent()


async def run() -> None:
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
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[MAIN] interrupted.")


if __name__ == "__main__":
    main()

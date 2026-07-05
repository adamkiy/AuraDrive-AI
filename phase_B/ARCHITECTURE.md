# AuraDrive (merged) Architecture — AHP Cold + Context-aware Agent

This is the **merged** architecture: the partner's v8 concurrent infrastructure
(SensorDB, context builder, frame-id rendezvous, head pose, hardened agent,
arbiter) with the **graded AHP reference engine** as the deterministic cold layer
(replacing the earlier binary `cold_decision.py`).

```text
T1 task_sensor
  Camera + MediaPipe face-mesh overlay + solvePnP head pose (pose.py)
  ├─ SensorDB (db.py): every frame retained for 10 minutes (frame_id, timestamp_ms)
  ├─ Reflex latch: Eyes Closed Duration >= 2000 ms -> ONE EMERGENCY_ALERT (T1, immediate)
  └─ latest-only sensor_queue
          ↓
T2 task_cold
  Deterministic GRADED AHP fusion  (reference_engine.get_cold_decision)
  ├─ AHP weights perclos .51 / closure .27 / yawn .12 / blink .10  (CR = 0.006)
  ├─ context multipliers: time-of-day (circadian) × trip-duration  (jointly capped)
  ├─ EMERGENCY floor: context alone cannot manufacture EMERGENCY
  ├─ head-pose corroboration: sustained head-down escalates one tier (cap URGENT)
  ├─ non-reflex maximum: URGENT_ALERT
  └─ latest-only cold_queue
          ↓
T5 task_arbiter
  ├─ publishes the Cold baseline IMMEDIATELY (real-time protection, never waits on the LLM)
  ├─ NO_ACTION       -> no agent request
  ├─ GENTLE/MODERATE/URGENT -> reserve + wake T4
  └─ waits asynchronously for the SAME frame_id
          ↓
T4 task_agent
  Native Ollama, TEXT-ONLY evidence (System Log + Facts + Frame Narrative) -> structured JSON
  guards: EMERGENCY is reflex-only (LLM EMERGENCY capped to URGENT);
          Post-Microsleep Recovery requires a deterministically-confirmed >=2 s event;
          schema additionalProperties:false; "examples are fictional" anti-hallucination.
          ↓
AgentDecisionLog (exact frame_id rendezvous, db.py)
          ↓
T5 evaluate(cold, agent)  (evaluator.py)
  ├─ agent may NOT weaken cold (cold is a hard safety floor) -> safety_override
  ├─ agent may add AT MOST one command tier over cold        -> escalation cap
  └─ TemporalGuard: bounded recovery — alert may rise instantly but fall <= one tier / step
  timeout / error -> Cold fallback (baseline already published)

T3 task_context runs in parallel with T2/T4/T5:
  SensorDB -> 10-minute System Log + Deterministic Facts + Frame Narrative (context_summary.py)
```

## Why AHP cold + one-tier cap compose correctly

The two halves are complementary and neither is sufficient alone:

- The **one-tier escalation cap** (evaluator) bounds the LLM's influence to a
  single tier above the deterministic floor — a single model over-reaction can
  only nudge, never max out the alert. This is safe **only if** the cold floor
  is itself appropriately high in dangerous situations.
- The **context-aware graded AHP cold** makes the floor rise correctly under a
  circadian trough or a long trip (e.g. PERCLOS 0.09 at 03:00 after 4 h →
  URGENT), so the cap leaves the agent the room it needs exactly when it matters.

A binary, context-blind cold (the earlier `cold_decision.py`) would lock the
agent low in precisely the dangerous-but-subtle cases; the graded AHP cold fixes
that while keeping the validated weights / consistency ratio defensible.

## Deterministic-first, agent-second

The deterministic layers (reflex + AHP cold) always set a safety floor that the
LLM can refine but never weaken. All inference is local (Ollama); raw video never
leaves the machine.

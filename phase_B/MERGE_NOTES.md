# MERGE NOTES — AuraDrive (merged build)

This document is the honest map of the merged project: which files are your
partner's, which are yours, what was written fresh during the merge, and what
you still need to do. Read the **TODO** section before you run or defend it.

---

## The merge decision (one line)

**Base = the partner's v8 concurrent infrastructure. Deterministic core = your
graded AHP reference engine (replacing his binary cold). Gate = your widened
version. New modality = his head pose, wired into both the gate and the cold.**

Rationale: his infrastructure (SensorDB, frame-id rendezvous, context builder,
head pose, hardened agent guards, validation suite, asyncio orchestration) is
more mature than yours; your AHP deterministic core is more complete and
defensible than his binary `cold_decision` (which was context-blind, ignored
blink rate, and was binary on PERCLOS: 0.08 and 0.90 both mapped to MODERATE).

---

## File-by-file provenance

| File | Origin | Status |
|------|--------|--------|
| `decision.py` | Partner v8 | as-is (canonical schema, single source of truth) |
| `db.py` | Partner v8 | as-is (SensorDB + AgentDecisionLog rendezvous + FinalDecisionLog) |
| `pose.py` | Partner v8 | as-is (calibrated head-pose tracker) |
| `context_summary.py` | Partner v8 | as-is (10-min text blocks for the LLM) |
| `evaluator.py` | Partner v8 | as-is (`evaluate` one-tier cap + `TemporalGuard`) |
| `agent.py` | Partner v8 (`agent-2.py`) | as-is (hardened: EMERGENCY-reflex-only, Post-Microsleep guard, anti-hallucination, strict schema) |
| `tasks.py` | Partner v8 | **1 line changed** — imports `get_cold_decision` from `reference_engine` instead of `cold_decision` |
| `sensor.py` | Partner v8 (has head pose) | **gate widened** (PERCLOS>0.08, yawn>=1, blink<10/>30, + head-down). Eye/mouth core and head pose unchanged. |
| `reference_engine.py` | **Yours** (AHP, CR=0.006) | **kept intact** + a `get_cold_decision()` adapter appended that wires it into v8 and adds head-pose corroboration |
| `main.py` | **Written fresh during merge** | asyncio entry point reconstructed from `tasks.py` (the partner archive shipped the task coroutines but not the top-level main) |
| `ARCHITECTURE.md` | Rewritten | reflects AHP cold (the partner's said "no reference engine" — no longer true) |
| `requirements.txt`, `README.md`, `TESTING_GUIDE.md`, `MANUAL_VALIDATION_CHECKLIST.md`, `.gitignore` | Partner v8 | as-is |

**Removed:** `cold_decision.py` (binary cold) — replaced by `reference_engine.py`.

---

## What the merge actually changed (only three code edits)

1. **`tasks.py`**: one import line now points at `reference_engine`.
2. **`sensor.py`**: the risk gate was widened and now also fires on a sustained
   head-down nod, so the agent engages early on subtle/contextual signals.
3. **`reference_engine.py`**: a `get_cold_decision(frame)` adapter was appended
   (your `compute_reference_decision` is untouched). The adapter:
   - calls your AHP scorer (which reads `time_of_day` / `trip_duration_min`),
   - adds **head-pose corroboration**: a sustained head-down nod escalates the
     fused command by one tier, capped at URGENT — the validated AHP matrix is
     left untouched and pose is an independent posture signal,
   - maps the output to the shape `tasks.cold_baseline()` and `context_summary`
     expect, and
   - keeps a microsleep safety net for parity with the partner's cold.

Everything else is the partner's code, unmodified.

---

## TODO — before you run or defend this

1. **Carry over the partner's validation harness.** `run_validation_suite.py`,
   `run_smoke_tests.py`, `run_full_validation.bat` / `run_windows.bat` are NOT
   in this build (they were not in the files shared with me, and I will not
   fabricate a test suite). They do not touch decision logic, so copy them in
   from the partner's archive as-is. Then re-run `python run_validation_suite.py`
   against this merged tree and confirm it still passes (the cold-threshold
   tests may need their expected values updated, since the cold layer changed
   from binary to graded AHP).

2. **Reconcile / tune the head-pose integration.** The adapter escalates one
   tier on a sustained head-down nod. This is the conservative default that
   keeps your AHP weights/CR intact. If you and your partner prefer head pose
   to be a *fifth/sixth AHP criterion*, re-derive the Saaty pairwise matrix for
   6 criteria and renormalise the weights — that is a methodology decision you
   should own, not something I should invent.

3. **Confirm `decision.py` is the version you want.** This build uses the
   partner's v8 `decision.py` (schema_version 8.0, `from_agent` only). Your
   earlier `decision.py` also had `from_reflex` / `from_reference`; they are not
   needed here because `tasks.py` builds reflex/cold decisions via
   `dec.make_decision` directly.

4. **Model choice / latency.** `agent.py` defaults to `llama3.2:1b` (fast) with
   `llama3.2:latest` as fallback, selected at runtime via `/api/tags`. If you
   want the larger model as primary, set `AURADRIVE_MODEL`. The 70 s agent wait
   (`AURADRIVE_AGENT_WAIT_TIMEOUT`) is generous — the cold baseline is already
   shown, but you may want to lower it.

5. **Decide whose name leads which subsystem** for the report/viva. This is a
   joint project; be explicit about who authored the AHP core vs the
   infrastructure when you present it.

---

## Verification already performed (in this build)

- `python -m py_compile` passes on every `.py` file.
- Import chain verified for all pure-Python modules (`decision`,
  `reference_engine`, `evaluator`, `context_summary`, `agent`, `pose`, `db`).
- The graded-AHP cold was exercised across PERCLOS / context / head-pose /
  microsleep scenarios and behaves as designed (context amplification, graded
  PERCLOS, head-down escalation, microsleep net).
- The full decision path (cold → agent canonicalisation → `evaluate`
  arbitration → `TemporalGuard` recovery → `context_summary`) was exercised on
  synthetic data without a camera or Ollama.

Not verifiable here (needs your machine): the live camera loop, MediaPipe, the
solvePnP head-pose numbers, and the Ollama round-trip. Run the manual checklist
in `MANUAL_VALIDATION_CHECKLIST.md` for those.

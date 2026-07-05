"""AuraDrive — audio actuation """
from __future__ import annotations

import hashlib
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

# Per-severity phrase text and speech rate (words/min) — higher = more urgent.
_PHRASES = {
    "GENTLE_ALERT":    ("Gentle alert. Early fatigue signs detected.", 170),
    "MODERATE_ALERT":  ("Moderate alert. Fatigue detected. Take a break soon.", 185),
    "URGENT_ALERT":    ("Urgent alert. Pull over safely.", 205),
    "EMERGENCY_ALERT": ("Emergency! Emergency! Pull over immediately.", 215),
}
# Lead-in sound per severity (None = phrase only, "SIREN" = generated file).
_LEAD_IN = {
    "GENTLE_ALERT":    None,
    "MODERATE_ALERT":  "/System/Library/Sounds/Glass.aiff",
    "URGENT_ALERT":    "/System/Library/Sounds/Sosumi.aiff",
    "EMERGENCY_ALERT": "SIREN",
}
_DOWNGRADE_TICK = "/System/Library/Sounds/Pop.aiff"
# Face-tracking loss is not an alert severity — it gets its own phrase.
_NO_FACE_PHRASE = ("Driver not visible. Please face the camera.", 190)
_RANK = {"NO_ACTION": 0, "GENTLE_ALERT": 1, "MODERATE_ALERT": 2,
         "URGENT_ALERT": 3, "EMERGENCY_ALERT": 4}


def _write_siren(path: Path, seconds: float = 1.6) -> None:
    """Synthesize an alarming two-tone siren (900/650 Hz, 0.2 s alternation)."""
    rate = 44_100
    frames = bytearray()
    for i in range(int(seconds * rate)):
        t = i / rate
        freq = 900.0 if int(t / 0.2) % 2 == 0 else 650.0
        sample = math.sin(2 * math.pi * freq * t)
        sample = max(-1.0, min(1.0, sample * 1.6))          # mild overdrive: harsher
        env = min(1.0, i / (0.01 * rate), (seconds * rate - i) / (0.05 * rate))
        frames += struct.pack("<h", int(sample * env * 32000))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


class AlertSounder:
    """Announce alert transitions without ever blocking the caller.

    Input : the canonical decision dict on every publish (any source).
    Output: side effect only — severity-shaped audio per the module rules.
    """

    def __init__(self) -> None:
        self.enabled = (
            os.getenv("AURADRIVE_AUDIO", "1") != "0"
            and shutil.which("afplay") is not None
            and shutil.which("say") is not None
        )
        self.voice = os.getenv("AURADRIVE_VOICE", "Samantha")
        self.speak_full = os.getenv("AURADRIVE_SPEAK_FULL", "0") == "1"
        self.repeat_seconds = float(os.getenv("AURADRIVE_AUDIO_REPEAT_SEC", "8"))
        self.cache = Path(tempfile.gettempdir()) / "auradrive_audio"
        self.siren = self.cache / "siren.wav"
        self.last_command = "NO_ACTION"
        self.last_play_at = 0.0
        self._gen = 0                      # playback generation: bumping it stops the chain
        self._player: Optional[subprocess.Popen] = None
        self._speech: Optional[subprocess.Popen] = None
        self._phrase_files: Dict[str, Path] = {}
        self.no_face_active = False
        self.last_no_face_at = 0.0
        self.no_face_repeat = float(os.getenv("AURADRIVE_NO_FACE_REPEAT_SEC", "10"))
        # Announcement cooldown: when a boundary-hovering command flips back to
        # a level that was already announced moments ago, stay silent instead
        # of repeating it. EMERGENCY is exempt (always announced).
        self.announce_cooldown = float(os.getenv("AURADRIVE_AUDIO_COOLDOWN_SEC", "45"))
        self._announced_at: Dict[str, float] = {}
        if self.enabled:
            for cmd, (text, rate) in _PHRASES.items():
                tag = hashlib.md5(f"{self.voice}|{rate}|{text}".encode()).hexdigest()[:8]
                self._phrase_files[cmd] = self.cache / f"{cmd}-{tag}.aiff"
            text, rate = _NO_FACE_PHRASE
            tag = hashlib.md5(f"{self.voice}|{rate}|{text}".encode()).hexdigest()[:8]
            self._phrase_files["NO_FACE"] = self.cache / f"NO_FACE-{tag}.aiff"
            threading.Thread(target=self._prerender, daemon=True).start()

    # ── rendering (once, off the event loop) ──────────────────────────────
    def _prerender(self) -> None:
        try:
            self.cache.mkdir(parents=True, exist_ok=True)
            if not self.siren.exists():
                _write_siren(self.siren)
        except OSError:
            return
        texts = {**_PHRASES, "NO_FACE": _NO_FACE_PHRASE}
        for cmd, path in self._phrase_files.items():
            if path.exists():
                continue
            text, rate = texts[cmd]
            for args in (["say", "-v", self.voice], ["say"]):   # voice fallback
                try:
                    subprocess.run([*args, "-r", str(rate), "-o", str(path), text],
                                   check=True, capture_output=True, timeout=30)
                    break
                except (subprocess.SubprocessError, OSError):
                    continue

    # ── playback (all non-blocking; sequences run in a daemon thread) ─────
    def _is_playing(self) -> bool:
        return self._player is not None and self._player.poll() is None

    def _play_sequence(self, files: List[str], interrupt: bool) -> None:
        if interrupt:
            self._gen += 1
            if self._is_playing():
                self._player.terminate()
        gen = self._gen

        def run() -> None:
            for f in files:
                if gen != self._gen or not os.path.exists(f):
                    return
                proc = subprocess.Popen(["afplay", f],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._player = proc
                proc.wait()
                if proc.returncode not in (0, None):
                    return
        threading.Thread(target=run, daemon=True).start()

    def _sequence_for(self, command: str) -> List[str]:
        files: List[str] = []
        lead = _LEAD_IN.get(command)
        if lead == "SIREN":
            files.append(str(self.siren))
        elif lead:
            files.append(lead)
        phrase = self._phrase_files.get(command)
        if phrase is not None and phrase.exists():
            files.append(str(phrase))
        return [f for f in files if os.path.exists(f)]

    def _speak_dynamic(self, text: str) -> None:
        """Optional live TTS of the LLM's personalized message (~3 s lag)."""
        if not text:
            return
        if self._speech is not None and self._speech.poll() is None:
            self._speech.terminate()
        self._speech = subprocess.Popen(["say", "-v", self.voice, text[:200]],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── public API ────────────────────────────────────────────────────────
    def notify(self, decision: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        command = str(decision.get("command", "NO_ACTION"))
        now = time.monotonic()

        if command == "NO_ACTION":
            # Recovery to normal: stay silent and let any current phrase finish.
            self.last_command = command
            return

        changed = command != self.last_command
        escalated = _RANK.get(command, 0) > _RANK.get(self.last_command, 0)
        held_emergency = (not changed and command == "EMERGENCY_ALERT"
                          and now - self.last_play_at >= self.repeat_seconds)

        if changed and escalated:
            # Cooldown: a level announced within the window is not repeated —
            # boundary flapping must not chatter. EMERGENCY always announces.
            recently_announced = (
                command != "EMERGENCY_ALERT"
                and now - self._announced_at.get(command, float("-inf")) < self.announce_cooldown
            )
            if not recently_announced:
                # Higher-priority news: interrupt whatever is playing.
                self._play_sequence(self._sequence_for(command), interrupt=True)
                self.last_play_at = now
                self._announced_at[command] = now
                if self.speak_full:
                    self._speak_dynamic(str(decision.get("message", "")))
        elif changed:
            # De-escalation: never cut speech mid-word. Soft tick only if idle.
            if not self._is_playing() and os.path.exists(_DOWNGRADE_TICK):
                self._play_sequence([_DOWNGRADE_TICK], interrupt=False)
                self.last_play_at = now
        elif held_emergency and not self._is_playing():
            # Sustained EMERGENCY: periodic re-announcement.
            self._play_sequence(self._sequence_for(command), interrupt=False)
            self.last_play_at = now

        self.last_command = command

    def notify_no_face(self, active: bool) -> None:
        """Spoken cue for sustained face-tracking loss (not an alert severity).

        Speaks once on loss and every `no_face_repeat` seconds while it lasts;
        never interrupts an alert announcement; silent on regain.
        """
        if not self.enabled:
            self.no_face_active = active
            return
        now = time.monotonic()
        rising = active and not self.no_face_active
        due = active and now - self.last_no_face_at >= self.no_face_repeat
        if (rising or due) and not self._is_playing():
            phrase = self._phrase_files.get("NO_FACE")
            if phrase is not None and phrase.exists():
                self._play_sequence([str(phrase)], interrupt=False)
                self.last_no_face_at = now
        self.no_face_active = active

    def stop(self) -> None:
        self._gen += 1
        for proc in (self._player, self._speech):
            if proc is not None and proc.poll() is None:
                proc.terminate()

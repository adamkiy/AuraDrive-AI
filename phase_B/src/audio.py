"""AuraDrive — graded alert audio actuation for Windows and macOS.

Public API consumed by tasks.py (unchanged):
    sounder = AlertSounder()
    sounder.notify(decision)          # canonical decision dict, any source
    sounder.notify_no_face(active)    # sensing state, not an alert severity
    sounder.stop()                    # shutdown

Every call returns immediately. Tones and speech run on daemon threads, so the
camera loop, the reflex latch and the asyncio event loop never wait for audio.
Playback is cancelled through a generation counter, never by killing a thread.

Backends (resolved once, at construction)
-----------------------------------------
windows : winsound.Beep for tones; PowerShell + System.Speech for speech.
macos   : pre-rendered tone WAVs played with afplay; `say` for speech.
none    : audio disables itself and reports why on stderr. The system stays
          fully functional — the deterministic layers and the on-screen banner
          are unaffected; only the audible channel is absent.

Fixed phrases are pre-rendered into a temp cache at startup so an alert never
pays a text-to-speech cold start. On Windows in particular, every PowerShell +
System.Speech launch costs a fresh process start; paying that per alert would
delay each spoken warning behind its tone pattern.

Environment variables
---------------------
AURADRIVE_AUDIO=0                Disable all audio.
AURADRIVE_TTS=0                  Keep alert tones, disable speech.
AURADRIVE_SPEAK_FULL=1           Speak the LLM's dynamic message instead of the
                                 fixed per-severity phrase.
AURADRIVE_VOICE=<name>           Installed SAPI (Windows) or `say` (macOS) voice.
AURADRIVE_AUDIO_REPEAT_SEC=8     Re-announce a held EMERGENCY after this long.
AURADRIVE_AUDIO_COOLDOWN_SEC=45  Cooldown for a repeated non-emergency escalation.
AURADRIVE_NO_FACE_REPEAT_SEC=10  Repeat the sustained no-face notice.
"""
from __future__ import annotations

import base64
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import winsound
except ImportError:  # non-Windows: the import must still succeed
    winsound = None  # type: ignore[assignment]


_RANK = {
    "NO_ACTION": 0,
    "GENTLE_ALERT": 1,
    "MODERATE_ALERT": 2,
    "URGENT_ALERT": 3,
    "EMERGENCY_ALERT": 4,
}

# (spoken text, rate in words per minute). One table drives both backends: macOS
# `say -r` takes words/minute directly, and _speech_script maps it onto SAPI's
# -10..10 scale. The text is fixed by default so it is pre-renderable and starts
# instantly; AURADRIVE_SPEAK_FULL=1 substitutes the LLM's message instead.
_PHRASES: Dict[str, Tuple[str, int]] = {
    "GENTLE_ALERT":    ("Gentle alert. Early fatigue signs detected.", 155),
    "MODERATE_ALERT":  ("Moderate alert. Fatigue detected. Take a break soon.", 180),
    "URGENT_ALERT":    ("Urgent alert. Pull over safely now.", 215),
    "EMERGENCY_ALERT": ("Emergency. Pull over immediately.", 250),
    "NO_FACE":         ("Driver not visible. Please face the camera.", 185),
}

# (frequency Hz, duration ms). Severity is audible before a word is spoken: the
# count and pitch rise with urgency, and EMERGENCY alternates high/low so it is
# unmistakably different from an ordinary warning chime.
_PATTERNS: Dict[str, List[Tuple[int, int]]] = {
    "GENTLE_ALERT":    [(740, 130)],
    "MODERATE_ALERT":  [(740, 140), (880, 140)],
    "URGENT_ALERT":    [(1050, 160), (1050, 160), (1050, 230)],
    "EMERGENCY_ALERT": [(1150, 180), (650, 180), (1150, 180), (650, 180),
                        (1150, 180), (650, 250)],
    "NO_FACE":         [(500, 120), (500, 120)],
}
_TONE_GAP_MS = 45        # silence between tones, so a pattern stays countable
_TONE_RATE_HZ = 44_100   # sample rate of the rendered macOS tone WAVs


class AlertSounder:
    """Graded alert audio for Windows and macOS behind a single interface.

    The backend is resolved once in __init__; every method dispatches internally,
    so tasks.py never learns which platform it is on. The announcement policy is
    identical on both: escalations interrupt, de-escalations stay quiet, and a
    held EMERGENCY re-announces periodically.
    """

    def __init__(self) -> None:
        """Resolve the audio backend once and prepare the playback state.

        Backend selection happens here rather than per alert so the cost is
        paid at startup and every later call is immediate. If neither platform
        backend is available the object disables itself and reports why on
        stderr: silent failure would be the dangerous outcome, since the
        deterministic layers would keep working while the driver heard nothing.

        Returns
        -------
        None
            The constructor only resolves the backend and prepares state.
        """
        if sys.platform == "win32" and winsound is not None:
            self.backend = "windows"
        elif sys.platform == "darwin" and shutil.which("afplay") is not None:
            self.backend = "macos"
        else:
            self.backend = "none"

        self.enabled = os.getenv("AURADRIVE_AUDIO", "1") != "0" and self.backend != "none"
        self.tts_enabled = self.enabled and os.getenv("AURADRIVE_TTS", "1") != "0"
        self.speak_full = os.getenv("AURADRIVE_SPEAK_FULL", "0") == "1"
        self.voice = os.getenv("AURADRIVE_VOICE", "").strip()

        self.repeat_seconds = float(os.getenv("AURADRIVE_AUDIO_REPEAT_SEC", "8"))
        self.announce_cooldown = float(os.getenv("AURADRIVE_AUDIO_COOLDOWN_SEC", "45"))
        self.no_face_repeat = float(os.getenv("AURADRIVE_NO_FACE_REPEAT_SEC", "10"))

        # Resolve the speech engine for this backend. PowerShell ships with every
        # normal Windows install and exposes System.Speech without adding a pip
        # dependency; `say` is standard on macOS. Either one missing degrades to
        # tones only, never to silence.
        self._powershell: Optional[str] = None
        if self.backend == "windows":
            self._powershell = (
                shutil.which("powershell.exe")
                or shutil.which("powershell")
                or shutil.which("pwsh.exe")
                or shutil.which("pwsh")
            )
            if self._powershell is None:
                self.tts_enabled = False
        elif self.backend == "macos":
            if not self.voice:
                self.voice = "Samantha"
            if shutil.which("say") is None:
                self.tts_enabled = False

        # Never fail silently: a mute system that reports nothing is
        # indistinguishable from a working one until the demo is already running.
        if not self.enabled:
            sys.stderr.write(
                f"[AUDIO] disabled (platform={sys.platform}, backend={self.backend}) — "
                "alerts remain visible on the UI banner.\n"
            )
        elif not self.tts_enabled:
            sys.stderr.write(
                f"[AUDIO] {self.backend}: tones only, speech engine unavailable.\n"
            )

        self.last_command = "NO_ACTION"
        self.last_play_at = 0.0
        self.no_face_active = False
        self.last_no_face_at = 0.0
        self._announced_at: Dict[str, float] = {}

        self._lock = threading.Lock()
        self._generation = 0
        self._tone_active = False
        self._player: Optional[subprocess.Popen] = None   # macOS tone playback
        self._speech: Optional[subprocess.Popen] = None   # live TTS process

        self._cache = Path(tempfile.gettempdir()) / "auradrive_audio"
        self._tone_files: Dict[str, Path] = {}    # command     -> rendered tone WAV
        self._phrase_files: Dict[str, Path] = {}  # spoken text -> rendered audio
        if self.enabled:
            threading.Thread(target=self._prerender, name="AuraDriveAudioPrerender",
                             daemon=True).start()

    # ------------------------------------------------------------------
    # Startup rendering
    # ------------------------------------------------------------------
    def _prerender(self) -> None:
        """Render tones and fixed phrases into a temp cache once, off the event loop.

        Windows synthesises tones live (winsound.Beep is immediate) and only
        pre-renders speech; macOS pre-renders both. Files persist between runs,
        so the cost is paid once. Every failure here is non-fatal: _play_tones
        skips a missing tone file and _speak falls back to live synthesis.
        """
        try:
            self._cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        if self.backend == "macos":
            for command, pattern in _PATTERNS.items():
                path = self._cache / f"tone-{command}.wav"
                if not path.exists():
                    frames = bytearray()
                    for frequency, duration_ms in pattern:
                        samples = int(_TONE_RATE_HZ * duration_ms / 1000)
                        fade = max(1, int(0.005 * _TONE_RATE_HZ))  # 5 ms: no clicks
                        for i in range(samples):
                            envelope = min(1.0, i / fade, (samples - i) / fade)
                            value = math.sin(2 * math.pi * frequency * (i / _TONE_RATE_HZ))
                            frames += struct.pack("<h", int(value * envelope * 26_000))
                        frames += b"\x00\x00" * int(_TONE_RATE_HZ * _TONE_GAP_MS / 1000)
                    try:
                        with wave.open(str(path), "wb") as handle:
                            handle.setnchannels(1)
                            handle.setsampwidth(2)
                            handle.setframerate(_TONE_RATE_HZ)
                            handle.writeframes(bytes(frames))
                    except (OSError, wave.Error):
                        continue
                self._tone_files[command] = path

        if not self.tts_enabled:
            return

        suffix = "wav" if self.backend == "windows" else "aiff"
        tag = self.voice or "default"
        for command, (text, wpm) in _PHRASES.items():
            path = self._cache / f"phrase-{command}-{tag}.{suffix}"
            if path.exists():
                self._phrase_files[text] = path
                continue
            if self.backend == "macos":
                for args in (["say", "-v", self.voice], ["say"]):  # voice fallback
                    try:
                        subprocess.run([*args, "-r", str(wpm), "-o", str(path), text],
                                       check=True, capture_output=True, timeout=30)
                        self._phrase_files[text] = path
                        break
                    except (subprocess.SubprocessError, OSError):
                        continue
            elif self._powershell is not None:
                try:
                    subprocess.run(
                        [self._powershell, "-NoProfile", "-NonInteractive", "-Command",
                         self._speech_script(text, wpm, out_path=path)],
                        check=True, capture_output=True, timeout=30,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if path.exists():
                        self._phrase_files[text] = path
                except (subprocess.SubprocessError, OSError):
                    continue

    # ------------------------------------------------------------------
    # Internal lifecycle helpers
    # ------------------------------------------------------------------
    def _is_current(self, generation: int) -> bool:
        """Report whether a playback token still refers to the active alert.

        Playback is cancelled by invalidating a token rather than by killing a
        thread, which cannot leave the audio device in a half-open state. A
        worker checks this between stages and abandons its remaining work if a
        newer alert has superseded it.

        Parameters
        ----------
        generation : int
            The token the calling worker was started with.

        Returns
        -------
        bool
            True while this worker is still the current one.
        """
        with self._lock:
            return generation == self._generation

    def _is_playing(self) -> bool:
        """Report whether a tone or a spoken phrase is currently sounding.

        Used by the announcement policy to decide whether a new alert should
        interrupt or wait, which is what keeps escalations immediate while
        stopping routine updates from talking over each other.

        Returns
        -------
        bool
            True if either the tone player or the speech process is active.
        """
        with self._lock:
            speech_alive = self._speech is not None and self._speech.poll() is None
            player_alive = self._player is not None and self._player.poll() is None
            return self._tone_active or speech_alive or player_alive

    def _interrupt_current(self) -> int:
        """Invalidate queued playback and stop anything currently sounding."""
        with self._lock:
            self._generation += 1
            generation = self._generation
            processes = [self._player, self._speech]
            self._player = None
            self._speech = None
        for process in processes:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        if self.backend == "windows" and winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except RuntimeError:
                pass
        return generation

    @staticmethod
    def _safe_text(text: Any, limit: int = 200) -> str:
        """Normalise text before it reaches a speech engine.

        Model output reaches this point, so it is treated as untrusted: control
        characters are removed and whitespace collapsed, then the result is
        truncated. Truncation is a safety property as much as a formatting one,
        because an over-long utterance would occupy the audio channel while the
        driver's state continues to change.

        Parameters
        ----------
        text : Any
            The message to be spoken.
        limit : int
            Maximum number of characters to keep.

        Returns
        -------
        str
            Text safe to hand to the speech backend.
        """
        return " ".join(str(text or "").replace("\x00", " ").split())[:limit]

    def _speech_script(self, text: str, wpm: int, out_path: Optional[Path] = None) -> str:
        """Build the PowerShell System.Speech command for one utterance.

        Base64 removes every quoting/escaping hazard for LLM-authored text. Rate
        is SAPI's -10..10 scale, where 0 is roughly 200 wpm and each step is
        about ten percent, so the single words-per-minute column in _PHRASES
        drives both backends. With out_path the speech is rendered to a WAV
        instead of the speakers (used by _prerender).
        """
        sapi_rate = max(-10, min(10, round(math.log(max(wpm, 60) / 200.0) / math.log(1.1))))
        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        encoded_voice = base64.b64encode(self.voice.encode("utf-8")).decode("ascii")
        if out_path is None:
            sink = "$s.SetOutputToDefaultAudioDevice();"
        else:
            escaped = str(out_path).replace("'", "''")  # PowerShell single-quote escape
            sink = f"$s.SetOutputToWaveFile('{escaped}');"
        return (
            "Add-Type -AssemblyName System.Speech;"
            f"$t=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_text}'));"
            f"$v=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_voice}'));"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"$s.Rate={sapi_rate};"
            "if($v){try{$s.SelectVoice($v)}catch{}};"
            f"{sink}"
            "try{$s.Speak($t)}finally{$s.Dispose()}"
        )

    # ------------------------------------------------------------------
    # Tones and speech
    # ------------------------------------------------------------------
    def _play_tones(self, command: str, generation: int) -> None:
        """Play the graded tone pattern for a command; blocks this daemon thread only.

        Windows beeps the pattern live; macOS plays the WAV rendered by
        _prerender. Either way the generation check lets a newer, higher-priority
        announcement cut this one off.
        """
        if not self.enabled or not self._is_current(generation):
            return
        with self._lock:
            self._tone_active = True
        try:
            if self.backend == "windows":
                for frequency, duration_ms in _PATTERNS.get(command, []):
                    if not self._is_current(generation):
                        return
                    try:
                        winsound.Beep(frequency, duration_ms)  # type: ignore[union-attr]
                    except RuntimeError:
                        return  # remote/headless session with no tone device
                    time.sleep(_TONE_GAP_MS / 1000.0)
                return

            path = self._tone_files.get(command)
            if path is None or not path.exists():
                return
            try:
                process = subprocess.Popen(["afplay", str(path)],
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
            except OSError:
                return
            with self._lock:
                if generation != self._generation:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                    return
                self._player = process
            process.wait()
            with self._lock:
                if self._player is process:
                    self._player = None
        finally:
            with self._lock:
                self._tone_active = False

    def _speak(self, text: str, rate: int, generation: int) -> None:
        """Speak one message; blocks this daemon thread only.

        A pre-rendered file is used when the text matches a fixed phrase (the
        default path, so speech starts immediately). Live synthesis is the
        fallback, and is the path taken by the LLM's dynamic message under
        AURADRIVE_SPEAK_FULL, which cannot be rendered ahead of time.
        """
        if not self.tts_enabled or not self._is_current(generation):
            return
        text = self._safe_text(text)
        if not text:
            return

        cached = self._phrase_files.get(text)
        if cached is not None and cached.exists():
            if self.backend == "windows" and winsound is not None:
                try:
                    with wave.open(str(cached)) as handle:
                        seconds = handle.getnframes() / float(handle.getframerate())
                    winsound.PlaySound(str(cached),
                                       winsound.SND_FILENAME | winsound.SND_ASYNC)
                except (OSError, wave.Error, RuntimeError):
                    return
                # SND_ASYNC returns at once, so poll the known duration and purge
                # the moment a higher-priority announcement supersedes this one.
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    if not self._is_current(generation):
                        try:
                            winsound.PlaySound(None, winsound.SND_PURGE)
                        except RuntimeError:
                            pass
                        return
                    time.sleep(0.05)
                return
            command = ["afplay", str(cached)]
        elif self.backend == "windows":
            if self._powershell is None:
                return
            command = [self._powershell, "-NoProfile", "-NonInteractive", "-Command",
                       self._speech_script(text, rate)]
        else:
            command = ["say", "-v", self.voice, "-r", str(rate), text]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            return

        with self._lock:
            if generation != self._generation:
                try:
                    process.terminate()
                except OSError:
                    pass
                return
            self._speech = process
        try:
            process.wait()
        finally:
            with self._lock:
                if self._speech is process:
                    self._speech = None

    def _start_announcement(
        self,
        command: str,
        *,
        dynamic_message: str = "",
        interrupt: bool,
    ) -> None:
        """Start tones and speech on a daemon thread so the event loop never blocks."""
        if not self.enabled:
            return
        if interrupt:
            generation = self._interrupt_current()
        else:
            with self._lock:
                generation = self._generation

        fixed_text, wpm = _PHRASES.get(command, ("AuraDrive warning.", 190))
        chosen_text = self._safe_text(dynamic_message) if self.speak_full else fixed_text
        if not chosen_text:
            chosen_text = fixed_text

        def run() -> None:
            """Play the tone pattern and then the phrase, on a daemon thread.

            Runs off the event loop so neither the camera nor the reflex latch
            ever waits on audio. The token is rechecked between the tones and
            the speech, so a newer alert cancels the phrase instead of queueing
            behind it.

            Returns
            -------
            None
                The function performs an action without returning a value.
            """
            self._play_tones(command, generation)
            if self._is_current(generation):
                self._speak(chosen_text, wpm, generation)

        threading.Thread(target=run, name=f"AuraDriveAudio-{command}", daemon=True).start()

    # ------------------------------------------------------------------
    # Public API expected by tasks.py
    # ------------------------------------------------------------------
    def notify(self, decision: Dict[str, Any]) -> None:
        """Announce only meaningful alert transitions.

        Escalations interrupt lower-priority audio. De-escalations stay quiet, so
        recovery never produces a cascade of tones. A held EMERGENCY
        re-announces every AURADRIVE_AUDIO_REPEAT_SEC seconds.
        """
        if not self.enabled:
            return

        command = str(decision.get("command", "NO_ACTION"))
        if command not in _RANK:
            command = "NO_ACTION"
        now = time.monotonic()

        if command == "NO_ACTION":
            # Recovery to normal: stay silent, let any current phrase finish.
            self.last_command = command
            return

        changed = command != self.last_command
        escalated = _RANK[command] > _RANK.get(self.last_command, 0)
        held_emergency = (
            not changed
            and command == "EMERGENCY_ALERT"
            and now - self.last_play_at >= self.repeat_seconds
        )

        if changed and escalated:
            # Cooldown: a level announced within the window is not repeated, so a
            # command hovering on a band edge cannot chatter. EMERGENCY is exempt
            # and always announces.
            recently_announced = (
                command != "EMERGENCY_ALERT"
                and now - self._announced_at.get(command, float("-inf")) < self.announce_cooldown
            )
            if not recently_announced:
                self._start_announcement(
                    command,
                    dynamic_message=self._safe_text(decision.get("message", "")),
                    interrupt=True,
                )
                self.last_play_at = now
                self._announced_at[command] = now
        elif held_emergency and not self._is_playing():
            self._start_announcement(
                command,
                dynamic_message=self._safe_text(decision.get("message", "")),
                interrupt=False,
            )
            self.last_play_at = now

        self.last_command = command

    def notify_no_face(self, active: bool) -> None:
        """Announce sustained face-tracking loss without overriding an alert.

        Speaks on the rising edge and every AURADRIVE_NO_FACE_REPEAT_SEC seconds
        while it lasts; never interrupts an alert; silent on regain.
        """
        if not self.enabled:
            self.no_face_active = active
            return

        now = time.monotonic()
        rising = active and not self.no_face_active
        due = active and now - self.last_no_face_at >= self.no_face_repeat
        if (rising or due) and not self._is_playing():
            self._start_announcement("NO_FACE", interrupt=False)
            self.last_no_face_at = now
        self.no_face_active = active

    def stop(self) -> None:
        """Stop current playback and prevent any further announcement (shutdown)."""
        self.enabled = False
        self._interrupt_current()

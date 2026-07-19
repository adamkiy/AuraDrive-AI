@echo off
setlocal EnableDelayedExpansion
rem
rem AuraDrive - launcher for Windows 10/11
rem =====================================
rem The counterpart to run.sh. Starts the system and, if needed, installs the
rem Python dependencies and makes sure a local Ollama server and model are ready.
rem
rem Layout this script assumes:
rem   phase_B\run.bat     this file
rem   phase_B\run.sh      the macOS and Linux equivalent
rem   phase_B\src\        the eleven modules and requirements.txt
rem   phase_B\logs\       every JSONL log the run produces
rem   phase_B\.venv\      created by --venv
rem
rem Usage:
rem   run.bat                Run. Installs deps only if missing; ensures Ollama.
rem   run.bat --venv         Use/create an isolated .venv and install into it.
rem   run.bat --install      Force a (re)install of requirements before running.
rem   run.bat --setup        Do setup (deps + model pull) and exit without running.
rem   run.bat --skip-ollama  Do not manage Ollama (agent calls fall back to the
rem                          deterministic cold layer if Ollama is unavailable).
rem   run.bat --help         Show this help.
rem
rem Environment overrides (all optional):
rem   AURADRIVE_MODEL           primary model      (default llama3.2:1b)
rem   AURADRIVE_FALLBACK_MODEL  fallback model     (default llama3.2:latest)
rem   AURADRIVE_OLLAMA_URL      Ollama endpoint    (default http://localhost:11434)
rem   AURADRIVE_CAMERA_INDEX    camera index       (default 0)
rem   AURADRIVE_PYTHON          python interpreter (default: py -3 or python)
rem
rem Notes:
rem   * Requires Python 3.10+ (the perception layer uses `X ^| None` signatures).
rem   * Audio on Windows uses winsound for tones and PowerShell System.Speech for
rem     speech. Both ship with the OS, so there is nothing extra to install.
rem   * Unlike run.sh this script does not stop an Ollama server it started;
rem     close the console window it opened, or leave the service running.
rem

rem ---------- defaults ----------
if not defined AURADRIVE_MODEL           set "AURADRIVE_MODEL=llama3.2:1b"
if not defined AURADRIVE_FALLBACK_MODEL  set "AURADRIVE_FALLBACK_MODEL=llama3.2:latest"
if not defined AURADRIVE_OLLAMA_URL      set "AURADRIVE_OLLAMA_URL=http://localhost:11434"
if not defined AURADRIVE_CAMERA_INDEX    set "AURADRIVE_CAMERA_INDEX=0"

set "USE_VENV=0"
set "FORCE_INSTALL=0"
set "SETUP_ONLY=0"
set "MANAGE_OLLAMA=1"

rem ---------- was this double-clicked in Explorer? ----------
rem Explorer launches a .bat through "cmd /c", which also closes the window the
rem moment the script ends. Detecting that lets the failure path pause, and lets
rem a double-click default to an isolated environment, matching what
rem AuraDrive.command does on macOS. A run from an existing prompt is untouched.
set "DOUBLE_CLICK=0"
echo %cmdcmdline% | find /i "/c" >nul && set "DOUBLE_CLICK=1"
if "%DOUBLE_CLICK%"=="1" if "%~1"=="" set "USE_VENV=1"

rem ---------- arg parsing ----------
:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--venv"        set "USE_VENV=1"        & shift & goto parse
if /I "%~1"=="--install"     set "FORCE_INSTALL=1"   & shift & goto parse
if /I "%~1"=="--setup"       set "SETUP_ONLY=1" & set "FORCE_INSTALL=1" & shift & goto parse
if /I "%~1"=="--skip-ollama" set "MANAGE_OLLAMA=0"   & shift & goto parse
if /I "%~1"=="--help"        goto help
if /I "%~1"=="-h"            goto help
echo [run][ERROR] Unknown option: %~1 (try --help) 1>&2
goto :die2
:parsed

rem ---------- resolve the layout ----------
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "SRC=%ROOT%\src"
set "LOGS=%ROOT%\logs"

if not exist "%SRC%\main.py" (
    echo [run][ERROR] src\main.py not found under %ROOT%. 1>&2
    goto :die
)
if not exist "%SRC%\requirements.txt" (
    echo [run][ERROR] src\requirements.txt not found. 1>&2
    goto :die
)
if not exist "%LOGS%" mkdir "%LOGS%"

rem ---------- pick a Python interpreter (>= 3.10) ----------
set "BASE_PY="
if defined AURADRIVE_PYTHON call :try_py "%AURADRIVE_PYTHON%"
if not defined BASE_PY      call :try_py "py -3"
if not defined BASE_PY      call :try_py "python"
if not defined BASE_PY      call :try_py "python3"
if not defined BASE_PY (
    echo [run][ERROR] Need Python 3.10+ on PATH. Install from python.org, or set 1>&2
    echo [run][ERROR] AURADRIVE_PYTHON to point at a suitable interpreter. 1>&2
    goto :die
)
echo [run] Using interpreter: %BASE_PY%

rem ---------- virtual environment ----------
rem An existing .venv is used whichever way the script was invoked. Ignoring one
rem that is already provisioned, merely because a flag was omitted, would send
rem pip at the system interpreter instead. --venv therefore means "create one if
rem it is missing". AURADRIVE_PYTHON overrides everything.
rem Kept as flat statements rather than nested blocks: %VAR% inside a
rem parenthesised block expands when the block is parsed, not when it runs.
set "PY="
if defined AURADRIVE_PYTHON goto venv_explicit
if exist "%ROOT%\.venv\Scripts\python.exe" goto venv_existing
if "%USE_VENV%"=="1" goto venv_create
set "PY=%BASE_PY%"
echo [run] No .venv present; using the system interpreter. Use --venv for an isolated one.
goto venv_done

:venv_explicit
set "PY=%BASE_PY%"
echo [run] Using the interpreter from AURADRIVE_PYTHON (ignoring any .venv).
goto venv_done

:venv_existing
set "PY=%ROOT%\.venv\Scripts\python.exe"
echo [run] Using existing virtualenv %ROOT%\.venv
goto venv_done

:venv_create
echo [run] Creating virtualenv .venv ...
%BASE_PY% -m venv "%ROOT%\.venv" || goto :die
set "FORCE_INSTALL=1"
set "PY=%ROOT%\.venv\Scripts\python.exe"
echo [run] Using virtualenv %ROOT%\.venv

:venv_done

rem ---------- Python dependencies ----------
rem Functional check, not a presence check: mediapipe can import successfully
rem and still be missing mp.solutions on a partial install.
if "%FORCE_INSTALL%"=="1" goto install
call :deps_ok && goto deps_done

:install
echo [run] Installing Python requirements ...
%PY% -m pip install --upgrade pip >nul 2>&1
%PY% -m pip install -r "%SRC%\requirements.txt"
call :deps_ok && goto deps_done
echo [run] Still not functional - forcing a clean reinstall ...
%PY% -m pip install --force-reinstall --no-cache-dir -r "%SRC%\requirements.txt"
call :deps_ok && goto deps_done
echo [run][ERROR] MediaPipe/OpenCV are installed but NOT functional in this interpreter. 1>&2
echo [run][ERROR] Most reliable fix: run.bat --venv 1>&2
goto :die

:deps_done
echo [run] Dependencies functional.

rem ---------- Ollama ----------
if "%MANAGE_OLLAMA%"=="0" (
    echo [run] Skipping Ollama management (--skip-ollama).
    echo [run] If Ollama is down, every agent call falls back to the cold layer.
    goto ollama_done
)

where ollama >nul 2>&1
if errorlevel 1 (
    echo [run][ERROR] Ollama is not installed - the LLM agent layer needs it. 1>&2
    echo [run][ERROR] Install from https://ollama.com/download, then re-run. 1>&2
    echo [run][ERROR] (Or re-run with --skip-ollama for the deterministic layer only.) 1>&2
    goto :die
)

ollama list >nul 2>&1
if errorlevel 1 (
    echo [run] Starting 'ollama serve' in a background window ...
    start "AuraDrive Ollama" /min ollama serve
    for /L %%i in (1,1,30) do (
        timeout /t 1 /nobreak >nul
        ollama list >nul 2>&1 && goto ollama_up
    )
    echo [run][ERROR] Ollama did not become ready. 1>&2
    goto :die
) else (
    echo [run] Ollama server already running.
)
:ollama_up

set "MODEL_FOUND=0"
for /f "skip=1 tokens=1" %%m in ('ollama list 2^>nul') do (
    if /I "%%m"=="%AURADRIVE_MODEL%"          set "MODEL_FOUND=1"
    if /I "%%m"=="%AURADRIVE_FALLBACK_MODEL%" set "MODEL_FOUND=1"
)
if "%MODEL_FOUND%"=="1" (
    echo [run] Model available (%AURADRIVE_MODEL% or %AURADRIVE_FALLBACK_MODEL%).
) else (
    echo [run] No model found; pulling '%AURADRIVE_MODEL%' (one-time, ~1.3 GB) ...
    ollama pull "%AURADRIVE_MODEL%" || goto :die
)
:ollama_done

rem The agent runs as a subprocess with its own cwd, so point its log at logs\
rem explicitly; the other three follow the working directory set below.
set "AURADRIVE_AGENT_LOG=%LOGS%\agent_decision_log.jsonl"

if "%SETUP_ONLY%"=="1" (
    echo [run] Setup complete (--setup). Not launching.
    exit /b 0
)

rem ---------- run ----------
echo [run] Launching AuraDrive. Press 'q' in the video window to quit.
echo [run] Logs for this session: %LOGS%
pushd "%LOGS%"
%PY% "%SRC%\main.py"
set "RC=%ERRORLEVEL%"
popd
echo [run] AuraDrive exited (code %RC%).
call :hold %RC%
exit /b %RC%

rem ---------- subroutines ----------
:try_py
%~1 -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" >nul 2>&1
if not errorlevel 1 set "BASE_PY=%~1"
exit /b 0

:deps_ok
%PY% -c "import numpy, cv2, mediapipe as mp; mp.solutions.face_mesh" >nul 2>&1
exit /b %ERRORLEVEL%

:die
call :hold 1
exit /b 1

:die2
call :hold 1
exit /b 2

:hold
rem Hold the console open after a failure, but only when this file was started
rem by double-clicking it. Explorer launches a .bat through "cmd /c", which
rem closes the window the moment the script ends and takes the error message
rem with it. A run from an existing prompt has no such problem and must not be
rem left waiting for a keypress, which would break scripted use.
if "%~1"=="0" exit /b 0
if not "%DOUBLE_CLICK%"=="1" exit /b 0
echo.
echo Press any key to close this window.
pause >nul
exit /b 0

:help
echo.
for /f "tokens=1,* delims=:" %%a in ('findstr /n "^rem" "%~f0"') do (
    set "line=%%b"
    setlocal EnableDelayedExpansion
    set "line=!line:~4!"
    echo(!line!
    endlocal
)
exit /b 0

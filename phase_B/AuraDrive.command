#!/usr/bin/env bash
#
# AuraDrive - double-clickable launcher for macOS
# ===============================================
# Finder opens a .command file in Terminal, so this is the whole shortcut: no
# Automator action, no .app bundle, no extra tooling. It is a thin wrapper that
# hands off to run.sh, which does the real work.
#
# Why the cd matters
# ------------------
# A double-clicked .command starts with the working directory set to the user's
# home folder, not to the folder the file lives in. Everything below therefore
# resolves relative to this script's own location, which means the whole
# repository can be moved or renamed and the shortcut keeps working, as long as
# this file stays next to run.sh.
#
# The window is held open on failure so an error stays readable instead of
# vanishing with the Terminal window.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || { echo "Cannot enter $HERE"; read -r -n 1 -s; exit 1; }

if [ ! -x ./run.sh ]; then
    if [ -f ./run.sh ]; then
        # The executable bit does not survive a ZIP download from GitHub.
        chmod +x ./run.sh
    else
        echo "run.sh not found next to this shortcut."
        echo "Keep AuraDrive.command in the phase_B folder, beside run.sh."
        echo
        echo "Press any key to close."
        read -r -n 1 -s
        exit 1
    fi
fi

printf '\033]0;AuraDrive\007'          # name the Terminal window
echo "AuraDrive"
echo "Project folder: $HERE"
echo

# --venv either way: run.sh reuses an existing environment and creates one only
# when it is missing. The message is the only thing that differs, since the
# first run also downloads the model.
if [ ! -d .venv ]; then
    echo "First run: setting up the environment and downloading the model."
    echo "This takes a few minutes once, then later launches start in seconds."
    echo
fi
./run.sh --venv "$@"

status=$?
echo
if [ $status -eq 0 ]; then
    echo "Session ended. This window can be closed."
else
    echo "AuraDrive exited with code $status. The messages above explain why."
    echo
    echo "Press any key to close."
    read -r -n 1 -s
fi
exit $status

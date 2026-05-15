#!/bin/bash
# LegalMind -- macOS / Linux launcher.
# Double-click in Finder or run ./LegalMind.command from terminal.
set -e

SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_PATH" ]; do
    LINK_TARGET="$(readlink "$SCRIPT_PATH")"
    case "$LINK_TARGET" in
        /*) SCRIPT_PATH="$LINK_TARGET" ;;
        *) SCRIPT_PATH="$(cd -P "$(dirname "$SCRIPT_PATH")" && pwd)/$LINK_TARGET" ;;
    esac
done
SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$SCRIPT_DIR"

PY=""
for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "[LegalMind] Python 3.11+ required but not found."
    echo "  macOS: brew install python@3.11"
    echo "  Linux: install python3.11 from your distro"
    read -n 1 -s -r -p "Press any key..."
    exit 1
fi

"$PY" scripts/bootstrap.py "$@"

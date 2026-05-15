"""Development server launcher for LegalMind.

Run: python scripts/launch_server.py [--port 8000] [--reload]

This script is a lightweight wrapper for development use. It adds the project
root to sys.path and launches uvicorn directly. For production or first-run
setup, use bootstrap.py instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path so imports resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Parse arguments and launch the development server."""
    parser = argparse.ArgumentParser(
        description="LegalMind development server launcher.",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind the server on (default: 8000).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes (development mode).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1). Incompatible with --reload.",
    )

    args = parser.parse_args()

    if args.reload and args.workers > 1:
        print(
            "[LegalMind] WARNING: --reload is incompatible with --workers > 1. "
            "Using single worker with reload enabled.",
            file=sys.stderr,
        )
        args.workers = 1

    try:
        import uvicorn
    except ImportError:
        print(
            "[LegalMind] ERROR: uvicorn is not installed.\n"
            "  Run: pip install uvicorn[standard]\n"
            "  Or use bootstrap.py for automatic setup.",
            file=sys.stderr,
        )
        return 1

    print(f"[LegalMind] Starting development server at http://{args.host}:{args.port}")
    if args.reload:
        print("[LegalMind] Auto-reload enabled -- watching for file changes.")

    uvicorn.run(
        "webapp.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

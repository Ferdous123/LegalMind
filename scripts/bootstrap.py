"""LegalMind first-run bootstrap + launcher (cross-platform, stdlib-only).

Steps:
  1. Verify Python >= 3.11
  2. Create .venv/ if needed
  3. Install requirements.txt (CUDA-aware for llama-cpp-python)
  4. Launch: uvicorn webapp.main:app --host 127.0.0.1 --port 8000
  5. Open default browser to http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import venv
import webbrowser
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_PY = (3, 11)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
HOST = "127.0.0.1"
PORT = 8000
APP_MODULE = "webapp.main:app"

# llama-cpp-python CUDA wheel indexes by CUDA major.minor
_LLAMA_CPP_WHEEL_INDEXES: dict[str, str] = {
    "12.6": "https://abetlen.github.io/llama-cpp-python/whl/cu126",
    "12.5": "https://abetlen.github.io/llama-cpp-python/whl/cu125",
    "12.4": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    "12.3": "https://abetlen.github.io/llama-cpp-python/whl/cu123",
    "12.2": "https://abetlen.github.io/llama-cpp-python/whl/cu122",
    "12.1": "https://abetlen.github.io/llama-cpp-python/whl/cu121",
    "11.8": "https://abetlen.github.io/llama-cpp-python/whl/cu118",
    "11.7": "https://abetlen.github.io/llama-cpp-python/whl/cu117",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a formatted status message."""
    print(f"[LegalMind] {msg}", flush=True)


def _error(msg: str) -> None:
    """Print a formatted error message and exit."""
    print(f"\n[LegalMind] ERROR: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(1)


def _run(
    cmd: list[str],
    *,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run a subprocess with consistent settings."""
    merged_env = {**os.environ, **(env or {})}
    try:
        return subprocess.run(
            cmd,
            env=merged_env,
            check=check,
            capture_output=capture,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        if capture and e.stderr:
            _log(f"Command failed: {' '.join(cmd)}")
            _log(f"stderr: {e.stderr.strip()}")
        raise


def _get_venv_python() -> Path:
    """Return path to the venv Python executable."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _get_venv_pip() -> list[str]:
    """Return pip invocation command within the venv."""
    return [str(_get_venv_python()), "-m", "pip"]


# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------

def _check_python_version() -> None:
    """Verify we are running on Python >= MIN_PY."""
    current = sys.version_info[:2]
    if current < MIN_PY:
        _error(
            f"Python {MIN_PY[0]}.{MIN_PY[1]}+ is required, "
            f"but this is Python {current[0]}.{current[1]}.\n"
            f"  Install a newer Python from https://www.python.org/downloads/"
        )


# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------

def _venv_is_valid() -> bool:
    """Check if the existing venv is usable (correct Python, not stale)."""
    py = _get_venv_python()
    if not py.exists():
        return False

    # Verify the venv Python is actually functional
    try:
        result = _run(
            [str(py), "-c", "import sys; print(sys.version_info[:2])"],
            capture=True,
            check=True,
        )
        version_str = result.stdout.strip()
        # Parse tuple like (3, 11)
        match = re.match(r"\((\d+),\s*(\d+)\)", version_str)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            if (major, minor) < MIN_PY:
                return False
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def _create_venv() -> None:
    """Create a fresh virtual environment."""
    if VENV_DIR.exists():
        _log("Removing stale virtual environment...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    _log("Creating virtual environment...")
    venv.create(str(VENV_DIR), with_pip=True, clear=True)

    # Upgrade pip to avoid compatibility issues
    _log("Upgrading pip...")
    _run([*_get_venv_pip(), "install", "--upgrade", "pip", "--quiet"])


# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------

def _detect_cuda_version() -> Optional[str]:
    """Detect CUDA version from nvidia-smi output.

    Returns a version string like '12.6' or None if no GPU / nvidia-smi not found.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = _run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None

    # nvidia-smi might not give CUDA version directly from that query,
    # so try parsing the full output instead
    try:
        result = _run([nvidia_smi], capture=True, check=True)
        output = result.stdout
    except (subprocess.CalledProcessError, OSError):
        return None

    # Look for "CUDA Version: X.Y" in the output
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
    if match:
        return match.group(1)

    return None


def _get_wheel_index(cuda_version: str) -> Optional[str]:
    """Map a CUDA version to the appropriate llama-cpp-python wheel index URL."""
    # Try exact match first
    if cuda_version in _LLAMA_CPP_WHEEL_INDEXES:
        return _LLAMA_CPP_WHEEL_INDEXES[cuda_version]

    # Try major.minor prefix match (e.g., 12.6.1 -> 12.6)
    parts = cuda_version.split(".")
    if len(parts) >= 2:
        short = f"{parts[0]}.{parts[1]}"
        if short in _LLAMA_CPP_WHEEL_INDEXES:
            return _LLAMA_CPP_WHEEL_INDEXES[short]

    # Try closest lower version within same major
    major = parts[0]
    minor = int(parts[1]) if len(parts) >= 2 else 0
    candidates = []
    for ver, url in _LLAMA_CPP_WHEEL_INDEXES.items():
        v_parts = ver.split(".")
        if v_parts[0] == major:
            candidates.append((int(v_parts[1]), url))

    if candidates:
        # Pick the highest minor version that is <= our minor
        candidates.sort(key=lambda x: x[0], reverse=True)
        for c_minor, url in candidates:
            if c_minor <= minor:
                return url
        # If none is <=, pick the lowest available
        return candidates[-1][1]

    return None


# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------

def _deps_installed() -> bool:
    """Quick check: are key dependencies already importable?

    This is a fast-path check to avoid running pip on every launch.
    """
    py = str(_get_venv_python())
    check_script = (
        "import fastapi, uvicorn, jinja2, pdfplumber, chromadb, "
        "sentence_transformers, llama_cpp, tiktoken"
    )
    try:
        result = _run([py, "-c", check_script], capture=True, check=False)
        return result.returncode == 0
    except OSError:
        return False


def _install_deps() -> None:
    """Install all requirements, handling llama-cpp-python CUDA builds specially."""
    pip = _get_venv_pip()

    # First, install everything EXCEPT llama-cpp-python
    _log("Installing dependencies (this may take a few minutes on first run)...")

    # Create a filtered requirements file without llama-cpp-python
    filtered_lines: list[str] = []
    with open(REQUIREMENTS, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if "llama-cpp-python" in stripped.lower() or "llama_cpp_python" in stripped.lower():
                continue
            filtered_lines.append(stripped)

    # Install non-GPU deps
    if filtered_lines:
        _run([*pip, "install", "--quiet", *filtered_lines])

    # Now handle llama-cpp-python with CUDA awareness
    _install_llama_cpp(pip)


def _install_llama_cpp(pip: list[str]) -> None:
    """Install llama-cpp-python with CUDA support if a GPU is detected."""
    cuda_version = _detect_cuda_version()

    if cuda_version:
        _log(f"Detected CUDA {cuda_version} -- installing llama-cpp-python with GPU support...")
        wheel_index = _get_wheel_index(cuda_version)

        if wheel_index:
            # Try prebuilt wheel from index
            _log(f"Trying prebuilt CUDA wheel from {wheel_index}...")
            result = _run(
                [
                    *pip, "install", "llama-cpp-python>=0.3.20",
                    "--extra-index-url", wheel_index,
                    "--quiet",
                ],
                check=False,
                capture=True,
            )
            if result.returncode == 0:
                _log("Successfully installed llama-cpp-python with CUDA wheel.")
                return

            _log("Prebuilt wheel not available, attempting source compile with CUDA...")

        # Fallback: source compile with CUDA enabled
        env_extras = {"CMAKE_ARGS": "-DGGML_CUDA=on"}
        if platform.system() == "Windows":
            env_extras["FORCE_CMAKE"] = "1"

        result = _run(
            [*pip, "install", "llama-cpp-python>=0.3.20", "--no-binary", ":all:", "--quiet"],
            env=env_extras,
            check=False,
            capture=True,
        )
        if result.returncode == 0:
            _log("Successfully compiled llama-cpp-python with CUDA support.")
            return

        _log("CUDA build failed. Falling back to CPU-only llama-cpp-python...")

    else:
        _log("No CUDA GPU detected -- installing CPU-only llama-cpp-python...")

    # CPU fallback
    result = _run(
        [*pip, "install", "llama-cpp-python>=0.3.20", "--quiet"],
        check=False,
        capture=True,
    )
    if result.returncode == 0:
        _log("Installed llama-cpp-python (CPU mode).")
    else:
        _log(
            "WARNING: Could not install llama-cpp-python. "
            "LLM inference will not be available until it is installed manually."
        )


# ---------------------------------------------------------------------------
# Server launch
# ---------------------------------------------------------------------------

def _launch_server(port: int, open_browser: bool = True) -> int:
    """Launch the uvicorn server and optionally open the browser.

    Returns the server process exit code.
    """
    py = str(_get_venv_python())
    url = f"http://{HOST}:{port}"

    _log(f"Starting LegalMind server at {url}")

    # Build uvicorn command
    cmd = [
        py, "-m", "uvicorn", APP_MODULE,
        "--host", HOST,
        "--port", str(port),
    ]

    if open_browser:
        # Open browser after a short delay to give server time to start
        _open_browser_delayed(url, delay=2.0)

    # Run server (blocks until Ctrl+C or shutdown)
    os.chdir(str(PROJECT_ROOT))
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return proc.returncode
    except KeyboardInterrupt:
        _log("Server stopped.")
        return 0


def _open_browser_delayed(url: str, delay: float = 2.0) -> None:
    """Open the default browser after a delay.

    Uses a separate thread so we do not block the server start.
    """
    import threading

    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass  # Non-critical: user can navigate manually

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the full bootstrap and launch sequence."""
    _check_python_version()

    _log(f"Project root: {PROJECT_ROOT}")
    _log(f"Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor})")

    # Step 1: Ensure virtual environment exists
    if not _venv_is_valid():
        _create_venv()
    else:
        _log("Virtual environment OK.")

    # Step 2: Install dependencies if needed
    if not _deps_installed():
        _install_deps()
    else:
        _log("Dependencies already installed (fast path).")

    # Step 3: Parse CLI arguments
    port = PORT
    open_browser = True
    args = sys.argv[1:]

    i = 0
    while i < len(args):
        if args[i] in ("--port", "-p") and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                _error(f"Invalid port: {args[i + 1]}")
            i += 2
        elif args[i] == "--no-browser":
            open_browser = False
            i += 1
        elif args[i] in ("--help", "-h"):
            print("Usage: bootstrap.py [--port PORT] [--no-browser]")
            print("  --port PORT     Server port (default: 8000)")
            print("  --no-browser    Do not open browser automatically")
            return 0
        else:
            _log(f"Unknown argument: {args[i]} (ignoring)")
            i += 1

    # Step 4: Launch server
    _log("All systems ready.")
    return _launch_server(port, open_browser=open_browser)


if __name__ == "__main__":
    sys.exit(main())

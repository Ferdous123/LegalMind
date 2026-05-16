"""Filesystem path constants for LegalMind.

All paths resolve relative to the project root (one level above this file).
Never hard-code absolute paths in other modules — import from here.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_YAML = CONFIG_DIR / "models.yaml"
LEARNED_RULES_YAML = CONFIG_DIR / "learned_rules.yaml"

# Data storage
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
CORRECTIONS_DIR = DATA_DIR / "corrections"
SAMPLE_DIR = DATA_DIR / "sample"
PALACE_DIR = DATA_DIR / "palace"

# Exemplar bank
EXEMPLARS_DIR = PROJECT_ROOT / "exemplars"

# Prompts
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Webapp
WEBAPP_DIR = PROJECT_ROOT / "webapp"
TEMPLATES_DIR = WEBAPP_DIR / "templates"
STATIC_DIR = WEBAPP_DIR / "static"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Model cache (GGUF files) — configurable via env var or models.yaml
_DEFAULT_MODEL_DIR = Path("F:/Research_Paper_Projects/LLMs")


def _resolve_model_cache_dir() -> Path:
    """Resolve model directory from environment, config, or default.

    Priority:
    1. LEGALMIND_MODEL_DIR environment variable
    2. model_cache_dir from config/models.yaml
    3. Default path
    """
    env_dir = os.environ.get("LEGALMIND_MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    if MODELS_YAML.exists():
        import yaml
        try:
            data = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
            cfg_dir = data.get("model_cache_dir", "")
            if cfg_dir:
                return Path(cfg_dir)
        except Exception:
            pass

    return _DEFAULT_MODEL_DIR


MODEL_CACHE_DIR = _resolve_model_cache_dir()

# Vector database
CHROMA_DIR = DATA_DIR / "chromadb"

# PDF page images (for human audit)
PAGES_DIR = DATA_DIR / "pages"


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [
        UPLOADS_DIR, PROCESSED_DIR, CORRECTIONS_DIR, SAMPLE_DIR,
        PALACE_DIR, EXEMPLARS_DIR, LOGS_DIR, CHROMA_DIR, PAGES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def check_model_availability() -> dict:
    """Check which models are available at the configured path.

    Returns dict with model status info for diagnostics.
    """
    result = {
        "model_dir": str(MODEL_CACHE_DIR),
        "dir_exists": MODEL_CACHE_DIR.exists(),
        "models": {},
    }

    if not MODEL_CACHE_DIR.exists():
        return result

    if MODELS_YAML.exists():
        import yaml
        try:
            data = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
            for key, val in data.get("local_models", {}).items():
                if isinstance(val, dict):
                    gguf_file = val.get("gguf_model", "")
                    file_path = MODEL_CACHE_DIR / gguf_file if gguf_file else None
                    result["models"][key] = {
                        "file": gguf_file,
                        "exists": file_path.exists() if file_path else False,
                        "size_gb": round(file_path.stat().st_size / (1024**3), 1) if file_path and file_path.exists() else 0,
                    }
        except Exception:
            pass

    return result

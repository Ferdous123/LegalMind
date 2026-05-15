"""Filesystem path constants for LegalMind.

All paths resolve relative to the project root (one level above this file).
Never hard-code absolute paths in other modules — import from here.
"""

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

# Model cache (GGUF files)
MODEL_CACHE_DIR = Path("F:/Research_Paper_Projects/LLMs")

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

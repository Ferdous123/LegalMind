# LegalMind — Hugging Face Spaces (Docker SDK) — READ-ONLY DEMO
# Builds a CPU-only image with the populated demo data baked in.
# Heavy operations (upload, draft generation, rule extraction) are blocked
# by the middleware in hf_space/app.py; this image deliberately ships
# without llama-cpp-python and without any GGUF weights.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LEGALMIND_DEMO=1

# Minimal system deps (poppler/pdfplumber don't strictly need extras for
# read-only browsing of already-processed JSON, but we keep a few here in
# case the corpus endpoint touches PDFs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the demo-only requirements first (better Docker layer caching),
# then the rest of the repo.
COPY hf_space/requirements.txt /app/hf_space/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/hf_space/requirements.txt

# The full repo tree is copied — but heavy/secret stuff (.env, model
# weights, .venv, etc.) is excluded by hf_space/.dockerignore.
COPY . /app

# Drop any uvicorn --reload hints; explicit launch, no auto-reload.
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "hf_space.app:app", \
     "--host", "0.0.0.0", "--port", "7860"]

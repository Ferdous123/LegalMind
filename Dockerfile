FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# llama-cpp-python with CUDA
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --force-reinstall llama-cpp-python>=0.3.20

COPY . .

# Model cache mounted as volume at runtime
VOLUME ["/models", "/app/data"]

ENV LEGALMIND_MODEL_DIR=/models
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/api/v1/system/status || exit 1

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]

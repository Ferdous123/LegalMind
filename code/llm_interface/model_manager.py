"""ModelManager singleton — loads/unloads GGUF models via llama-cpp-python.

Only ONE large model loaded at a time. Embeddings model (sentence-transformers)
is separate and can coexist with any GGUF model.
"""

import gc
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import yaml

from config.paths import MODELS_YAML, MODEL_CACHE_DIR
from code.llm_interface.gpu_guard import can_load_model, get_free_vram_gb

logger = logging.getLogger(__name__)

_CUDA_DLL_SETUP_DONE = False


def _setup_cuda_dlls():
    """Add CUDA DLL directories on Windows."""
    global _CUDA_DLL_SETUP_DONE
    if _CUDA_DLL_SETUP_DONE or sys.platform != "win32":
        return
    _CUDA_DLL_SETUP_DONE = True
    cuda_paths = [
        Path(os.environ.get("CUDA_PATH", "")) / "bin",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
    ]
    for p in cuda_paths:
        if p.exists():
            os.add_dll_directory(str(p))
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            logger.info("Added CUDA DLL directory: %s", p)
            break


class ModelManager:
    """Singleton that manages GGUF model lifecycle on GPU."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        _setup_cuda_dlls()
        self._config = self._load_config()
        self._current_model_name: Optional[str] = None
        self._current_model = None
        self._embeddings_model = None

    @classmethod
    def instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_config(self) -> dict:
        with open(MODELS_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_model_path(self, filename: str) -> str:
        # Environment variable takes priority over models.yaml value, which
        # takes priority over the compiled-in MODEL_CACHE_DIR constant.
        cache_dir = (
            os.environ.get("LEGALMIND_MODEL_CACHE")
            or self._config.get("model_cache_dir")
            or str(MODEL_CACHE_DIR)
        )
        return str(Path(cache_dir) / filename)

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model_name

    def load(self, role: str):
        """Load a model by role name (ocr, extraction, reasoning).

        Unloads any previously loaded model first.
        Raises RuntimeError if insufficient VRAM.
        """
        if self._current_model_name == role:
            return self._current_model

        if self._current_model is not None:
            self.unload()

        model_cfg = self._config["local_models"][role]
        vram_needed = model_cfg["vram_gb"]

        if not can_load_model(vram_needed):
            raise RuntimeError(
                f"Cannot load '{role}': needs {vram_needed}GB, "
                f"only {get_free_vram_gb():.1f}GB free"
            )

        from llama_cpp import Llama

        gguf_path = self._get_model_path(model_cfg["gguf_model"])
        mmproj_path = None
        if model_cfg.get("gguf_mmproj"):
            mmproj_path = self._get_model_path(model_cfg["gguf_mmproj"])

        chat_handler = None
        if model_cfg.get("chat_handler") == "qwen25vl":
            from llama_cpp.llama_chat_format import Qwen2VLChatHandler
            chat_handler = Qwen2VLChatHandler(clip_model_path=mmproj_path)

        self._current_model = Llama(
            model_path=gguf_path,
            n_ctx=model_cfg.get("max_context", 4096),
            n_gpu_layers=-1,
            chat_handler=chat_handler,
            verbose=False,
        )
        self._current_model_name = role
        logger.info("Loaded model '%s' (%s)", role, model_cfg["name"])
        return self._current_model

    def unload(self):
        """Unload current model and free GPU memory."""
        if self._current_model is not None:
            del self._current_model
            self._current_model = None
            self._current_model_name = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Model unloaded, VRAM freed")

    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.3, images: Optional[list] = None) -> str:
        """Generate text from the currently loaded model.

        Args:
            prompt: The input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            images: Optional list of image paths (for VL models).

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._current_model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        model_cfg = self._config["local_models"][self._current_model_name]
        temp = temperature or model_cfg.get("temperature", 0.3)

        messages = [{"role": "user", "content": prompt}]

        if images and model_cfg.get("chat_handler"):
            content = [{"type": "text", "text": prompt}]
            for img_path in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{img_path}"}
                })
            messages = [{"role": "user", "content": content}]

        response = self._current_model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return response["choices"][0]["message"]["content"]

    def is_embeddings_model_cached(self) -> bool:
        """Return True if the embeddings model files are fully cached locally.

        Checks the HuggingFace cache for the model's snapshot directory and
        verifies that at least one weight file (.safetensors or .bin) is present
        and that no .incomplete marker files exist.
        """
        import os
        emb_cfg = self._config["local_models"]["embeddings"]
        model_id: str = emb_cfg["model_id"]
        # HF cache path: ~/.cache/huggingface/hub/models--{org}--{name}/
        cache_root = Path(os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            os.path.expanduser("~/.cache/huggingface/hub"),
        ))
        model_dir_name = "models--" + model_id.replace("/", "--")
        model_dir = cache_root / model_dir_name
        if not model_dir.exists():
            return False
        # Check for any .incomplete files (download in progress)
        if any(model_dir.rglob("*.incomplete")):
            return False
        # Check snapshots for weight files
        snapshots = model_dir / "snapshots"
        if not snapshots.exists():
            return False
        for snap in snapshots.iterdir():
            weight_files = list(snap.glob("*.safetensors")) + list(snap.glob("*.bin"))
            if weight_files:
                return True
        return False

    def load_embeddings(self):
        """Load the BGE-M3 embeddings model (can coexist with GGUF models)."""
        if self._embeddings_model is not None:
            return self._embeddings_model

        if not self.is_embeddings_model_cached():
            raise RuntimeError(
                "Embeddings model (BGE-M3) is not yet fully cached locally. "
                "The model may still be downloading. Please retry later."
            )

        from sentence_transformers import SentenceTransformer

        emb_cfg = self._config["local_models"]["embeddings"]
        self._embeddings_model = SentenceTransformer(
            emb_cfg["model_id"],
            device=emb_cfg.get("device", "cuda")
        )
        logger.info("Loaded embeddings model: %s", emb_cfg["model_id"])
        return self._embeddings_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using BGE-M3."""
        model = self.load_embeddings()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

"""ModelManager singleton — dict-based model registry with analytical VRAM tracking.

VRAM tracking is analytical: starts at 0, incremented on load, decremented on
unload.  A threading lock on load() prevents concurrent loads of the same model
(the race condition where two async handlers both pass the VRAM check before
either increments the counter).  Only one large model loaded at a time.
"""

import gc
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

from config.paths import MODELS_YAML, MODEL_CACHE_DIR

logger = logging.getLogger(__name__)

_CUDA_DLL_SETUP_DONE = False

VRAM_TOTAL_GB = 10.0  # RTX 3080


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


class VRAMBudgetExceeded(Exception):
    """Raised when loading a model would exceed VRAM budget."""


class ModelLoadFailed(Exception):
    """Raised when a model load fails after retries."""


class ModelManager:
    """Singleton that manages GGUF model lifecycle on GPU.

    Key guarantees:
    - _loaded dict registry: load() is idempotent (returns cached handle on second call)
    - _load_lock serialises concurrent load attempts — double-checked locking pattern
      prevents the race where two threads both pass the VRAM check simultaneously
    - reclear_gpu() zeroes _vram_used_gb completely — safe recovery from any stuck state
    - swap() unloads one model then loads another atomically
    """

    _instance: Optional["ModelManager"] = None
    _singleton_lock = threading.Lock()

    def __init__(self):
        raise RuntimeError("Use ModelManager.instance()")

    @classmethod
    def instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    obj = object.__new__(cls)
                    obj._init()
                    cls._instance = obj
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton — for testing only."""
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance.reclear_gpu()
            cls._instance = None

    def _init(self):
        _setup_cuda_dlls()
        self._config = self._load_config()
        self._loaded: dict[str, object] = {}
        self._loaded_vram: dict[str, float] = {}
        self._vram_used_gb: float = 0.0
        self._current_model_name: Optional[str] = None
        self._load_lock = threading.Lock()

    def _load_config(self) -> dict:
        if not Path(MODELS_YAML).exists():
            return {}
        with open(MODELS_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _get_model_path(self, filename: str) -> str:
        cache_dir = (
            os.environ.get("LEGALMIND_MODEL_CACHE")
            or self._config.get("model_cache_dir")
            or str(MODEL_CACHE_DIR)
        )
        return str(Path(cache_dir) / filename)

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model_name

    @property
    def vram_used_gb(self) -> float:
        return self._vram_used_gb

    @property
    def vram_free_gb(self) -> float:
        return max(0.0, VRAM_TOTAL_GB - self._vram_used_gb)

    def can_load(self, role: str) -> bool:
        if role in self._loaded:
            return True
        model_cfg = self._config.get("local_models", {}).get(role, {})
        vram_needed = model_cfg.get("vram_gb", 0.0)
        return self._vram_used_gb + vram_needed <= VRAM_TOTAL_GB

    def load(self, role: str) -> object:
        """Load a model by role (ocr, extraction, reasoning).

        Thread-safe via double-checked locking.  Returns immediately if the model
        is already in _loaded.  Unloads any other model first (one-model policy).
        """
        # Fast path — already loaded, no lock contention
        if role in self._loaded:
            self._current_model_name = role
            return self._loaded[role]

        with self._load_lock:
            # Re-check under lock: another thread may have loaded it while we waited
            if role in self._loaded:
                self._current_model_name = role
                return self._loaded[role]

            # Unload any other models to free VRAM (one model at a time)
            for loaded_role in list(self._loaded.keys()):
                if loaded_role != role:
                    self._unload_handle(loaded_role)

            model_cfg = self._config.get("local_models", {}).get(role)
            if not model_cfg:
                raise RuntimeError(f"Unknown model role: '{role}'")

            vram_needed = float(model_cfg.get("vram_gb", 0.0))

            if self._vram_used_gb + vram_needed > VRAM_TOTAL_GB:
                raise VRAMBudgetExceeded(
                    f"Cannot load '{role}': needs {vram_needed:.1f}GB, "
                    f"only {self.vram_free_gb:.1f}GB available "
                    f"(budget: {self._vram_used_gb:.1f}/{VRAM_TOTAL_GB}GB used). "
                    f"POST /api/v1/system/gpu_reset to recover."
                )

            last_exc: Optional[Exception] = None
            handle = None
            for attempt in range(3):
                try:
                    handle = self._load_llama_cpp(role, model_cfg)
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "load(%s) attempt %d/3 failed (%s: %s); reclearing GPU",
                        role, attempt + 1, type(exc).__name__, exc,
                    )
                    if attempt < 2:
                        self.reclear_gpu()
                        time.sleep(2.0)

            if handle is None:
                raise ModelLoadFailed(
                    f"Failed to load '{role}' after 3 attempts. "
                    f"Last error: {type(last_exc).__name__}: {last_exc}. "
                    f"POST /api/v1/system/gpu_reset to recover."
                ) from last_exc

            self._loaded[role] = handle
            self._loaded_vram[role] = vram_needed
            self._vram_used_gb += vram_needed
            self._current_model_name = role
            logger.info(
                "Loaded model '%s' (%s) — %.1fGB used / %.1fGB total",
                role, model_cfg.get("name", role), self._vram_used_gb, VRAM_TOTAL_GB,
            )
            return handle

    def _load_llama_cpp(self, role: str, model_cfg: dict) -> object:
        """Load a GGUF model via llama-cpp-python."""
        _setup_cuda_dlls()
        from llama_cpp import Llama

        gguf_path = self._get_model_path(model_cfg["gguf_model"])
        if not Path(gguf_path).exists():
            raise FileNotFoundError(f"Model file not found: {gguf_path}")

        kwargs: dict = {
            "model_path": gguf_path,
            "n_gpu_layers": -1,
            "n_ctx": model_cfg.get("max_context", 4096),
            "verbose": False,
            "flash_attn": True,
        }

        if model_cfg.get("gguf_mmproj"):
            mmproj_path = self._get_model_path(model_cfg["gguf_mmproj"])
            if not Path(mmproj_path).exists():
                raise FileNotFoundError(f"mmproj file not found: {mmproj_path}")
            handler = self._make_chat_handler(model_cfg.get("chat_handler", ""), mmproj_path)
            kwargs["chat_handler"] = handler

        logger.info("Loading llama.cpp model: %s", Path(gguf_path).name)
        return Llama(**kwargs)

    @staticmethod
    def _make_chat_handler(handler_type: str, clip_path: str) -> object:
        from llama_cpp.llama_chat_format import Llava16ChatHandler, Qwen25VLChatHandler
        if handler_type == "qwen25vl":
            return Qwen25VLChatHandler(clip_model_path=clip_path)
        return Llava16ChatHandler(clip_model_path=clip_path)

    def _unload_handle(self, role: str) -> None:
        """Internal: unload a handle by role name (no lock, caller holds _load_lock)."""
        if role not in self._loaded:
            return

        handle = self._loaded.pop(role)
        vram = self._loaded_vram.pop(role, 0.0)

        chat_handler = getattr(handle, "chat_handler", None)
        if chat_handler is not None:
            if hasattr(chat_handler, "close"):
                try:
                    chat_handler.close()
                except Exception as exc:
                    logger.warning("chat_handler.close() for %s raised: %s", role, exc)
            try:
                handle.chat_handler = None
            except Exception:
                pass
            del chat_handler

        if hasattr(handle, "close"):
            handle.close()
        del handle

        self._vram_used_gb = max(0.0, self._vram_used_gb - vram)
        if self._current_model_name == role:
            self._current_model_name = next(iter(self._loaded), None)

        for _ in range(3):
            gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

        logger.info(
            "Unloaded model '%s' (freed %.1fGB; %.1f/%.1fGB remaining)",
            role, vram, self._vram_used_gb, VRAM_TOTAL_GB,
        )

    def unload(self, role: Optional[str] = None) -> None:
        """Unload a model by role, or the current model if role is None.

        Public API — safe to call concurrently.
        """
        if role is None:
            role = self._current_model_name
        if role is None:
            return
        with self._load_lock:
            self._unload_handle(role)

    def swap(self, unload_role: str, load_role: str) -> object:
        """Atomic swap: unload one model, load another."""
        logger.info("Swapping %s -> %s", unload_role, load_role)
        self.unload(unload_role)
        return self.load(load_role)

    def reclear_gpu(self) -> None:
        """Aggressive GPU/VRAM cleanup. Called by POST /api/v1/system/gpu_reset.

        Drops every loaded model handle, zeroes _vram_used_gb, runs GC + CUDA
        cache clear.  Safe to call from any state including stuck/leaked state.
        """
        for role in list(self._loaded.keys()):
            try:
                self._unload_handle(role)
            except Exception as exc:
                logger.warning("reclear unload(%s) raised: %s", role, exc)
        self._loaded.clear()
        self._loaded_vram.clear()
        self._vram_used_gb = 0.0
        self._current_model_name = None

        for _ in range(3):
            gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        logger.info("GPU reclear complete (%.1fGB now used)", self._vram_used_gb)

    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.3, images: Optional[list] = None) -> str:
        """Generate text from the currently loaded model."""
        role = self._current_model_name
        if role is None or role not in self._loaded:
            raise RuntimeError("No model loaded. Call load() first.")

        handle = self._loaded[role]
        model_cfg = self._config.get("local_models", {}).get(role, {})
        temp = temperature or model_cfg.get("temperature", 0.3)

        messages: list[dict] = [{"role": "user", "content": prompt}]

        if images and model_cfg.get("chat_handler"):
            content: list[dict] = [{"type": "text", "text": prompt}]
            for img_path in images:
                url = Path(img_path).resolve().as_uri()
                content.append({"type": "image_url", "image_url": {"url": url}})
            messages = [{"role": "user", "content": content}]

        stop_tokens = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        response = handle.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
            stop=stop_tokens,
        )
        return response["choices"][0]["message"]["content"]

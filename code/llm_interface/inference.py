"""Unified inference API — wraps ModelManager for clean per-task access."""

import json
import logging
from typing import Optional

from code.llm_interface.model_manager import ModelManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-level inference interface used by pipeline components.

    Handles model loading/unloading transparently. Callers specify the role
    they need (ocr, extraction, reasoning) and this engine ensures the correct
    model is loaded before inference.
    """

    def __init__(self):
        self._mgr = ModelManager.instance()

    def generate_text(self, prompt: str, role: str = "extraction",
                      max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Generate text using the specified model role."""
        self._mgr.load(role)
        return self._mgr.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def generate_structured(self, prompt: str, schema: dict,
                           role: str = "extraction", max_tokens: int = 2048) -> dict:
        """Generate structured JSON output conforming to schema.

        Appends JSON instruction to prompt, parses response as JSON.
        Falls back to extracting JSON from markdown code blocks if needed.
        """
        json_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )
        self._mgr.load(role)
        raw = self._mgr.generate(json_prompt, max_tokens=max_tokens, temperature=0.1)
        return self._parse_json_response(raw)

    def generate_with_image(self, prompt: str, image_path: str,
                           role: str = "ocr", max_tokens: int = 2048) -> str:
        """Generate text from an image input (for VL models)."""
        self._mgr.load(role)
        return self._mgr.generate(prompt, max_tokens=max_tokens, images=[image_path])

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self._mgr.embed([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return self._mgr.embed(texts)

    def get_current_model(self) -> Optional[str]:
        """Return the currently loaded model role, or None."""
        return self._mgr.current_model

    def unload_current(self):
        """Explicitly unload the current model to free VRAM."""
        self._mgr.unload()

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """Parse JSON from model response, handling markdown code blocks."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error("Failed to parse JSON from model response: %s...", text[:200])
            return {"_parse_error": True, "_raw_response": raw}

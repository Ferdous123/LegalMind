"""LLM interface layer. All model access goes through ModelManager and InferenceEngine."""

from code.llm_interface.model_manager import ModelManager
from code.llm_interface.inference import InferenceEngine

__all__ = ["ModelManager", "InferenceEngine"]

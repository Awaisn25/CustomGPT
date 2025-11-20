"""Utility to manage model configurations for Ollama hot-swapping."""

from dataclasses import dataclass

from private_gpt.settings.settings import settings


@dataclass
class ModelConfig:
    """Represents an Ollama model configuration."""

    model_name: str  # e.g., "llama3.1", "gemma3:4b-it-qat"
    display_name: str  # e.g., "Ollama: llama3.1"

    def __str__(self) -> str:
        return self.display_name


def get_available_models() -> list[ModelConfig]:
    """Get available Ollama models from settings.dropdown_models."""
    config_settings = settings()
    if config_settings is None:
        return []

    model_names = config_settings.ui.dropdown_models
    return [
        ModelConfig(
            model_name=model_name,
            display_name=f"Ollama: {model_name}",
        )
        for model_name in model_names
    ]

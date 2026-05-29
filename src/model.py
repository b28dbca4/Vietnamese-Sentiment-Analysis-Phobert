"""Model loading, saving, and inspection utilities."""

from __future__ import annotations

import os
from typing import Any

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
except ImportError as exc:  # pragma: no cover - exercised only when dependencies are missing.
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    PreTrainedModel = Any
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from src import config
from src.utils import count_parameters, get_model_size_mb


def _require_transformers() -> None:
    """Raise a clear error if Hugging Face Transformers is unavailable."""
    if AutoModelForSequenceClassification is None or AutoTokenizer is None:
        raise ImportError(
            "The transformers package is required for model loading. "
            "Install project dependencies with 'pip install -r requirements.txt'."
        ) from _TRANSFORMERS_IMPORT_ERROR


def load_model_for_training(model_name: str) -> PreTrainedModel:
    """Load a pretrained Transformer with a sequence classification head.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Model ready for supervised fine-tuning.
    """
    _require_transformers()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    return model


def load_tokenizer(model_name: str):
    """Load the tokenizer paired with a specific pretrained model.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Hugging Face tokenizer instance.
    """
    _require_transformers()
    return AutoTokenizer.from_pretrained(model_name)


def _directory_size_mb(path: str) -> float:
    """Calculate the total file size of a directory in megabytes."""
    total_bytes = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total_bytes += os.path.getsize(os.path.join(root, filename))
    return total_bytes / (1024**2)


def save_model(model: PreTrainedModel, tokenizer, save_dir: str) -> None:
    """Save a fine-tuned model and tokenizer to disk.

    Args:
        model: Fine-tuned Transformer model.
        tokenizer: Tokenizer associated with the model.
        save_dir: Destination directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved model to {save_dir} ({_directory_size_mb(save_dir):.1f} MB)")


def load_model_for_inference(model_dir: str) -> tuple[PreTrainedModel, Any]:
    """Load a saved model and tokenizer for inference.

    Args:
        model_dir: Directory containing a saved Hugging Face checkpoint.

    Returns:
        Pair of model and tokenizer in evaluation mode.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    _require_transformers()
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def _get_config_value(model_config, *names: str, default: Any = None) -> Any:
    """Read the first available attribute from a model config object."""
    for name in names:
        if hasattr(model_config, name):
            return getattr(model_config, name)
    return default


def _tokenizer_type(model_name: str, model_config) -> str:
    """Infer a tokenizer family from model metadata."""
    model_type = str(_get_config_value(model_config, "model_type", default="")).lower()
    lower_name = model_name.lower()
    if "roberta" in lower_name or model_type == "roberta":
        return "BPE"
    if "bert" in lower_name or model_type in {"bert", "distilbert"}:
        return "WordPiece"
    return "Unknown"


def get_model_summary(model: PreTrainedModel, model_name: str) -> dict[str, Any]:
    """Generate a compact model architecture summary.

    Args:
        model: Transformer model to inspect.
        model_name: Human-readable model identifier.

    Returns:
        Dictionary with architecture, parameter, and tokenizer metadata.
    """
    parameters = count_parameters(model)
    model_config = model.config
    return {
        "name": model_name,
        "hf_identifier": getattr(model_config, "_name_or_path", model_name),
        "architecture_type": _get_config_value(model_config, "model_type", default="unknown"),
        "num_layers": _get_config_value(model_config, "num_hidden_layers", "n_layers"),
        "hidden_size": _get_config_value(model_config, "hidden_size", "dim"),
        "num_attention_heads": _get_config_value(model_config, "num_attention_heads", "n_heads"),
        "vocab_size": _get_config_value(model_config, "vocab_size"),
        "total_params": parameters["total"],
        "trainable_params": parameters["trainable"],
        "non_trainable_params": parameters["non_trainable"],
        "model_size_mb": get_model_size_mb(model),
        "tokenizer_type": _tokenizer_type(model_name, model_config),
    }

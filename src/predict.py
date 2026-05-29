"""Inference pipeline for trained IMDb sentiment models."""

from __future__ import annotations

from typing import Any

import torch

from src import config
from src import model as model_utils
from src.preprocessing import clean_text
from src.utils import count_parameters, get_device, get_model_size_mb


class SentimentPredictor:
    """Production-oriented sentiment prediction pipeline."""

    def __init__(self, model_dir: str = config.BEST_MODEL_DIR) -> None:
        """Load a saved model, tokenizer, and device.

        Args:
            model_dir: Directory containing a saved Hugging Face checkpoint.
        """
        self.model_dir = model_dir
        self.model, self.tokenizer = model_utils.load_model_for_inference(model_dir)
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

        parameters = count_parameters(self.model)
        print(
            f"Loaded model from {model_dir} on {self.device}; "
            f"parameters={parameters['total']:,}"
        )

    def preprocess(self, text: str) -> str:
        """Clean raw input text with the shared training pipeline."""
        return clean_text(text)

    def predict(self, text: str) -> dict[str, Any]:
        """Predict sentiment for one review.

        Args:
            text: Raw review text.

        Returns:
            Human-readable prediction dictionary.
        """
        cleaned = self.preprocess(text)
        full_tokens = self.tokenizer(cleaned, add_special_tokens=True, truncation=False)["input_ids"]
        encoding = self.tokenizer(
            cleaned,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            probabilities_tensor = torch.softmax(outputs.logits, dim=1).squeeze(0)

        probabilities = probabilities_tensor.detach().cpu().numpy()
        prediction_id = int(probabilities.argmax())
        label = config.ID2LABEL[prediction_id]
        return {
            "label": label,
            "confidence": float(probabilities[prediction_id]),
            "probabilities": {
                "negative": float(probabilities[config.LABEL2ID["negative"]]),
                "positive": float(probabilities[config.LABEL2ID["positive"]]),
            },
            "num_tokens": int(len(full_tokens)),
            "was_truncated": bool(len(full_tokens) > config.MAX_LENGTH),
            "cleaned_text": cleaned,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict sentiment for multiple reviews."""
        return [self.predict(text) for text in texts]

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata used by the app."""
        parameters = count_parameters(self.model)
        model_type = str(getattr(self.model.config, "model_type", "")).lower()
        display_names = {
            "distilbert": "DistilBERT",
            "bert": "BERT-base",
            "roberta": "RoBERTa",
        }
        return {
            "model_name": getattr(self.model.config, "_name_or_path", self.model_dir),
            "model_type": model_type,
            "display_name": display_names.get(model_type, "Transformer"),
            "total_params": parameters["total"],
            "trainable_params": parameters["trainable"],
            "max_length": config.MAX_LENGTH,
            "device": str(self.device),
            "size_mb": get_model_size_mb(self.model),
        }

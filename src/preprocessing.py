"""Text preprocessing shared by training and inference."""

from __future__ import annotations

import re


def remove_html(text: str) -> str:
    """Remove HTML tags while preserving word boundaries.

    Args:
        text: Raw review text.

    Returns:
        Text with HTML tags replaced by spaces.
    """
    cleaned = re.sub(r"<br\s*/?>", " ", str(text))
    cleaned = re.sub(r"<.*?>", " ", cleaned)
    return cleaned


def remove_urls(text: str) -> str:
    """Remove URLs that start with ``http`` or ``www``.

    Args:
        text: Review text.

    Returns:
        Text with URL substrings replaced by spaces.
    """
    return re.sub(r"http\S+|www\S+", " ", str(text))


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading or trailing spaces.

    Args:
        text: Review text.

    Returns:
        Text with normalized whitespace.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_text(text: str) -> str:
    """Apply the Transformer-safe preprocessing pipeline.

    Args:
        text: Raw review text.

    Returns:
        Cleaned text matching the notebook 01 preprocessing behavior.
    """
    cleaned = remove_html(text)
    cleaned = remove_urls(cleaned)
    cleaned = normalize_whitespace(cleaned)
    return cleaned


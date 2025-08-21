"""Clarity/readability metrics package."""

from .metrics import (
    flesch_reading_ease,
    flesch_kincaid_grade,
    smog_index,
    gunning_fog_index,
)

__all__ = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "smog_index",
    "gunning_fog_index",
]

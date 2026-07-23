from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RUBRIC_DIMENSIONS = ("Cin", "Pur", "Mot", "Phy")
RATING_MIN = 1.0
RATING_MAX = 5.0


def objective_score(correct_count: int, question_count: int) -> float:
    """Return the dependency-aware QA accuracy for one prompt-video pair."""
    if question_count <= 0:
        raise ValueError("question_count must be positive.")
    if correct_count < 0 or correct_count > question_count:
        raise ValueError("correct_count must be between 0 and question_count.")
    return correct_count / question_count


def normalize_rating(value: float) -> float:
    """Normalize a 1-5 rubric rating by the maximum rating of 5."""
    rating = float(value)
    if not RATING_MIN <= rating <= RATING_MAX:
        raise ValueError(f"Rubric rating must be in [1, 5], got {value!r}.")
    return rating / RATING_MAX


def extract_rubric_ratings(score_map: Mapping[str, Any]) -> dict[str, float]:
    ratings: dict[str, float] = {}
    missing: list[str] = []
    for key in RUBRIC_DIMENSIONS:
        value = score_map.get(key)
        if isinstance(value, Mapping):
            value = value.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            missing.append(key)
            continue
        ratings[key] = float(value)
    if missing:
        raise ValueError(f"Missing rubric dimensions: {', '.join(missing)}")
    return ratings


def extract_rubric_ratings_from_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    """Extract the four paper dimensions from either supported result schema."""
    score_map: dict[str, Any] = {}
    results = payload.get("results")
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, Mapping) or row.get("id") not in RUBRIC_DIMENSIONS:
                continue
            score_map[str(row["id"])] = row.get("score")

    scores = payload.get("scores")
    if isinstance(scores, Mapping):
        for key in RUBRIC_DIMENSIONS:
            if key not in scores:
                continue
            value = scores[key]
            candidate = value.get("score") if isinstance(value, Mapping) else value
            existing = score_map.get(key)
            if existing is not None and candidate != existing:
                raise ValueError(
                    f"Conflicting rubric score for {key}: {existing!r} versus {candidate!r}."
                )
            score_map[key] = candidate

    return extract_rubric_ratings(score_map)


def mean_rubric_rating(score_map: Mapping[str, Any]) -> float:
    ratings = extract_rubric_ratings(score_map)
    return sum(ratings.values()) / len(RUBRIC_DIMENSIONS)


def mean_rubric_score(score_map: Mapping[str, Any]) -> float:
    """Return mean(rating) / 5 over Cin, Pur, Mot, and Phy."""
    ratings = extract_rubric_ratings(score_map)
    return sum(normalize_rating(value) for value in ratings.values()) / len(RUBRIC_DIMENSIONS)


def sample_vgif_score(
    objective: float,
    subjective: float,
    *,
    objective_weight: float = 0.5,
    subjective_weight: float = 0.5,
) -> float:
    """Combine objective and subjective scores for one prompt-video pair."""
    if not 0.0 <= objective <= 1.0 or not 0.0 <= subjective <= 1.0:
        raise ValueError("Objective and subjective scores must be in [0, 1].")
    if objective_weight < 0 or subjective_weight < 0:
        raise ValueError("VGIF component weights must be non-negative.")
    total_weight = objective_weight + subjective_weight
    if total_weight <= 0:
        raise ValueError("VGIF component weights must sum to a positive value.")
    return (
        objective * objective_weight + subjective * subjective_weight
    ) / total_weight


def mean_sample_score(values: list[float]) -> float:
    """Macro-average a score that has already been computed per sample."""
    if not values:
        raise ValueError("At least one sample score is required.")
    return sum(values) / len(values)


def to_percent(value: float, digits: int = 2) -> float:
    return round(float(value) * 100.0, digits)

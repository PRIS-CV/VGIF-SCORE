from __future__ import annotations

import sys
import unittest
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "code" / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

from scoring import (
    extract_rubric_ratings_from_payload,
    mean_rubric_rating,
    mean_rubric_score,
    mean_sample_score,
    normalize_rating,
    objective_score,
    sample_vgif_score,
)


class ScoringTests(unittest.TestCase):
    def test_normalize_rating_endpoints(self) -> None:
        self.assertEqual(normalize_rating(1), 0.2)
        self.assertEqual(normalize_rating(5), 1.0)

    def test_four_dimension_reported_formula(self) -> None:
        ratings = {"Cin": 1, "Pur": 2, "Mot": 4, "Phy": 5}
        self.assertEqual(mean_rubric_rating(ratings), 3.0)
        self.assertEqual(mean_rubric_score(ratings), 0.6)

    def test_nested_score_payloads_are_supported(self) -> None:
        ratings = {key: {"score": 3} for key in ("Cin", "Pur", "Mot", "Phy")}
        self.assertEqual(mean_rubric_score(ratings), 0.6)

    def test_list_result_payload_ignores_legacy_rub_score(self) -> None:
        payload = {
            "average_score": 1,
            "results": [
                {"id": key, "score": 3}
                for key in ("Cin", "Pur", "Mot", "Phy")
            ] + [{"id": "Rub", "score": 1}],
        }
        ratings = extract_rubric_ratings_from_payload(payload)
        self.assertEqual(mean_rubric_score(ratings), 0.6)

    def test_scores_are_computed_per_sample_then_macro_averaged(self) -> None:
        objectives = [objective_score(1, 1), objective_score(1, 3)]
        subjectives = [0.2, 1.0]
        per_sample = [
            sample_vgif_score(objective, subjective)
            for objective, subjective in zip(objectives, subjectives)
        ]
        self.assertAlmostEqual(mean_sample_score(objectives), 2.0 / 3.0)
        self.assertAlmostEqual(mean_sample_score(subjectives), 0.6)
        self.assertAlmostEqual(mean_sample_score(per_sample), 19.0 / 30.0)

        micro_objective = (1 + 1) / (1 + 3)
        self.assertNotAlmostEqual(mean_sample_score(objectives), micro_objective)

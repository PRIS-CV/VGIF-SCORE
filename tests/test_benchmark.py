from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "code" / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from build_project_page_data import score_pair
from build_vgif_bench import DEFAULT_SOURCE, EXPECTED_COUNTS, load_entries, validate_entries


class BenchmarkTests(unittest.TestCase):
    def test_camera_ready_counts_and_schema(self) -> None:
        summary, errors = validate_entries(load_entries(DEFAULT_SOURCE))
        self.assertEqual(errors, [])
        for key, expected in EXPECTED_COUNTS.items():
            self.assertEqual(summary[key], expected)

    def test_sample_score_uses_normalized_four_dimension_mean(self) -> None:
        qa_payload = {"correct_count": 3, "question_count": 4}
        rubric_payload = {
            "results": [
                {"id": "Cin", "score": 5},
                {"id": "Pur", "score": 4},
                {"id": "Mot", "score": 3},
                {"id": "Phy", "score": 2},
            ]
        }
        objective, subjective, vgif = score_pair(qa_payload, rubric_payload)
        self.assertAlmostEqual(objective, 0.75)
        self.assertAlmostEqual(subjective, 14 / 20)
        self.assertAlmostEqual(vgif, (0.75 + 0.70) / 2)

    def test_public_case_library_covers_all_macro_domains(self) -> None:
        path = Path(__file__).resolve().parents[1] / "docs" / "data" / "case_study.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["cases"]), 8)
        self.assertEqual(sum(len(case["models"]) for case in payload["cases"]), 16)
        self.assertEqual(payload["model_count"], 8)
        self.assertEqual({case["id"] for case in payload["cases"]}, {
            "product", "narrative", "surreal", "physics",
            "emotion", "spatial", "performance", "nature",
        })
        for case in payload["cases"]:
            for model in case["models"]:
                self.assertAlmostEqual(
                    model["vgif"],
                    (model["objective"] + model["subjective"]) / 2,
                    delta=0.01,
                )

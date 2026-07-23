from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "code" / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    requests_stub = types.ModuleType("requests")
    requests_stub.Session = object
    requests_stub.Response = object
    sys.modules["requests"] = requests_stub

from evaluate_video_qa_accuracy import (
    build_autorubric_prompt,
    evaluate_dependency_expression,
    evaluate_predictions,
)


class DependencyTests(unittest.TestCase):
    def test_autorubric_prompt_requests_only_paper_dimensions(self) -> None:
        dimension = {
            "dimension_goal": "goal",
            "focus_points": ["focus"],
            "score_anchors": {str(value): "anchor" for value in range(1, 6)},
        }
        prompt = build_autorubric_prompt(
            {
                "prompt": "test prompt",
                "dimensions": {
                    "cinematography": dimension,
                    "purity": dimension,
                    "motion_smoothness": dimension,
                    "physics_adherence": dimension,
                },
            }
        )
        self.assertIn("exactly these four ids once each: Cin, Pur, Mot, Phy", prompt)
        self.assertNotIn("Rub (", prompt)

    def test_and_or_dependency_expression(self) -> None:
        values = {"q1": True, "q2": False, "q3": True}
        passed, failed = evaluate_dependency_expression(
            "q1 AND (q2 OR q3)", values.__getitem__
        )
        self.assertTrue(passed)
        self.assertEqual(failed, ["q2"])

    def test_failed_parent_short_circuits_child(self) -> None:
        qa_pairs = [
            {"id": "q1", "question": "parent", "type": "entity", "dependency": "None", "expected_answer": "Yes"},
            {"id": "q2", "question": "child", "type": "action", "dependency": "q1", "expected_answer": "Yes"},
        ]
        answers = {
            "q1": {"answer": "No", "reason": "missing"},
            "q2": {"answer": "Yes", "reason": "visible"},
        }
        rows, raw_correct, correct, blocked = evaluate_predictions(qa_pairs, answers)
        self.assertEqual(raw_correct, 1)
        self.assertEqual(correct, 0)
        self.assertEqual(blocked, 1)
        self.assertEqual(rows[1]["dependency_failed_ids"], ["q1"])

    def test_acyclic_forward_reference_is_evaluated_recursively(self) -> None:
        qa_pairs = [
            {"id": "q1", "question": "child", "type": "action", "dependency": "q2", "expected_answer": "Yes"},
            {"id": "q2", "question": "parent", "type": "entity", "dependency": "None", "expected_answer": "Yes"},
        ]
        answers = {
            "q1": {"answer": "Yes", "reason": "visible"},
            "q2": {"answer": "Yes", "reason": "visible"},
        }
        rows, _, correct, _ = evaluate_predictions(qa_pairs, answers)
        self.assertEqual(correct, 2)
        self.assertTrue(all(row["correct"] for row in rows))

    def test_dependency_cycle_is_rejected(self) -> None:
        qa_pairs = [
            {"id": "q1", "question": "one", "type": "action", "dependency": "q2", "expected_answer": "Yes"},
            {"id": "q2", "question": "two", "type": "action", "dependency": "q1", "expected_answer": "Yes"},
        ]
        answers = {
            "q1": {"answer": "Yes", "reason": "visible"},
            "q2": {"answer": "Yes", "reason": "visible"},
        }
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            evaluate_predictions(qa_pairs, answers)

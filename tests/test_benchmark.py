from __future__ import annotations

import sys
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "code" / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from build_vgif_bench import DEFAULT_SOURCE, EXPECTED_COUNTS, load_entries, validate_entries


class BenchmarkTests(unittest.TestCase):
    def test_camera_ready_counts_and_schema(self) -> None:
        summary, errors = validate_entries(load_entries(DEFAULT_SOURCE))
        self.assertEqual(errors, [])
        for key, expected in EXPECTED_COUNTS.items():
            self.assertEqual(summary[key], expected)

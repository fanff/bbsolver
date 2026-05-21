import unittest

import numpy as np

from buildXng_validators import (
    build_min_max_validator_2,
    make_minimum_top_count_validators,
)


def _cols(*rows: str) -> list[list[int]]:
    return [[ord(c) - ord("A") for c in row] for row in rows]


A, B, C = 0, 1, 2


class TestMinimumTopCountValidators(unittest.TestCase):
    def test_4wires_depth_1(self):
        bconfig = _cols("AC", "B")
        segments = make_minimum_top_count_validators(bconfig, wire_count=4, color_count=3)

        self.assertEqual(len(segments), 1)
        counter, start_idx, end_idx = segments[0]
        self.assertEqual((start_idx, end_idx), (0, 4))
        self.assertEqual(counter[A], 1)
        self.assertEqual(counter[C], 1)
        self.assertEqual(counter[B], 1)

    def test_6wires_depth_1(self):
        bconfig = _cols("ACA", "BA")
        segments = make_minimum_top_count_validators(bconfig, wire_count=6, color_count=3)

        self.assertEqual(len(segments), 2)

        counter_0, start_idx, end_idx = segments[0]
        self.assertEqual((start_idx, end_idx), (0, 4))
        self.assertEqual(counter_0[A], 1)
        self.assertEqual(counter_0[C], 1)
        self.assertEqual(counter_0[B], 1)

        counter_1, start_idx, end_idx = segments[1]
        self.assertEqual((start_idx, end_idx), (2, 6))
        self.assertEqual(counter_1[A], 1)
        self.assertEqual(counter_1[C], 1)
        self.assertEqual(counter_1.get(B, 0), 0)


class TestBuildMinMaxValidator2(unittest.TestCase):
    def test_returns_consistent_bounds(self):
        bconfig = _cols("ACA", "BA")
        fima = np.array([2, 2, 2], dtype=np.int8)

        lower, upper, missing = build_min_max_validator_2(
            bconfig, fima, color_count=3
        )

        self.assertEqual(missing.total(), 0)
        for seg, lb in lower.items():
            self.assertIn(seg, upper)
            self.assertTrue(np.all(upper[seg] >= lb))

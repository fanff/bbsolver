"""Tests for buildXng_validators (current integer-based API)."""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from buildXng_validators import (
    build_col_iterator,
    build_min_max_validator_2,
    counter_to_vec,
    make_minimum_top_count_validators,
)

from tests.helpers import A, B, C, cols, segment_by_range, segments_at_depth, wire_count_for


class TestMinimumTopCountValidators(unittest.TestCase):
    def test_4wires_depth_1(self):
        segments = segments_at_depth("AC", "B", depth=2)

        self.assertEqual(len(segments), 1)
        counter, start_idx, end_idx = segments[0]
        self.assertEqual((start_idx, end_idx), (0, 4))
        self.assertEqual(counter[A], 1)
        self.assertEqual(counter[C], 1)
        self.assertEqual(counter[B], 1)

    def test_6wires_depth_1(self):
        segments = segments_at_depth("ACA", "BA", depth=2)

        self.assertEqual(len(segments), 2)

        counter_0 = segment_by_range(segments, 0, 4)
        self.assertEqual(counter_0[A], 1)
        self.assertEqual(counter_0[C], 1)
        self.assertEqual(counter_0[B], 1)

        counter_1 = segment_by_range(segments, 2, 6)
        self.assertEqual(counter_1[A], 1)
        self.assertEqual(counter_1[C], 1)
        self.assertEqual(counter_1.get(B, 0), 0)

    def test_6wires_depth_1_reverse(self):
        segments = segments_at_depth("BA", "ACA", depth=2)

        self.assertEqual(len(segments), 3)
        self.assertEqual(segment_by_range(segments, 0, 3)[A], 1)
        self.assertEqual(segment_by_range(segments, 1, 5)[B], 1)
        self.assertEqual(segment_by_range(segments, 3, 6)[A], 1)

    def test_6wires_depth_2(self):
        segments = segments_at_depth("ACA", "BA", "ACA", depth=3)

        self.assertEqual(len(segments), 3)

        counter_0 = segment_by_range(segments, 0, 4)
        self.assertEqual(counter_0[A], 1)
        self.assertEqual(counter_0[B], 1)
        self.assertEqual(counter_0[C], 1)

        counter_1 = segment_by_range(segments, 0, 6)
        self.assertEqual(counter_1[A], 2)
        self.assertEqual(counter_1[B], 1)
        self.assertEqual(counter_1[C], 1)

        counter_2 = segment_by_range(segments, 2, 6)
        self.assertEqual(counter_2[A], 1)
        self.assertEqual(counter_2.get(B, 0), 0)
        self.assertEqual(counter_2[C], 1)

    def test_6wires_depth_4_segment_counts(self):
        bconfig_rows = ("ACA", "CA", "ACA", "CA", "ABA")
        wire_count = wire_count_for(*bconfig_rows)

        vd1 = make_minimum_top_count_validators(cols(*bconfig_rows[:2]), wire_count, 3)
        vd2 = make_minimum_top_count_validators(cols(*bconfig_rows[:3]), wire_count, 3)
        vd3 = make_minimum_top_count_validators(cols(*bconfig_rows[:4]), wire_count, 3)
        vd4 = make_minimum_top_count_validators(cols(*bconfig_rows[:5]), wire_count, 3)

        self.assertEqual(len(vd1), 2)
        self.assertEqual(len(vd2), 3)
        self.assertEqual(len(vd3), 2)
        self.assertEqual(len(vd4), 3)

    def test_8wires_depth_2(self):
        segments = segments_at_depth("ACAA", "BAA", "ACAA", depth=3)

        self.assertEqual(len(segments), 4)

        self.assertEqual(segment_by_range(segments, 0, 4)[A], 1)
        self.assertEqual(segment_by_range(segments, 0, 4)[B], 1)
        self.assertEqual(segment_by_range(segments, 0, 4)[C], 1)

        self.assertEqual(segment_by_range(segments, 0, 6)[A], 2)
        self.assertEqual(segment_by_range(segments, 0, 6)[C], 1)
        self.assertEqual(segment_by_range(segments, 0, 6)[B], 1)

        self.assertEqual(segment_by_range(segments, 2, 8)[A], 2)
        self.assertEqual(segment_by_range(segments, 2, 8)[C], 1)
        self.assertEqual(segment_by_range(segments, 2, 8).get(B, 0), 0)

        self.assertEqual(segment_by_range(segments, 4, 8)[A], 2)
        self.assertEqual(segment_by_range(segments, 4, 8).get(C, 0), 0)
        self.assertEqual(segment_by_range(segments, 4, 8).get(B, 0), 0)


class TestBuildMinMaxValidator2(unittest.TestCase):
    def test_returns_consistent_bounds(self):
        bconfig = cols("ACA", "BA")
        fima = np.array([2, 2, 2], dtype=np.int8)

        lower, upper, missing = build_min_max_validator_2(bconfig, fima, color_count=3)

        self.assertEqual(missing.total(), 0)
        for seg, lb in lower.items():
            self.assertIn(seg, upper)
            self.assertTrue(np.all(upper[seg] >= lb))

    def test_detects_missing_colors(self):
        bconfig = cols("ACA", "BA")
        # FIMA wire count matches design, but color A is under-supplied vs constraints
        fima = np.array([1, 2, 3], dtype=np.int8)

        _, _, missing = build_min_max_validator_2(bconfig, fima, color_count=3)

        self.assertGreater(missing.total(), 0)
        self.assertGreater(missing.get(A, 0), 0)


class TestBuildColIterator(unittest.TestCase):
    def test_yields_column_solutions(self):
        bconfig = cols("ACA", "BA")
        fima = np.array([2, 2, 2], dtype=np.int8)
        wire_count = int(fima.sum())

        lower, upper, missing = build_min_max_validator_2(bconfig, fima, color_count=3)
        self.assertEqual(missing.total(), 0)

        solutions = list(
            build_col_iterator(
                bconfig[0],
                lower,
                upper,
                None,
                None,
                wire_count=wire_count,
                color_count=3,
            )
        )

        self.assertGreater(len(solutions), 0)
        top, bottom, inner = solutions[0]
        self.assertEqual(len(top), wire_count)
        self.assertEqual(len(bottom), wire_count)


class TestCounterToVec(unittest.TestCase):
    def test_round_trip(self):
        c = Counter({A: 2, B: 1})
        vec = counter_to_vec(c, color_count=3)
        self.assertEqual(vec.tolist(), [2, 1, 0])

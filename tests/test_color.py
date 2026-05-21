"""Tests for color.py type factories."""

from __future__ import annotations

import unittest

from color import make_color_types


class TestColorTuple(unittest.TestCase):
    def test_equality_and_prefix(self):
        ColorEnum, ColorTuple, _ = make_color_types(tuple_len=10)

        ct1 = ColorTuple.from_colors([ColorEnum.C0] * 10)
        ct1_alias = ColorTuple.from_colors([ColorEnum(0)] * 10)
        ct2 = ColorTuple.from_colors([ColorEnum.C1] * 10)

        self.assertNotEqual(ct1, ct2)
        self.assertTrue(ct1.equal_prefix(ct2, prefix_len=0))
        self.assertFalse(ct1.equal_prefix(ct2, prefix_len=1))
        self.assertEqual(ct1, ct1_alias)


class TestColorCounter(unittest.TestCase):
    def test_from_tuple_counts(self):
        ColorEnum, ColorTuple, ColorCounter = make_color_types(tuple_len=10)

        ct1 = ColorTuple.from_colors([ColorEnum.C0] * 10)
        ct2 = ColorTuple.from_colors([ColorEnum.C1] * 10)

        self.assertEqual(ColorCounter.from_tuple(ct1).as_list()[0], 10)
        self.assertEqual(ColorCounter.from_tuple(ct2).as_list()[0], 0)
        self.assertEqual(ColorCounter.from_tuple(ct2).as_list()[1], 10)

    def test_window_is_leq_all_prepared(self):
        ColorEnum, ColorTuple, ColorCounter = make_color_types(
            tuple_len=50,
            num_colors=3,
        )

        cc1 = ColorCounter.zeros()
        cc1.inc(0, 2)

        cc2 = ColorCounter.zeros()
        cc2.inc(1, 1)

        window_set = ColorCounter.prepare_windows([(cc1, 0, 2), (cc2, 2, 4)])

        t = ColorTuple.from_values([0, 0, 1, 2] + [0] * (50 - 4))
        scratch = bytearray(ColorCounter.NUM_COLORS)

        self.assertTrue(
            ColorCounter.window_is_leq_all_prepared(t, window_set, scratch)
        )

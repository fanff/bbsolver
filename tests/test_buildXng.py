"""Tests for buildXng solver utilities."""

from __future__ import annotations

import unittest

import numpy as np

from buildXng import take_with_exhaustion, wire_generator


class TestTakeWithExhaustion(unittest.TestCase):
    def test_partial_batch_not_exhausted(self):
        it = iter(range(10))
        items, exhausted, remaining = take_with_exhaustion(it, n=3)

        self.assertEqual(items, [0, 1, 2])
        self.assertFalse(exhausted)
        self.assertEqual(list(remaining), list(range(3, 10)))

    def test_full_exhaustion(self):
        it = iter([1, 2])
        items, exhausted, remaining = take_with_exhaustion(it, n=10)

        self.assertEqual(items, [1, 2])
        self.assertTrue(exhausted)
        self.assertEqual(list(remaining), [])


class TestWireGenerator(unittest.TestCase):
    def test_multiset_permutation(self):
        fima = np.array([2, 1], dtype=int)
        gen = wire_generator(fima, as_list=True, rng=np.random.default_rng(0))
        sample = next(gen)

        self.assertEqual(len(sample), 3)
        self.assertEqual(sorted(sample), [0, 0, 1])

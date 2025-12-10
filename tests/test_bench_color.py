from __future__ import annotations

import pytest

from color import make_color_types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_TUPLES = 1_000_000
TUPLE_LEN = 30
NUM_COLORS = 3  # must be >= 2


def index_to_values(i: int) -> list[int]:
    """
    Deterministically map i -> a length-TUPLE_LEN list of base-NUM_COLORS digits.

    This guarantees uniqueness for i in [0, NUM_TUPLES) as long as
    NUM_TUPLES <= NUM_COLORS ** TUPLE_LEN.
    """
    vals = [0] * TUPLE_LEN
    for pos in range(TUPLE_LEN):
        vals[pos] = i % NUM_COLORS
        i //= NUM_COLORS
    return vals


# ---------------------------------------------------------------------------
# Fixtures: build types and dataset once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def color_types():
    """
    Create the ColorEnum and ColorTuple types once.
    """
    ColorEnum, ColorTuple = make_color_types(
        tuple_len=TUPLE_LEN,
        num_colors=NUM_COLORS,
        enum_name="BenchColorEnum",
        tuple_class_name="BenchColorTuple",
    )
    return ColorEnum, ColorTuple


@pytest.fixture(scope="session")
def color_dataset(color_types):
    """
    Build:
      - a list of NUM_TUPLES distinct ColorTuple instances
      - a set of those tuples
      - one element that is in the set
      - one element that is NOT in the set
    """
    _ColorEnum, ColorTuple = color_types

    # Build the list of distinct tuples
    tuples = []
    for i in range(NUM_TUPLES):
        vals = index_to_values(i)
        ct = ColorTuple.from_values(vals)
        tuples.append(ct)

    s = set(tuples)

    # Pick one existing element from the middle
    existing = tuples[len(tuples) // 2]

    # Build a missing element: index NUM_TUPLES (never inserted), then tweak a digit
    missing_vals = index_to_values(NUM_TUPLES)
    missing_vals[0] = (missing_vals[0] + 1) % NUM_COLORS
    missing = ColorTuple.from_values(missing_vals)

    # Sanity checks to avoid benchmarking garbage
    assert existing in s
    assert missing not in s

    return {
        "set": s,
        "existing": existing,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

N_CHECKS = 1_000_000


def test_membership_present(benchmark, color_dataset):
    """
    Benchmark membership checks where the element IS in the set.
    """
    s = color_dataset["set"]
    x = color_dataset["existing"]

    def run():
        for _ in range(N_CHECKS):
            if x not in s:  # sanity; should never happen
                raise RuntimeError("existing element not found in set")

    benchmark(run)


def test_membership_missing(benchmark, color_dataset):
    """
    Benchmark membership checks where the element IS NOT in the set.
    """
    s = color_dataset["set"]
    x = color_dataset["missing"]

    def run():
        for _ in range(N_CHECKS):
            if x in s:  # sanity; should never happen
                raise RuntimeError("missing element found in set")

    benchmark(run)

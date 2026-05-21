"""Shared helpers for bbsolver unit tests."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from buildXng_validators import make_minimum_top_count_validators

# Color indices used in string-based configs (A=0, B=1, C=2, ...)
A, B, C = 0, 1, 2


def cols(*rows: str) -> list[list[int]]:
    """Convert column strings like 'ACA' to lists of color indices."""
    return [[ord(c) - ord("A") for c in row] for row in rows]


def wire_count_for(*rows: str) -> int:
    """Wire count for a normal (non-small) column of max node width."""
    return max(len(row) for row in rows) * 2


def segments_at_depth(
    *rows: str,
    depth: int | None = None,
    color_count: int = 3,
) -> list[tuple[Counter, int, int]]:
    """
    Minimum top-count segments for the first ``depth`` columns.

    If depth is None, use all provided rows.
    """
    bconfig = cols(*rows)
    if depth is not None:
        bconfig = bconfig[:depth]
    wire_count = wire_count_for(*rows)
    return make_minimum_top_count_validators(
        bconfig, wire_count=wire_count, color_count=color_count
    )


def segment_by_range(
    segment_list: Iterable[tuple[Counter, int, int]],
    start: int,
    end: int,
) -> Counter:
    """Pick the counter for wire segment (start, end), or empty Counter if missing."""
    for counter, s, e in segment_list:
        if s == start and e == end:
            return counter
    return Counter()

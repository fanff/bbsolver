from __future__ import annotations

from enum import Enum
from typing import (
    Generic,
    Sequence,
    Type,
    TypeVar,
    Tuple,
)


E = TypeVar("E", bound=Enum)


def int_to_color_str(i: int) -> str:
    # Simple mapping for demonstration; adjust as needed
    return {
        0: "red",
        1: "blue",
        2: "green",
        3: "yellow",
        4: "magenta",
        5: "cyan",
        7: "white",
    }[int(i)]


class ColorTupleBase(Generic[E]):
    """
    Generic base for a fixed-length tuple of colors, stored as bytes.

    Concrete subclasses are created via `make_color_types`, which binds:
      - a specific ColorEnum (subclass of Enum)
      - a specific tuple length (TUPLE_LEN)
    """

    __slots__ = ("_data",)

    # To be set on each concrete subclass
    ColorEnum: Type[E]
    TUPLE_LEN: int

    def __init__(self, data: bytes):
        if len(data) != self.TUPLE_LEN:
            raise ValueError(f"Expected length {self.TUPLE_LEN}, got {len(data)}")
        self._data = data

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_colors(cls, colors: Sequence[E]) -> ColorTupleBase[E]:
        if len(colors) != cls.TUPLE_LEN:
            raise ValueError(f"Expected length {cls.TUPLE_LEN}, got {len(colors)}")

        # Optional safety check; remove if you want max speed
        for c in colors:
            if not isinstance(c, cls.ColorEnum):
                raise TypeError("Wrong enum type in ColorTupleBase.from_colors")

        data = bytes(c.value for c in colors)
        return cls(data)

    @classmethod
    def from_values(cls, values: Sequence[int]) -> ColorTupleBase[E]:
        if len(values) < cls.TUPLE_LEN:
            # pad with zeros
            values = list(values) + [0] * (cls.TUPLE_LEN - len(values))
        elif len(values) > cls.TUPLE_LEN:
            raise ValueError(f"Expected length {cls.TUPLE_LEN}, got {len(values)}")

        if any(not (0 <= v <= 255) for v in values):
            raise ValueError("Color values must be in 0..255 (1 byte each)")

        return cls(bytes(values))

    # ------------------------------------------------------------------
    # Introspection / decoding
    # ------------------------------------------------------------------
    def to_colors(self) -> list[E]:
        # Assumes enum values are 0..N-1 and match the order of iteration
        members = list(self.ColorEnum)
        return [members[b] for b in self._data]

    def values(self) -> bytes:
        return self._data

    # ------------------------------------------------------------------
    # Comparison / hashing / prefix-compare
    # ------------------------------------------------------------------
    def equal_prefix(self, other: ColorTupleBase[E], prefix_len: int) -> bool:
        if not isinstance(other, ColorTupleBase):
            return False
        if prefix_len > self.TUPLE_LEN:
            raise ValueError("prefix_len larger than tuple length")
        if prefix_len == 0:
            return True
        return self._data[:prefix_len] == other._data[:prefix_len]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColorTupleBase):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        return hash(self._data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @classmethod
    def all_color_ints(cls):
        return range(cls.ColorEnum.__len__())

    def __getitem__(self, idx: int) -> E:
        if not (0 <= idx < self.TUPLE_LEN):
            raise IndexError(idx)
        b = self._data[idx]
        members = list(self.ColorEnum)
        try:
            return members[b]
        except IndexError as exc:
            raise ValueError(f"No ColorEnum member for value {b}") from exc

    def __len__(self) -> int:
        return self.TUPLE_LEN

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self._data)})"


def _make_default_color_names(num_colors: int) -> list[str]:
    return [f"C{i}" for i in range(num_colors)]


class WindowSet(Generic[E]):
    """
    Preprocessed representation of many (counter, start, end) triples.

    This stores the minimal wire count requirements for some portion of the column.
    (Oldy known as AllowedTopsValidator)

    This is not created directly; use 'ColorCounter.prepare_windows()'.

    - c_data_list: list of underlying counter bytearrays (live views)
    - windows: bytes of length 2 * n_windows:
        windows[2*i]   = start_i
        windows[2*i+1] = end_i
    """

    __slots__ = ("c_data_list", "windows")

    def __init__(
        self,
        c_data_list: list[bytearray],
        windows: bytes,
    ):
        self.c_data_list = c_data_list
        self.windows = windows


# ----------------------------------------------------------------------
# ColorCounter
# ----------------------------------------------------------------------


class ColorCounterBase(Generic[E]):
    """
    Counts how many times each color occurs.

    - Internally: bytearray of length NUM_COLORS
      (each count fits in 0..255; your use case is < 50).
    - Concrete subclass is bound to:
        - a ColorEnum (ColorEnum)
        - number of colors (NUM_COLORS)
    """

    __slots__ = ("_data",)

    ColorEnum: Type[E]
    NUM_COLORS: int

    def __init__(self, data: bytearray):
        if len(data) != self.NUM_COLORS:
            raise ValueError(f"Expected {self.NUM_COLORS} counts, got {len(data)}")
        self._data = data

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def zeros(cls) -> ColorCounterBase[E]:
        """All counts = 0."""
        return cls(bytearray(cls.NUM_COLORS))

    @classmethod
    def from_tuple(cls, t: ColorTupleBase[E]) -> ColorCounterBase[E]:
        """Count all colors in the given ColorTuple."""
        counts = bytearray(cls.NUM_COLORS)
        vals = t.values()  # bytes, each 0..NUM_COLORS-1
        for b in vals:
            counts[b] += 1
        return cls(counts)

    @classmethod
    def from_tuple_window(
        cls,
        t: ColorTupleBase[E],
        start: int,
        end: int,
    ) -> ColorCounterBase[E]:
        """
        Count colors only in t[start:end].
        """
        if not (0 <= start <= end <= len(t)):
            raise ValueError("Invalid window (start, end)")
        counts = bytearray(cls.NUM_COLORS)
        vals = t.values()[start:end]
        for b in vals:
            counts[b] += 1
        return cls(counts)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: ColorCounterBase[E]) -> ColorCounterBase[E]:
        if not isinstance(other, ColorCounterBase):
            return NotImplemented
        if self.NUM_COLORS != other.NUM_COLORS:
            raise ValueError("Mismatched NUM_COLORS in ColorCounterBase.__add__")
        data = bytearray(self.NUM_COLORS)
        for i in range(self.NUM_COLORS):
            data[i] = self._data[i] + other._data[i]
        return self.__class__(data)

    def __iadd__(self, other: ColorCounterBase[E]) -> ColorCounterBase[E]:
        if not isinstance(other, ColorCounterBase):
            return NotImplemented
        if self.NUM_COLORS != other.NUM_COLORS:
            raise ValueError("Mismatched NUM_COLORS in ColorCounterBase.__iadd__")
        for i in range(self.NUM_COLORS):
            self._data[i] += other._data[i]
        return self

    # ------------------------------------------------------------------
    # Comparisons / checks
    # ------------------------------------------------------------------
    def is_leq(self, other: ColorCounterBase[E]) -> bool:
        """
        Check self <= other elementwise:
        return False if *any* color has self[c] > other[c].
        """
        if not isinstance(other, ColorCounterBase):
            return False
        if self.NUM_COLORS != other.NUM_COLORS:
            return False
        for i in range(self.NUM_COLORS):
            if self._data[i] > other._data[i]:
                return False
        return True

    # ------------------------------------------------------------------
    # Mutations on a specific color index (byte key)
    # ------------------------------------------------------------------
    def inc(self, color_index: int, delta: int = 1) -> None:
        """Add delta (default +1) to the count of color_index."""
        if not (0 <= color_index < self.NUM_COLORS):
            raise IndexError(color_index)
        self._data[color_index] += delta  # assumes no overflow >255

    def dec(self, color_index: int, delta: int = 1) -> None:
        """Subtract delta (default -1) from the count of color_index."""
        if not (0 <= color_index < self.NUM_COLORS):
            raise IndexError(color_index)
        self._data[color_index] -= delta  # assumes count stays >= 0

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------
    def total(self) -> int:
        """Sum of all counts."""
        return sum(self._data)

    def argmin(self) -> int:
        """
        Return the color index (0..NUM_COLORS-1) with the smallest count.
        """
        best_idx = 0
        best_val = self._data[0]
        for i in range(1, self.NUM_COLORS):
            v = self._data[i]
            if v < best_val:
                best_val = v
                best_idx = i
        return best_idx

    def min_color(self) -> E:
        """Return the ColorEnum member with the smallest count."""
        idx = self.argmin()
        members = list(self.ColorEnum)
        return members[idx]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def as_list(self) -> list[int]:
        return list(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self._data)})"

    @classmethod
    def prepare_windows(
        cls,
        counters: Sequence[Tuple["ColorCounterBase[E]", int, int]],
    ) -> WindowSet[E]:
        """
        Preprocess a sequence of (counter, start, end) into a WindowSet.

        Call this once and reuse the WindowSet in the `window_is_leq_all` method.
        """
        if not counters:
            return WindowSet([], b"")

        c_data_list: list[bytearray] = []
        # packed as [start0, end0, start1, end1, ...]
        windows_bytes = bytearray(2 * len(counters))

        for idx, (c, start, end) in enumerate(counters):
            if not isinstance(c, ColorCounterBase):
                raise TypeError("counters must contain ColorCounterBase instances")
            if c.NUM_COLORS != cls.NUM_COLORS:
                raise ValueError("Mismatched NUM_COLORS in counters")
            if not (0 <= start <= end <= 255):
                raise ValueError("start/end must be in 0..255 (and start <= end)")

            c_data_list.append(c._data)  # type: ignore[attr-defined]
            windows_bytes[2 * idx] = start
            windows_bytes[2 * idx + 1] = end

        # freeze windows as immutable bytes
        return WindowSet(c_data_list, bytes(windows_bytes))

    # ------------------------------------------------------------------
    # Fused operation: window counting + compare to many counters
    # ------------------------------------------------------------------
    @classmethod
    def window_is_leq_all_prepared(
        cls,
        t: ColorTupleBase[E],
        window_set: WindowSet[E],
        scratch: bytearray,
    ) -> bool:
        """
        Use a preprocessed WindowSet for maximum performance.

        For each window i:
          - Count colors in t[start_i:end_i] into `scratch`
          - Check scratch >= c_data_list[i] elementwise

        Return False on first violation, otherwise True.

        No allocations here, assuming `scratch` is preallocated.
        """
        data = t.values()  # bytes
        c_data_list = window_set.c_data_list
        windows = window_set.windows

        n_windows = len(c_data_list)
        if len(scratch) != cls.NUM_COLORS:
            raise ValueError("scratch has wrong length")

        for idx_window in range(n_windows):
            start = windows[2 * idx_window]
            end = windows[2 * idx_window + 1]

            # optional safety; if you trust prepare_windows, you can drop this:
            # if not (0 <= start <= end <= len(t)):
            #     raise ValueError("Invalid window (start, end)")

            # zero scratch
            for i in range(cls.NUM_COLORS):
                scratch[i] = 0

            # count t[start:end]
            for pos in range(start, end):
                color_index = data[pos]  # 0..NUM_COLORS-1
                scratch[color_index] += 1

            # compare with this counter
            c_data = c_data_list[idx_window]
            for i in range(cls.NUM_COLORS):
                if scratch[i] < c_data[i]:
                    return False

        return True


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------
def _make_default_color_names(num_colors: int) -> list[str]:
    return [f"C{i}" for i in range(num_colors)]


def make_color_types(
    tuple_len: int,
    *,
    num_colors: int = 16,
    enum_name: str = "ColorEnum",
    tuple_class_name: str = "ColorTuple",
    counter_class_name: str = "ColorCounter",
) -> Tuple[Type[Enum], Type[ColorTupleBase[Enum]], Type[ColorCounterBase[Enum]]]:
    """
    Create ColorEnum, ColorTuple, ColorCounter types wired together.

    Usage:
        ColorEnum, ColorTuple, ColorCounter = make_color_types(tuple_len=10)

    - ColorEnum: Enum with values 0..num_colors-1
    - ColorTuple: fixed-length tuple of bytes (len = tuple_len)
    - ColorCounter: vector of counts (len = num_colors)
    """
    if tuple_len <= 0:
        raise ValueError("tuple_len must be positive")
    if not (1 <= num_colors <= 256):
        raise ValueError("num_colors must be between 1 and 256")

    # Enum
    names = _make_default_color_names(num_colors)
    mapping = {name: i for i, name in enumerate(names)}
    ColorEnum = Enum(enum_name, mapping)

    # ColorTuple subclass
    class ColorTuple(ColorTupleBase):  # type: ignore[type-arg]
        pass

    ColorTuple.__name__ = tuple_class_name
    ColorTuple.ColorEnum = ColorEnum  # type: ignore[assignment]
    ColorTuple.TUPLE_LEN = tuple_len  # type: ignore[assignment]

    # ColorCounter subclass
    class ColorCounter(ColorCounterBase):  # type: ignore[type-arg]
        pass

    ColorCounter.__name__ = counter_class_name
    ColorCounter.ColorEnum = ColorEnum  # type: ignore[assignment]
    ColorCounter.NUM_COLORS = num_colors  # type: ignore[assignment]

    return ColorEnum, ColorTuple, ColorCounter

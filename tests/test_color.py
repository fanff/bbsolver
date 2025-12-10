# world_config.py (for example)
import unittest
from color import make_color_types


class TestColor(unittest.TestCase):

    def test_same_color(self):
        ColorEnum, ColorTuple, _ = make_color_types(tuple_len=10)

        # Now use them everywhere:
        ct1 = ColorTuple.from_colors([ColorEnum.C0] * 10)
        ct1_1 = ColorTuple.from_colors([ColorEnum(0)] * 10)
        ct2 = ColorTuple.from_colors([ColorEnum.C1] * 10)

        assert ct1 != ct2
        assert ct1.equal_prefix(ct2, prefix_len=0)  # True
        assert not ct1.equal_prefix(ct2, prefix_len=1)

        assert ct1 == ct1_1


class TestColorCounter(unittest.TestCase):

    def test_same_color(self):
        ColorEnum, ColorTuple, ColorCounter = make_color_types(tuple_len=10)

        # Now use them everywhere:
        ct1 = ColorTuple.from_colors([ColorEnum.C0] * 10)
        ct1_1 = ColorTuple.from_colors([ColorEnum(0)] * 10)
        ct2 = ColorTuple.from_colors([ColorEnum.C1] * 10)

        cc = ColorCounter.from_tuple(ct1)
        assert cc.as_list()[0] == 10

        cc = ColorCounter.from_tuple(ct1_1)
        assert cc.as_list()[0] == 10

        cc = ColorCounter.from_tuple(ct2)
        assert cc.as_list()[0] == 0
        assert cc.as_list()[1] == 10

    def test_allowed_tops_validators(self):

        ColorEnum, ColorTuple, ColorCounter = make_color_types(
            tuple_len=50,
            num_colors=3,
        )

        cc1 = ColorCounter.zeros()
        cc1.inc(0, 2)  # color 0 has count 1

        cc2 = ColorCounter.zeros()
        cc2.inc(1, 1)  # color 1 has count 1 between 2 and 4

        raw_counters = [
            (cc1, 0, 2),
            (cc2, 2, 4),
        ]

        window_set = ColorCounter.prepare_windows(raw_counters)

        t = ColorTuple.from_values([0, 0, 1, 2] + [0] * (50 - 4))
        # now we have the color 0 at the beginning
        # color 1 next then color 2

        scratch = bytearray(ColorCounter.NUM_COLORS)
        assert ColorCounter.window_is_leq_all_prepared(
            t,
            window_set,
            scratch,
        )

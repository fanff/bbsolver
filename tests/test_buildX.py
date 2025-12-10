import unittest
from buildX import ColorEnum, quad_node_types


class TestQuadNode_1(unittest.TestCase):

    def test_same_color(self):

        allowed_tops = set()

        allowed_tops.add(
            (ColorEnum.from_str("A"), ColorEnum.from_str("A")),
        )
        res = quad_node_types(ColorEnum.from_str("A"), allowed_tops)

        self.assertEqual(len(res), 1)

    def test_half_same(self):

        allowed_tops = set()
        allowed_tops.add(
            (ColorEnum.from_str("A"), ColorEnum.from_str("B")),
        )
        res = quad_node_types(ColorEnum.from_str("A"), allowed_tops)

        self.assertEqual(len(res), 2)

        # now revers A and B
        allowed_tops = set()
        allowed_tops.add(
            (ColorEnum.from_str("B"), ColorEnum.from_str("A")),
        )
        res = quad_node_types(ColorEnum.from_str("A"), allowed_tops)

        self.assertEqual(len(res), 2)

    def test_different_colors(self):
        allowed_tops = set()
        allowed_tops.add(
            (ColorEnum.from_str("B"), ColorEnum.from_str("C")),
        )
        res = quad_node_types(ColorEnum.from_str("A"), allowed_tops)

        self.assertEqual(len(res), 0)


class TestQuadNode_2(unittest.TestCase):

    def test_same_color(self):

        allowed_tops = set()

        allowed_tops.add(
            (ColorEnum.from_str("A"), ColorEnum.from_str("A")),  # 1 solution
        )
        allowed_tops.add(
            (ColorEnum.from_str("A"), ColorEnum.from_str("B")),  # 2 solutions,
        )
        res = quad_node_types(ColorEnum.from_str("A"), allowed_tops)

        self.assertEqual(len(res), 3)

    def test_half_same(self):

        allowed_tops = set()
        allowed_tops.add(
            (ColorEnum.from_str("A"), ColorEnum.from_str("B")),  # 2 solutions
        )
        allowed_tops.add(
            (ColorEnum.from_str("B"), ColorEnum.from_str("A")),  # 2 solutions,
        )
        res = quad_node_types(ColorEnum.from_str("A"), allowed_tops)

        self.assertEqual(len(res), 4)

        # now revers A and B
        allowed_tops = set()
        allowed_tops.add(
            (ColorEnum.from_str("B"), ColorEnum.from_str("A")),
        )
        res = quad_node_types(ColorEnum.from_str("A"), allowed_tops)

        self.assertEqual(len(res), 2)

    def test_different_colors(self):
        allowed_tops = set()
        allowed_tops.add(
            (ColorEnum.from_str("B"), ColorEnum.from_str("C")),
        )
        allowed_tops.add(
            (ColorEnum.from_str("C"), ColorEnum.from_str("B")),
        )
        allowed_tops.add(
            (ColorEnum.from_str("B"), ColorEnum.from_str("B")),
        )
        allowed_tops.add(
            (ColorEnum.from_str("C"), ColorEnum.from_str("C")),
        )
        res = quad_node_types(ColorEnum.from_str("A"), allowed_tops)

        self.assertEqual(len(res), 0)


class TestMakeTopValidator(unittest.TestCase):
    def test_4wires_depth_1(self):
        from buildX import (
            make_minimum_top_count_validator,
            ColorEnum,
            AllowedTopsValidator,
        )

        bconfig = ["AC", "B"]

        v = make_minimum_top_count_validator(bconfig)

        self.assertTrue(isinstance(v, AllowedTopsValidator))
        self.assertEqual(
            len(v.segment_counters), 1
        )  # one segment because one node in the last layer
        counter, start_idx, end_idx = v.segment_counters[0]
        self.assertEqual(start_idx, 0)
        self.assertEqual(
            end_idx, 4
        )  # (0,+4) because we roll back from 1 column, so we encounter 2 nodes, and each have 2 input color;

        # those two are the minimals for the layer 0
        self.assertEqual(counter[ColorEnum.from_str("A")], 1)
        self.assertEqual(counter[ColorEnum.from_str("C")], 1)
        # this one the layer 1 minimal requirement
        self.assertEqual(counter[ColorEnum.from_str("B")], 1)

    def test_6wires_depth_1(self):
        from buildX import (
            make_minimum_top_count_validator,
            ColorEnum,
            AllowedTopsValidator,
        )

        bconfig = ["ACA", "BA"]

        v = make_minimum_top_count_validator(bconfig)

        self.assertTrue(isinstance(v, AllowedTopsValidator))
        self.assertEqual(
            len(v.segment_counters), 2
        )  # Two segments because two nodes in the last layer

        # checking first Counter
        counter_0, start_idx, end_idx = v.segment_counters[0]
        self.assertEqual(start_idx, 0)
        self.assertEqual(
            end_idx, 4
        )  # (0,+4) because we roll back from 1 column, so we encounter 2 nodes, and each have 2 input color;

        # those two are the minimals for the layer 0 'AC' start
        self.assertEqual(counter_0[ColorEnum.from_str("A")], 1)
        self.assertEqual(counter_0[ColorEnum.from_str("C")], 1)
        # this one the layer 1 minimal requirement for 'B'
        self.assertEqual(counter_0[ColorEnum.from_str("B")], 1)

        # checking second Counter
        counter_1, start_idx, end_idx = v.segment_counters[1]
        self.assertEqual(start_idx, 2)
        self.assertEqual(
            end_idx, 6
        )  # (2,+4) because we roll back from 1 column, so we encounter 2 nodes, and each have 2 input color;

        # those two are the minimals for the layer 0 'CA'
        self.assertEqual(counter_1[ColorEnum.from_str("A")], 1)  # necessary for layer 0
        self.assertEqual(counter_1[ColorEnum.from_str("C")], 1)  # necessary for layer 0
        # check there is no 'B' count in this segment
        self.assertEqual(counter_1.get(ColorEnum.from_str("B"), 0), 0)

    def test_6wires_depth_1_reverse(self):
        from buildX import (
            make_minimum_top_count_validator,
            ColorEnum,
            AllowedTopsValidator,
        )

        bconfig = ["BA", "ACA"]

        v = make_minimum_top_count_validator(bconfig)

        self.assertTrue(isinstance(v, AllowedTopsValidator))
        self.assertEqual(len(v.segment_counters), 3)

        # checking first Counter
        counter_0, start_idx, end_idx = v.segment_counters[0]
        self.assertEqual(start_idx, 0)
        self.assertEqual(
            end_idx, 4
        )  # (0,+4) because we roll back from 1 column, so we encounter 2 nodes, and each have 2 input color;

    def test_6wires_depth_2(self):
        from buildX import (
            make_minimum_top_count_validator,
            ColorEnum,
            AllowedTopsValidator,
        )

        bconfig = ["ACA", "BA", "ACA"]
        # A C A
        #  B A
        # A C A

        v = make_minimum_top_count_validator(bconfig)

        self.assertTrue(isinstance(v, AllowedTopsValidator))
        self.assertEqual(len(v.segment_counters), 3)

        # checking first Counter
        counter_0, start_idx, end_idx = v.segment_counters[0]
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 4)  # not 6 because we are on the border

        self.assertEqual(counter_0[ColorEnum.from_str("A")], 1)
        self.assertEqual(counter_0[ColorEnum.from_str("B")], 1)
        self.assertEqual(counter_0[ColorEnum.from_str("C")], 1)

        # checking second Counter
        counter_1, start_idx, end_idx = v.segment_counters[1]
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 6)  #

        self.assertEqual(counter_1[ColorEnum.from_str("A")], 2)
        self.assertEqual(counter_1[ColorEnum.from_str("B")], 1)
        self.assertEqual(counter_1[ColorEnum.from_str("C")], 1)

        # checking last Counter
        counter_2, start_idx, end_idx = v.segment_counters[2]
        self.assertEqual(start_idx, 2)
        self.assertEqual(end_idx, 6)  #

        self.assertEqual(counter_2[ColorEnum.from_str("A")], 1)
        self.assertEqual(counter_2[ColorEnum.from_str("C")], 1)

    def test_6wires_depth_4(self):
        from buildX import (
            make_minimum_top_count_validator,
            ColorEnum,
            AllowedTopsValidator,
        )

        bconfig = ["ACA", "CA", "ACA", "CA", "ABA"]
        # A C A
        #  C A    # d1
        # A C A   # d2
        #  C A    # d3
        # A B A   # d4

        vd1 = make_minimum_top_count_validator(bconfig[:2])
        vd2 = make_minimum_top_count_validator(bconfig[:3])
        vd3 = make_minimum_top_count_validator(bconfig[:4])
        vd4 = make_minimum_top_count_validator(bconfig[:5])

        self.assertEqual(len(vd1.segment_counters), 2)
        self.assertEqual(len(vd2.segment_counters), 3)
        self.assertEqual(len(vd3.segment_counters), 2)
        self.assertEqual(len(vd4.segment_counters), 3)

        # index the segments :
        segment_index = {}
        for vd in [vd1, vd2, vd3, vd4]:
            for counter, start_idx, end_idx in vd.segment_counters:
                seg = (start_idx, end_idx)
                if seg in segment_index:
                    segment_index[seg] = counter.__or__(segment_index[seg])
                else:
                    segment_index[seg] = counter
        for k, v in segment_index.items():
            start_idx, end_idx = k
            pass

    def test_8wires_depth_2(self):
        from buildX import (
            make_minimum_top_count_validator,
            ColorEnum,
            AllowedTopsValidator,
        )

        bconfig = ["ACAA", "BAA", "ACAA"]
        # A C A A
        #  B A A
        # A C A A

        v = make_minimum_top_count_validator(bconfig)

        self.assertTrue(isinstance(v, AllowedTopsValidator))
        self.assertEqual(
            len(v.segment_counters), 4
        )  # 4 segments because 4 nodes in the last layer

        # checking first Counter
        counter_0, start_idx, end_idx = v.segment_counters[0]
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 4)  # not 6 because we are on the border

        self.assertEqual(counter_0[ColorEnum.from_str("A")], 1)
        self.assertEqual(counter_0[ColorEnum.from_str("B")], 1)
        self.assertEqual(counter_0[ColorEnum.from_str("C")], 1)

        # checking second Counter
        counter_1, start_idx, end_idx = v.segment_counters[1]
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 6)  #
        # those two are the minimals for the layer 0 'CA'
        self.assertEqual(counter_1[ColorEnum.from_str("A")], 2)
        self.assertEqual(counter_1[ColorEnum.from_str("C")], 1)
        self.assertEqual(counter_1[ColorEnum.from_str("B")], 1)

        # checking third Counter
        counter_2, start_idx, end_idx = v.segment_counters[2]
        self.assertEqual(start_idx, 2)
        self.assertEqual(end_idx, 8)  #
        # those two are the minimals for the layer 0 'CA'
        self.assertEqual(counter_2[ColorEnum.from_str("A")], 2)
        self.assertEqual(counter_2[ColorEnum.from_str("C")], 1)
        self.assertEqual(counter_2.get(ColorEnum.from_str("B"), 0), 0)

        # checking last Counter
        counter_2, start_idx, end_idx = v.segment_counters[3]
        self.assertEqual(start_idx, 4)
        self.assertEqual(end_idx, 8)  #
        self.assertEqual(counter_2[ColorEnum.from_str("A")], 2)
        self.assertEqual(counter_2.get(ColorEnum.from_str("C"), 0), 0)
        self.assertEqual(counter_2.get(ColorEnum.from_str("B"), 0), 0)

    def test_8wires_depth_10(self):
        from buildX import (
            make_minimum_top_count_validator,
            ColorEnum,
            AllowedTopsValidator,
        )
        from rich.console import Console

        console = Console(force_interactive=False)
        bconfig = [
            "BCAA",
            "BAA",
            "ACAA",
            "BAA",
            "ACAB",
            "BAA",
            "ACAA",
            "BAA",
            "ACAA",
            "BAA",
            "ACAA",
            "BAA",
        ]
        # B C A A
        #  B A A
        # A C A A
        #  B A A
        # A C A B
        validators = [
            make_minimum_top_count_validator(bconfig[: i + 2]) for i in range(8)
        ]
        # index the segments :
        segment_index = {}
        for vd in validators:
            for counter, start_idx, end_idx in vd.segment_counters:
                seg = (start_idx, end_idx)
                if seg in segment_index:
                    segment_index[seg] = counter.__or__(segment_index[seg])
                else:
                    segment_index[seg] = counter

        for _ in range(10):
            for k, c1 in segment_index.items():
                start_idx, end_idx = k
                if start_idx == 0 and end_idx != 8:
                    for k2, c2 in segment_index.items():
                        if end_idx == k2[0]:

                            k2_end_idx = k2[1]
                            add = (c1 + c2) | segment_index[(0, k2_end_idx)]
                            segment_index[(0, k2_end_idx)] = add
        # now check segment index
        segment_index
        # now the 0, 8 range is A:3, B1 C1
        # assuming we discover it should be B2 elsewhere ...
        segment_index[(0, 8)][ColorEnum.from_str("B")] = 2

        # minimum assortmen viable
        mina = segment_index[(0, 8)]
        from rich.text import Text

        def counter_to_rich(c):
            atoms = []
            for col in ColorEnum:
                v = c.get(col, 0)
                color = col.to_color()
                atoms.append(f"[{color}]**{v}**[/{color}]")
            return Text("".join(atoms))

        console.print("Minimum assortment:", mina, counter_to_rich(mina))
        segment_index

        # console.file.flush()


class Test_Validator_validate(unittest.TestCase):
    def test_6_w_d2(self):
        from buildX import make_minimum_top_count_validator, ColorEnum

        bconfig = ["ACA", "CA", "ACB"]
        # A C A
        #  C A
        # A C B

        v = make_minimum_top_count_validator(bconfig)

        tops = [ColorEnum.from_str(c) for c in "ACABAC"]
        self.assertTrue(v.validate(tops))
        tops = [ColorEnum.from_str(c) for c in "ACBCAC"]
        self.assertTrue(v.validate(tops))

        # B is too far to be used in the last layer
        tops = [ColorEnum.from_str(c) for c in "BCACAC"]
        self.assertFalse(v.validate(tops))
        tops = [ColorEnum.from_str(c) for c in "CBACAC"]
        self.assertFalse(v.validate(tops))

    def test_6_w_d4(self):
        from buildX import make_minimum_top_count_validator, ColorEnum

        bconfig = [
            "ACA",
            "CA",
            "ACB",
            "CA",
            "BCA",
        ]
        # A C A
        #  C A
        # A C B
        #  C A
        # B C A

        v = make_minimum_top_count_validator(bconfig)

        tops = [
            ColorEnum.from_str(c) for c in "ACABAC"
        ]  # this is technically not OK, but the validator is at depth
        self.assertTrue(v.validate(tops))
        tops = [ColorEnum.from_str(c) for c in "ACBCAC"]
        self.assertTrue(v.validate(tops))
        tops = [ColorEnum.from_str(c) for c in "ACBCAC"]
        self.assertTrue(v.validate(tops))
        tops = [ColorEnum.from_str(c) for c in "ACBCAC"]
        self.assertTrue(v.validate(tops))

        # B is not presented
        tops = [ColorEnum.from_str(c) for c in "ACACAC"]
        self.assertFalse(v.validate(tops))

    def test_6_w_d0_d4(self):
        from buildX import make_minimum_top_count_validator, ColorEnum

        bconfig = [
            "ACA",
            "CA",
            "ACA",
            "CA",
            "BCA",
        ]
        # A C A
        #  C A
        # A C A
        #  C A
        # B C A

        v0 = make_minimum_top_count_validator(bconfig[:2])
        v4 = make_minimum_top_count_validator(bconfig)
        tops = [
            ColorEnum.from_str(c) for c in "ACABAC"
        ]  # this is technically not OK, but the validator is at depth
        self.assertTrue(v4.validate(tops))

        tops = [
            ColorEnum.from_str(c) for c in "ACABAC"
        ]  # this is technically not OK, but the validator is at depth
        self.assertFalse(v0.validate(tops))

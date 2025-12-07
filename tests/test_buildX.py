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
        from buildX import make_top_validator, ColorEnum, AllowedTopsValidator

        bconfig = ["AC", "B"]

        v = make_top_validator(bconfig)

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
        from buildX import make_top_validator, ColorEnum, AllowedTopsValidator

        bconfig = ["ACA", "BA"]

        v = make_top_validator(bconfig)

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

    def test_6wires_depth_2(self):
        from buildX import make_top_validator, ColorEnum, AllowedTopsValidator

        bconfig = ["ACA", "BA", "ACA"]
        # A C A
        #  B A
        # A C A

        v = make_top_validator(bconfig)

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

    def test_8wires_depth_2(self):
        from buildX import make_top_validator, ColorEnum, AllowedTopsValidator

        bconfig = ["ACAA", "BAA", "ACAA"]
        # A C A A
        #  B A A
        # A C A A

        v = make_top_validator(bconfig)

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


class Test_Validator_validate(unittest.TestCase):
    def test_6_w_d2(self):
        from buildX import make_top_validator, ColorEnum

        bconfig = ["ACA", "CA", "ACB"]
        # A C A
        #  C A
        # A C B

        v = make_top_validator(bconfig)

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
        from buildX import make_top_validator, ColorEnum

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

        v = make_top_validator(bconfig)

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
        from buildX import make_top_validator, ColorEnum

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

        v0 = make_top_validator(bconfig[:2])
        v4 = make_top_validator(bconfig)
        tops = [
            ColorEnum.from_str(c) for c in "ACABAC"
        ]  # this is technically not OK, but the validator is at depth
        self.assertTrue(v4.validate(tops))

        tops = [
            ColorEnum.from_str(c) for c in "ACABAC"
        ]  # this is technically not OK, but the validator is at depth
        self.assertFalse(v0.validate(tops))

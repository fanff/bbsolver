from collections import Counter
from dataclasses import dataclass
import itertools
from typing import Dict, Iterator, List, Set, Tuple, TypeVar
from itertools import product
from enum import Enum, auto
from rich.text import Text
from rich.console import Console


T = TypeVar("T")


def take_with_exhaustion(it: Iterator[T], n=1000) -> tuple[List[T], bool, Iterator[T]]:
    # Take up to n items
    items = list(itertools.islice(it, n))

    # Check if iterator is exhausted by trying one more element
    try:
        peek = next(it)
        exhausted = False
    except StopIteration:
        exhausted = True
    else:
        # Put the peeked value back by creating a new iterator
        it = itertools.chain([peek], it)

    return items, exhausted, it


class ColorEnum(Enum):
    A = auto()
    B = auto()
    C = auto()

    def to_color(self) -> str:
        if self == ColorEnum.A:
            return "red"
        elif self == ColorEnum.B:
            return "blue"
        elif self == ColorEnum.C:
            return "green"

    @staticmethod
    def from_str(s: str):
        if s == "A":
            return ColorEnum.A
        elif s == "B":
            return ColorEnum.B
        elif s == "C":
            return ColorEnum.C


@dataclass
class NodeColor:
    TL: ColorEnum
    TR: ColorEnum
    BL: ColorEnum
    BR: ColorEnum
    I: ColorEnum

    def __hash__(self):
        return hash((self.TL, self.TR, self.BL, self.BR, self.I))


class Assortment(dict):
    """A dictionary subclass representing an assortment of ColorEnum items with positive counts.

    For example, Assortment({ColorEnum.A: 2, ColorEnum.B: 3}) indicates 2 items of type A and 3 items of type B.
    Only positive counts are stored; zero or negative counts are omitted.


    """

    def __init__(self, input_dict=None):
        super().__init__({k: v for k, v in input_dict.items() if v > 0})

    def __eq__(self, other):
        if not isinstance(other, dict):
            return False
        if len(self) != len(other):
            return False
        if self.keys() != other.keys():
            return False

        for k in self.keys():
            if self[k] != other[k]:
                return False
        return True


def quad_node_types(
    input: ColorEnum, allowed_tops: Set[Tuple[ColorEnum, ColorEnum]] = None
) -> Set[NodeColor]:
    """Build all possible NodeColor configurations for a given input color."""
    enum_cls = type(input)
    values = list(enum_cls)
    results = set()
    if allowed_tops is not None:
        for TL, TR in allowed_tops:
            if input not in (TL, TR):
                continue
            results.add(NodeColor(TL=TL, TR=TR, BL=TL, BR=TR, I=input))
            if TL != TR:
                results.add(NodeColor(TL=TL, TR=TR, BL=TR, BR=TL, I=input))
    else:
        for TL in values:
            for TR in values:
                # Top must contain I
                if input not in (TL, TR):
                    continue

                # Two possible bottom patterns: same or swapped
                results.add(NodeColor(TL=TL, TR=TR, BL=TL, BR=TR, I=input))
                if TL != TR:
                    results.add(NodeColor(TL=TL, TR=TR, BL=TR, BR=TL, I=input))

    return results


@dataclass
class Bcol:
    """Represents a column of nodes with top (T), bottom (B) colors for a Column."""

    T: Tuple[ColorEnum, ...]
    B: Tuple[ColorEnum, ...]
    # B and T have the same length

    # for reference we keep I
    I: Tuple[ColorEnum, ...]

    # this is import as its indicate how to validate the column against assortment
    is_small_col: bool = False

    def __to_rich__(self) -> Text:
        if self.is_small_col:
            return self.__rich__small()
        else:
            return self.__rich__normal()

    def __rich__small(self) -> Text:
        txt = ""
        for w_idx, c in enumerate(self.T):
            color = c.to_color()

            if w_idx == 0:
                txt += f"[{color}]|[/{color}] "
            elif w_idx == len(self.T) - 1:
                txt += f"[{color}]|[/{color}] "
            else:
                txt += f"[{color}]|[/{color}] "
        txt += "\n   "
        for w_idx, c in enumerate(self.I):
            color = c.to_color()
            txt += f"[{color}]▅[/{color}]   "
        txt += "\n"
        for w_idx, c in enumerate(self.B):
            color = c.to_color()
            if w_idx == 0:
                txt += f"[{color}]|[/{color}] "
            elif w_idx == len(self.B) - 1:
                txt += f"[{color}]|[/{color}] "
            else:
                txt += f"[{color}]|[/{color}] "
        return Text.from_markup(f"{txt}")

    def __rich__normal(self) -> Text:
        txt = ""
        for w_idx, c in enumerate(self.T):
            color = c.to_color()

            if w_idx % 2 == 0:
                txt += f"[{color}]\\\\[/{color}] "
            else:
                txt += f"[{color}]/[/{color}] "
        txt += "\n"
        for w_idx, c in enumerate(self.I):
            color = c.to_color()
            txt += f" [{color}]▅[/{color}]  "
        txt += "\n"
        for w_idx, c in enumerate(self.B):
            color = c.to_color()
            if w_idx % 2 == 0:
                txt += f"[{color}]/[/{color}] "
            else:
                txt += f"[{color}]\\\\[/{color}] "
        return Text.from_markup(f"{txt}")


def search_column(
    nodes_per_pos: List[List[NodeColor]],
    assortment: Assortment,
    is_small_col=False,
    allowed_tops: Set[Tuple[ColorEnum, ...]] = None,
    extra_lr_pairs: Set[Tuple[ColorEnum, ColorEnum]] = None,
) -> Iterator[Bcol]:

    L = len(nodes_per_pos)
    target = Counter(assortment)  # target counts per color
    allowed_tops_partials = None
    # building partial allowed tops to prune search
    if allowed_tops is not None:
        if is_small_col:
            allowed_tops_partials = {}
            for node_idx in range(len(nodes_per_pos)):
                allowed_tops_partials[node_idx] = set(
                    (
                        tuple(
                            [allowed_top[0], allowed_top[-1]]
                            + list(allowed_top[1 : node_idx * 2 + 3]),
                        )
                        for allowed_top in allowed_tops
                    )
                )
        else:
            allowed_tops_partials = {}
            for node_idx in range(len(nodes_per_pos)):
                allowed_tops_partials[node_idx] = set(
                    (allowed_top[: node_idx * 2 + 2] for allowed_top in allowed_tops)
                )

    def backtrack(
        pos, top_counts, bottom_counts, T_seq, B_seq, I_seq
    ) -> Iterator[Bcol]:
        if pos == L:
            # Finished all nodes; check exact match
            if (Counter(T_seq) == target) and (
                Counter(B_seq) == target
            ):  # this if can be useless because we did check the < at the previous backtrack
                if is_small_col:
                    # add extra wires at beginning and end
                    T_seq = tuple(T_seq[:1] + T_seq[2:] + T_seq[1:2])
                    B_seq = tuple(B_seq[:1] + B_seq[2:] + B_seq[1:2])
                else:
                    T_seq = tuple(T_seq)
                    B_seq = tuple(B_seq)

                if allowed_tops is not None and T_seq not in allowed_tops:
                    return  # prune tops that are not allowed
                yield Bcol(
                    T=T_seq,
                    B=B_seq,
                    I=tuple(I_seq),
                    is_small_col=is_small_col,
                )

                return
            else:
                raise ValueError("Should not happen")

        # we are not finished yet building the column
        for node in nodes_per_pos[pos]:
            # Contribution of this node
            t_add = [node.TL, node.TR]
            b_add = [node.BL, node.BR]

            if allowed_tops_partials is not None:
                # in can validate the partial tops i am about to add
                partials = allowed_tops_partials[pos]
                candidate_top = tuple(T_seq + t_add)
                if not is_small_col:
                    pass
                if candidate_top not in partials:
                    continue  # prune
            else:
                # Quick upper bound check against the assortment
                valid = True
                for c in ColorEnum:
                    if top_counts[c] + t_add.count(c) > target[c]:
                        valid = False
                        break
                    if bottom_counts[c] + b_add.count(c) > target[c]:
                        valid = False
                        break
                if not valid:
                    continue  # prune

            # Extend
            new_top_counts = top_counts.copy()
            new_bottom_counts = bottom_counts.copy()
            for c in t_add:
                new_top_counts[c] += 1
            for c in b_add:
                new_bottom_counts[c] += 1

            yield from backtrack(
                pos + 1,
                new_top_counts,
                new_bottom_counts,
                T_seq + t_add,
                B_seq + b_add,
                I_seq + [node.I],
            )

    # start recursion
    if is_small_col:

        for left_pad, right_pad in extra_lr_pairs:
            top_c = {c: 0 for c in ColorEnum}
            top_c[left_pad] += 1
            top_c[right_pad] += 1
            top_c = Counter(top_c)

            bottom_c = {c: 0 for c in ColorEnum}
            bottom_c[left_pad] += 1
            bottom_c[right_pad] += 1
            bottom_c = Counter(bottom_c)

            valid = True
            for c in ColorEnum:
                if top_c[c] > target[c]:
                    valid = False
                    break
                if bottom_c[c] > target[c]:
                    valid = False
                    break
            if not valid:
                continue  # prune
            yield from backtrack(
                pos=0,
                top_counts=top_c,
                bottom_counts=bottom_c,
                T_seq=[left_pad, right_pad],
                B_seq=[left_pad, right_pad],
                I_seq=[],
            )
    else:
        yield from backtrack(
            pos=0,
            top_counts=Counter({c: 0 for c in ColorEnum}),
            bottom_counts=Counter({c: 0 for c in ColorEnum}),
            T_seq=[],
            B_seq=[],
            I_seq=[],
        )


def make_all_possible_solution(
    col_config: str,
    assortment: Assortment,
    is_small_col=False,
    allowed_tops: Set[Tuple[ColorEnum, ...]] = None,
) -> Iterator[Bcol]:

    if allowed_tops is None:

        # we will craft an allowed top that has big probability of working
        allowed_tops_by_node = [set() for _ in range(len(col_config))]
        for node_id, k in enumerate(col_config):
            for c in ColorEnum:
                allowed_tops_by_node[node_id].add((ColorEnum.from_str(k), c))

        nodes = [
            quad_node_types(ColorEnum.from_str(k), allowed_top)
            for k, allowed_top in zip(col_config, allowed_tops_by_node)
        ]
        return search_column(nodes, assortment, is_small_col=is_small_col)
    else:
        if is_small_col:
            allowed_tops_by_node = [set() for _ in range(len(col_config))]
            extra_lr_pairs = set()
            for allowed_top in allowed_tops:
                for node_id in range(0, len(col_config)):
                    allowed_tops_by_node[node_id].add(
                        (allowed_top[node_id * 2 + 1], allowed_top[node_id * 2 + 2])
                    )
                    extra_lr_pairs.add(
                        (allowed_top[0], allowed_top[-1])
                    )  # the extreme left & right values are extra constraints

            nodes = [
                quad_node_types(ColorEnum.from_str(k), allowed_top)
                for k, allowed_top in zip(col_config, allowed_tops_by_node)
            ]
            # all nodes must have at least one option, if any is empty, then no solution possible
            for node_options in nodes:
                if len(node_options) == 0:
                    return iter([])
            return search_column(
                nodes,
                assortment,
                is_small_col=is_small_col,
                allowed_tops=allowed_tops,
                extra_lr_pairs=extra_lr_pairs,
            )
        else:
            # every allowed top is splitted into list of couples , because theyr are L R L R L R L R

            allowed_tops_by_node = [set() for _ in range(len(col_config))]
            for allowed_top in allowed_tops:
                for node_id in range(0, len(col_config)):
                    allowed_tops_by_node[node_id].add(
                        (allowed_top[node_id * 2], allowed_top[node_id * 2 + 1])
                    )

            nodes = [
                quad_node_types(ColorEnum.from_str(k), allowed_top)
                for k, allowed_top in zip(col_config, allowed_tops_by_node)
            ]
            # all nodes must have at least one option, if any is empty, then no solution possible
            for allowed_top in allowed_tops_by_node:
                if len(allowed_top) == 0:
                    return iter([])
            return search_column(
                nodes, assortment, is_small_col=is_small_col, allowed_tops=allowed_tops
            )


def make_assortment_from_col_configs(col_configs: List[str]) -> Assortment:
    wire_count = len(col_configs[0]) * 2
    current_assortment = {k: 0 for k in ColorEnum}
    for row in col_configs:
        total_counts = dict(Counter([ColorEnum.from_str(k) for k in row]))
        for k, v in total_counts.items():
            current_assortment[k] = max(current_assortment[k], v)
        assortment_wc = sum(current_assortment.values())
        if assortment_wc > wire_count:
            raise ValueError(
                f"Row config {col_configs} exceeds wire count {wire_count}."
            )
    asso = Assortment(current_assortment)
    missing_wires = wire_count - sum(asso.values())
    if missing_wires > 0:
        # assign missing wires iteratively to the color with least count
        for _ in range(missing_wires):
            least_color = min(asso, key=asso.get)
            asso[least_color] += 1
    return asso


# structure to hold the allowed_tops validator
# its contains a list of N "Counters" associated with a segment of a column (between start and end)
# each counter indicate how many time each color must appear at least in this segment,
# this is a minimal requirement to be able to build a column with this allowed_tops
#
# It provides a method to validate if a given allowed_top tuple is compatible with this allowed tops validator
#
# It is created using the col_configs of a couple of columns (the one we define the validator for, followed
# by the next M layers, M depends on the width of the braclet, the more width, the more depth),
# we calculate and the counters, and then create the AllowedTopsValidator, see function make_top_validator
class AllowedTopsValidator:
    segment_counters: List[Tuple[Counter[ColorEnum, int], int, int]] = []

    def __init__(
        self, segment_counters: List[Tuple[Counter[ColorEnum, int], int, int]] = None
    ):
        self.segment_counters = segment_counters

    def validate(self, allowed_top: Tuple[ColorEnum, ...]):
        for segment_counter, start_idx, end_idx in self.segment_counters:
            # extract the segment from allowed_top
            segment = allowed_top[start_idx:end_idx]
            segment_count = Counter(segment)
            for color, count in segment_counter.items():
                if segment_count[color] < count:
                    return False
        return True


def make_minimum_top_count_validator(
    col_configs: List[str], wire_count=None
) -> AllowedTopsValidator:
    if len(col_configs) < 2:
        if wire_count is None:
            raise ValueError(
                "At least two column configs are needed to build a validator."
            )
        else:
            col = col_configs[0]
            if len(col) * 2 == wire_count:  # normal column
                sc = []
                for cidx, c in enumerate(col):
                    sc.append(
                        (Counter({ColorEnum.from_str(c): 1}), cidx * 2, cidx * 2 + 2)
                    )
                return AllowedTopsValidator(sc)
            elif len(col) * 2 + 2 == wire_count:  # small column
                sc = []
                for cidx, c in enumerate(col):
                    sc.append(
                        (
                            Counter({ColorEnum.from_str(c): 1}),
                            cidx * 2 + 1,
                            cidx * 2 + 3,
                        )
                    )
                return AllowedTopsValidator(sc)
            else:
                raise ValueError(
                    f"Wrong wire count. {wire_count} not matching column length {len(col)}."
                )

    small_col_size = min(len(col_configs[0]), len(col_configs[1]))
    # we determine this because it does change the segment indexing.
    if len(col_configs[0]) > len(col_configs[1]):
        is_small_col = False
        node_count = len(col_configs[0])
        wire_count = node_count * 2

        max_depth = wire_count - 1
    else:
        is_small_col = True
        node_count = len(col_configs[0])
        wire_count = (node_count + 1) * 2

        max_depth = wire_count - 1

    if len(col_configs) < max_depth + 1:
        max_depth = len(col_configs) - 1
    segment_counters = []
    # we start at the max_depth layer
    max_depth_config = col_configs[max_depth]
    for node_idx, c in enumerate(col_configs[max_depth]):
        is_small_col = len(max_depth_config) == small_col_size

        dot_color = ColorEnum.from_str(c)
        # current_counter for this node
        counter = Counter({k: 0 if k != dot_color else 1 for k in ColorEnum})

        if is_small_col:
            start_idx = node_idx * 2 + 1
            end_idx = start_idx + 2
        else:
            start_idx = node_idx * 2
            end_idx = start_idx + 2
        # now iteratively jump to the previous layer
        for depth in range(max_depth - 1, -1, -1):
            col_config_at_depth = col_configs[depth]
            is_small_col = len(col_config_at_depth) == small_col_size

            if is_small_col:
                color_peek_start = max(0, start_idx // 2 - 1)
                color_peek_end = (end_idx) // 2
                count_for_depth = Counter(
                    col_config_at_depth[color_peek_start:color_peek_end]
                )
                for c, v in count_for_depth.items():
                    dot_color = ColorEnum.from_str(c)
                    if counter[dot_color] < v:
                        counter[dot_color] = v
                # update start_idx and end_idx for next depth
                start_idx = start_idx - 1
                end_idx = end_idx + 1
            else:
                color_peek_start = max(0, (start_idx - 1) // 2)
                color_peek_end = (end_idx // 2) + 1
                count_for_depth = Counter(
                    col_config_at_depth[color_peek_start:color_peek_end]
                )
                for c, v in count_for_depth.items():
                    dot_color = ColorEnum.from_str(c)
                    if counter[dot_color] < v:
                        counter[dot_color] = v
                # update start_idx and end_idx for next depth
                start_idx = start_idx - 1
                end_idx = end_idx + 1
            start_idx = max(0, start_idx)
            end_idx = min(wire_count, end_idx)
        segment_counters.append((counter, start_idx, end_idx))
    return AllowedTopsValidator(segment_counters)


if __name__ == "__main__":

    columns = ["AA", "A", "AB", "B"]
    columns = ["AB", "A", "AA", "A", "BB"]
    columns = ["AAB", "AB", "AAB", "AB"]  # 3 nodes

    columns = ["BABB", "ABA", "AABA", "AAA", "AAAA"]  # 4 nodes
    columns = ["BABBA", "ABAA", "AABAA", "AAAA", "AAAAA"]  # 5 nodes
    columns = ["BABBAC", "ABAAC", "AABAAC", "AAAAC", "AAAAAC"]  # 6 nodes
    columns = ["CABBBAC", "CABBAC", "CAABAAC", "CAAAAC", "BCAAACB"]  # 7 nodes
    columns = ["CCBABABCC", "CCAAAACC", "CCAABAACC", "CCAAAACC", "BCAAAAACB"]  # 9 nodes
    columns = ["CCBABBACCB", "CCABAACCB", "CCAABAACCB", "CCABAACCB"]  # 10 nodes

    columns = [
        "ACCBABBACCBAA",
        "ACCABAACCBAA",
        "ACCAABAACCBAA",
        "ACCABAACCBAA",
        "ACCABAACCBAAA",
    ]  # 13

    columns = [
        "ACCBABBACCBAACC",
        "ACCABAACCBAACC",
        "ACCAABAACCBAACC",
        "ACCABAACCBAACC",
        "ACCABAACCBAAACC",
        "BCCABAACCBAACB",
    ]  # 15

    columns = [
        "ABCABBBCBBBBACCABACC",
        "ABCABAACAAAACCBAACC",
        "ABCABAACAAAAACCBAACC",
        "ABCABAACAAAACCBAACC",
        "ABCABAACAAAAACCBAAAC",
        "ABCABAACAAAACCBAACC",
        "ABCABAACAAAAACCBAACC",
        "ABCABAACAAAACCBAACC",
        "ABCABAACAAAACCBAAACC",
    ]  # 20
    # columns = [
    #    "BBBCACBBB",
    #    "AABCACAA",
    #    "AABCACBAA",
    #    "ABACACBA",
    #    "ABACACABA",
    #    "BAACACAB",
    #    "BAACACAAB",
    #    "BAACACAB",
    #    "ABACACABA",
    #    "ABACACBA",
    #    "AABCACBAA",
    #    "AABCACAA",
    #    "BBBCACBBB",
    # ]  #

    columns = [
        "CCBABBACCBAA",
        "CCABAACCBAA",
        "CCAABAACCBAA",
        "CCABAACCBAA",
        "CCABAACCBAAA",
    ]  # 12
    columns = [
        "CCBABACCBACCB",
        "CCABAACCACCB",
        
    ]*30
    # 0. determine assortment from column configurations, this ensure the
    # assortment is compatible with all columns and there is the minimum number of wires
    init_wireing = tuple(
        [
            ColorEnum.from_str(c)
            for c in ("".join((l + r for l, r in zip(columns[0], columns[1] + "A"))))
        ]
    )
    assortment = Assortment(dict(Counter(init_wireing)))
    init_wireing = None
    assortment = make_assortment_from_col_configs(columns)
    wire_count = sum(assortment.values())
    console = Console()

    console.print("Assortment:", assortment)
    # I am now building a Top Validator from the column configurations

    top_validators: List[AllowedTopsValidator] = []
    for col_idx in range(len(columns) - 1):
        v = []
        for depth in range(2, wire_count - 1, 2):
            v += make_minimum_top_count_validator(
                columns[col_idx : col_idx + depth + 1]
            ).segment_counters

        top_validators.append(AllowedTopsValidator(v))
    top_validators.append(None)  # last column has no validator

    # 1. building a global "possible" solution by unfolding a product of all
    # the possible element within the columns
    # 2. filtering across layers to keep only consistent solutions
    # essentially the B of layer N must match the T of layer N+1
    # solution is updated accordingly at each step to give
    # a usable feedback to the "search column" function

    # containing the current possible solution for each columns
    assortment_solutions: List[List[Bcol]] = [[] for _ in columns]
    # contain the bound between two layers, inserted with False, it becomes True when the next layer has evaluated this bound.
    bound_check: List[Dict[Tuple[ColorEnum, ...], bool]] = [{} for _ in columns]

    # columns iterators of potential Bcol solution for each column
    columns_iterators: List[Iterator[Bcol] | None] = [None for _ in columns]
    # contain the current evaluated head for each column iterator , (essentially a set containing ONE element )
    iterators_head: List[Set[Tuple[ColorEnum, ...]] | None] = [None for _ in columns]

    # columns iterators of potential Bcol solution for each column
    columns_fullyExhausted: List[bool] = [False for _ in columns]
    solution_found = False

    col_idx = 0  # we start at columns_idx
    while not solution_found:
        # iterate over the columns
        # for col_idx, col_config in enumerate(columns):

        col_config = columns[col_idx]
        is_small_col = col_idx % 2 == 1

        console.print(f"Column {col_idx} ")

        column_iterator = columns_iterators[col_idx]
        if (column_iterator is None) or (column_iterator is True):
            if col_idx > 0:
                prev_bounds = bound_check[col_idx - 1]
                allowed_tops = set((k for k, v in prev_bounds.items() if not v))

            else:
                allowed_tops = None if init_wireing is None else set([init_wireing])
            # the allowed tops can be filtered more by looking at the next layer requirements.
            # An allowed top is also made so that it also almost fit the next layer col_config requirements.
            # more precisely, every color of the layer N+1 col_config must be present in this color allowed tops
            # and located "close" to the node.
            top_validatr = top_validators[col_idx]
            filtered_allowed_tops = set()
            if top_validatr is not None and allowed_tops is not None:

                for k in allowed_tops:
                    if top_validatr.validate(k):
                        filtered_allowed_tops.add(k)
                    else:
                        bound_check[col_idx - 1][k] = True  # mark as evaluated

                allowed_tops = filtered_allowed_tops

            console.print(f" done filtering")
            if col_idx > 0 and len(allowed_tops) == 0:
                # no possible tops from previous layer
                console.print(f" No TOPS,  can move next")
                if columns_fullyExhausted[col_idx - 1]:
                    columns_fullyExhausted[col_idx] = True
                    console.print(f" Parent fully exhausted, move next")
                    col_idx += 1
                    if col_idx >= len(columns):
                        console.print(f" Finished last columns")
                        break
                else:
                    console.print(f" Parent not exhausted, move previous")
                    col_idx -= 1
                continue
            if col_idx == 0 and column_iterator is True:
                console.print(f"Column {col_idx} can't remake iterator")
                columns_fullyExhausted[col_idx] = True
                col_idx += 1
                continue
            console.print(f" New Iterator (From {column_iterator})")
            iterators_head[col_idx] = allowed_tops
            column_iterator = make_all_possible_solution(
                col_config,
                assortment,
                is_small_col=is_small_col,
                allowed_tops=allowed_tops,
            )

        else:
            # continue with existing iterator
            console.print(f" Continue current iterator...")

        # pull next N elements from the iterator
        current_head = iterators_head[col_idx]
        current_head_size = len(current_head) if current_head is not None else "NoSize"
        console.print(f" with head size:", current_head_size, " elements")
        all_possible, exhausted, column_iterator = take_with_exhaustion(
            column_iterator, n=10000
        )

        if exhausted:
            console.print(f" exhausted with ", len(all_possible), " elements")
            columns_iterators[col_idx] = True
            # getting the current head of the iterator
            current_iter_head = iterators_head[col_idx]
            if current_iter_head is not None:

                # mark all bounds check with previous columns as evaluated
                for current_allowed_top in current_iter_head:
                    bound_check[col_idx - 1][
                        current_allowed_top
                    ] = True  # mark as evaluated

                eval_tops = set((b for b, v in bound_check[col_idx - 1].items() if v))
                # if all_possible is empty, then we can ask the parent columns to forget about the solutions that led to this
                if len(all_possible) == 0 and col_idx > 0:
                    before = len(assortment_solutions[col_idx - 1])

                    assortment_solutions[col_idx - 1] = [
                        bcol
                        for bcol in assortment_solutions[col_idx - 1]
                        if bcol.B not in eval_tops
                    ]

                    after_len = len(assortment_solutions[col_idx - 1])
                    console.print(
                        f" no possible solution, cleaning parent solutions from {before} to {after_len} elements. will move to previous"
                    )

                    col_idx -= 1
                    continue  # go back to previous column

        else:
            console.print(f" not exhausted with ", len(all_possible), " elements")
            # there will be more elements to pull later
            columns_iterators[col_idx] = column_iterator

        # getting the status of every bound for this columns
        column_bounds = bound_check[col_idx]

        added_seeds_for_next_layer = 0
        for bb in set(
            (bcol.B for bcol in all_possible)
        ):  # contain possible Tops for next Layer (Bottoms for my new all possible values)
            if (
                bb not in column_bounds
            ):  # if not yet present in the bound, add it as not evaluated
                bound_check[col_idx][bb] = False
                added_seeds_for_next_layer += 1
        console.print(f" added seeds for next layer:", added_seeds_for_next_layer)
        if added_seeds_for_next_layer == 0:
            console.print(f"  no new seeds for next layer, will move previous layer")
            col_idx -= 1
            continue
        # store all possible solutions for this column
        assortment_solutions[col_idx] = assortment_solutions[col_idx] + all_possible

        current_solution_found = len(assortment_solutions[col_idx])

        if (current_solution_found > 0) and (col_idx == len(columns) - 1):
            solution_found = True

        console.print(
            f" over with total:",
            current_solution_found,
            " elements, moving to next column",
        )
        col_idx += 1
        if col_idx >= len(columns):
            console.print(" Finished last columns")
            break

    if not solution_found:
        console.print("No solution found :(")
        quit(1)
    import pickle

    with open("assortment_solutions.pkl", "wb") as f:
        pickle.dump(assortment_solutions, f)

    # 3. printing sexy results with rich lib...

    picked_bcol = [None for _ in columns]

    # we will be picking backwards, (like a backward search)
    current_col = len(columns) - 1

    def find_solution_up_to_columns(
        col_idx, target_top: Tuple[ColorEnum, ...]
    ) -> List[Bcol]:
        if col_idx == 0:
            for bcol in assortment_solutions[col_idx]:
                if bcol.B == target_top:
                    return [bcol]
            console.print("No solution found at col 0")
            return None
        else:
            for bcol in assortment_solutions[col_idx]:
                if target_top is not None and bcol.B != target_top:
                    continue
                prev_solution = find_solution_up_to_columns(col_idx - 1, bcol.T)
                if prev_solution is not None:
                    return prev_solution + [bcol]
            # console.print("No solution found at col", col_idx)
            return None

    sols = find_solution_up_to_columns(len(columns) - 1, None)
    for s in sols:
        console.print(s.__to_rich__())
    quit(0)
    prev_bcol = None
    for idx, layer in reversed(list(enumerate(assortment_solutions))):
        if idx == len(assortment_solutions) - 1:
            bcol = layer[0]  # pick first solution
            console.print(bcol.__to_rich__() + Text(f"  total={len(layer)}"))
            prev_bcol = bcol
        else:
            foundsolution_here = False
            for bcol in layer:
                if bcol.B == prev_bcol.T:
                    console.print(bcol.__to_rich__() + Text(f"  total={len(layer)}"))
                    prev_bcol = bcol
                    foundsolution_here = True
                    break
            if not foundsolution_here:
                print("no solution found :()")

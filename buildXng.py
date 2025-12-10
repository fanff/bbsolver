from collections import Counter
from dataclasses import dataclass
import itertools
from typing import Dict, Generator, Iterator, List, Set, Tuple, TypeVar
from itertools import product
from enum import Enum, auto
import numpy as np
from rich.text import Text
from rich.console import Console

from buildXng_validators import (
    build_col_iterator,
    build_min_max_validator,
    build_min_max_validator_2,
    counter_to_rich,
    counter_to_vec,
)
from color import ColorCounterBase, ColorTupleBase, int_to_color_str, make_color_types


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


AnyColorType = TypeVar("AnyColorType")
AnyColumnColors = TypeVar("AnyColumnColors")
AnyNodesColorTuple = TypeVar("AnyNodesColorTuple")
ColorCounterType = TypeVar("ColorCounterType")


# TODO DROP
@dataclass
class NodeColor:
    TL: AnyColorType
    TR: AnyColorType
    BL: AnyColorType
    BR: AnyColorType
    I: AnyColorType

    def __hash__(self):
        return hash((self.TL, self.TR, self.BL, self.BR, self.I))


# TODO DROP
def quad_node_types(
    input: AnyColorType, allowed_tops: Set[Tuple[AnyColorType, AnyColorType]] = None
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

    T: AnyColumnColors
    B: AnyColumnColors
    # B and T have the same length

    # for reference we keep I
    I: AnyNodesColorTuple

    # this is important as it indicates how to validate the column against assortment
    is_small_col: bool = False

    def __to_rich__(self) -> Text:
        if self.is_small_col:
            return self.__rich__small()
        else:
            return self.__rich__normal()

    def __rich__small(self) -> Text:
        txt = ""
        for w_idx, c in enumerate(self.T):
            color = int_to_color_str(c)

            if w_idx == 0:
                txt += f"[{color}]|[/{color}] "
            elif w_idx == len(self.T) - 1:
                txt += f"[{color}]|[/{color}] "
            else:
                txt += f"[{color}]|[/{color}] "
        txt += "\n   "
        for w_idx, c in enumerate(self.I):
            color = int_to_color_str(c)
            txt += f"[{color}]▅[/{color}]   "
        txt += "\n"
        for w_idx, c in enumerate(self.B):
            color = int_to_color_str(c)
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
            color = int_to_color_str(c)

            if w_idx % 2 == 0:
                txt += f"[{color}]" + "\\\\" + f"[/{color}] "
            else:
                txt += f"[{color}]/[/{color}] "
        txt += "\n"
        for w_idx, c in enumerate(self.I):
            color = int_to_color_str(c)
            txt += f" [{color}]▅[/{color}]  "
        txt += "\n"
        for w_idx, c in enumerate(self.B):
            color = int_to_color_str(c)
            if w_idx % 2 == 0:
                txt += f"[{color}]/[/{color}] "
            else:
                txt += f"[{color}]" + "\\\\" + f"[/{color}] "
        return Text.from_markup(f"{txt}")


def make_bcol_iterator(
    unchecked_bounds: Set[ColorTupleBase],
    colors_required: List[int],
    top_min_validators: Dict[Tuple[int, int], np.ndarray],
    top_max_validators: Dict[Tuple[int, int], np.ndarray],
    bottom_min_validators: Dict[Tuple[int, int], np.ndarray],
    bottom_max_validators: Dict[Tuple[int, int], np.ndarray],
    wire_count: int,
    color_count,
) -> Iterator[Bcol]:
    is_small_col = len(colors_required) * 2 + 2 == wire_count

    if unchecked_bounds is None:
        allowed_tops = None
    else:
        # unchecked_bounds gives another constraint on T of the column during the build.
        if is_small_col:
            # the constraints are evaluated iteratively 0--3,0--5,0--7...
            # so, for the set of unchecked_bounds, we build sets of partial tops
            allowed_tops: Dict[Tuple[int, int], Set[bytes]] = {}
            for wire_max in range(3, wire_count + 1, 2):
                partial_tops = set()
                for bound in unchecked_bounds:
                    partial_tops.add(bound[:wire_max])
                allowed_tops[(0, wire_max)] = partial_tops
            # if wire_count == 8 , we have partials for 0-3,0-5,0-7
            # we must add the 0-8 full tops too
            allowed_tops[(0, wire_count)] = unchecked_bounds

        else:
            # the constraints are evaluated iteratively 0--2,0--4,0--6...
            # so, for the set of unchecked_bounds, we build sets of partial tops
            allowed_tops: Dict[Tuple[int, int], Set[bytes]] = {}
            for wire_max in range(2, wire_count + 1, 2):
                partial_tops = set()
                for bound in unchecked_bounds:
                    partial_tops.add(bound[:wire_max])
                allowed_tops[(0, wire_max)] = partial_tops

    for t, b, i in build_col_iterator(
        colors_required,
        top_min_validators,
        top_max_validators,
        bottom_min_validators,
        bottom_max_validators,
        wire_count,
        color_count,
        allowed_tops,
    ):
        yield Bcol(bytes(t), bytes(b), i, is_small_col)


def color_int_from_char(c: str) -> int:
    # A -> 0, B -> 1, C -> 2, ...
    return ord(c) - ord("A")


def colors_int_from_str(s: str) -> Generator[int, None, None]:
    return (color_int_from_char(c) for c in s)


def wire_generator(fima, as_list=False, rng=None):
    """
    Yield random permutations of labels [0, 1, ..., len(fima)-1]
    with counts specified by fima.
    """
    fima = np.asarray(fima)
    labels = np.arange(len(fima))
    base = np.repeat(labels, fima)  # fixed multiset of labels

    # use a given RNG or the global one
    rng = rng or np.random

    while True:
        arr = base.copy()
        rng.shuffle(arr)
        yield arr.tolist() if as_list else arr


def solve_bracelet(
    bconfig_ints: List[List[int]], wire_count: int, num_colors: int, fima: np.ndarray
):
    CFG_INITIAL_SAMPLING_BATCH_SIZE = 100000
    CFG_TAKE_N = 1000000
    ColorEnum, ColorTuple, ColorCounter = make_color_types(
        tuple_len=wire_count,
        num_colors=num_colors,
    )
    init_wireing = None

    # 0. determine assortment from column configurations, this ensure the
    # assortment is compatible with all columns and there is the minimum number of wires
    if init_wireing == True:
        init_wireing = []
        for l, r in zip(bconfig_ints[0], bconfig_ints[1] + [1]):
            init_wireing += [l, r]
        fima = counter_to_vec(Counter(init_wireing), color_count=num_colors)
        init_wireing = bytes(init_wireing)

    # 1. We calculate validators for each columns;
    # Validators are a list of minimal & maximal color counters for each sub-segment of the columns starting from left
    # that is  0--2, 0--4, 0--6, ... and also the partials 2-4 , 4-6, ...

    constraints_list = []
    for col_idx in range(len(columns)):
        min_constraints, max_constraints, missing_colors = build_min_max_validator_2(
            bconfig_ints[col_idx:],
            fima,
            color_count=num_colors,
        )
        if missing_colors.total() > 0:
            raise ValueError(f"Column {col_idx} missing colors: {missing_colors}")

        constraints_list.append((min_constraints, max_constraints))

    # then the algorithm works with a backtracking approach accross columns
    # we start from the first column, and build all possible local solutions
    # based on the min/max constraints of this columns.
    # each solution comes with a Bottom wiring (Bcol.B) that is kept as input for the next columns (Bound set)
    # we can jump to the next column , that will work on the Bound set do :
    #   - filter Bounds that do not match its Validators (this filter as well the previous column solution that were inserted)
    #   - build all possible local solutions (Bcol) that satisfy its own constraints and Bcol.T is in the Bound set.
    #   - iterate if any solution found; else backtrack to the previous column to get a new solution

    # the initial columns can expire in solution and is marked as fullyExhausted
    # when a column is fullyExhausted, we move to the next column directly
    # when any column exhaust and the previous columns is fullyExhausted, it is marked as fullyExhausted too
    # this continue until the last column is fullyExhausted or a solution is found

    # a solution is found when the last column is able to build at least one solution
    # at this point we have a valid Bcol for each column that can be printed

    # we have then variables to hold the current state of the search
    console = Console()

    class SolutionState:
        wire_count: int

        columns_solutions: List[List[Bcol]]
        columns_fullyExhausted: List[bool]

        # state of the current iterators for each column
        columns_iterators: List[Iterator[Bcol] | None]
        iterators_head: List[Set[ColorTupleBase] | None]

        # bound_check between columns, if false, it means this bound has not been evaluated yet by the next columns
        # if true, it means this bound has been evaluated already
        bound_check: List[Dict[ColorTupleBase, bool]]

        # columns constraints (min/max) for each column
        columns_constraints: List[
            Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]
        ]

        # bracelet configuration by column, as list of list of int
        b_configs: List[List[int]]

        def __init__(
            self,
            wire_count: int,
            columns_constraints,
            b_configs: List[List[int]],
            fima,
            init_wireing: bytes = None,
            initial_sampling_batch_size=100000,
            iteration_batch_size=1000000,
        ):
            self.wire_count = wire_count
            self.columns_constraints = columns_constraints
            self.b_configs = b_configs

            self.columns_solutions = [[] for _ in columns]
            self.columns_fullyExhausted = [False for _ in columns]
            self.columns_iterators = [None for _ in columns]
            self.iterators_head = [None for _ in columns]
            self.bound_check = [{} for _ in columns]

            self.init_wireing = init_wireing
            self.wire_gen = wire_generator(fima, as_list=True)
            self.INITIAL_SAMPLING_BATCH_SIZE = initial_sampling_batch_size
            self.ITERATION_BATCH_SIZE = iteration_batch_size

        @classmethod
        def color_count(cls):
            return ColorEnum.__len__()

        def search_column(self, idx: int):
            # search column at idx
            # depending on the state of the column:

            # if we have unchecked bounds from previous layer
            # we create a new iterator with the allowed tops from previous layer bounds that are not yet checked

            # if we don't have unchecked bounds from previous layer
            # we check if the previous layer is fully exhausted
            # - if yes, we mark this layer as fully exhausted and move to next layer
            # - if no, we move to previous layer

            # if we have an existing iterator
            #   we continue pulling from it some solutions
            #   we pull N solutions from the iterator
            #   if we have some solutions
            #     - we update the bound_check with previous layer bound to mark as evaluated
            #     - we update the bound_check for this layer with the Bcol.B of the solutions
            #     - we store the solutions in columns_solutions and move to the next columns (ok finish because we are the last)
            #   if we don't have solutions after pulling N solutions
            #     - if we exhausted the iterator
            #        - we mark all the the evaluated bounds as evaluated in the previous layer and invalid (filtering the parent solutions)
            #     - we check if the previous layer is fully exhausted
            #        - if yes, we can relunch this column
            #        - if no, we move to previous layer
            if (
                self.columns_iterators[idx] is None
                or self.columns_iterators[idx] is True
            ):
                if idx == 0:
                    # self.iterators_head[idx] = build...
                    if self.columns_fullyExhausted[idx]:
                        console.print(
                            f" Column {idx} exhausted, no more solutions here. moving next"
                        )

                        return idx + 1  # move to next layer

                    unchecked_bounds = (
                        {self.init_wireing} if self.init_wireing is not None else None
                    )
                    if unchecked_bounds is None:
                        unchecked_bounds = {
                            bytes(next(self.wire_gen))
                            for _ in range(self.INITIAL_SAMPLING_BATCH_SIZE)
                        }
                    console.print(
                        f" Column {idx} Create iterator for {len(unchecked_bounds) if unchecked_bounds is not None else 0} bounds"
                    )
                    self.iterators_head[idx] = unchecked_bounds
                    self.columns_iterators[idx] = make_bcol_iterator(
                        unchecked_bounds=unchecked_bounds,
                        colors_required=self.b_configs[idx],
                        top_min_validators=self.columns_constraints[idx][0],
                        top_max_validators=self.columns_constraints[idx][1],
                        bottom_min_validators=self.columns_constraints[idx + 1][0],
                        bottom_max_validators=self.columns_constraints[idx + 1][1],
                        wire_count=self.wire_count,
                        color_count=self.color_count(),
                    )
                else:
                    prev_bounds = self.bound_check[idx - 1]
                    unchecked_bounds = set((k for k, v in prev_bounds.items() if not v))
                    if len(unchecked_bounds) == 0:
                        # mm, nothing to work on yet. maybe I can jumpt to another layer
                        if self.columns_fullyExhausted[idx - 1]:
                            self.columns_fullyExhausted[idx] = True
                            return idx + 1
                        else:
                            return idx - 1
                    else:
                        # create an iterator with these bounds
                        self.iterators_head[idx] = unchecked_bounds

                        if idx < len(columns) - 1:  # identify if we have a next layer
                            bottom_min_validators = self.columns_constraints[idx + 1][0]
                            bottom_max_validators = self.columns_constraints[idx + 1][1]
                        else:
                            bottom_min_validators = None
                            bottom_max_validators = None

                        self.columns_iterators[idx] = make_bcol_iterator(
                            unchecked_bounds=unchecked_bounds,
                            colors_required=self.b_configs[idx],
                            top_min_validators=self.columns_constraints[idx][0],
                            top_max_validators=self.columns_constraints[idx][1],
                            bottom_min_validators=bottom_min_validators,
                            bottom_max_validators=bottom_max_validators,
                            wire_count=self.wire_count,
                            color_count=self.color_count(),
                        )

            # pull next N elements from the iterator
            current_head = self.iterators_head[idx]
            console.print(
                f" with head size:",
                len(current_head) if current_head is not None else "NoSize",
                " elements",
            )
            all_possible, exhausted, column_iterator = take_with_exhaustion(
                self.columns_iterators[idx], n=self.ITERATION_BATCH_SIZE
            )

            if exhausted:
                self.columns_iterators[idx] = True  # mark as exhausted
                if idx == 0:
                    console.print(
                        f" Column {idx} exhausted, no more solutions here. Reampling from scratch my friend "
                    )

                    # self.columns_fullyExhausted[idx] = True
            else:
                self.columns_iterators[idx] = (
                    column_iterator  # keep the iterator in a corner for the moment
                )

            next_layer_bounds = self.bound_check[idx]
            added_solution_downward = 0
            for sol in all_possible:

                # add the new bound for this layer
                if sol.B not in next_layer_bounds:
                    next_layer_bounds[sol.B] = False  # not yet evaluated
                    added_solution_downward += 1

            # keep all the new solutions on my side.
            self.columns_solutions[idx] = self.columns_solutions[idx] + all_possible
            console.print(
                f" Pulled {len(all_possible)} solutions, exhausted={exhausted}, total solutions now={len(self.columns_solutions[idx])}"
            )
            if added_solution_downward:
                return idx + 1  # move to next layer
            else:
                if exhausted:
                    if idx == 0:
                        console.print(
                            f" Column {idx} exhausted, no more solutions here. Will resample more"
                        )
                        # self.columns_fullyExhausted[idx] = True
                        return idx
                    else:
                        # mark all the evaluated bounds as evaluated in the previous layer
                        prev_bounds = self.bound_check[idx - 1]
                        for k in current_head:
                            prev_bounds[k] = True  # mark as evaluated

                        # and filtering the parent solutions based on my current solutions Tops
                        current_tops = set(s.T for s in self.columns_solutions[idx])
                        self.columns_solutions[idx - 1] = [
                            s
                            for s in self.columns_solutions[idx - 1]
                            if s.B in current_tops
                        ]
                        # i can relaunch myself
                        console.print(
                            f" Column {idx} self relaunch after exhaustion, filtered previous column to {len(self.columns_solutions[idx - 1])} solutions"
                        )
                        return idx

                else:
                    return idx  # we can relaunch this column

        def find_solution_up_to_columns(
            self, col_idx, target_top: Tuple[ColorTupleBase, ...] | None
        ) -> List[Bcol]:
            if col_idx == 0:
                for bcol in self.columns_solutions[col_idx]:
                    if bcol.B == target_top:
                        return [bcol]
                console.print("No solution found at col 0")
                return None
            else:
                for bcol in self.columns_solutions[col_idx]:
                    if target_top is not None and bcol.B != target_top:
                        continue
                    prev_solution = self.find_solution_up_to_columns(
                        col_idx - 1, bcol.T
                    )
                    if prev_solution is not None:
                        return prev_solution + [bcol]
                # console.print("No solution found at col", col_idx)
                return None

    console.print("Fima:", counter_to_rich(fima, color_count=num_colors))

    console.print("Building solution state...")
    solution_state = SolutionState(
        wire_count,
        constraints_list,
        bconfig_ints,
        fima,
        init_wireing=init_wireing,
        initial_sampling_batch_size=CFG_INITIAL_SAMPLING_BATCH_SIZE,
        iteration_batch_size=CFG_TAKE_N,
    )

    # we start at column 0
    current_col = 0
    solution_found = False
    while not solution_found:
        console.print(f"Searching column {current_col}...")
        next_idx = solution_state.search_column(current_col)

        if next_idx >= len(bconfig_ints):
            # solution found
            solution_found = True
        current_col = next_idx

    # 3. printing sexy results with rich lib...

    sols = solution_state.find_solution_up_to_columns(len(bconfig_ints) - 1, None)
    for s in sols:
        console.print(s.__to_rich__())
    return solution_state


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
        "CCBABBACCBAA",
        "CCABAACCBAA",
        "CCAABAACCBAA",
        "CCABAACCBAA",
        "CCABAACCBAAA",
    ]  # 12
    fima = np.array([8, 8, 8], dtype=np.int8)

    # columns = [
    #     "ACCBABBACCBAA",
    #     "ACCABAACCBAA",
    #     "ACCAABAACCBAA",
    #     "ACCABAACCBAA",
    #     "ACCABAACCBAAA",
    # ]  # 13

    columns = [
        "ACCBABBACAACCBAACC",
        "ACCABABACCACBAACC",
        "ACCAABBACAACCBAACC",
        "ACCABABACCACBAACC",
        "ACCABABACCACBAAACC",
        "BCCABABACCACBAACB",
    ]  # 18

    columns = [
        "CCBABACCBACCB",
        "CCABAACCACCB",
    ] * 30  # 13 * 60
    fima = np.array([9, 9, 8], dtype=np.int8)
    columns = [
        "ACCBABAACCBAACC",
        "ACCABACACBAACC",
        "ACCAABAACCBAACC",
        "ACCABACACBAACC",
        "ACCABACACBAAACC",
        "BCCABACACBAACB",
    ]  # 15 (30 wires)
    fima = np.array([8, 5, 17], dtype=np.int8)
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
        "ABCABAACAAAACCBAACC",
        "ABCABAACAAAAACCBAACC",
        "ABCABAACAAAACCBAACC",
        "ABCABAACAAAACCBAAACC",
        "ABCABAACAAAACCBAACC",
        "ABCABAACAAAAACCBAACC",
        "ABCABAACAAAACCBAACC",
        "ABCABAACAAAACCBAAACC",
    ]  # 20
    fima = np.array([16, 12, 12], dtype=np.int8)

    columns = [
        "CCBABBACCBAA",
        "CCABAACCBAA",
        "CCAABAACCBAA",
        "CCABAACCBAA",
        "CCABAACCBAAA",
    ]  # 12
    fima = np.array([8, 8, 8], dtype=np.int8)
    columns = ["BABB", "ABA", "AABA", "AAA", "AAAA"]  # 4 nodes
    fima = np.array([4, 4], dtype=np.int8)

    wire_count = max(len(col) for col in columns) * 2
    num_colors = len(set("".join(columns)))
    bconfig_ints = [[ord(c) - ord("A") for c in bc] for bc in columns]
    solve_bracelet(bconfig_ints, wire_count, num_colors, fima)

from typing import List
from qiskit_aer import AerSimulator
from qiskit import transpile

from qiskit.primitives import StatevectorSampler
from rich.text import Text
from build0 import (
    COLOR_DICT,
    Assortment,
    build_chunk,
    build_multi_instance_or_circuit,
    build_or_grover_circuit,
    clist_to_rich,
    decode_multi_counts,
    pairs_to_rich,
    process_dual_layer,
)
from pprint import pprint
from rich.console import Console


class BCol:
    def __init__(self, c_list: List[str]):
        self.c_list = c_list
        self.bra = None
        self.is_intermediate: bool = False

    def assortment(self) -> Assortment:
        return Assortment((c, self.c_list.count(c)) for c in set(self.c_list))

    def to_rich(self, solution=None) -> Text:
        if solution is not None:
            return pairs_to_rich(solution) + "\n" + self.to_rich()
            txt = ""
            for w_idx, c in enumerate(solution):
                if w_idx % 2 == 0:
                    color = COLOR_DICT[c]
                    txt += f"[{color}]╰[/{color}]"
                    color = COLOR_DICT[self.c_list[w_idx // 2]]
                    txt += f"[{color}]▅[/{color}]"
                else:
                    color = COLOR_DICT[c]
                    txt += f"[{color}]╯[/{color}] "
            return Text.from_markup(f"{txt}")

        else:
            C_str = "   ".join(
                [f"[{c}]▅[/{c}]" for c in [COLOR_DICT[p] for p in self.c_list]]
            )
            if self.is_intermediate:
                C_str = f"  {C_str}"
            return Text.from_markup(f" {C_str}")


class Bra:
    def __init__(self, c_lists: List[BCol], assortment: Assortment = None):
        self.c_lists = c_lists
        for i, b in enumerate(c_lists):
            b.bra = self
            b.is_intermediate = i % 2 == 1
        self.assortment = assortment
        self.sol_list = None

    def to_rich(self) -> Text:

        if self.sol_list is not None:
            rich_text = Text()
            for i, b in enumerate(self.c_lists):
                if b.is_intermediate:
                    rich_text.append(b.to_rich())
                    rich_text.append("\n")
                else:
                    # pick first solution for this layer
                    layer_idx = i // 2
                    sols, _ = self.sol_list[layer_idx]
                    first_sol = sols[0][0]
                    rich_text.append(
                        b.to_rich(first_sol)
                        + Text.from_markup(f" > {len(sols)} solutions")
                    )
                    rich_text.append("\n")
            return rich_text
        else:
            rich_text = Text()
            for i, b in enumerate(self.c_lists):
                rich_text.append(b.to_rich())
                rich_text.append("\n")
            return rich_text

    @property
    def wire_count(self) -> int:
        return sum(self.assortment.values())

    @property
    def row_count(self) -> int:
        return self.wire_count // 2

    def solve(self):
        C_list_prev = None
        prev_extrema_set = None
        self.sol_list = []
        for i in range(len(self.c_lists) // 2):
            C_list_layer0 = self.c_lists[2 * i].c_list
            C_list_layer1 = self.c_lists[2 * i + 1].c_list
            sols, prev_extrema_set = process_dual_layer(
                C_list_layer0,
                C_list_layer1,
                extremas=None,
                C_list_prev=C_list_prev,
                prev_set_extremas=(
                    set(e for e, _ in prev_extrema_set)
                    if prev_extrema_set is not None
                    else None
                ),
                assortment=self.assortment,
            )
            C_list_prev = C_list_layer1

            self.sol_list.append((sols, prev_extrema_set))

        return self.sol_list


b = Bra(
    [
        BCol(["00", "00", "00", "00"]),
        BCol(["11", "11", "11"]),
        BCol(["00", "00", "10", "00"]),
        BCol(["11", "11", "00"]),
        BCol(["10", "00", "00", "00"]),
        BCol(["11", "11", "00"]),
    ],
    assortment=Assortment({"00": 4, "01": 0, "10": 1, "11": 3}.items()),
)

console = Console()
console.print(b.to_rich())
sol_list = b.solve()
console.print(b.to_rich())

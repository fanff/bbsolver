from rich.text import Text
import numpy as np
from bracelet_maker import load_bracelet_from_image
from buildXng import solve_bracelet

from rich.console import Console

from enriching import bracelet_int_config_to_rich

if __name__ == "__main__":
    from pathlib import Path

    console = Console()
    # bracelet_design, wire_count, color_to_index, index_to_color, indexed_color_count = (
    #    load_bracelet_from_image(
    #        Path("bra_design\\10\\1030_test_purple.png"),
    #        target_bracelet_node_width=None,
    #        crop_top_n_cols=-1,
    #    )
    # )
    # fima = np.array([11, 9], dtype=int)

    bracelet_design, wire_count, color_to_index, index_to_color, indexed_color_count = (
        load_bracelet_from_image(
            Path("bra_design/20/bratest.png"),
            target_bracelet_node_width=None,
            crop_top_n_cols=34,
        )
    )
    fima = np.array([20, 12, 8], dtype=int)

    console.print("Wire count:", wire_count)
    console.print("Indexed color count:", indexed_color_count)
    console.print("FIMA:", fima)

    rich_bracelet = bracelet_int_config_to_rich(
        bracelet_design, wire_count, index_to_color, character_width=2
    )
    for col_idx, col in enumerate(rich_bracelet):
        is_small_col = col_idx % 2 == 1
        if is_small_col:
            prefix_text = Text(" ")
        else:
            prefix_text = Text("")

        console.print(prefix_text + Text.assemble(*col))
    solve_bracelet(bracelet_design, wire_count, len(color_to_index), fima)

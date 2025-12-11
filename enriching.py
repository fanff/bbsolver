from typing import List
from rich.color import Color
from rich.style import Style
from rich.text import Text


def bracelet_int_config_to_rich(
    bcol: List[List[int]], wire_count, index_to_color, character_width=1
) -> List[List[Text]]:
    """Convert a bracelet column configuration from integer indices to rich text formated using RGB colors in the markup
    and the full glyph as character"""

    rich_col: List[List[Text]] = []
    for col_idx, col in enumerate(bcol):
        current_row_str_atoms: List[Text] = []
        for node_idx, color_index in enumerate(col):

            color_rgb = index_to_color[color_index]
            color_hex = "#{:02x}{:02x}{:02x}".format(
                color_rgb[0], color_rgb[1], color_rgb[2]
            )
            style = Style(color=Color.parse(color_hex))
            rich_glyph = f"\u2588"
            if character_width > 1:
                rich_glyph = rich_glyph * character_width
            text = Text(rich_glyph)
            text.stylize(style)
            current_row_str_atoms.append(text)
        rich_col.append(current_row_str_atoms)
    return rich_col

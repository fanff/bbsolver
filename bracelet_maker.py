from collections import Counter
from PIL import Image

from pathlib import Path


def load_bracelet_from_image(
    image_path: Path, target_bracelet_node_width: int, crop_top_n_cols: int = -1
) -> list:
    """
    Load a bracelet design from an image file.
    """
    image = Image.open(str(image_path))
    image = image.convert("RGB")  # Ensure image is in RGB format
    width, height = image.size

    # if width > height, image is given horizontally, rotate it
    if width > height:
        image = image.rotate(90, expand=True)
        width, height = image.size

    if target_bracelet_node_width != None:
        wire_count = target_bracelet_node_width * 2
        if width != target_bracelet_node_width:
            # Now calculate a proportional scale factor, so that width is 20 pixels
            scale_factor = target_bracelet_node_width / width
            new_width = target_bracelet_node_width
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.NEAREST)
            width, height = image.size
    else:
        wire_count = width * 2

    if 0 < crop_top_n_cols < height:
        # Crop the top n columns
        image = image.crop((0, 0, width, crop_top_n_cols))
        width, height = image.size
    # save the image back for debugging
    debug_image_name = image_path.stem + "_processed.png"
    image.save(debug_image_name)

    all_colors = []
    bracelet_design = []
    for col_idx in range(height):
        row = []

        is_small_col = col_idx % 2 == 1
        shift = -1 if is_small_col else 0
        for x in range(width + shift):
            pixel = image.getpixel((x, col_idx))
            all_colors.append(pixel)
            row.append(pixel)  # Store RGB tuple
        bracelet_design.append(row)

    # find how many colors are used :
    all_color_count = Counter(all_colors)
    print("All colors used (with counts):", all_color_count)
    # sort by most common and index them , such that 0 is most common color
    sorted_colors = [color for color, count in all_color_count.most_common()]
    color_to_index = {color: idx for idx, color in enumerate(sorted_colors)}
    index_to_color = {idx: color for idx, color in enumerate(sorted_colors)}
    # now convert bracelet design to indexed design
    for col_idx in range(len(bracelet_design)):
        for x in range(len(bracelet_design[col_idx])):
            color = bracelet_design[col_idx][x]
            color_index = color_to_index[color]
            bracelet_design[col_idx][x] = color_index

    # indexed color count
    indexed_color_count = Counter()
    for color, count in all_color_count.items():
        color_index = color_to_index[color]
        indexed_color_count[color_index] = count
    return (
        bracelet_design,
        wire_count,
        color_to_index,
        index_to_color,
        indexed_color_count,
    )


if __name__ == "__main__":
    bracelet_design, wire_count, color_to_index, index_to_color, indexed_color_count = (
        load_bracelet_from_image(
            Path("bra_design\\10\\1030_test_purple.png"),
            target_bracelet_node_width=None,
            crop_top_n_cols=-1,
        )
    )

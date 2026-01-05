import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

from glyphogen.dataset import get_hierarchical_data, collate_fn
from glyphogen.nodeglyph import NodeGlyph
from glyphogen.command_defs import NodeCommand


def plot_alignments(image, glyph, alignment_data, alignment_axis, title):
    """
    Displays a glyph raster and overlays colored circles for alignment groups.
    """
    plt.figure(figsize=(10, 10))
    # The raster image is normalized, but for display, the raw shape is fine.
    # It's also (C, H, W), so we need to squeeze it for imshow.
    plt.imshow(image.squeeze()[0, :, :], cmap="gray_r", origin="lower")
    plt.title(title)

    # Generate a set of unique colors for the alignment groups
    colors = [
        "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
        for i in range(sum(len(contour_aligns) for contour_aligns in alignment_data))
    ]
    color_idx = 0

    for contour_idx, contour in enumerate(glyph.contours):
        contour_alignments = alignment_data[contour_idx]
        for group in contour_alignments:
            if not group:
                continue

            color = colors[color_idx]
            # Gather coordinates of all nodes in the current alignment group
            coords = [contour.nodes[node_idx].coordinates for node_idx in group]

            # Plot each point in the group
            for x, y in coords:
                # Note: plt.scatter uses standard cartesian coordinates, but imshow
                # places the origin at the top-left. We set origin='lower' in imshow
                # to align them so (0,0) is at the bottom-left for both.
                plt.scatter(
                    x, y, color=color, s=150, alpha=0.7, edgecolors="w", linewidths=1.5
                )

            # Optionally, draw a line to emphasize the alignment
            if len(coords) > 1:
                coords_arr = np.array(coords)
                if alignment_axis == "x":
                    # For X-alignment, the line is vertical
                    mean_x = coords_arr[:, 0].mean()
                    min_y = coords_arr[:, 1].min()
                    max_y = coords_arr[:, 1].max()
                    plt.plot(
                        [mean_x, mean_x], [min_y, max_y], color=color, linestyle="--"
                    )
                else:
                    # For Y-alignment, the line is horizontal
                    mean_y = coords_arr[:, 1].mean()
                    min_x = coords_arr[:, 0].min()
                    max_x = coords_arr[:, 0].max()
                    plt.plot(
                        [min_x, max_x], [mean_y, mean_y], color=color, linestyle="--"
                    )

            color_idx += 1

    plt.savefig("plot.png")
    _ = input("Press Enter to continue...")
    plt.close()


def main():
    """
    Main function to load data and generate alignment plots.
    """
    print("Loading dataset...")
    train_dataset, _ = get_hierarchical_data()
    data_loader = DataLoader(
        train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True
    )

    print("Fetching one batch...")
    batch = next(iter(data_loader))

    if batch is None:
        print("Failed to get a batch. Exiting.")
        return

    # --- Process the first glyph in the batch ---
    for glyph_idx in range(len(batch["images"])):
        print(f"Visualizing alignments for glyph index {glyph_idx} in the batch...")

        # Find all contours belonging to the first glyph
        first_glyph_mask = batch["contour_image_idx"] == glyph_idx

        # Get the raster image for the glyph
        image = batch["images"][glyph_idx]

        # Get the ground truth sequences and decode them to get absolute coordinates
        gt_sequences = [
            seq.cpu()
            for i, seq in enumerate(batch["target_sequences"])
            if first_glyph_mask[i]
        ]
        if not gt_sequences:
            print("Selected glyph has no contours. Exiting.")
            return

        glyph = NodeGlyph.decode(gt_sequences, NodeCommand)

        # Get the alignment data for the glyph
        x_alignments = [
            align
            for i, align in enumerate(batch["x_aligned_point_indices"])
            if first_glyph_mask[i]
        ]
        y_alignments = [
            align
            for i, align in enumerate(batch["y_aligned_point_indices"])
            if first_glyph_mask[i]
        ]

        # --- Plot X-Alignments ---
        plot_alignments(image, glyph, x_alignments, "x", "X-Axis Alignments")

        # --- Plot Y-Alignments ---
        plot_alignments(image, glyph, y_alignments, "y", "Y-Axis Alignments")


if __name__ == "__main__":
    main()

import tensorflow as tf
from glyphogen.dataset import get_pretrain_data
from glyphogen.rasterizer import rasterize_batch
import numpy as np
import pydiffvg_tensorflow as pydiffvg


def main():
    """
    Test script for loading a batch from the pre-train dataset and preparing it for rasterization.
    """
    train_dataset, _ = get_pretrain_data()

    # Take a single batch from the dataset
    batch = train_dataset.skip(100).batch(16).take(1)

    # Iterate over the batch
    for (rasters, seq), (cmds, coords) in batch:
        rendered_images = rasterize_batch(cmds, coords)
        # imwrite wants 3 channels, so we convert our single channel image to RGB
        img_to_save = tf.image.grayscale_to_rgb(rasters[0])
        pydiffvg.imwrite(img_to_save, "target.png", gamma=2.2)
        img_to_save = tf.image.grayscale_to_rgb(rendered_images[0])
        pydiffvg.imwrite(img_to_save, "produced.png", gamma=2.2)
        loss = tf.reduce_mean(tf.square(rendered_images - rasters))
        print(f"Raster loss: {loss.numpy()}")

        # Create a comparison image
        for ix, (target_raster, produced_raster) in enumerate(
            zip(rasters, rendered_images)
        ):

            # Red channel for target, Green for produced
            red_channel = target_raster
            green_channel = produced_raster
            blue_channel = tf.zeros_like(target_raster)

            # Alpha is 50% of the intensity
            alpha_channel = (
                tf.clip_by_value(target_raster + produced_raster, 0.0, 1.0) * 0.5
            )

            comparison_image = tf.concat(
                [red_channel, green_channel, blue_channel, alpha_channel], axis=-1
            )

            pydiffvg.imwrite(comparison_image, "comparison-%02i.png" % ix)


if __name__ == "__main__":
    main()

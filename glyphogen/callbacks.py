import tensorflow as tf
import keras
from .glyph import NodeGlyph, NodeCommand
import numpy as np


class ImageGenerationCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, test_dataset, num_images=3, pre_train=False):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + "/images")
        self.test_dataset = test_dataset.unbatch().take(num_images)
        self.pre_train = pre_train

    def on_epoch_end(self, epoch, logs=None):
        if self.pre_train:
            # Don't generate images during pre-training
            return

        for i, (inputs, outputs) in enumerate(self.test_dataset):
            (style_image, glyph_id, target_sequence) = inputs
            true_raster = outputs["raster"]
            # Add batch dimension
            style_image = tf.expand_dims(style_image, axis=0)
            glyph_id = tf.expand_dims(glyph_id, axis=0)
            target_sequence = tf.expand_dims(target_sequence, axis=0)

            # Predict generated raster
            generated_raster = self.model((style_image, glyph_id, target_sequence))[
                "raster"
            ]
            true_raster = tf.expand_dims(true_raster, axis=0)

            assert true_raster.shape == generated_raster.shape, (
                f"Shape mismatch: true_raster {true_raster.shape}, "
                f"generated_raster {generated_raster.shape}"
            )

            with self.file_writer.as_default():
                tf.summary.image(
                    f"True Glyph Raster - Sample {i}",
                    true_raster,
                    step=epoch,
                )
                tf.summary.image(
                    f"Generated Glyph Raster - Sample {i}", generated_raster, step=epoch
                )


class SVGGenerationCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, test_dataset, num_samples=3, pre_train=False):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + "/svgs")
        self.test_dataset = test_dataset.unbatch().take(num_samples)
        self.pre_train = pre_train

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5:
            return
        model = self.model

        for i, (inputs, outputs) in enumerate(self.test_dataset):
            if self.pre_train:
                (raster_image_input, target_sequence_input) = inputs

                # Add batch dimension
                raster_image_input = tf.expand_dims(raster_image_input, axis=0)
                target_sequence_input = tf.expand_dims(target_sequence_input, axis=0)

                output = model(
                    (raster_image_input, target_sequence_input), training=False
                )
            else:
                (style_image, glyph_id, target_sequence_input) = inputs

                # Add batch dimension
                style_image = tf.expand_dims(style_image, axis=0)
                glyph_id = tf.expand_dims(glyph_id, axis=0)
                target_sequence_input = tf.expand_dims(target_sequence_input, axis=0)

                output = model(
                    (style_image, glyph_id, target_sequence_input), training=False
                )

            command_output = output["command"]
            coord_output = output["coord"]

            # a single batch
            command_tensor = command_output[0]
            coord_tensor = coord_output[0]

            try:
                decoded_glyph = NodeGlyph.from_numpy(
                    command_tensor.numpy(), coord_tensor.numpy()
                )
                try:
                    svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
                except Exception:
                    svg_string = (
                        "Couldn't convert to SVG: " + decoded_glyph.to_debug_string()
                    )
            except Exception:
                command_keys = list(NodeCommand.grammar.keys())
                svg_string = "Invalid Node Sequence " + " ".join(
                    [
                        command_keys[np.argmax(command_tensor[i])]
                        for i in range(command_tensor.shape[0])
                    ]
                )

            with self.file_writer.as_default():
                tf.summary.text(f"Generated SVG - Sample {i}", svg_string, step=epoch)

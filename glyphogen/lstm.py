#!/usr/bin/env python
import keras
from keras import layers, ops
import tensorflow as tf

from glyphogen.glyph import (
    NODE_COMMAND_WIDTH,
    COORDINATE_WIDTH,
)
from glyphogen.hyperparameters import MAX_COMMANDS

MAX_COORDINATE = 1500.0  # We scale the coordinates to be in the range [-1, 1]

@keras.saving.register_keras_serializable()
class LSTMDecoder(layers.Layer):
    def __init__(self, d_model, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.rate = rate

        self.command_embedding = layers.Dense(d_model)
        self.coord_embedding = layers.Dense(d_model)
        self.lstm = layers.LSTM(d_model, return_sequences=True)
        self.dropout = layers.Dropout(rate)
        self.output_command = layers.Dense(NODE_COMMAND_WIDTH, activation="softmax")
        self.output_coords = layers.Dense(COORDINATE_WIDTH, activation="tanh")

    def call(self, x, context=None, training=False):
        command_input = x[:, :, :NODE_COMMAND_WIDTH]
        coord_input = ops.cast(x[:, :, NODE_COMMAND_WIDTH:], dtype="float32") / MAX_COORDINATE
        command_emb = self.command_embedding(command_input)
        coord_emb = self.coord_embedding(coord_input)
        x = command_emb + coord_emb
        x = self.dropout(x, training=training)

        if context is not None:
            context_tiled = tf.tile(context, [1, tf.shape(x)[1], 1])
            x = layers.concatenate([x, context_tiled])

        x = self.lstm(x)

        command_output = self.output_command(x)
        coord_output = self.output_coords(x)
        coord_output = coord_output * MAX_COORDINATE
        return command_output, coord_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "rate": self.rate,
            }
        )
        return config

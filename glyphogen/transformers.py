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
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "position": self.position,
                "d_model": self.d_model,
            }
        )
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            ops.arange(position, dtype="float32")[:, None],
            ops.arange(d_model, dtype="float32")[None, :],
            d_model,
        )
        sines = ops.sin(angle_rads[:, 0::2])
        cosines = ops.cos(angle_rads[:, 1::2])
        pos_encoding = ops.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[None, ...]
        return ops.cast(pos_encoding, dtype=self.dtype)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / ops.power(
            10000, (2 * (i // 2)) / ops.cast(d_model, "float32")
        )
        return pos * angle_rates

    def call(self, x):
        return x + self.pos_encoding[:, : ops.shape(x)[1], :]


@keras.saving.register_keras_serializable()
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential(
            [
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "rate": self.rate,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.command_embedding = layers.Dense(d_model)
        self.coord_embedding = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(MAX_COMMANDS, d_model)
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training):
        command_input = x[:, :, :NODE_COMMAND_WIDTH]
        coord_input = ops.cast(x[:, :, NODE_COMMAND_WIDTH:], dtype="float32") / MAX_COORDINATE
        command_emb = self.command_embedding(command_input)
        coord_emb = self.coord_embedding(coord_input)
        x = command_emb + coord_emb
        x *= ops.sqrt(ops.cast(self.d_model, "float32"))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "rate": self.rate,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential(
            [
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, context=None, look_ahead_mask=None, training=False):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2 = self.mha2(out1, context, context)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "rate": self.rate,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class TransformerDecoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.command_embedding = layers.Dense(d_model)
        self.coord_embedding = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(MAX_COMMANDS, d_model)
        self.dec_layers = [
            TransformerDecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(rate)
        self.output_command = layers.Dense(NODE_COMMAND_WIDTH, activation="softmax")
        self.output_coords = layers.Dense(COORDINATE_WIDTH, activation="tanh")

    def call(self, x, context=None, look_ahead_mask=None, training=False):
        command_input = x[:, :, :NODE_COMMAND_WIDTH]
        coord_input = x[:, :, NODE_COMMAND_WIDTH:] / MAX_COORDINATE
        command_emb = self.command_embedding(command_input)
        coord_emb = self.coord_embedding(coord_input)
        x = command_emb + coord_emb
        x *= ops.sqrt(ops.cast(self.d_model, "float32"))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, context=context, look_ahead_mask=look_ahead_mask, training=training
            )

        command_output = self.output_command(x)
        coord_output = self.output_coords(x)
        coord_output = coord_output * MAX_COORDINATE
        return command_output, coord_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "rate": self.rate,
            }
        )
        return config

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

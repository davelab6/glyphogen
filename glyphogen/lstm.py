#!/usr/bin/env python
import torch
from torch import nn
import torch.nn.functional as F

from glyphogen.command_defs import (
    NODE_COMMAND_WIDTH,
    COORDINATE_WIDTH,
)


class LSTMDecoder(nn.Module):
    def __init__(self, d_model_cmd, d_model_coord, latent_dim, rate=0.1):
        super().__init__()
        self.d_model_cmd = d_model_cmd
        self.d_model_coord = d_model_coord
        self.latent_dim = latent_dim
        self.rate = rate

        self.command_embedding = nn.Linear(NODE_COMMAND_WIDTH, d_model_cmd)
        self.coord_embedding = nn.Linear(COORDINATE_WIDTH, d_model_coord)
        self.lstm_cmd = nn.LSTM(d_model_cmd + latent_dim, d_model_cmd, batch_first=True)
        self.lstm_coord = nn.LSTM(
            d_model_coord + latent_dim, d_model_coord, batch_first=True
        )
        self.dropout_cmd = nn.Dropout(rate)
        self.dropout_coord = nn.Dropout(rate)
        self.output_command = nn.Linear(d_model_cmd, NODE_COMMAND_WIDTH)
        self.output_coords = nn.Linear(
            d_model_coord + NODE_COMMAND_WIDTH, COORDINATE_WIDTH
        )
        # Initialize bias to zeros to encourage zero-centered output for untrained model
        nn.init.zeros_(self.output_coords.bias)
        self.coord_output_scale = nn.Parameter(torch.tensor(0.1))
        # self.softmax = F.log_softmax(dim=-1)
        self.tanh = nn.Tanh()

    @torch.compiler.disable
    def forward(self, x, context=None):
        command_input = x[:, :, :NODE_COMMAND_WIDTH].float()
        coord_input = x[:, :, NODE_COMMAND_WIDTH:].float()

        command_emb = self.command_embedding(command_input)
        coord_emb = self.coord_embedding(coord_input)

        x_cmd = command_emb
        x_coord = coord_emb
        x_cmd = self.dropout_cmd(x_cmd)
        x_coord = self.dropout_coord(x_coord)

        if context is not None:
            # The original context is (batch, 1, latent_dim). We need to tile it.
            context_tiled = context.repeat(1, x.shape[1], 1)
            x_cmd = torch.cat([x_cmd, context_tiled], dim=-1)
            x_coord = torch.cat([x_coord, context_tiled], dim=-1)

        # LSTM expects input of shape (batch, seq_len, input_size)
        # which is what we have.
        x_cmd, _ = self.lstm_cmd(x_cmd)
        x_coord, _ = self.lstm_coord(x_coord)

        command_output = self.output_command(x_cmd)

        # Condition coordinate output on the command output
        coord_head_input = torch.cat([x_coord, command_output], dim=-1)
        coord_output = self.output_coords(coord_head_input)
        coord_output = self.tanh(coord_output * self.coord_output_scale)

        return command_output, coord_output

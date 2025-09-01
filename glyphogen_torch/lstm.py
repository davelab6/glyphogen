#!/usr/bin/env python
import torch
from torch import nn

from glyphogen.glyph import (
    NODE_COMMAND_WIDTH,
    COORDINATE_WIDTH,
)

MAX_COORDINATE = 1500.0  # We scale the coordinates to be in the range [-1, 1]

class LSTMDecoder(nn.Module):
    def __init__(self, d_model, latent_dim, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.rate = rate

        self.command_embedding = nn.Linear(NODE_COMMAND_WIDTH, d_model)
        self.coord_embedding = nn.Linear(COORDINATE_WIDTH, d_model)
        self.lstm = nn.LSTM(d_model + latent_dim, d_model, batch_first=True)
        self.dropout = nn.Dropout(rate)
        self.output_command = nn.Linear(d_model, NODE_COMMAND_WIDTH)
        self.output_coords = nn.Linear(d_model, COORDINATE_WIDTH)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, x, context=None):
        command_input = x[:, :, :NODE_COMMAND_WIDTH].float()
        coord_input = x[:, :, NODE_COMMAND_WIDTH:].float() / MAX_COORDINATE
        
        command_emb = self.command_embedding(command_input)
        coord_emb = self.coord_embedding(coord_input)
        
        x = command_emb + coord_emb
        x = self.dropout(x)

        if context is not None:
            # The original context is (batch, 1, latent_dim). We need to tile it.
            context_tiled = context.repeat(1, x.shape[1], 1)
            x = torch.cat([x, context_tiled], dim=-1)

        # LSTM expects input of shape (batch, seq_len, input_size)
        # which is what we have. 
        x, _ = self.lstm(x)

        command_output = self.output_command(x)
        command_output = self.softmax(command_output)

        coord_output = self.output_coords(x)
        coord_output = self.tanh(coord_output)
        coord_output = coord_output * MAX_COORDINATE
        
        return command_output, coord_output
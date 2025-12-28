#!/usr/bin/env python
import random
import torch
from torch import nn
import torch.nn.functional as F

from glyphogen.hyperparameters import PROJ_SIZE

from glyphogen.command_defs import (
    NODE_COMMAND_WIDTH,
    COORDINATE_WIDTH,
    PREDICTED_COORDINATE_WIDTH,
)


class LSTMDecoder(nn.Module):
    def __init__(self, d_model, latent_dim, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.proj_size = PROJ_SIZE
        self.rate = rate

        self.command_embedding = nn.Linear(NODE_COMMAND_WIDTH, d_model)
        self.coord_embedding = nn.Linear(COORDINATE_WIDTH, d_model)
        self.lstm = nn.LSTM(
            d_model + latent_dim, d_model, batch_first=True, proj_size=self.proj_size
        )
        self.layer_norm = nn.LayerNorm(self.proj_size)
        self.dropout = nn.Dropout(rate)
        self.output_command = nn.Linear(self.proj_size, NODE_COMMAND_WIDTH)
        self.output_command_activation = nn.ReLU()
        self.output_coords = nn.Linear(
            self.proj_size + NODE_COMMAND_WIDTH, PREDICTED_COORDINATE_WIDTH
        )
        nn.init.zeros_(self.output_coords.bias)
        self.coord_output_scale = nn.Parameter(torch.tensor(0.1))
        self.tanh = nn.Tanh()

    def _forward_step(self, input_token, context, hidden_state=None):
        """
        Performs a single decoding step.
        """
        command_input = input_token[:, :, :NODE_COMMAND_WIDTH].float()
        coord_input = input_token[
            :, :, NODE_COMMAND_WIDTH : NODE_COMMAND_WIDTH + COORDINATE_WIDTH
        ].float()

        command_emb = self.command_embedding(command_input)
        coord_emb = self.coord_embedding(coord_input)
        x = command_emb + coord_emb
        x = self.dropout(x)

        if context is not None:
            x = torch.cat([x, context], dim=-1)

        x, hidden_state = self.lstm(x, hidden_state)
        x = self.layer_norm(x)

        command_logits = self.output_command(x)
        coord_head_input = torch.cat([x, command_logits], dim=-1)
        coord_output = self.output_coords(coord_head_input)
        coord_output = self.tanh(coord_output * self.coord_output_scale)

        return command_logits, coord_output, hidden_state

    def forward(self, x, context=None, teacher_forcing_ratio=1.0):
        """
        Training forward pass with scheduled sampling.
        Unrolls the sequence step by step, choosing between teacher forcing
        and autoregressive input based on the teacher_forcing_ratio.
        """
        batch_size, seq_len, _ = x.shape
        # First input is always the SOS token from the ground truth
        current_input = x[:, 0:1, :]
        hidden_state = None
        
        all_command_logits = []
        all_coord_outputs = []

        for i in range(seq_len):
            command_logits, coord_output, hidden_state = self._forward_step(
                current_input, context, hidden_state
            )
            all_command_logits.append(command_logits)
            all_coord_outputs.append(coord_output)

            # Decide on the next input
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if i + 1 < seq_len:
                if use_teacher_forcing:
                    # Use next ground truth token
                    current_input = x[:, i + 1 : i + 2, :]
                else:
                    # Use model's own prediction
                    command_probs = F.softmax(command_logits.squeeze(1), dim=-1)
                    predicted_command_idx = torch.argmax(command_probs, dim=1, keepdim=True)
                    next_command_onehot = F.one_hot(
                        predicted_command_idx, num_classes=NODE_COMMAND_WIDTH
                    ).float()
                    
                    coord_padded = torch.zeros(batch_size, 1, COORDINATE_WIDTH, device=x.device)
                    coord_padded[:, :, :PREDICTED_COORDINATE_WIDTH] = coord_output
                    
                    current_input = torch.cat([next_command_onehot, coord_padded], dim=-1)

        command_output = torch.cat(all_command_logits, dim=1)
        coord_output = torch.cat(all_coord_outputs, dim=1)
        return command_output, coord_output

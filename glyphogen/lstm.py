#!/usr/bin/env python
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
        Args:
            input_token (Tensor): Shape (batch_size, 1, NODE_COMMAND_WIDTH + COORDINATE_WIDTH)
            context (Tensor): Shape (batch_size, 1, latent_dim)
            hidden_state (tuple, optional): Previous hidden state of the LSTM.
        Returns:
            command_logits (Tensor): Shape (batch_size, 1, NODE_COMMAND_WIDTH)
            coord_output (Tensor): Shape (batch_size, 1, PREDICTED_COORDINATE_WIDTH)
            hidden_state (tuple): New hidden state of the LSTM.
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

    def forward(self, x, context=None):
        """
        Teacher-forcing training. Unrolls the sequence step by step.
        Args:
            x (Tensor): Ground truth sequence, shape (batch_size, seq_len, feature_size)
            context (Tensor): Context tensor, shape (batch_size, 1, latent_dim)
        """
        batch_size, seq_len, _ = x.shape
        all_command_logits = []
        all_coord_outputs = []
        hidden_state = None

        for i in range(seq_len):
            input_token = x[:, i : i + 1, :]
            command_logits, coord_output, hidden_state = self._forward_step(
                input_token, context, hidden_state
            )
            all_command_logits.append(command_logits)
            all_coord_outputs.append(coord_output)

        command_output = torch.cat(all_command_logits, dim=1)
        coord_output = torch.cat(all_coord_outputs, dim=1)
        return command_output, coord_output

    def generate_sequence(self, context, max_length=200, temperature=1.0):
        """
        Autoregressively generate a sequence of commands and coordinates.
        """
        from .command_defs import NodeCommand

        device = context.device
        batch_size = context.shape[0]
        eos_index = NodeCommand.encode_command("EOS")
        sos_index = NodeCommand.encode_command("SOS")

        # Start with SOS token
        current_token_full = torch.zeros(
            batch_size, 1, NODE_COMMAND_WIDTH + COORDINATE_WIDTH, device=device
        )
        current_token_full[:, 0, sos_index] = 1.0

        # Don't include SOS in outputs - only include predictions
        all_commands = []
        all_coords = []

        hidden_state = None

        for _ in range(max_length):
            command_logits, coord_pred, hidden_state = self._forward_step(
                current_token_full, context, hidden_state
            )

            # Get command prediction from logits
            command_probs = F.softmax(command_logits.squeeze(1) / temperature, dim=-1)
            predicted_command_idx = torch.argmax(command_probs, dim=1, keepdim=True)

            # Append predicted command and coords to lists
            next_command_onehot = F.one_hot(
                predicted_command_idx, num_classes=NODE_COMMAND_WIDTH
            ).float()
            all_commands.append(next_command_onehot)
            all_coords.append(coord_pred)

            # Check for EOS token
            if (predicted_command_idx.squeeze(1) == eos_index).all():
                break

            # Prepare next input token
            coord_padded = torch.zeros(batch_size, 1, COORDINATE_WIDTH, device=device)
            coord_padded[:, :, :PREDICTED_COORDINATE_WIDTH] = coord_pred
            current_token_full = torch.cat([next_command_onehot, coord_padded], dim=-1)

        command_output = torch.cat(all_commands, dim=1)
        coord_output = torch.cat(all_coords, dim=1)

        return command_output, coord_output

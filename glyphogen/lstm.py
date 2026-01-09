#!/usr/bin/env python
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from glyphogen.command_defs import NodeCommand
from glyphogen.hyperparameters import PROJ_SIZE


class LSTMDecoder(nn.Module):
    def __init__(self, d_model, latent_dim, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.proj_size = PROJ_SIZE
        self.rate = rate

        self.command_embedding = nn.Linear(NodeCommand.command_width, d_model)
        self.coord_embedding = nn.Linear(NodeCommand.coordinate_width, d_model)
        self.lstm = nn.LSTM(
            d_model + latent_dim, d_model, batch_first=True, proj_size=self.proj_size
        )
        self.layer_norm = nn.LayerNorm(self.proj_size)
        self.dropout = nn.Dropout(rate)
        self.output_command = nn.Linear(self.proj_size, NodeCommand.command_width)
        self.output_coords = nn.Linear(
            self.proj_size + NodeCommand.command_width, NodeCommand.coordinate_width
        )
        nn.init.zeros_(self.output_coords.bias)

        # Load and process coordinate statistics for normalization
        self._initialize_stats_buffers()

    def _initialize_stats_buffers(self):
        """Load stats and create broadcastable tensors for standardization."""
        try:
            stats = torch.load("coord_stats.pt")
        except FileNotFoundError:
            print(
                "Warning: coord_stats.pt not found. Using default (0,1) stats. "
                "Run analyze_dataset_stats.py to generate it."
            )
            stats = defaultdict(lambda: {"mean": 0.0, "std": 1.0})

        mean_tensor = torch.zeros(
            NodeCommand.command_width, NodeCommand.coordinate_width
        )
        std_tensor = torch.ones(
            NodeCommand.command_width, NodeCommand.coordinate_width
        )
        cmd_indices = {
            cmd: NodeCommand.encode_command(cmd)
            for cmd in NodeCommand.grammar.keys()
        }

        for cmd_name, cmd_idx in cmd_indices.items():
            if cmd_name == "M":
                mean_tensor[cmd_idx, 0] = stats["M_abs_x"]["mean"]
                std_tensor[cmd_idx, 0] = stats["M_abs_x"]["std"]
                mean_tensor[cmd_idx, 1] = stats["M_abs_y"]["mean"]
                std_tensor[cmd_idx, 1] = stats["M_abs_y"]["std"]
            elif cmd_name in ["L", "N", "NS", "NH", "NV", "NCI", "NCO"]:
                mean_tensor[cmd_idx, 0] = stats["L_rel_dx"]["mean"]
                std_tensor[cmd_idx, 0] = stats["L_rel_dx"]["std"]
                mean_tensor[cmd_idx, 1] = stats["L_rel_dy"]["mean"]
                std_tensor[cmd_idx, 1] = stats["L_rel_dy"]["std"]
            elif cmd_name == "LH":
                mean_tensor[cmd_idx, 0] = stats["L_rel_dx"]["mean"]
                std_tensor[cmd_idx, 0] = stats["L_rel_dx"]["std"]
            elif cmd_name == "LV":
                mean_tensor[cmd_idx, 0] = stats["L_rel_dy"]["mean"]
                std_tensor[cmd_idx, 0] = stats["L_rel_dy"]["std"]

            if cmd_name == "N":
                mean_tensor[cmd_idx, 2] = stats["C_in_dx"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_in_dx"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_in_dy"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_in_dy"]["std"]
                mean_tensor[cmd_idx, 4] = stats["C_out_dx"]["mean"]
                std_tensor[cmd_idx, 4] = stats["C_out_dx"]["std"]
                mean_tensor[cmd_idx, 5] = stats["C_out_dy"]["mean"]
                std_tensor[cmd_idx, 5] = stats["C_out_dy"]["std"]
            elif cmd_name == "NCI":
                mean_tensor[cmd_idx, 2] = stats["C_in_dx"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_in_dx"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_in_dy"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_in_dy"]["std"]
            elif cmd_name == "NCO":
                mean_tensor[cmd_idx, 2] = stats["C_out_dx"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_out_dx"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_out_dy"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_out_dy"]["std"]
            elif cmd_name == "NH":
                mean_tensor[cmd_idx, 2] = stats["C_in_dx"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_in_dx"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_out_dx"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_out_dx"]["std"]
            elif cmd_name == "NV":
                mean_tensor[cmd_idx, 2] = stats["C_in_dy"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_in_dy"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_out_dy"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_out_dy"]["std"]
            elif cmd_name == "NS":
                mean_tensor[cmd_idx, 2] = stats["NS_angle"]["mean"]
                std_tensor[cmd_idx, 2] = stats["NS_angle"]["std"]
                mean_tensor[cmd_idx, 3] = stats["NS_len_in"]["mean"]
                std_tensor[cmd_idx, 3] = stats["NS_len_in"]["std"]
                mean_tensor[cmd_idx, 4] = stats["NS_len_out"]["mean"]
                std_tensor[cmd_idx, 4] = stats["NS_len_out"]["std"]

        self.register_buffer("mean_tensor", mean_tensor)
        self.register_buffer("std_tensor", std_tensor)

    def get_stats_for_sequence(self, command_indices):
        means = self.mean_tensor[command_indices]
        stds = self.std_tensor[command_indices]
        return means, stds

    def standardize(self, coords, means, stds):
        return (coords - means) / stds

    def de_standardize(self, coords_std, means, stds):
        return coords_std * stds + means

    def _forward_step(self, input_token_std, context, hidden_state=None):
        """
        Performs a single decoding step in standardized coordinate space.
        """
        command_input = input_token_std[:, :, : NodeCommand.command_width].float()
        coord_input_std = input_token_std[
            :,
            :,
            NodeCommand.command_width : NodeCommand.command_width
            + NodeCommand.coordinate_width,
        ].float()

        command_emb = self.command_embedding(command_input)
        coord_emb = self.coord_embedding(coord_input_std)
        x = command_emb + coord_emb
        x = self.dropout(x)

        if context is not None:
            x = torch.cat([x, context], dim=-1)

        x, hidden_state = self.lstm(x, hidden_state)
        x = self.layer_norm(x)

        command_logits = self.output_command(x)
        coord_head_input = torch.cat([x, command_logits], dim=-1)
        coord_output_std = self.output_coords(coord_head_input)

        return command_logits, coord_output_std, hidden_state

    def forward(self, x_std, context=None, teacher_forcing_ratio=1.0):
        """
        Training forward pass with scheduled sampling.
        Operates entirely in STANDARDIZED coordinate space.
        `x_std` is the ground truth sequence, already standardized.
        The returned `coord_output_std` is also in STANDARDIZED space.
        """
        batch_size, seq_len, _ = x_std.shape
        current_input_std = x_std[:, 0:1, :]
        hidden_state = None

        all_command_logits = []
        all_coord_outputs_std = []

        for i in range(seq_len):
            command_logits, coord_output_std, hidden_state = self._forward_step(
                current_input_std, context, hidden_state
            )
            all_command_logits.append(command_logits)
            all_coord_outputs_std.append(coord_output_std)

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if i + 1 < seq_len:
                if use_teacher_forcing:
                    current_input_std = x_std[:, i + 1 : i + 2, :]
                else:
                    command_probs = F.softmax(command_logits.squeeze(1), dim=-1)
                    predicted_command_idx = torch.argmax(
                        command_probs, dim=1, keepdim=True
                    )
                    next_command_onehot = F.one_hot(
                        predicted_command_idx, num_classes=NodeCommand.command_width
                    ).float()
                    current_input_std = torch.cat(
                        [next_command_onehot, coord_output_std], dim=-1
                    )

        command_output = torch.cat(all_command_logits, dim=1)
        coord_output_std = torch.cat(all_coord_outputs_std, dim=1)

        return command_output, coord_output_std

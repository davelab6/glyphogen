#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch import nn

from glyphogen.command_defs import (
    NODE_GLYPH_COMMANDS,
    MAX_COORDINATE,
)
from glyphogen.lstm import LSTMDecoder
from glyphogen.losses import losses, unroll_relative_coords


@torch.compile(fullgraph=True)
class VectorizationGenerator(nn.Module):
    def __init__(self, d_model, latent_dim=32, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.rate = rate

        self.conv1 = nn.Conv2d(1, 16, 7, padding=3, stride=2)
        self.norm1 = nn.LayerNorm([16, 256, 256])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2, stride=2)
        self.norm2 = nn.LayerNorm([32, 128, 128])
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.norm3 = nn.LayerNorm([64, 64, 64])
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.norm4 = nn.LayerNorm([128, 32, 32])
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.norm5 = nn.LayerNorm([256, 16, 16])
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(rate)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256 * 16 * 16, latent_dim)
        self.norm_dense = nn.LayerNorm(latent_dim)
        self.sigmoid = nn.Sigmoid()
        self.output_dense = nn.Linear(latent_dim, latent_dim)
        self.contour_head = nn.Linear(latent_dim, 1)
        torch.nn.init.ones_(self.contour_head.bias)

        self.decoder = torch.compile(
            LSTMDecoder(d_model=d_model, latent_dim=latent_dim, rate=rate)
        )
        self.arg_counts = torch.tensor(
            list(NODE_GLYPH_COMMANDS.values()), dtype=torch.long
        )

        self.use_raster = False

    @torch.compile
    def encode(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.norm_dense(x)
        x = self.sigmoid(x)
        z = self.output_dense(x)
        return z

    @torch.compile
    def forward(self, inputs):
        raster_image_input = inputs["raster_image"]
        target_sequence_input = inputs["target_sequence"]
        z = self.encode(raster_image_input)

        pred_contour_count = self.contour_head(z)

        z_unsq = z.unsqueeze(1)

        command_output, coord_output_relative = self.decoder(
            target_sequence_input, context=z_unsq
        )

        # Scale up relative coordinates
        coord_output_relative_scaled = coord_output_relative * MAX_COORDINATE

        coord_output_absolute = unroll_relative_coords(
            command_output, coord_output_relative_scaled
        )
        return {
            "command": command_output,
            "coord_relative": coord_output_relative,
            "coord_absolute": coord_output_absolute,
            "pred_contour_count": pred_contour_count,
        }


def step(model, batch, step, val=False):
    # with torch.profiler.record_function("h2d copy"):
    device = next(model.parameters()).device
    (inputs, y) = batch
    inputs = {k: v.to(device) for k, v in inputs.items()}
    y = {k: v.to(device) for k, v in y.items()}

    # with torch.profiler.record_function("forward step"):
    outputs = model(inputs)
    # with torch.profiler.record_function("losses"):
    loss_values = losses(y, inputs, outputs, step, device, model.arg_counts, val=val)
    return loss_values, outputs

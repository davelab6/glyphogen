#!/usr/bin/env python
from collections import defaultdict
from glyphogen.glyph import NodeGlyph
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from glyphogen.command_defs import (
    NODE_GLYPH_COMMANDS,
    COORDINATE_WIDTH,
    PREDICTED_COORDINATE_WIDTH,
)
from glyphogen.coordinate import (
    image_space_to_mask_space,
    mask_space_to_image_space,
)
from glyphogen.hyperparameters import GEN_IMAGE_SIZE
from glyphogen.lstm import LSTMDecoder
from glyphogen.losses import losses
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

DEBUG = True


def get_model_instance_segmentation(num_classes, load_pretrained=True):
    if load_pretrained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="MaskRCNN_ResNet50_FPN_Weights.DEFAULT"
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


class VectorizationGenerator(nn.Module):
    def __init__(self, segmenter_state, d_model, latent_dim=32, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.rate = rate
        self.img_size = GEN_IMAGE_SIZE[0]

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

        # Load and freeze the segmentation model
        self.segmenter = get_model_instance_segmentation(
            num_classes=3, load_pretrained=False
        )
        self.segmenter.load_state_dict(segmenter_state)
        self.segmenter.eval()
        for param in self.segmenter.parameters():
            param.requires_grad = False

        # The new context will be the latent vector of the normalized mask
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
    def forward(self, raster_image, gt_targets=None, teacher_forcing_ratio=1.0):
        # This model implements the canonical mask normalization strategy.
        # It processes one image at a time (batch size should be 1).
        if raster_image.shape[0] > 1:
            raise NotImplementedError(
                "Canonical model currently supports a batch size of 1."
            )

        # Step 1: Image goes through segmentation model, we get boxes and masks
        if gt_targets is not None:
            # During training, we use the ground truth boxes and masks for stability.
            if "gt_contours" not in gt_targets:
                raise ValueError("During training, gt_targets must be provided.")
            gt_contours = gt_targets["gt_contours"]
            contour_boxes = torch.stack([c["box"] for c in gt_contours])
            contour_masks = torch.stack([c["mask"] for c in gt_contours])
            target_sequences = [c["sequence"] for c in gt_contours]
        else:  # Inference
            with torch.no_grad():
                segmenter_output = self.segmenter(raster_image)[0]
            if not segmenter_output["masks"].numel():
                return {"pred_commands": [], "pred_coords": []}

            areas = (
                segmenter_output["boxes"][:, 2] - segmenter_output["boxes"][:, 0]
            ) * (segmenter_output["boxes"][:, 3] - segmenter_output["boxes"][:, 1])
            sorted_indices = torch.argsort(areas, descending=True)
            contour_boxes = segmenter_output["boxes"][sorted_indices]
            contour_masks = segmenter_output["masks"][sorted_indices].squeeze(1)
            target_sequences = None

        # Step 2: For each contour, get a latent vector and decode the sequence.
        glyph_pred_commands = []
        glyph_pred_coords_img_space = []

        command_width = len(NODE_GLYPH_COMMANDS)

        for i in range(len(contour_boxes)):
            box = contour_boxes[i].clamp(min=0, max=raster_image.shape[-1] - 1)
            mask = contour_masks[i]
            x1, y1, x2, y2 = box.long()
            if x1 >= x2 or y1 >= y2:
                continue

            cropped_mask = mask[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
            normalized_mask = F.interpolate(
                cropped_mask.to(torch.float32),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            z = self.encode(normalized_mask).unsqueeze(1)

            if target_sequences is not None:  # Training
                gt_sequence = target_sequences[i].unsqueeze(0)
                gt_commands = gt_sequence[:, :, :command_width]
                gt_coords_img_space = gt_sequence[:, :, command_width:]

                # Normalize GT coords for teacher forcing
                gt_coords_norm = image_space_to_mask_space(
                    gt_coords_img_space.squeeze(0), box
                ).unsqueeze(0)

                # Prepare decoder input
                decoder_input_norm = torch.cat([gt_commands, gt_coords_norm], dim=-1)
                decoder_input = decoder_input_norm[:, :-1, :]

                pred_commands, pred_coords_norm = self.decoder(
                    decoder_input,
                    context=z,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )
            else:  # Inference
                # Autoregressive generation loop
                hidden_state = None

                from .command_defs import NodeCommand

                device = raster_image.device
                batch_size = 1  # We process one image at a time
                sos_index = NodeCommand.encode_command("SOS")

                command_part = torch.zeros(batch_size, 1, command_width, device=device)
                command_part[:, 0, sos_index] = 1.0

                coords_part_img_space = torch.zeros(
                    batch_size, 1, COORDINATE_WIDTH, device=device
                )
                coords_part_norm = image_space_to_mask_space(
                    coords_part_img_space.squeeze(0), box
                ).unsqueeze(0)

                current_input = torch.cat([command_part, coords_part_norm], dim=-1)

                # Lists for the current contour's timesteps
                contour_timestep_commands = []
                contour_timestep_coords_norm = []

                for _ in range(50):  # max_length
                    command_logits, coord_output_norm, hidden_state = (
                        self.decoder._forward_step(current_input, z, hidden_state)
                    )

                    contour_timestep_commands.append(command_logits)
                    contour_timestep_coords_norm.append(coord_output_norm)

                    command_probs = F.softmax(command_logits.squeeze(1), dim=-1)
                    predicted_command_idx = torch.argmax(
                        command_probs, dim=1, keepdim=True
                    )

                    if predicted_command_idx.item() == NodeCommand.encode_command(
                        "EOS"
                    ):
                        break

                    next_command_onehot = F.one_hot(
                        predicted_command_idx, num_classes=command_width
                    ).float()

                    coord_padded = torch.zeros(
                        batch_size, 1, COORDINATE_WIDTH, device=device
                    )
                    coord_padded[:, :, :PREDICTED_COORDINATE_WIDTH] = coord_output_norm

                    current_input = torch.cat(
                        [next_command_onehot, coord_padded], dim=-1
                    )

                if not contour_timestep_commands:
                    pred_commands = torch.empty(
                        batch_size, 0, command_width, device=device
                    )
                    pred_coords_norm = torch.empty(
                        batch_size, 0, PREDICTED_COORDINATE_WIDTH, device=device
                    )
                else:
                    pred_commands = torch.cat(contour_timestep_commands, dim=1)
                    pred_coords_norm = torch.cat(contour_timestep_coords_norm, dim=1)

            # Denormalize predicted coordinates back to image space for loss calculation
            pred_coords_img_space = mask_space_to_image_space(
                pred_coords_norm.squeeze(0), box
            )

            glyph_pred_commands.append(pred_commands.squeeze(0))
            glyph_pred_coords_img_space.append(pred_coords_img_space)

        return {
            "pred_commands": glyph_pred_commands,
            "pred_coords_img_space": glyph_pred_coords_img_space,
            "used_teacher_forcing": target_sequences is not None,
        }


def step(model, batch, writer, global_step, teacher_forcing_ratio=1.0):
    device = next(model.parameters()).device
    images, targets = batch

    # Accumulate losses and outputs over the batch
    batch_losses = defaultdict(list)
    all_outputs = []

    # Process each item in the batch individually, as the model's
    # forward pass is designed for a single, complex sample.
    for i in range(len(images)):
        img = images[i].to(device)
        y = targets[i]

        # Move nested ground truth tensors to device
        gt_contours = y["gt_contours"]
        for contour in gt_contours:
            for key, value in contour.items():
                if isinstance(value, torch.Tensor):
                    contour[key] = value.to(device)

        # Run the model for a single item
        # For debugging, we use teacher forcing for validation as well.
        if model.training:
            outputs = model(
                img.unsqueeze(0),
                gt_targets=y,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
        else:  # Validation/Inference
            # Always use teacher forcing during validation for speed
            outputs = model(img.unsqueeze(0), gt_targets=y, teacher_forcing_ratio=1.0)
        all_outputs.append(outputs)

        # Calculate loss for the single item
        loss_values = losses(y, outputs, device, validation=not model.training)

        if DEBUG and writer is not None and global_step % 100 == 0:
            dump_debug_sequences(writer, global_step, i, gt_contours, outputs)
        # Append loss tensors to lists for later averaging
        if "total_loss" in loss_values:
            for key, val in loss_values.items():
                batch_losses[key].append(val)

    # Create final loss dictionary for the training loop
    if batch_losses:
        # Average losses across the batch
        final_losses = {k: torch.stack(v).mean() for k, v in batch_losses.items()}
    else:
        print("Warning: No valid samples in batch.")
        # Handle empty or invalid batch
        final_losses = {}
        final_losses["total_loss"] = torch.tensor(
            0.0, device=device, requires_grad=True
        )
        final_losses["command_loss"] = torch.tensor(0.0)
        final_losses["coord_loss"] = torch.tensor(0.0)

    return final_losses, all_outputs


def dump_debug_sequences(writer, global_step, i, gt_contours, outputs):
    """For debugging, dump ground truth and predicted sequences"""
    pred_commands_and_coords = [
        (
            outputs["pred_commands"][idx].detach(),
            outputs["pred_coords_img_space"][idx].detach(),
        )
        for idx in range(len(outputs["pred_commands"]))
    ]
    gt_commands_and_coords = []
    for contour_idx in range(len(gt_contours)):
        gt_sequence = gt_contours[contour_idx]["sequence"]
        command_width = len(NODE_GLYPH_COMMANDS)
        gt_command = gt_sequence[:, :command_width].detach()
        gt_coords_img_space = gt_sequence[:, command_width:].detach()
        gt_commands_and_coords.append((gt_command, gt_coords_img_space))
    pred_glyph = NodeGlyph.from_numpy(pred_commands_and_coords)
    debug_string = pred_glyph.to_svg_glyph().to_svg_string()
    gt_glyph = NodeGlyph.from_numpy(gt_commands_and_coords)
    gt_debug_string = gt_glyph.to_svg_glyph().to_svg_string()
    writer.add_text(
        f"SVG/Debug_{i}",
        f"GT: {gt_debug_string}\nPred: {debug_string}",
        global_step,
    )

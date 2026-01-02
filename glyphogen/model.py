#!/usr/bin/env python
from collections import defaultdict
from typing import Optional, Tuple, List
from jaxtyping import Float

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN

from glyphogen.command_defs import NodeCommand
from glyphogen.typing import (
    GroundTruthContour,
    LossDictionary,
    ModelResults,
    SegmenterOutput,
    Target,
)
from glyphogen.hyperparameters import GEN_IMAGE_SIZE
from glyphogen.losses import (
    dump_debug_sequences,
    losses,
)
from glyphogen.lstm import LSTMDecoder

DEBUG = True


class Mask:
    bounds: Tuple[float, float, float, float]
    mask_tensor: torch.Tensor

    def __init__(self, bounds, mask_tensor):
        self.bounds = bounds
        self.mask_tensor = mask_tensor


def get_model_instance_segmentation(num_classes, load_pretrained=True) -> MaskRCNN:
    if load_pretrained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="MaskRCNN_ResNet50_FPN_Weights.DEFAULT"
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


class VectorizationGenerator(nn.Module):
    def __init__(
        self, segmenter_state, d_model: int, latent_dim: int = 32, rate: float = 0.1
    ):
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
        self.segmenter: MaskRCNN = get_model_instance_segmentation(
            num_classes=3, load_pretrained=False
        )
        self.segmenter.load_state_dict(segmenter_state)
        self.segmenter.eval()
        for param in self.segmenter.parameters():
            param.requires_grad = False

        # The new context will be the latent vector of the normalized mask
        self.decoder = LSTMDecoder(d_model=d_model, latent_dim=latent_dim, rate=rate)
        self.arg_counts: torch.Tensor = torch.tensor(
            list(NodeCommand.grammar.values()), dtype=torch.long
        )

        self.use_raster = False

    def encode(self, inputs):
        """Return a latent vector encoding of the input mask images."""
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

    @torch.compile(dynamic=True)
    def get_boxes_and_masks(
        self, raster_image: torch.Tensor, gt_targets: Optional[Target] = None
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[list[torch.Tensor]]
    ]:
        if gt_targets is not None:
            # During training, we use the ground truth boxes and masks for stability.
            gt_contours = gt_targets["gt_contours"]
            if not gt_contours:
                return None, None, None
            contour_boxes = torch.stack([c["box"] for c in gt_contours])
            contour_masks = torch.stack([c["mask"] for c in gt_contours])
            target_sequences = [c["sequence"] for c in gt_contours]
        else:  # Inference
            with torch.no_grad():
                segmenter_output: SegmenterOutput = self.segmenter(raster_image)[0]
            if not segmenter_output["masks"].numel():
                return None, None, None

            # Sort by mask area descending
            areas = (
                segmenter_output["boxes"][:, 2] - segmenter_output["boxes"][:, 0]
            ) * (segmenter_output["boxes"][:, 3] - segmenter_output["boxes"][:, 1])
            sorted_indices = torch.argsort(areas, descending=True)
            contour_boxes = segmenter_output["boxes"][sorted_indices]
            contour_masks = segmenter_output["masks"][sorted_indices].squeeze(1)
            target_sequences = None
        return contour_boxes, contour_masks, target_sequences

    @torch.compile
    def forward(
        self, raster_image, gt_targets=None, teacher_forcing_ratio=1.0
    ) -> ModelResults:
        # This model implements the canonical mask normalization strategy.
        # It processes one image at a time (batch size should be 1).
        if raster_image.shape[0] > 1:
            raise NotImplementedError(
                "Canonical model currently supports a batch size of 1."
            )

        # Step 1: Image goes through segmentation model, we get boxes and masks
        contour_boxes, contour_masks, target_sequences = self.get_boxes_and_masks(
            raster_image, gt_targets
        )
        if contour_boxes is None or contour_masks is None or target_sequences is None:
            return ModelResults.empty()

        # Step 2: Batch process all contours

        # Collect and normalize masks
        normalized_masks = []
        valid_boxes = []
        for i in range(len(contour_boxes)):
            box = contour_boxes[i].clamp(min=0, max=raster_image.shape[-1] - 1)
            mask = contour_masks[i]
            x1, y1, x2, y2 = box.long()
            if x1 >= x2 or y1 >= y2:
                continue

            valid_boxes.append(box)
            cropped_mask = mask[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
            normalized_mask = F.interpolate(
                cropped_mask.to(torch.float32),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            normalized_masks.append(normalized_mask)

        if not normalized_masks:
            return ModelResults.empty()

        # Send the normalized masks through the latent vector encoder
        normalized_masks_batch = torch.cat(normalized_masks, dim=0)
        z_batch = self.encode(normalized_masks_batch).unsqueeze(1)

        # and then decode a seuence for each contour
        if target_sequences is not None:  # Training
            return self.teacher_forcing(
                target_sequences, valid_boxes, z_batch, teacher_forcing_ratio
            )
        else:  # Inference
            return self.autoregression(raster_image, z_batch, valid_boxes)

    def teacher_forcing(
        self, target_sequences, valid_boxes, z_batch, teacher_forcing_ratio
    ) -> ModelResults:
        glyph_pred_commands = []
        glyph_pred_coords_norm = []
        glyph_pred_coords_img_space = []
        # Prepare batch for decoder
        decoder_inputs_norm = [
            NodeCommand.image_space_to_mask_space(seq, box)
            for seq, box in zip(target_sequences, valid_boxes)
        ]

        # Pad sequences
        padded_decoder_inputs = torch.nn.utils.rnn.pad_sequence(
            decoder_inputs_norm, batch_first=True, padding_value=0.0
        )

        # Prepare decoder input (all but last token)
        decoder_input_batch = padded_decoder_inputs[:, :-1, :]

        pred_commands_batch, pred_coords_norm_batch = self.decoder(
            decoder_input_batch,
            context=z_batch,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # Unpad and denormalize
        for i in range(len(valid_boxes)):
            box = valid_boxes[i]
            seq_len = decoder_inputs_norm[i].shape[0] - 1  # Original length

            pred_commands = pred_commands_batch[i, :seq_len, :]
            pred_coords_norm = pred_coords_norm_batch[i, :seq_len, :]

            glyph_pred_commands.append(pred_commands)
            glyph_pred_coords_norm.append(pred_coords_norm)

            # Also create image space version for logging/decoding
            pred_sequence_norm = torch.cat([pred_commands, pred_coords_norm], dim=-1)
            # To convert back, we need a full sequence including SOS
            sos_token = decoder_inputs_norm[i][0:1, :]
            full_pred_sequence_norm = torch.cat([sos_token, pred_sequence_norm], dim=0)

            pred_sequence_img_space = NodeCommand.mask_space_to_image_space(
                full_pred_sequence_norm, box
            )
            # Return the coordinate part, excluding the SOS token's coords
            glyph_pred_coords_img_space.append(
                pred_sequence_img_space[1:, NodeCommand.command_width :]
            )
        return ModelResults(
            pred_commands=glyph_pred_commands,
            pred_coords_norm=glyph_pred_coords_norm,
            pred_coords_img_space=glyph_pred_coords_img_space,
            used_teacher_forcing=True,
            contour_boxes=valid_boxes,
        )

    def autoregression(
        self, raster_image, z_batch, valid_boxes: List[Float[torch.Tensor, "4"]]
    ) -> ModelResults:
        glyph_pred_commands = []
        glyph_pred_coords_norm = []
        glyph_pred_coords_img_space = []
        # Autoregressive generation for each contour in the batch
        # This part is still sequential per time step, but batched over contours
        batch_size = len(valid_boxes)
        device = raster_image.device
        sos_index = NodeCommand.encode_command("SOS")

        command_part = torch.zeros(
            batch_size, 1, NodeCommand.command_width, device=device
        )
        command_part[:, 0, sos_index] = 1.0

        coords_part_img_space = torch.zeros(
            batch_size, 1, NodeCommand.coordinate_width, device=device
        )

        # This part of image_space_to_mask_space is tricky to batch as boxes are different
        # However, since coords are all zero, we can simplify
        coords_part_norm = (coords_part_img_space * 2) - 1

        current_input = torch.cat([command_part, coords_part_norm], dim=-1)
        hidden_state = None

        # Lists to store the full sequences for each item in the batch
        batch_contour_commands = [[] for _ in range(batch_size)]
        batch_contour_coords_norm = [[] for _ in range(batch_size)]

        active_indices = list(range(batch_size))

        for _ in range(50):  # max_length
            if not active_indices:
                break

            active_z = z_batch[active_indices]

            command_logits, coord_output_norm, hidden_state = (
                self.decoder._forward_step(current_input, active_z, hidden_state)
            )

            # Store results for active contours
            for i, original_idx in enumerate(active_indices):
                batch_contour_commands[original_idx].append(command_logits[i : i + 1])
                batch_contour_coords_norm[original_idx].append(
                    coord_output_norm[i : i + 1]
                )

            command_probs = F.softmax(command_logits.squeeze(1), dim=-1)
            predicted_command_idx = torch.argmax(command_probs, dim=1, keepdim=True)

            # Check for EOS and remove finished sequences from the active batch
            eos_mask = predicted_command_idx.squeeze(1) == NodeCommand.encode_command(
                "EOS"
            )

            if any(eos_mask):
                active_indices_mask = ~eos_mask
                active_indices = [
                    idx
                    for i, idx in enumerate(active_indices)
                    if active_indices_mask[i]
                ]
                current_input = current_input[active_indices_mask]
                if hidden_state is not None:
                    h, c = hidden_state
                    hidden_state = (
                        h[:, active_indices_mask, :],
                        c[:, active_indices_mask, :],
                    )
                if not active_indices:
                    break
                predicted_command_idx = predicted_command_idx[active_indices_mask]
                coord_output_norm = coord_output_norm[active_indices_mask]

            next_command_onehot = F.one_hot(
                predicted_command_idx,
                num_classes=NodeCommand.command_width,
            ).float()

            coord_padded = torch.zeros(
                next_command_onehot.shape[0],
                1,
                NodeCommand.coordinate_width,
                device=device,
            )
            coord_padded[:, :, : NodeCommand.coordinate_width] = coord_output_norm

            current_input = torch.cat([next_command_onehot, coord_padded], dim=-1)

        # Process and denormalize results
        for i in range(batch_size):
            if not batch_contour_commands[i]:
                pred_commands = torch.empty(0, NodeCommand.command_width, device=device)
                pred_coords_norm = torch.empty(
                    0, NodeCommand.coordinate_width, device=device
                )
            else:
                # Squeeze out the time dimension from the list of tensors
                pred_commands = torch.cat(batch_contour_commands[i], dim=1).squeeze(0)
                pred_coords_norm = torch.cat(
                    batch_contour_coords_norm[i], dim=1
                ).squeeze(0)

            glyph_pred_commands.append(pred_commands)
            glyph_pred_coords_norm.append(pred_coords_norm)

            # Also create image space version for logging/decoding
            pred_sequence_norm = torch.cat([pred_commands, pred_coords_norm], dim=-1)

            # We need a full sequence to convert back to image space.
            # The first token is always SOS with zero coordinates in normalized space.
            sos_cmd = (
                F.one_hot(
                    torch.tensor(NodeCommand.encode_command("SOS")),
                    num_classes=NodeCommand.command_width,
                )
                .float()
                .to(device)
            )
            sos_coords = torch.zeros(NodeCommand.coordinate_width, device=device)
            sos_token = torch.cat([sos_cmd, sos_coords]).unsqueeze(0)

            full_pred_sequence_norm = torch.cat([sos_token, pred_sequence_norm], dim=0)

            pred_sequence_img_space = NodeCommand.mask_space_to_image_space(
                full_pred_sequence_norm, valid_boxes[i]
            )
            glyph_pred_coords_img_space.append(
                pred_sequence_img_space[1:, NodeCommand.command_width :]
            )
        return ModelResults(
            pred_commands=glyph_pred_commands,
            pred_coords_norm=glyph_pred_coords_norm,
            pred_coords_img_space=glyph_pred_coords_img_space,
            used_teacher_forcing=False,
            contour_boxes=valid_boxes,
        )


def step(
    model, batch, writer, global_step, teacher_forcing_ratio=1.0
) -> Tuple[LossDictionary, List[ModelResults]]:
    device = next(model.parameters()).device
    images, targets = batch

    # Accumulate losses and outputs over the batch
    batch_losses = defaultdict(list)
    all_outputs = []

    # Process each item in the batch individually, as the model's
    # forward pass is designed for a single, complex sample.
    for i in range(len(images)):
        img = images[i].to(device)
        y: Target = targets[i]

        # Move nested ground truth tensors to device
        gt_contours: List[GroundTruthContour] = y["gt_contours"]
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
            dump_debug_sequences(
                writer, global_step, i, gt_contours, outputs, loss_values
            )
        # Append loss tensors to lists for later averaging
        if "total_loss" in loss_values:
            for key, val in loss_values.items():
                batch_losses[key].append(val)

    # Create final loss dictionary for the training loop
    if batch_losses:
        # Average losses across the batch
        final_losses = {
            "total_loss": torch.stack(batch_losses["total_loss"]).mean(),
            "command_loss": torch.stack(batch_losses["command_loss"]).mean(),
            "coord_loss": torch.stack(batch_losses["coord_loss"]).mean(),
            "signed_area_loss": torch.stack(batch_losses["signed_area_loss"]).mean(),
            "command_accuracy_metric": torch.stack(
                batch_losses["command_accuracy_metric"]
            ).mean(),
            "coordinate_mae_metric": torch.stack(
                batch_losses["coordinate_mae_metric"]
            ).mean(),
        }
    else:
        print("Warning: No valid samples in batch.")
        # Handle empty or invalid batch
        final_losses: LossDictionary = {
            "total_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "command_loss": torch.tensor(0.0, device=device),
            "coord_loss": torch.tensor(0.0, device=device),
            "signed_area_loss": torch.tensor(0.0, device=device),
            "command_accuracy_metric": torch.tensor(0.0, device=device),
            "coordinate_mae_metric": torch.tensor(0.0, device=device),
        }

    return final_losses, all_outputs

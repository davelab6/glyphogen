#!/usr/bin/env python
import os
import torch
import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pkbar

from glyphogen_torch.dataset import get_full_model_data, get_pretrain_data, collate_fn
from glyphogen_torch.model import (
    GlyphGenerator,
    VectorizationGenerator,
)
from glyphogen_torch.callbacks import log_images, log_svgs, log_pretrain_rasters
from glyphogen_torch.rasterizer import rasterize_batch

from glyphogen_torch.hyperparameters import (
    BATCH_SIZE,
    NUM_GLYPHS,
    LATENT_DIM,
    D_MODEL,
    RASTER_LOSS_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
    RATE,
    EPOCHS,
    LEARNING_RATE,
    SCHEDULER_STEP,
    SCHEDULER_GAMMA
)


def main(
    model_name="glyphogen.pt",
    pre_train=False,
    epochs=EPOCHS,
    vectorizer_model_name=None,
    single_batch=False,
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # torch.autograd.set_detect_anomaly(True)

    # Model
    if os.path.exists(model_name) and not vectorizer_model_name:
        model = torch.load(model_name, map_location=device)
        print(f"Loaded model from {model_name}")
    else:
        model = GlyphGenerator(
            num_glyphs=NUM_GLYPHS,
            d_model=D_MODEL,
            latent_dim=LATENT_DIM,
            rate=RATE,
        ).to(device)

    if vectorizer_model_name:
        model.vectorizer.load_state_dict(
            torch.load(vectorizer_model_name, map_location=device)
        )
        print(f"Loaded vectorizer from {vectorizer_model_name}")

    model_to_train = model.vectorizer if pre_train else model
    model_save_name = (
        model_name.replace(".pt", ".vectorizer.pt") if pre_train else model_name
    )

    # Data
    if pre_train:
        train_dataset, test_dataset = get_pretrain_data()
    else:
        train_dataset, test_dataset = get_full_model_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=not isinstance(train_dataset, torch.utils.data.IterableDataset),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    if single_batch:
        print("Reducing dataset to a single batch for overfitting test")
        train_loader = [next(iter(train_loader))] * 32  # Repeat the same batch
        test_loader = train_loader

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)

    # Training Loop
    writer = SummaryWriter(
        f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    writer.add_text(
        f"Hyperparameters", open("glyphogen_torch/hyperparameters.py").read(), 0
    )
    best_val_loss = float("inf")
    global_step = 0
    LOSSES = ["total_loss", "command_loss", "coord_loss", "raster_loss"]
    if pre_train:
        LOSSES += [
            "point_placement_contour_loss",
            "point_placement_eos_loss",
            "point_placement_handle_loss",
        ]

    for epoch in range(epochs):
        print()
        model_to_train.train()
        total_train_loss = 0
        kbar = pkbar.Kbar(
            target=len(train_loader),
            epoch=epoch,
            num_epochs=epochs,
            width=8,
            always_stateful=False,
        )
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            losses = model_to_train.step(batch, step=global_step)
            losses["total_loss"].backward()
            optimizer.step()
            total_train_loss += losses["total_loss"].item()

            kbar.update(
                i,
                values=[
                    (label.replace("_loss", ""), losses[label]) for label in LOSSES
                ],
            )

            for label in LOSSES:
                writer.add_scalar(
                    f"Loss/{label.replace('loss','batch')}",
                    losses[label].item(),
                    global_step,
                )
            if i % 10 == 0:
                writer.flush()

            global_step += 1
            if global_step % SCHEDULER_STEP == 0:
                scheduler.step()

        avg_train_loss = total_train_loss / (i + 1)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        # Validation
        if not single_batch:
            model_to_train.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    losses = model_to_train.step(batch, step=global_step)
                    total_val_loss += losses["total_loss"].item()

            avg_val_loss = (
                total_val_loss / len(test_loader) if len(test_loader) > 0 else 0
            )
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

            # Checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model_to_train.state_dict(), model_save_name)
                print(f"Saved best model to {model_save_name}")

        # Callbacks
        if pre_train:
            log_pretrain_rasters(model_to_train, test_loader, writer, epoch)
        else:
            log_images(model_to_train, test_loader, writer, epoch, pre_train)
        log_svgs(model_to_train, test_loader, writer, epoch, pre_train)

        # Log the learning rate
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the Glyph Generator model in PyTorch."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="glyphogen.pt",
        help="Name of the model to save.",
    )
    parser.add_argument(
        "--pre-train",
        action="store_true",
        help="Whether to pre-train the vectorizer.",
    )
    parser.add_argument(
        "--vectorizer_model_name",
        type=str,
        default=None,
        help="Name of the pre-trained vectorizer model to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--single-batch",
        action="store_true",
        help="Whether to use a single batch for training.",
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        pre_train=args.pre_train,
        epochs=args.epochs,
        single_batch=args.single_batch,
        vectorizer_model_name=args.vectorizer_model_name,
    )

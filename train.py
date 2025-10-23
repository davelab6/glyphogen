#!/usr/bin/env python
from collections import defaultdict
import datetime
import os

import pkbar
import torch
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.tensorboard import SummaryWriter
import random

from glyphogen.callbacks import (
    log_images,
    log_pretrain_rasters,
    log_svgs,
    init_confusion_matrix_state,
    collect_confusion_matrix_data,
    log_confusion_matrix,
)
from glyphogen.dataset import collate_fn, get_full_model_data, get_pretrain_data
from glyphogen.hyperparameters import (
    BATCH_SIZE,
    D_MODEL,
    EPOCHS,
    LATENT_DIM,
    LEARNING_RATE,
    NUM_GLYPHS,
    RATE,
    FINAL_LEARNING_RATE,
)
from glyphogen.model import (
    GlyphGenerator,
    VectorizationGenerator,
    SKIP_RASTERIZATION,
)


def dump_accumulators(accumulators, writer, epoch, batch_idx, val=False):
    prefix = "Val" if val else "Train"
    for key, value in accumulators.items():
        avg_value = value / (batch_idx + 1)
        if key.endswith("_loss"):
            key = key.replace("_loss", "")
            writer.add_scalar(f"{prefix}Loss/{key}", avg_value, epoch)
        else:
            key = key.replace("_metric", "")
            writer.add_scalar(f"{prefix}Metric/{key}", avg_value, epoch)


def write_gradient_norms(model, losses, writer, step):
    for key in losses.keys():
        if key == "total_loss":
            continue
        if losses[key].grad_fn is None:
            continue
        coord_grads = torch.autograd.grad(
            losses[key],
            model.parameters(),
            retain_graph=True,
            allow_unused=True,
        )
        norm = torch.norm(
            torch.stack([torch.norm(g, 2.0) for g in coord_grads if g is not None]),
            2.0,
        )
        writer.add_scalar(f"GradNorm/{key}", norm, step)


def main(
    model_name="glyphogen.pt",
    pre_train=False,
    epochs=EPOCHS,
    vectorizer_model_name=None,
    single_batch=False,
    debug_grads=False,
    augmentations=20,
    roll_augmentations=2,
):
    random.seed(1234)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch._dynamo.config.capture_scalar_outputs = True
    # torch._dynamo.config.capture_dynamic_output_shape_ops = True
    print(f"Using device: {device}")
    # torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(1234)

    # Model
    if os.path.exists(model_name) and not vectorizer_model_name:
        model: GlyphGenerator = torch.load(model_name, map_location=device)
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
        train_dataset, test_dataset = get_pretrain_data(
            augmentations=augmentations, roll_augmentations=roll_augmentations
        )
    else:
        train_dataset, test_dataset = get_full_model_data()

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        worker_init_fn = seed_worker,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        worker_init_fn = seed_worker,
        num_workers=2,
    )

    if single_batch:
        print("Reducing dataset for overfitting test")
        loader_iter = iter(train_loader)
        train_loader = [next(loader_iter)] * 32  # for _ in range(32)]
        test_loader = train_loader

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=LEARNING_RATE)
    # Work out gamma from number of steps and start/end learning rate
    # final_learning_rate = LEARNING_RATE * (gamma ** epochs)
    gamma = (FINAL_LEARNING_RATE / LEARNING_RATE) ** (1 / epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Training Loop
    writer = SummaryWriter(
        f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    writer.add_text(
        "Hyperparameters", open("glyphogen/hyperparameters.py").read(), 0
    )
    best_val_metric = 0
    global_step = 0
    LOSSES = ["total_loss", "command_loss", "coord_loss"]
    if pre_train:
        LOSSES += [
            "contour_count_loss",
            "node_count_loss",
            "handle_smoothness_loss",
        ]
        if not SKIP_RASTERIZATION:
            LOSSES += ["raster_metric"]
    if debug_grads:
        torch._functorch.config.donated_buffer = False
    for epoch in range(epochs):
        print()
        model_to_train.train()
        loss_accumulators = defaultdict(lambda: 0.0)
        i = 0
        kbar = pkbar.Kbar(
            target=len(train_loader),
            epoch=epoch,
            num_epochs=epochs,
            width=8,
            always_stateful=False,
        )
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            losses, _ = model_to_train.step(batch, step=global_step)

            if debug_grads:
                write_gradient_norms(model_to_train, losses, writer, global_step)

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), max_norm=1.0)
            optimizer.step()
            for loss_key, loss_value in losses.items():
                loss_accumulators[loss_key] += loss_value.item()

            kbar.update(
                i,
                values=[
                    (label.replace("_loss", ""), losses[label])
                    for label in losses.keys()
                ],
            )

            if i % 10 == 0:
                writer.flush()

            global_step += 1
        scheduler.step()

        dump_accumulators(loss_accumulators, writer, epoch, i, val=False)
        # Validation
        model_to_train.eval()
        total_val_loss = 0
        loss_accumulators = defaultdict(lambda: 0.0)
        i = 0
        cm_state = init_confusion_matrix_state()

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                losses, outputs = model_to_train.step(batch, step=global_step, val=True)
                for loss_key, loss_value in losses.items():
                    loss_accumulators[loss_key] += loss_value
                total_val_loss += losses["total_loss"].item()
                collect_confusion_matrix_data(cm_state, outputs, batch[1])

        avg_val_loss = total_val_loss / (0.1 + i)
        avg_val_metric = loss_accumulators["raster_metric"] / (0.1 + i)
        dump_accumulators(loss_accumulators, writer, epoch, i, val=True)
        print(
            f"Epoch {epoch}, Validation Loss: {avg_val_loss}; Metric: {avg_val_metric}"
        )

        # Checkpoint
        if avg_val_metric > best_val_metric:
            best_val_metric = avg_val_metric
            torch.save(model_to_train.state_dict(), model_save_name)
            print(f"Saved best model to {model_save_name}")

        # Callbacks
        if pre_train:
            log_pretrain_rasters(model_to_train, test_loader, writer, epoch)
            log_confusion_matrix(cm_state, writer, epoch)
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
    parser.add_argument(
        "--debug-grads",
        action="store_true",
        help="Whether to log gradient norms for debugging.",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=0,
        help="Number of augmentations to apply.",
    )
    parser.add_argument(
        "--roll-augmentations",
        type=int,
        default=2,
        help="Number of roll augmentations to apply.",
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        pre_train=args.pre_train,
        epochs=args.epochs,
        debug_grads=args.debug_grads,
        single_batch=args.single_batch,
        vectorizer_model_name=args.vectorizer_model_name,
        augmentations=args.augmentations,
        roll_augmentations=args.roll_augmentations,
    )

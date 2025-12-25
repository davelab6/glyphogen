import sys
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# This allows us to import engine, utils, etc.
sys.path.append("vision/references/detection/")

import engine
import utils
import torchvision.transforms.v2 as T
from glyphogen.dataset import GlyphCocoDataset


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="MaskRCNN_ResNet50_FPN_Weights.DEFAULT"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def get_transform(train):
    transforms = []
    if train:
        # A simple training transform
        transforms.append(T.RandomHorizontalFlip(0.5))
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    num_classes = 3  # 0: background, 1: outer, 2: hole

    # Define paths
    from pathlib import Path
    DATA_DIR = Path("data")
    TRAIN_IMG_DIR = DATA_DIR / "images_hierarchical" / "train"
    TEST_IMG_DIR = DATA_DIR / "images_hierarchical" / "test"
    TRAIN_JSON = DATA_DIR / "train_hierarchical.json"
    TEST_JSON = DATA_DIR / "test_hierarchical.json"

    # Use the new dataset
    dataset = GlyphCocoDataset(root=TRAIN_IMG_DIR, annFile=TRAIN_JSON, transforms=get_transform(train=True))
    dataset_test = GlyphCocoDataset(root=TEST_IMG_DIR, annFile=TEST_JSON, transforms=get_transform(train=False))

    # DataLoaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        engine.train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=100
        )
        lr_scheduler.step()
        engine.evaluate(model, data_loader_test, device=device)

    print("That's it! Saving model.")
    torch.save(model.state_dict(), "glyphogen.segmenter.pt")


if __name__ == "__main__":
    main()

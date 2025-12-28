import torch
from pathlib import Path
import pytest

# This import will fail if the main project is not in the path.
# Pytest handles this automatically.
from glyphogen.dataset import GlyphCocoDataset

@pytest.fixture
def hierarchical_test_dataset():
    """Provides a fixture for the hierarchical test dataset."""
    DATA_DIR = Path("data")
    TEST_IMG_DIR = DATA_DIR / "images_hierarchical" / "test"
    TEST_JSON = DATA_DIR / "test_hierarchical.json"
    
    if not TEST_JSON.exists():
        pytest.skip("Hierarchical test dataset not found. Run preprocess_for_hierarchical.py.")
        
    dataset = GlyphCocoDataset(root=TEST_IMG_DIR, annFile=TEST_JSON)
    return dataset

def test_dataset_loading(hierarchical_test_dataset):
    """Tests that the dataset can be loaded and has items."""
    assert len(hierarchical_test_dataset) > 0, "Dataset should not be empty."

def test_mask_properties(hierarchical_test_dataset):
    """
    Tests the properties of the masks loaded from the dataset to ensure
    they are binary torch.uint8 tensors.
    """
    # Get a sample item from the dataset
    # Let's take an item from the middle to get a typical glyph
    img, target = hierarchical_test_dataset[len(hierarchical_test_dataset) // 2]

    assert "gt_contours" in target, "Target dictionary should have 'gt_contours' key."
    
    gt_contours = target["gt_contours"]
    assert isinstance(gt_contours, list), "gt_contours should be a list."
    assert len(gt_contours) > 0, "Sample should have at least one contour."

    # Get the mask from the first contour
    first_contour = gt_contours[0]
    assert "mask" in first_contour, "Contour dictionary should have a 'mask' key."
    mask = first_contour["mask"]

    # 1. Check the type
    assert isinstance(mask, torch.Tensor), f"Mask should be a torch.Tensor, but got {type(mask)}."

    # 2. Check the dtype
    assert mask.dtype == torch.uint8, f"Mask dtype should be torch.uint8, but got {mask.dtype}."

    # 3. Check the value range
    unique_values = torch.unique(mask)
    is_binary = all(v in [0, 1] for v in unique_values)
    assert is_binary, f"Mask should be binary (contain only 0s and 1s), but found values: {unique_values.tolist()}"

    # 4. Check the shape
    assert mask.ndim == 2, f"Mask should be a 2D tensor [H, W], but has {mask.ndim} dimensions."
    assert mask.shape[0] > 0 and mask.shape[1] > 0, f"Mask dimensions should be greater than 0, but got shape {mask.shape}"


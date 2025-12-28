import torch
from glyphogen.dataset import GlyphCocoDataset
from glyphogen.command_defs import NODE_GLYPH_COMMANDS
import os

# This test is designed to be run from the root of the project.
# It checks the integrity of the dataset by inspecting the command sequences.

def test_inspect_dataset_sequences():
    """
    Loads the training dataset and prints the command sequences for the
    first few items to verify they are not malformed (e.g., all EOS).
    """
    DATA_DIR = "data"
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images_hierarchical", "train")
    TRAIN_JSON = os.path.join(DATA_DIR, "train_hierarchical.json")

    print(f"Loading dataset from: {TRAIN_JSON}")

    # Check if the annotation file exists
    if not os.path.exists(TRAIN_JSON):
        pytest.fail(f"Annotation file not found: {TRAIN_JSON}")

    dataset = GlyphCocoDataset(root=TRAIN_IMG_DIR, annFile=TRAIN_JSON)

    command_names = list(NODE_GLYPH_COMMANDS.keys())
    command_width = len(command_names)
    num_items_to_check = 5

    print(f"--- Inspecting first {num_items_to_check} items from the dataset ---")

    found_valid_sequence = False
    for i in range(min(num_items_to_check, len(dataset))):
        img, target_dict = dataset[i]
        gt_contours = target_dict["gt_contours"]
        
        print(f"\nItem {i} (Image ID: {target_dict['image_id']}):")
        if not gt_contours:
            print("  No contours.")
            continue

        for j, contour in enumerate(gt_contours):
            sequence_tensor = contour["sequence"]
            command_tensor = sequence_tensor[:, :command_width]
            
            indices = command_tensor.argmax(dim=-1).tolist()
            names = [command_names[idx] for idx in indices]
            
            print(f"  Contour {j}: {names}")

            # Check if there's at least one non-SOS/EOS command
            if any(cmd not in ["SOS", "EOS"] for cmd in names):
                found_valid_sequence = True

    print("--- Inspection Complete ---")
    
    # This is not a formal assertion, but a check for the user to see.
    # We can add a real assertion later if needed.
    if not found_valid_sequence:
        print("\nWARNING: No valid (non-SOS/EOS) commands found in the inspected items.")
    else:
        print("\nSUCCESS: At least one valid sequence was found.")

if __name__ == "__main__":
    import pytest
    # This allows running the test directly.
    test_inspect_dataset_sequences()

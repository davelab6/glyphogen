# Glyphogen

Glyphogen is a font glyph generation model designed to produce "designer-quality" vector representations of glyphs. 

## Project Status

The project is currently in a research and development phase focused on **designer-like vectorization**. While the long-term goal is full-glyphset style transfer and expansion (e.g., automatically generating missing currency symbols for a font library), the current codebase focuses on pre-training a **VectorizationGenerator** to accurately recover vector paths from raster images using typographic design principles (node-based representations rather than segment-based).

## Installation

This project uses `pyproject.toml` for dependency management. It is recommended to work within a virtual environment.

```bash
# Create and activate a virtual environment (if not already present)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install .

# Note: pydiffvg is required but not available on PyPI. 
# You must install it manually from: https://github.com/BachiLi/diffvg
```

## Dataset Preparation

The training pipeline requires access to a large collection of font files.

1.  **Clone Google Fonts:** Download or clone the [google/fonts](https://github.com/google/fonts) repository.
2.  **Configure Paths:** Update the `BASE_DIR` in `glyphogen/hyperparameters.py` to point to your local copy of the fonts (e.g., the `ofl` directory).
3.  **Preprocess:** Run the hierarchical preprocessing script. This generates the JSON datasets and normalized contour images.
    ```bash
    python preprocess_for_hierarchical.py
    ```
4.  **Analyze Statistics:** Run the statistics analyzer to compute coordinate means and standard deviations used for normalization.
    ```bash
    python analyze_dataset_stats.py
    ```

## Training

The model training is split into two parts: segmentation and vectorization.

### 1. Train the Segmenter

The segmenter (based on Mask R-CNN) identifies individual contours within a glyph raster.

```bash
python train_segmentation.py
```

This produces `glyphogen.segmenter.pt`.

### 2. Train the Vectorizer

Once the segmenter is trained, you can train the vectorization model (the LSTMs that produce the nodes):

```bash
python train.py --segmentation-model glyphogen.segmenter.pt
```

### Common Options for `train.py`

- `--load_model`: Resume training from an existing `glyphogen.vectorizer.pt`.
- `--canary 1`: Run a smoke test on a single batch to verify convergence and pipeline integrity.
- `--canary <N>`: Train on a subset of N batches.
- `--debug-grads`: Log gradient norms to TensorBoard for debugging optimization stability.

Monitor training progress via TensorBoard:
```bash
tensorboard --logdir logs/
```

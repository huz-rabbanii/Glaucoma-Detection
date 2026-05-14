# Glaucoma Detection — U-Net Optic Disc and Cup Segmentation

A U-Net based deep learning pipeline for detecting glaucoma by segmenting the optic disc and optic cup in retinal fundus images. The cup-to-disc ratio (CDR) is then used to assess glaucoma risk.

## How It Works

1. **Train two separate U-Net models** — one for optic disc segmentation and one for optic cup segmentation.
2. **Run inference** on a retinal fundus image to produce binary masks for the disc and cup.
3. **Calculate the CDR** (cup area / disc area). A CDR > 0.5 indicates a high risk of glaucoma.
4. **Visualize** the overlap between the predicted cup and disc masks.

## Project Structure

```
Glaucoma-Detection/
├── U-net.ipynb         # Main notebook: model, training, inference, and CDR analysis
├── requirements.txt
├── train_images/       # Training fundus images
├── train_masks/        # Ground truth masks for training
├── val_images/         # Validation fundus images
└── val_masks/          # Ground truth masks for validation
```

### Key Components (inside `U-net.ipynb`)

| Component | Description |
|---|---|
| `DoubleConv` | Two consecutive Conv2d → BatchNorm → ReLU blocks |
| `UNET` | Full encoder-decoder U-Net with skip connections |
| `Dataset` | PyTorch `Dataset` for loading images and binary masks |
| `get_loaders` | Returns `DataLoader` objects for train and validation sets |
| `train_fn` | Single-epoch training loop with mixed-precision (`GradScaler`) |
| `check_accuracy` | Computes pixel accuracy and Dice score on a data loader |
| `save_predictions_as_imgs` | Saves predicted masks to disk after each epoch |
| `save_checkpoint` / `load_checkpoint` | Saves and restores model weights |
| `calculate_white_area` | Counts white pixels in a binary mask |
| `compare_images` | Overlays cup (white) and disc-only (gray) for CDR visualization |

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | `1e-4` |
| Batch size | `16` |
| Epochs | `15` |
| Input size | `160 × 240` |
| Optimizer | Adam |
| Loss function | `BCEWithLogitsLoss` |
| Device | CUDA (falls back to CPU) |

## Usage

### 1. Set dataset paths

In the notebook, update the path variables to point to your dataset:

```python
TRAIN_IMG_DIR = "/path/to/train_images/"
TRAIN_MASK_DIR = "/path/to/train_masks/"
VAL_IMG_DIR   = "/path/to/val_images/"
VAL_MASK_DIR  = "/path/to/val_masks/"
```

### 2. Train the model

Run the `main()` cell. The model checkpoint is saved as `my_checkpoint.pth.tar` after each epoch along with sample predictions.

### 3. Run inference

Load the saved checkpoint and pass a fundus image through the model to generate a disc or cup mask:

```python
checkpoint = torch.load("my_checkpoint.pth.tar")
model = UNET(in_channels=3, out_channels=1)
model.load_state_dict(checkpoint["state_dict"])
```

The output is thresholded at `0.5` to produce a binary mask saved as `disk_output.png` or `cup_output.png`.

### 4. Calculate CDR and assess glaucoma risk

```python
cdr = white_area_cup / white_area_disk
# CDR > 0.5 → high chance of glaucoma
```

### 5. Visualize disc vs. cup overlap

The `compare_images()` function produces a combined image where:
- **White** pixels = optic cup (overlap region)
- **Gray** pixels = optic disc rim (disc only)

The result is saved as `CDR.png`.

## Data Augmentation

Training uses `albumentations` with:
- Random rotation (±35°)
- Horizontal flip (p=0.5)
- Vertical flip (p=0.1)
- Normalization (zero mean, unit std)

Validation uses only resize and normalization.

## Requirements

```
torch
torchvision
albumentations
Pillow
numpy
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

## Notes

- The notebook was originally developed on Google Colab with Google Drive paths. Update all `/content/drive/...` paths before running locally.
- Two separate model checkpoints are needed: one trained on disc masks and one on cup masks.
- Mixed-precision training (`torch.cuda.amp`) is used for faster training on CUDA-enabled GPUs.


# RAFI++ : Robust Adaptive Face Inpainting

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
   - [SegNet++](#segnet)
   - [RestoreNet++](#restorenet)
   - [Discriminators](#discriminators)
   - [Loss Functions](#loss-functions)
3. [Project Pipeline](#project-pipeline)
   - [Data Preparation](#1-data-preparation)
   - [Training](#2-training)
   - [Testing and Inference](#3-testing-and-inference)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [Dependencies](#dependencies)

---

## Overview

RAFI++ is a two-stage generative framework designed for lower-face inpainting, specifically targeting the restoration of face regions occluded by masks. The system decomposes the problem into two sequential sub-tasks:

1. **Segmentation** -- Accurately predict the occluded region (binary mask), its boundary contour, and a pixel-level confidence map.
2. **Restoration** -- Reconstruct the missing facial content guided by the segmentation outputs, then seamlessly blend the restored region back into the original image.

The framework is trained using a multi-stage curriculum strategy and supervised with a composite loss function that combines segmentation, reconstruction, perceptual, identity-preserving, and adversarial objectives.

Dataset: CelebA-HQ (resized to 256x256). The data preparation pipeline uses MediaPipe Face Mesh to automatically generate lower-face polygon masks, binary masks, and boundary maps.

---

## Model Architecture

The RAFI++ model (`RAFIpp`) consists of two main sub-networks -- `SegNet++` and `RestoreNet++` -- connected in a feedforward manner. SegNet++ first analyzes the masked input to produce segmentation maps, which then condition RestoreNet++ for image restoration.

```
Input (masked face)
       |
       v
  +-----------+
  |  SegNet++ |  -->  mask_pred, boundary_pred, confidence_pred
  +-----------+
       |
       v
  +--------------+
  | RestoreNet++ |  -->  restored image (Ir), synthesized image (Isyn)
  +--------------+
       |
       v
  +------------------+       +---------------------------+
  | PatchDiscriminator|      | FeaturePatchDiscriminator  |
  | (Image-level)     |      | (Feature-level, VGG relu4) |
  +------------------+       +---------------------------+
```

### SegNet++

**Purpose**: Predict three pixel-wise maps from the masked input image.

**Architecture**: U-Net encoder-decoder with skip connections.

- **Encoder**: Four encoding stages using `ConvINAct` blocks (Conv2d + InstanceNorm + LeakyReLU), each followed by a stride-2 downsampling convolution. Channel progression: 3 -> 64 -> 128 -> 256 -> 512 -> 1024 (bottleneck).
- **Decoder**: Four upsampling stages using nearest-neighbor interpolation followed by `ConvINAct`. Skip connections concatenate encoder features at matching resolutions. Channel progression reverses: 1024 -> 512 -> 256 -> 128 -> 64.
- **Output Heads**: Three independent 1x1 convolution heads, each followed by sigmoid activation:
  - `mask_pred` (1 channel) -- binary occlusion mask
  - `boundary_pred` (1 channel) -- boundary contour of the mask
  - `confidence_pred` (1 channel) -- per-pixel confidence map, trained to reflect segmentation accuracy

### RestoreNet++

**Purpose**: Restore the occluded lower-face region conditioned on segmentation outputs.

**Architecture**: Gated convolution encoder-decoder with multi-scale contextual attention and skip attention gates.

- **Input**: Concatenation of the masked image (3 channels) + mask_pred + boundary_pred + confidence_pred = 6 channels.
- **Encoder**: Four stages of `GatedConv2d` blocks. Gated convolution uses a learned sigmoid gate to modulate feature flow, which is particularly effective for inpainting tasks where valid/invalid regions differ. Channel progression: 6 -> 64 -> 128 -> 256 -> 512.
- **Bottleneck -- MCSAM (Multi-scale Channel-Spatial Attention Module)**: Three cascaded MCSAM blocks operate on the 512-channel bottleneck features.
  - Each MCSAM block uses four parallel dilated convolution branches (dilation rates 1, 2, 4, 8) to capture multi-scale context.
  - Channel attention is applied via a squeeze-and-excitation MLP on the global average pooled features.
  - Spatial attention combines average-pooled and max-pooled channel descriptors through a 7x7 convolution.
  - Residual connections link MCSAM blocks with learnable scaling parameters (alpha1, alpha2, alpha3).
- **Decoder**: Three upsampling stages using `UpGatedConv` (nearest-neighbor upsample + gated conv), each followed by a `SkipAttentionGate` and a `ResidualRefineBlock`.
  - **SkipAttentionGate**: Modulates encoder skip features by considering decoder features and all three segmentation maps (mask, boundary, confidence). This ensures the decoder focuses on relevant spatial regions.
  - **ResidualRefineBlock**: Two stacked `ConvINAct` layers with a residual shortcut, refining features at each scale.
- **Output**: A final 3x3 convolution projects to 3-channel RGB, activated by tanh (output range [-1, 1]).
- **Blending**: The final synthesized image is produced by soft blending: `Isyn = (1 - mask_pred) * input + mask_pred * Ir`, where `Ir` is the raw restored output.

### Discriminators

Two PatchGAN discriminators enforce adversarial quality:

1. **PatchDiscriminator (Dp)**: Operates on image-level inputs. Takes a 5-channel input (RGB image + predicted mask + predicted boundary) and produces a patch-wise realness map. Uses spectral normalization for training stability.

2. **FeaturePatchDiscriminator (Df)**: Operates on VGG feature-level inputs. Takes the 512-channel `relu4_3` feature maps extracted from VGG19 and produces a feature-level realness map. Also uses spectral normalization.

Both discriminators use the hinge loss formulation for stable adversarial training.

### Loss Functions

The total training objective combines multiple loss terms:

**Segmentation Losses (Stage 1 and 3)**:

| Loss | Description | Weight |
|------|-------------|--------|
| L_mask | BCE + Dice loss on mask prediction | lambda_m = 1.0 |
| L_boundary | BCE + Dice loss on boundary prediction | lambda_b = 1.0 |
| L_confidence | L1 loss against an exponential confidence target | lambda_c = 0.2 |

**Reconstruction Losses (Stage 2 and 3)**:

| Loss | Description | Weight |
|------|-------------|--------|
| L_rec | Region-weighted L1 loss (higher weight on masked and boundary regions) | lambda_rec = 10.0 |
| L_ssim | SSIM loss computed on the masked region | lambda_ssim = 5.0 |
| L_perc | Perceptual loss (L1 on VGG19 features at relu1_2, relu2_2, relu3_4, relu4_3) | lambda_perc = 1.0 |
| L_style | Style loss (L1 on Gram matrices of VGG features at relu2_2 through relu5_2) | lambda_style = 100.0 |
| L_id | Identity loss (cosine similarity via InceptionResNetV1 / fallback encoder) | lambda_id = 2.0 |
| L_edge | Edge loss (L1 on Sobel edge maps within the mask region) | lambda_edge = 2.0 |
| L_adv | Adversarial loss (hinge loss from both discriminators) | lambda_adv = 0.1 |

The region-weighted reconstruction uses: `weight = 1 + alpha * mask + gamma * boundary` (alpha = 3.0, gamma = 2.0).

---

## Project Pipeline

### 1. Data Preparation

**Script**: `Dataset/prepare_celeba_hq_masks.py`

This script transforms raw CelebA-HQ images into a structured dataset suitable for RAFI++ training:

1. **Face Landmark Detection**: Each image is processed through MediaPipe Face Mesh to extract 468 facial landmarks.
2. **Lower-Face Polygon Generation**: A subset of landmarks (cheeks, mouth, nose-bottom, chin) are selected and formed into a convex hull. The hull is clipped to the lower face using nose-bridge landmarks as the top boundary, then slightly expanded to ensure full coverage.
3. **Mask and Boundary Generation**:
   - The polygon is rasterized into a binary mask (white = occluded region).
   - A morphological boundary map is computed by taking the difference between dilated and eroded versions of the mask.
4. **Masked Image Creation**: The original image is blacked out within the mask region.
5. **Train/Val/Test Splitting**: Images are randomly split into train (90%), validation (5%), and test (5%) sets. Split lists are saved as text files.

```
python Dataset/prepare_celeba_hq_masks.py \
  --raw_dir ./celeba_hq_256 \
  --out_dir ./Dataset/CelebA/rafipp \
  --img_size 256 \
  --train_ratio 0.9 \
  --val_ratio 0.05 \
  --test_ratio 0.05
```

**Output structure**:

```
Dataset/CelebA/rafipp/
  gt/            -- Ground truth images (256x256 RGB)
  masked/        -- Images with lower face blacked out
  masks/         -- Binary occlusion masks (grayscale)
  boundaries/    -- Boundary maps of the masks (grayscale)
  splits/
    train.txt    -- Filenames for training set
    val.txt      -- Filenames for validation set
    test.txt     -- Filenames for test set
```

### 2. Training

**Script**: `train.py`

Training follows a three-stage curriculum:

**Stage 1 -- SegNet++ Pretraining** (default: 10 epochs)
- Only SegNet++ is trained; RestoreNet++ and discriminators are frozen.
- Objective: Segmentation loss (L_mask + L_boundary + L_confidence).
- Optimizer: Adam (lr = 2e-4) on SegNet++ parameters only.
- Validation metric: Dice score on mask prediction.

**Stage 2 -- RestoreNet++ Training with Frozen SegNet++** (default: 20 epochs)
- SegNet++ is frozen (no gradient); RestoreNet++ and both discriminators are trained.
- SegNet++ runs in inference mode to produce segmentation maps for conditioning.
- Discriminator update: Hinge loss on real vs. fake (image-level and feature-level).
- Generator update: All reconstruction losses + adversarial loss.
- Optimizer: Separate Adam optimizers for generator and discriminators (lr = 2e-4).
- Validation metric: Negative L1 distance (lower is better).

**Stage 3 -- Joint Fine-tuning** (default: 10 epochs)
- All components are trained end-to-end (SegNet++ + RestoreNet++ + discriminators).
- Full loss: Segmentation losses + reconstruction losses + adversarial loss.
- Optimizer: Joint Adam optimizer for both sub-networks; separate optimizer for discriminators.
- Validation metric: Negative L1 distance.

```
python train.py \
  --data_root ./Dataset/CelebA/rafipp \
  --batch_size 8 \
  --stage1_epochs 10 \
  --stage2_epochs 20 \
  --stage3_epochs 10
```

**Outputs**:
- Checkpoints saved to `./checkpoints/<run_name>/` (latest, best per stage, periodic)
- Training history saved to `./logs/<run_name>/history.json` and `metrics.csv`
- Visual samples saved to `./outputs/samples/<run_name>/` (per-epoch grids)

### 3. Testing and Inference

**Script**: `test.py`

- Loads a trained checkpoint and runs inference on the test split.
- For each test image, saves: input (masked), ground truth mask, predicted mask, raw restored output, synthesized (blended) output, and ground truth.
- Computes and reports aggregate metrics: L1, PSNR, SSIM, Dice score, and IoU.

```
python test.py \
  --data_root ./Dataset/CelebA/rafipp \
  --checkpoint ./checkpoints/rafipp_run/best_stage3.pt \
  --save_dir ./outputs/test_predictions
```

---

## Project Structure

```
RAFIpp_Project/
  celeba_hq_256/                        -- Raw dataset folder (CelebA-HQ 256x256)
  Dataset/
    CelebA/rafipp/                      -- Processed dataset (generated by preparation script)
      gt/
      masked/
      masks/
      boundaries/
      splits/
    __init__.py
    datasets.py                         -- RAFIppCelebA Dataset class (PyTorch)
    prepare_celeba_hq_masks.py          -- Data preparation script (MediaPipe + OpenCV)
  Experiments/
    __init__.py
    configs.py                          -- Argument parser and config management
  Model/
    __init__.py
    blocks.py                           -- Building blocks (ConvINAct, GatedConv2d, MCSAM, etc.)
    loss.py                             -- Loss functions (SSIMLoss, Dice, Gram, Sobel, LossFactory)
    networks.py                         -- Network definitions (SegNet++, RestoreNet++, discriminators, VGG19, IdentityEncoder)
    RAFIpp.py                           -- RAFIppSystem (training loop logic, optimizers, checkpointing)
  Utils/
    __init__.py
    metrics.py                          -- Evaluation metrics (L1, PSNR, SSIM, Dice, IoU)
    seed.py                             -- Random seed utility
    visualization.py                    -- Visualization utilities (training grids, tensor-to-image)
  train.py                              -- Training entry point
  test.py                               -- Testing/inference entry point
  visualize_metrics.py                  -- Metric visualization script
  requirements.txt                      -- Python dependencies
```

---

## Getting Started

1. Place the raw CelebA-HQ dataset (256x256 images) in `./celeba_hq_256/`.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   ```
   python Dataset/prepare_celeba_hq_masks.py
   ```
4. Train the model:
   ```
   python train.py --data_root ./Dataset/CelebA/rafipp
   ```
5. Run inference:
   ```
   python test.py --checkpoint ./checkpoints/rafipp_run/best_stage3.pt
   ```

---

## Dependencies

- **PyTorch** -- Core deep learning framework
- **torchvision** -- VGG19 pretrained weights for perceptual and style losses
- **mediapipe** -- Face landmark detection during data preparation
- **OpenCV (cv2)** -- Image processing and morphological operations
- **facenet-pytorch** -- InceptionResNetV1 for identity loss (optional; falls back to pooled RGB features if unavailable)
- **Pillow, NumPy, tqdm, PyYAML** -- General utilities

All images are normalized to [-1, 1] for RGB and [0, 1] for masks/boundaries. Default resolution is 256x256.

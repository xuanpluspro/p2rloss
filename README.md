# P2RLoss

Official code for CVPR-2025 paper:"[Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](https://arxiv.org/abs/2505.21943)."

## Overview

This repository implements the P2RLoss method for semi-supervised point-based crowd counting. The project uses a teacher-student framework with VGG16BN backbone and proposes a novel Point-to-Region Loss to improve counting performance with limited labeled data.

## Features

- **Semi-supervised learning**: Utilizes both labeled and unlabeled data for training
- **Teacher-student framework**: Employs EMA-based teacher model for pseudo-labeling
- **P2RLoss**: Novel loss function that matches points to regions for better supervision
- **Strong augmentation**: Uses strong data augmentation for unlabeled samples
- **Multi-stage training**: Two-stage training strategy with progressive semi-supervised learning

## Requirements

- Python 3.x
- PyTorch >= 1.8
- torchvision
- timm
- yacs
- numpy
- PIL
- opencv-python
- tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/Elin24/P2RLoss.git
cd P2RLoss

# Install dependencies
pip install torch torchvision timm yacs numpy opencv-python tqdm
```

## Data Preparation

The project uses ShanghaiTech dataset. Please organize your data as follows:

```
data/
└── ShanghaiTech/
    └── part_A/
        ├── train_data/
        │   ├── images/
        │   └── new-anno/
        └── test_data/
            ├── images/
            └── new-anno/
```

You also need to prepare a protocol file that specifies which images are labeled for semi-supervised learning. The protocol file should contain image IDs (one per line).

## Training

### RiceW3999 fully supervised training on AutoDL

The RiceW3999 adapter uses **all train annotations** with `--label 100` and validates on `val.txt`; no `test.txt` or unlabeled protocol is required. Expected layout:

```text
/root/autodl-tmp/p2rloss/data/RiceW3999/
  ImageSets/train.txt
  ImageSets/val.txt
  JPEGImages/<image_id>.png   # also supports jpg/jpeg/bmp/tif/tiff
  PointAnnotations/<image_id>.txt
```

Split files contain image IDs without extensions, one per line. Point TXT files contain pixel coordinates `x y`, one point per line (not YOLO-normalized coordinates). An empty TXT represents an image without points.

Place the ImageNet VGG16-BN checkpoint `vgg16_bn-6c64b313.pth` in the repository root, then run:

```bash
cd /root/autodl-tmp/p2rloss
bash train_ricew3999.sh ricew3999_ep50_bs8
```

To reuse weights stored elsewhere:

```bash
WEIGHTS_PATH=/absolute/path/vgg16_bn-6c64b313.pth bash train_ricew3999.sh ricew3999_ep50_bs8
```

The script trains for 50 epochs, batch size 8, using the P2R supervised loss. Training uses **one online random 512×512 output patch per source image per visit**, with scale augmentation (0.7–1.3) and random horizontal flip; points are cropped/scaled/flipped together. This is not exhaustive non-overlapping tiling. Validation uses full images, batch size 1. Wheat 1024×1024 and rice 512×512 images can share a training batch because every training output is 512×512. The teacher/pseudo-label branch is skipped at 100% labels, although the current trainer still creates and EMA-updates a teacher model.

Logs: `exp/<tag>/train.log/running.log`; best checkpoint: `exp/<tag>/output/ckpt_epoch_best.pth`. Validation runs every 5 epoch indices after index 0 and at the last epoch. Use a new tag for a fresh run: the existing automatic checkpoint reload restores model weights, not full optimizer/epoch state.

GPU peak memory depends on batch size, precision, image size and point count. The trainer currently uses FP32. It logs PyTorch peak allocated memory as `mem ... MB`; no measured GPU-memory comparison with PET is provided here.

### Basic Usage

```bash
python main.py --data-path /path/to/data \
    --label 5 \
    --protocol /path/to/protocol.txt \
    --batch-size 16 \
    --tag experiment_name
```

### Using the provided script

Modify `run.sh` with your data paths and configurations:

```bash
datadir=/path/to/ShanghaiTech/part_A
name=sha
part=5
T=${name}-L${part}

mkdir exp
mkdir exp/$T
mkdir exp/$T/code
cp -r datasets exp/$T/code/datasets
cp -r models exp/$T/code/models
cp -r losses exp/$T/code/losses
cp ./*.py exp/$T/code/
cp run.sh exp/$T/code

mkdir exp/$T/train.log
python main.py --data-path $datadir \
    --label ${part} --protocol /path/to/protocol.txt \
    --batch-size 16 --tag $T 2>&1 | tee exp/$T/train.log/running.log
```

### Arguments

- `--data-path`: Path to the dataset
- `--label`: Percentage of labeled data (e.g., 5 for 5%)
- `--protocol`: Path to the protocol file
- `--batch-size`: Batch size for training
- `--tag`: Experiment tag for logging
- `--resume`: Path to checkpoint for resuming training
- `--eval`: Evaluation mode only

## Configuration

The training configuration is managed through `config.py`. Key parameters include:

- `DATA.BATCH_SIZE`: Batch size (default: 1)
- `DATA.DATASET`: Dataset name (default: 'shha')
- `MODEL.NAME`: Model architecture (default: 'VGG16BN')
- `MODEL.LOSS`: Loss function (default: 'P2R')
- `TRAIN.EPOCHS`: Total training epochs (default: 1500)
- `TRAIN.BASE_LR`: Base learning rate (default: 5e-5)
- `TRAIN.LR_SCHEDULER`: Learning rate scheduler settings

## Code Structure

```
P2RLoss/
├── config.py              # Configuration management
├── main.py                # Main training and evaluation script
├── run.sh                 # Training script
├── utils.py               # Utility functions
├── logger.py              # Logging utilities
├── lr_scheduler.py        # Learning rate scheduler
├── datasets/              # Dataset loaders
│   ├── __init__.py
│   ├── shha.py           # ShanghaiTech dataset
│   └── utils.py          # Dataset utilities
├── models/                # Model architectures
│   ├── __init__.py
│   ├── vgg16bn.py        # VGG16BN backbone
│   └── utils.py          # Model utilities
└── losses/                # Loss functions
    ├── __init__.py
    └── p2rloss.py        # Point-to-Region Loss
```

## Model Architecture

The model uses VGG16BN as the backbone with:
- Multi-scale feature extraction
- Feature fusion layer
- Decoder for density map generation
- Two output channels for density estimation

## Training Strategy

1. **Stage 1 (Epochs 0-25)**: Supervised training with labeled data only
2. **Stage 2 (Epochs 25-50)**: Semi-supervised training with both labeled and unlabeled data
   - Teacher model generates pseudo-labels for unlabeled data
   - Student model learns from both labeled and pseudo-labeled data
   - Progressive weight adjustment for semi-supervised loss

## Results

The model achieves competitive performance on ShanghaiTech dataset with limited labeled data. Training logs and curves are saved to `exp/<tag>/train.log/`.

## Citation

If you find this code useful for your research, please cite:

```bibtex
@inproceedings{lin2025point,
  title={Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting},
  author={Lin, Wei and Zhao, Chenyang and Chan, Antoni B},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29363--29373},
  year={2025}
}
```

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

## GPUFree / AutoDL 本地数据与启动

仓库仅提供 `data/` 目录。请将 RiceW3999 压缩包上传至该目录后解压，
使数据位于 `data/RiceW3999/`，其中包含 JPEGImages、PointAnnotations 和 ImageSets。
实际数据与压缩包均由 Git 忽略，不会上传。

```bash
cd /root/gpufree-data/p2rloss
git pull --ff-only origin main
BATCH_SIZE=16 EPOCHS=50 bash train_ricew3999.sh ricew3999_ep50_bs16
```

脚本默认 batch=8、epochs=50、workers=8、验证间隔=5，均可通过环境变量覆盖：
`BATCH_SIZE`、`EPOCHS`、`WORKERS`、`VAL_EVERY`。
`DATA_PATH` 和 `WEIGHTS_PATH` 可指向仓库外的数据及本地权重。
实验名称只决定输出目录，不会覆盖这些参数。请为新实验使用新名称；
启动脚本拒绝复用非空 output 目录，以免现有训练器自动加载旧模型却不恢复优化器与轮数。
模型、损失与学习率未改动：100% 标注、FP32、512×512 随机训练裁剪、整图验证 batch=1。
验证发生在大于 0 且可被 VAL_EVERY 整除的 epoch 索引，以及最后一轮。
现有训练器学习率保持固定；变量、日志与新曲线名称统一使用 RMSE；当前最佳模型统一按验证 MAE 最小选择。


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

Place the ImageNet VGG16-BN checkpoint at `pretrained/vgg16_bn-6c64b313.pth`, then run:

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

## 统一计数、定位与效率指标

新增 MAE、RMSE、Precision/Recall/F1@4/8/16px、逐图 CSV 与指标 JSON；
训练自动导出，参数量、延迟/FPS、峰值显存和可选 GFLOPs 由独立脚本统计。
详见 [评价说明](METRICS.md)。

P2R 训练默认已启用4/8/16像素定位评价，可通过 `--opts TEST.LOC_RADII '[4.0,8.0,16.0]'` 设置。
独立评估已有权重：
```bash
python main.py --eval --label 100 --data-path /path/to/RiceW3999 --resume /path/to/ckpt_epoch_best.pth --tag leaf_evaluation --opts DATA.DATASET ricew3999
```

## 预训练权重目录
默认使用 `pretrained/vgg16_bn-6c64b313.pth`，与 PET 的 ImageNet VGG16-BN 骨干权重相同。
若文件原先放在仓库根目录，更新代码后手动移入 pretrained；程序不会移动或删除已有文件。
原来的 WEIGHTS_PATH 和 --opts MODEL.PRETRAINED 显式路径仍有效。

## 训练与验证进度条

训练和验证默认显示 tqdm 单行进度条。训练显示当前/平均 loss、实际学习率（科学计数法）、
峰值显存、批次数和 ETA；每轮结束保留一行汇总，验证结束输出 MAE、RMSE 和定位指标。
启动脚本使用 tee 时，临时进度条直接写入控制终端，不污染 running.log。
非交互后台任务无终端时自动关闭进度条，仍保留日志。
PRINT_FREQ 控制详细文件日志频率，不再控制终端进度条刷新；详细批次记录见 output/log_rank0.txt。
可设置 P2R_PROGRESS=0 禁用进度条。缺少依赖时安装 tqdm（不会安装 torch）。
VGG16-BN 初始化改用 weights=None，再加载指定的本地预训练权重；
meshgrid 显式指定 indexing='ij'，保留原有(y,x)坐标顺序。
此次修改不改变网络、损失、学习率或数据增强。现有训练进程需重启后才使用新版界面。

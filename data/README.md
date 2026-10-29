# 本地数据目录

克隆仓库后已提供 RiceW3999 目录骨架；实际图像、标注及划分清单保持本地，不提交 Git。

```text
data/RiceW3999/
├── JPEGImages/          # 原图
├── PointAnnotations/   # 同名 TXT 点标注，每行 x y（原图像素坐标）
└── ImageSets/
    ├── train.txt       # 每行一个无扩展名样本 ID
    └── val.txt
```

请放入你已有的 train.txt / val.txt，不提供空白清单，以免误用空数据集。
当前读取器按空白分隔 ID，因此文件名不要包含空格。
图像与标注必须同名，点坐标应在原图范围内；空 TXT 表示无目标。
训练和验证样本应互不重叠。请先按原图/植株划分，再生成 patch，避免泄漏。

GPUFree 默认位置：/root/gpufree-data/p2rloss/data/RiceW3999/
AutoDL 默认位置：/root/autodl-tmp/p2rloss/data/RiceW3999/

数据也可以留在仓库外，通过 DATA_PATH 指定，不必复制：
```bash
DATA_PATH=/root/gpufree-data/RiceW3999 BATCH_SIZE=16 EPOCHS=50 bash train_ricew3999.sh ricew3999_bs16_ep50
```

预训练权重默认位于仓库根目录 vgg16_bn-6c64b313.pth；可用 WEIGHTS_PATH 指向其他位置。
训练脚本不会下载、移动、覆盖或删除数据和权重。

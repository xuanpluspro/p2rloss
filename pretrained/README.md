# 预训练骨干权重

将 torchvision 官方 ImageNet VGG16-BN 权重放在此目录：

```text
pretrained/vgg16_bn-6c64b313.pth
```

PET 和 P2RLoss 使用同一份骨干初始化权重，可以复用同一个文件。
官方地址：https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

这是骨干初始化权重，不是训练后的 PET 或 P2RLoss 检查点；
两个模型训练后的完整权重不能互换。权重二进制不提交 Git。

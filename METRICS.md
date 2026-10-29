# 计数、定位与效率评价（两个仓库统一口径）

## 训练中自动评价
- MAE、RMSE：逐图计数误差，越低越好。变量、终端日志、JSON/CSV字段和新曲线统一使用 rmse/RMSE，不再输出 mse 别名。
- Precision / Recall / F1：默认在 4、8、16 像素半径分别计算，数值为 0～1。
- 使用评估图像的像素坐标（y,x）、欧氏距离和一对一匹配。
  先最大化容差内匹配数，再最小化距离；重复预测只能匹配一次。
- 全数据集累加 TP/FP/FN 后计算 micro 指标，不是逐图 F1 平均。
  分母为零时该项为 0；空图仍参加 MAE/RMSE，空图上的误检计为 FP。
- 两个模型均保留概率阈值 >0.5；P2R 等价于 logit>0。
  PET 越界/填充区域预测会过滤；P2R 的 RiceW3999 适配器按原始尺寸过滤填充预测。
- PET 分布式验证收集逐图结果并去重 DistributedSampler 补齐的样本。
- 定位匹配增加 CPU 验证耗时，不改变训练损失。不同原图分辨率下相同像素容差
  不代表相同物理距离；对比时固定数据、图像尺度与容差，勿使用测试集调参。
- 两个仓库均按验证 MAE 最小选 best；P2R 从原来的 4×MAE+RMSE 改为 MAE。
  旧 best 仍可单独评估，但原选择规则不同，正式对比请记录区别。
- 训练历史记录训练阶段峰值已分配显存（MiB）与训练时间；不等于整机 nvidia-smi 占用。

导出位置：实验 output 目录中的 metrics_history.csv、best_metrics.json，
以及 evaluation/epoch_XXXX/metrics.json 和 per_image.csv（epoch 从1计数）。
PET 独立 eval.py 默认导出到 evaluation/。
P2R --eval 默认导出到本次实验 output/evaluation/standalone/，不会继续训练。
metrics.json 包含协议、参数量和汇总指标；per_image.csv 包含逐图数量、误差及各容差 TP/FP/FN。
best_metrics.json 与本次新规则保存的最佳检查点对应。
已有实验请勿覆盖重跑；使用新实验名。真实数据和 GPU 性能不随代码更新自动产生。

## 依赖与 CPU 指标检查
镜像已有 torch/torchvision 时不要重装。定位匹配需要 scipy：
```bash
uv pip install --python "$(which python)" scipy --index-url https://mirror.sjtu.edu.cn/pypi/web/simple
python -m unittest discover -s tests -p test_counting_metrics.py -v
```

## 独立效率测试（不在每轮验证中运行）
```bash
python benchmark_model.py --checkpoint /path/to/best.pth --size 512 --device cuda --output benchmark.json
```
默认 batch1、FP32、TF32关闭、预热20次、测量100次。记录总参数量、
可训练参数量、平均/中位延迟、FPS、推理峰值已分配/预留显存及设备版本。
使用同步墙钟时间，只包含模型前向（含模型内部输出筛选），
不含读图、CPU/GPU传输、归一化、定位匹配或可视化。不是 TensorRT 测速。
显存测量与训练独立；P2R 仅创建学生模型，不包含教师副本。
只加载你自己信任的检查点。

两个模型的构造器仍需要已有本地 VGG 预训练文件：
PET 使用 pretrained/vgg16_bn-6c64b313.pth；
P2R 使用根目录同名文件，可用 --pretrained /path/to/weights.pth 指定。
PET 检查点中的 args 用于恢复其架构；无 args 时使用当前 eval.py 默认架构，
若自定义训练配置不同，需补齐检查点 args，严格载入不兼容会报错。
PET 当前实现依赖 CUDA；不支持用该脚本在 CPU 上测 PET。

PET 的动态查询路径依赖输入内容。默认随机输入仅用于基准，
应另用 --image /path/to/representative.jpg 测代表图像并在报告中注明，
也可对多张代表图像重复运行。不要把一次512输入测试视为所有密度场景的速度。

## 可选计算量
```bash
uv pip install --python "$(which python)" fvcore --index-url https://mirror.sjtu.edu.cn/pypi/web/simple
python benchmark_model.py --checkpoint /path/to/best.pth --device cuda --size 512 --flops --output benchmark.json
```
fvcore 将一次乘加计为一次运算。unsupported_ops 非空时只保存 counted_gflops，
gflops 留空，禁止将部分统计冒充完整计算量。PET 仅代表当前输入触发的追踪路径。
即使可选追踪失败，已测延迟和显存仍保存在文件中。计算量与测速安排在 GPUFree 执行；
当前修改仅通过 CPU 指标单测与源码语法检查，未声称获得真实 GPU 测量值。

命名统一后，旧实验文件不会自动改写。旧 metrics_history.csv 若含 mse 列，请使用新实验目录重新评估，避免混合新旧表头。

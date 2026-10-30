# 断点续训与验证

## 启动和恢复

```bash
cd /root/gpufree-data/p2rloss
BATCH_SIZE=16 EPOCHS=50 bash train_ricew3999.sh ricew3999_ep50_bs16
```

第一次从头训练；中断后执行同一条命令自动恢复。使用你原来的 TAG、BATCH_SIZE、EPOCHS 等配置。
例如原实验名是 `ricew3999_full_ep50_bs16_run2`，恢复时也必须使用该名字。
脚本日志追加到 `exp/<tag>/train.log/running.log`，不会清空原日志。
独立实验使用新的 TAG。

也可在原 `python main.py ...` 命令后加：

```bash
--resume exp/<tag>/output/ckpt_epoch_latest.pth
```

`--resume` 也接受实验的 output 目录，显式路径优先于自动恢复。
自动恢复默认为 `TRAIN.AUTO_RESUME=True`，优先选择固定的 latest 文件，而非按修改时间误选 best。
没有 latest 时会检查旧版 pth；若旧文件缺少训练状态，将明确报错。

## 保存内容

输出位于 `exp/<tag>/output/`：

- `ckpt_epoch_latest.pth`：每个完整 epoch 后保存，包括该轮计划的验证。
- `ckpt_epoch_best.pth`：验证 MAE 最低的完整状态，保留旧文件名。
- `ckpt_epoch_stage2.pth`：第 50 轮结束的阶段快照。

保存教师、学生、优化器、学习率调度器、零起始 epoch、最佳 MAE/RMSE/loss、最佳轮数、配置和 Python/NumPy/PyTorch/CUDA 随机状态。
原训练器为 FP32，没有 AMP GradScaler 状态需要恢复；学习率仍保持原训练器的固定策略，未启用原来注释掉的 scheduler.step()。

使用同目录临时文件、flush/fsync 和原子替换，写入失败时保留旧文件。
这不会防止实例磁盘被删除。每次只保证单个检查点文件的原子写入。

## 恢复边界与兼容性

- `EPOCHS=50` 表示总计 50 轮，已完成 20 轮则从第 21 轮继续；保持原来的总轮数、模型、标注比例、损失和学习率配置。
- 仅在完整 epoch 边界恢复，当前未保存的一轮会重跑；第一个检查点保存前中断则从头开始。不保存爆显存时的部分更新状态。
- 可以调小 batch size 再恢复，程序会警告训练轨迹改变；严格复现应保持数据、batch、workers、设备和环境一致。GPU 非确定性仍可能造成差异。
- 旧版只有 teacher/student 权重的 pth 无法补回已丢失的优化器和轮数，因此不能声称完整续训。可继续使用 `--eval --resume <旧文件>` 做评估；重新训练请用新 TAG。
- 模型使用严格结构匹配，训练检查点包含 Python 对象，只加载可信文件。
- CSV 中晚于检查点的指标行在恢复时移除，曲线从保留的历史指标继续。文本日志保留历史记录；重跑的逐轮评估覆盖相应目录。

## 验证频率

`SAVE_FREQ` 为兼容旧命令保留，现明确表示验证间隔：默认 5，即第 5、10、15……轮验证，最后一轮始终验证，不重复执行。
训练 50 轮验证 10 次；训练 12 轮则在 5、10、12 轮验证。
`VAL_EVERY` 环境变量仍映射到此配置。最新检查点每轮保存，不受验证间隔影响。

当前半监督分支从零起始 epoch=50 开始，第一轮权重为零，下一轮开始增加；50 轮训练不会实际利用未标注样本。100% 标注始终跳过半监督分支。恢复保留此阶段逻辑。

## 测试

```bash
python -m unittest discover -s tests -p test_checkpointing.py -v
```

测试使用 CPU 小模型检验连续/恢复训练的教师和学生参数、优化器、调度器与随机状态，另覆盖原子写入失败、自动选择 latest、旧版评估兼容、配置检查、历史指标和验证时机。实际 CUDA 与 RiceW3999 训练仍需在服务器验证。

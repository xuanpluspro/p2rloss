import os
from pathlib import Path
import torch

CKPT_PATH = "/root/autodl-tmp/P2RLoss-main/exp/autodl_test_run_50/output/ckpt_epoch_best.pth"

def run_check():
    path = Path(CKPT_PATH)
    if not path.is_file():
        print(f"错误: 找不到文件 {path}")
        return

    print("=" * 75)
    print(f"分析权重文件: {path.name}")
    print(f"文件大小: {os.path.getsize(path) / (1024 * 1024):.2f} MB")
    print("=" * 75)

    ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, dict):
        print("权重不是标准字典格式。")
        return

    print("权重文件包含的根键 (Root Keys):", list(ckpt.keys()))

    if "max_accuracy" in ckpt:
        acc = ckpt["max_accuracy"]
        print("-" * 75)
        print(f"训练记录的最佳指标 (max_accuracy):")
        print(f"  * Best MAE : {acc[0]:.4f}")
        print(f"  * Best RMSE : {acc[1]:.4f}")
        if len(acc) > 2:
            print(f"  * 产出 Epoch: {int(acc[2])}")

    print("-" * 75)
    # 统计 teacher 和 student 各自的参数量
    for role in ["student", "teacher"]:
        if role in ckpt:
            state = ckpt[role]
            # 过滤掉统计批次的计数器，只计算权重张量参数
            params = sum(v.numel() for k, v in state.items() if "num_batches_tracked" not in k)
            print(f"[{role.upper()} 模型]")
            print(f"  * 参数总量 : {params:,} ({params / 1e6:.3f} M)")
            print(f"  * FP32 大小: {params * 4 / (1024 * 1024):.2f} MB")
        else:
            print(f"未找到 {role} 键")

    print("-" * 75)
    # 细分子模块统计（以实际推理用的 student 为准）
    target_state = ckpt.get("student", ckpt.get("teacher"))
    if target_state:
        groups = {
            "encoders.0 (VGG浅中层)": 0,
            "encoders.1 (VGG深层)": 0,
            "fuse_layer (特征对齐与融合)": 0,
            "decoders (密度回归头)": 0,
        }
        for k, v in target_state.items():
            if "num_batches_tracked" in k:
                continue
            if "encoders.0." in k:
                groups["encoders.0 (VGG浅中层)"] += v.numel()
            elif "encoders.1." in k:
                groups["encoders.1 (VGG深层)"] += v.numel()
            elif "fuse_layer" in k:
                groups["fuse_layer (特征对齐与融合)"] += v.numel()
            elif "decoders" in k:
                groups["decoders (密度回归头)"] += v.numel()

        total_single = sum(groups.values())
        print(f"{'子模块组件':<30} | {'参数量 (Params)':<15} | {'占比 (%)'}")
        print("-" * 75)
        for name, cnt in groups.items():
            pct = (cnt / total_single) * 100 if total_single > 0 else 0
            print(f"{name:<30} | {cnt:>15,} | {pct:>8.2f}%")

    print("=" * 75)
    total_stored = sum(
        sum(v.numel() for k, v in ckpt[r].items() if "num_batches_tracked" not in k)
        for r in ["student", "teacher"] if r in ckpt
    )
    print(f"单模型测试部署参数量 (Student) : {total_single:,} ({total_single / 1e6:.3f} M)")
    print(f".pth 实际保存的总参数量 (双模型): {total_stored:,} ({total_stored / 1e6:.3f} M)")
    print("=" * 75)

if __name__ == "__main__":
    run_check()

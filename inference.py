import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# 导入你项目原有的模块
from config import get_config
from models import build_model
import argparse


def get_inference_config():
    # 模拟终端参数，快速获取配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--opts', default=None, nargs='+')
    
    # 把我们造的万能 MagicArgs 喂给解析器
    args, _ = parser.parse_known_args()
    
    missing_attrs = [
        'batch_size', 'data_path', 'label', 'protocol', 'resume', 
        'accumulation_steps', 'use_checkpoint', 'output', 'tag', 
        'eval', 'throughput'
    ]

    for attr in missing_attrs:
        if not hasattr(args, attr):
            setattr(args, attr, None)

    
    config = get_config(args)
    
    # 必须把模型设置为 VGG16BN，与训练时保持绝对一致
    config.defrost()
    config.MODEL.NAME = 'VGG16BN'
    config.freeze()
    return config

def main():
    # 1. 路径准备 (请把 test_img_path 替换为你随便挑的一张测试集图片路径)
    weight_path = "/root/autodl-tmp/P2RLoss-main/exp/autodl_test_run_50/output/ckpt_epoch_best.pth"
    test_img_path = "/root/autodl-tmp/P2RLoss-main/SHTech_PartA/train_data/images/IMG_100.jpg" # 替换成真实存在的图
    output_img_path = "inference_result.jpg"

    # 2. 初始化模型
    print("⏳ 正在构建模型...")
    config = get_inference_config()
    # inference 只需要拿 teacher 就够了（更稳定）
    _, teacher = build_model(config.MODEL)
    teacher = teacher.cuda()
    teacher.eval() # 开启评估模式

    # 3. 加载巅峰权重
    print(f"📦 正在加载权重: {weight_path}")
    checkpoint = torch.load(weight_path, map_location='cuda:0')
    # 存档里通常存了 'teacher' 和 'student'，我们只取 teacher
    teacher.load_state_dict(checkpoint['teacher'])
    print("✅ 权重加载成功！")

    # 4. 图像预处理 (与训练时保持一致的归一化)
    img_pil = Image.open(test_img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0).cuda() # 增加 Batch 维度

    # 5. 执行推理 (开启静音结界榨干算力)
    print("🧠 正在执行前向推理...")
    with torch.inference_mode():
        output = teacher(img_tensor)
    
    # 6. 解析预测坐标 (借用 Stage 2 的伪标签魔法逻辑)
    down_rate = img_tensor.size(-1) // output.size(-1) # 计算下采样倍率 (VGG 通常是 8)
    
    # 获取第一张图的输出 (去除 batch 维度)
    pred_map = output[0].squeeze() 
    
    # 找出所有大于等于 0 的候选坐标
    tseq = torch.nonzero(pred_map >= 0) 
    peaks = pred_map[tseq[:, 0], tseq[:, 1]]
    
    # 使用 70% 置信度阈值 (sigmoid(0.847) ≈ 0.7)
    threshold = 0.847 
    valid_points = tseq[peaks > threshold]
    
    # 将特征图坐标映射回原图坐标 (乘以 down_rate 并加上偏移)
    original_coords = valid_points * down_rate + (down_rate - 1) / 2
    original_coords = original_coords.cpu().numpy()

    print(f"🎯 模型预测该图总共有 {len(original_coords)} 个目标。")

    # 7. 可视化：在原图上画红点
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    for y, x in original_coords: # 注意 numpy 坐标是 (y, x)
        cv2.circle(img_cv, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)
    
    cv2.imwrite(output_img_path, img_cv)
    print(f"🖼️ 可视化结果已保存至: {output_img_path}")

if __name__ == '__main__':
    main()
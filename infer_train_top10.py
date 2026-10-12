# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from models.vgg16bn import VGG16_BN


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(ckpt_path, device, weights="student"):
    model = VGG16_BN(None).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint keys:", ckpt.keys())

    if weights not in ckpt:
        raise KeyError(
            f"checkpoint 中没有 '{weights}'，实际 keys={list(ckpt.keys())}"
        )

    msg = model.load_state_dict(ckpt[weights], strict=True)
    print(f"Loaded {weights}: {msg}")

    model.eval()
    return model


def preprocess(img):
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])(img)

    h, w = x.shape[-2:]

    # 与项目测试阶段一致：pad 到 32 的倍数
    ph = (32 - h % 32) % 32
    pw = (32 - w % 32) % 32

    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))

    return x.unsqueeze(0), (h, w)


@torch.inference_mode()
def predict(model, image_path, device, threshold=0.0):
    # 与原项目保持一致，不额外进行 EXIF transpose
    img = Image.open(image_path).convert("RGB")
    x, (orig_h, orig_w) = preprocess(img)
    x = x.to(device)

    output = model(x)

    out_h, out_w = output.shape[-2:]
    in_h, in_w = x.shape[-2:]

    down_h = in_h / out_h
    down_w = in_w / out_w

    # 与 main.py validate() 一致：output > 0 即认为存在预测点
    ys, xs = torch.nonzero(
        output[0, 0] > threshold,
        as_tuple=True
    )

    px = xs.float() * down_w + (down_w - 1) / 2
    py = ys.float() * down_h + (down_h - 1) / 2

    points = torch.stack([px, py], dim=1).cpu().numpy()

    # 去掉 padding 区域中的预测点
    if len(points) > 0:
        keep = (
            (points[:, 0] >= 0) &
            (points[:, 0] < orig_w) &
            (points[:, 1] >= 0) &
            (points[:, 1] < orig_h)
        )
        points = points[keep]

    return img, points, output.detach().cpu().numpy()


def draw_result(img, pred_points, gt_points, save_path, radius=8):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)

    # GT：绿色空心圆
    for x, y in gt_points:
        x, y = float(x), float(y)
        r = radius + 4
        draw.ellipse(
            [x-r, y-r, x+r, y+r],
            outline=(0, 255, 0),
            width=4
        )

    # Prediction：红色实心点
    for x, y in pred_points:
        x, y = float(x), float(y)
        draw.ellipse(
            [x-radius, y-radius, x+radius, y+radius],
            fill=(255, 0, 0),
            outline=(255, 255, 255),
            width=2
        )

    pred_n = len(pred_points)
    gt_n = len(gt_points)
    err = pred_n - gt_n

    text = f"Pred: {pred_n}   GT: {gt_n}   Error: {err:+d}"
    box = draw.textbbox((0, 0), text)
    tw = box[2] - box[0]
    th = box[3] - box[1]

    draw.rectangle([15, 15, 40 + tw, 40 + th], fill=(0, 0, 0))
    draw.text((27, 27), text, fill=(255, 255, 0))

    vis.save(save_path, quality=95)


def find_image(images_dir, stem):
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        default="/root/autodl-tmp/P2RLoss-main/exp/sudan_100_e30/output/ckpt_epoch_best.pth"
    )
    parser.add_argument(
        "--images-dir",
        default="/root/autodl-tmp/P2RLoss-main/sudanPart/train_data/images"
    )
    parser.add_argument(
        "--anno-dir",
        default="/root/autodl-tmp/P2RLoss-main/sudanPart/train_data/new-anno"
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/P2RLoss-main/train_top10_inference"
    )
    parser.add_argument(
        "--weights",
        choices=["student", "teacher"],
        default="student"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    images_dir = Path(args.images_dir)
    anno_dir = Path(args.anno_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===============================
    # 1. 根据 GT 点数排序
    # ===============================
    samples = []

    for npy_path in sorted(anno_dir.glob("*.npy")):
        gt = np.load(npy_path)

        if gt.ndim == 1:
            gt = gt.reshape(-1, 2)

        count = len(gt)

        image_path = find_image(images_dir, npy_path.stem)
        if image_path is None:
            print(f"[跳过] 找不到对应图片: {npy_path.stem}")
            continue

        samples.append((count, image_path, npy_path))

    if not samples:
        raise RuntimeError("没有找到有效的 image/npy 配对。")

    samples.sort(key=lambda x: x[0], reverse=True)
    selected = samples[:args.topk]

    print("\nGT 点数最多的图片：")
    for i, (count, image_path, _) in enumerate(selected, 1):
        print(f"{i:02d}. {image_path.name} | GT={count}")

    # ===============================
    # 2. 加载模型
    # ===============================
    model = load_model(args.ckpt, device, args.weights)

    # ===============================
    # 3. 推理
    # ===============================
    rows = []

    print("\n开始推理：")

    for i, (gt_count, image_path, npy_path) in enumerate(selected, 1):
        gt_points = np.load(npy_path)[:, :2].astype(np.float32)

        img, pred_points, raw_output = predict(
            model,
            image_path,
            device,
            threshold=args.threshold
        )

        pred_count = len(pred_points)
        error = pred_count - gt_count
        abs_error = abs(error)

        # 输出值统计，用来排查为什么 MAE 不变
        out_min = float(raw_output.min())
        out_max = float(raw_output.max())
        out_mean = float(raw_output.mean())
        positive_ratio = float((raw_output > args.threshold).mean())

        save_path = output_dir / f"{i:02d}_{image_path.stem}_GT{gt_count}_Pred{pred_count}.jpg"

        draw_result(
            img,
            pred_points,
            gt_points,
            save_path
        )

        print(
            f"[{i:02d}/{len(selected)}] {image_path.name} "
            f"| GT={gt_count} "
            f"| Pred={pred_count} "
            f"| Error={error:+d} "
            f"| output[min={out_min:.4f}, max={out_max:.4f}, mean={out_mean:.4f}] "
            f"| positive={positive_ratio*100:.2f}%"
        )

        rows.append([
            image_path.name,
            gt_count,
            pred_count,
            error,
            abs_error,
            out_min,
            out_max,
            out_mean,
            positive_ratio
        ])

    # ===============================
    # 4. 保存 CSV
    # ===============================
    csv_path = output_dir / "top10_results.csv"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "image,gt_count,pred_count,error,abs_error,"
            "output_min,output_max,output_mean,positive_ratio\n"
        )

        for row in rows:
            f.write(",".join(map(str, row)) + "\n")

    mae = np.mean([r[4] for r in rows])

    print("\n============================")
    print(f"Top-{len(rows)} mean absolute error: {mae:.3f}")
    print(f"可视化结果: {output_dir}")
    print(f"CSV结果: {csv_path}")
    print("============================")


if __name__ == "__main__":
    main()

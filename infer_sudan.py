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

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if weights not in checkpoint:
        raise KeyError(
            f"checkpoint 中没有 '{weights}'，实际 keys={list(checkpoint.keys())}"
        )

    msg = model.load_state_dict(checkpoint[weights], strict=True)
    print(f"Loaded {weights}: {msg}")

    model.eval()
    return model


def preprocess(img):
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])(img)

    h, w = tensor.shape[-2:]

    # 与项目 test 阶段一致：pad 到 32 的倍数
    ph = (32 - h % 32) % 32
    pw = (32 - w % 32) % 32

    if ph or pw:
        tensor = F.pad(tensor, (0, pw, 0, ph))

    return tensor.unsqueeze(0), (h, w)


@torch.inference_mode()
def predict(model, image_path, device, threshold=0.0):
    # 与原项目一致，不额外做 EXIF transpose
    img = Image.open(image_path).convert("RGB")
    x, (orig_h, orig_w) = preprocess(img)
    x = x.to(device)

    output = model(x)  # [1, 1, Hout, Wout]

    out_h, out_w = output.shape[-2:]
    in_h, in_w = x.shape[-2:]

    down_h = in_h / out_h
    down_w = in_w / out_w

    # 原项目 validate(): outnum = (output > 0).sum(...)
    ys, xs = torch.nonzero(
        output[0, 0] > threshold,
        as_tuple=True
    )

    # 对应 P2RLoss 中网格中心的位置
    px = xs.float() * down_w + (down_w - 1) / 2
    py = ys.float() * down_h + (down_h - 1) / 2

    points = torch.stack([px, py], dim=1).cpu().numpy()

    # 去掉 pad 区域中的点
    keep = (
        (points[:, 0] >= 0) &
        (points[:, 0] < orig_w) &
        (points[:, 1] >= 0) &
        (points[:, 1] < orig_h)
    )
    points = points[keep]

    return img, points, output.shape, (down_h, down_w)


def draw_result(img, points, save_path, radius=6, gt_points=None):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)

    # GT：绿色空心圆
    if gt_points is not None:
        for x, y in gt_points:
            x, y = float(x), float(y)
            r = radius + 3
            draw.ellipse(
                [x-r, y-r, x+r, y+r],
                outline=(0, 255, 0),
                width=3,
            )

    # Prediction：红色实心点
    for x, y in points:
        x, y = float(x), float(y)
        draw.ellipse(
            [x-radius, y-radius, x+radius, y+radius],
            fill=(255, 0, 0),
            outline=(255, 255, 255),
            width=2,
        )

    if gt_points is None:
        label = f"Pred: {len(points)}"
    else:
        label = f"Pred: {len(points)}   GT: {len(gt_points)}   Error: {len(points)-len(gt_points):+d}"

    box = draw.textbbox((0, 0), label)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    draw.rectangle([15, 15, 35 + tw, 35 + th], fill=(0, 0, 0))
    draw.text((25, 25), label, fill=(255, 255, 0))

    vis.save(save_path, quality=95)


def get_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([x for x in p.iterdir() if x.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/root/autodl-tmp/P2RLoss-main/exp/sudan_100_e30/output/ckpt_epoch_best.pth"
    )
    parser.add_argument(
        "--input",
        default="/root/autodl-tmp/P2RLoss-main/sudanPart/test_data/images"
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/P2RLoss-main/inference_results"
    )
    parser.add_argument(
        "--gt-dir",
        default="/root/autodl-tmp/P2RLoss-main/sudanPart/test_data/new-anno",
        help="NPY真值目录；不想比较GT时传空字符串"
    )
    parser.add_argument(
        "--weights",
        choices=["student", "teacher"],
        default="student"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="与项目验证代码一致，默认 output > 0"
    )
    parser.add_argument("--radius", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device, args.weights)

    images = get_images(args.input)
    print(f"Found {len(images)} images")

    summary = []

    for i, image_path in enumerate(images, 1):
        img, points, out_shape, down = predict(
            model,
            image_path,
            device,
            args.threshold
        )

        gt_points = None
        gt_count = None
        if args.gt_dir:
            gt_path = Path(args.gt_dir) / f"{image_path.stem}.npy"
            if gt_path.exists():
                gt_points = np.load(gt_path)[:, :2]
                gt_count = len(gt_points)

        save_path = output_dir / f"{image_path.stem}_pred.jpg"
        draw_result(
            img,
            points,
            save_path,
            args.radius,
            gt_points=gt_points
        )

        gt_msg = f" | GT={gt_count} | error={len(points)-gt_count:+d}" if gt_count is not None else ""
        print(
            f"[{i}/{len(images)}] {image_path.name} "
            f"| Pred={len(points)}{gt_msg} "
            f"| output={tuple(out_shape)} "
            f"| down=({down[0]:.2f},{down[1]:.2f})"
        )

        summary.append((image_path.name, len(points), gt_count))

    csv_path = output_dir / "counts.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,pred_count,gt_count,error\n")
        for name, pred_count, gt_count in summary:
            if gt_count is None:
                f.write(f"{name},{pred_count},,\n")
            else:
                f.write(f"{name},{pred_count},{gt_count},{pred_count-gt_count}\n")

    print(f"\n完成。可视化结果：{output_dir}")
    print(f"计数汇总：{csv_path}")


if __name__ == "__main__":
    main()

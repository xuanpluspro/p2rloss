"""Single-model inference benchmark; never starts training or downloads torch."""
import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted local training checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', default='benchmark.json')
    parser.add_argument('--pretrained', default='vgg16_bn-6c64b313.pth', help='P2R local backbone weights')
    parser.add_argument('--image', help='Optional representative RGB image, resized to --size')
    parser.add_argument('--flops', action='store_true', help='Optional fvcore profiling after timing')
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32 or args.iterations <= 0 or args.warmup < 0:
        parser.error('size must be a positive multiple of 32; iterations positive; warmup nonnegative')
    device = torch.device(args.device)
    torch.manual_seed(2024)
    torch.backends.cudnn.benchmark = False
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    is_pet = Path(__file__).with_name('engine.py').exists()
    if is_pet:
        if device.type != 'cuda':
            raise ValueError('PET currently requires CUDA; use --device cuda')
        from eval import get_args_parser
        from models import build_model
        options = get_args_parser().parse_args([])
        saved = checkpoint.get('args')
        if saved is not None:
            for key, value in (saved.items() if isinstance(saved, dict) else vars(saved).items()):
                setattr(options, key, value)
        options.device = str(device)
        model, _ = build_model(options)
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        from config import _C
        from models.vgg16bn import VGG16_BN
        options = _C.clone()
        options.MODEL.PRETRAINED = args.pretrained
        model = VGG16_BN(options.MODEL)
        model.load_state_dict(checkpoint['student'], strict=True)
    del checkpoint
    gc.collect()
    model = model.to(device).float().eval()
    if args.image:
        from PIL import Image
        from torchvision import transforms
        image = Image.open(args.image).convert('RGB').resize((args.size, args.size))
        x = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            [.485, .456, .406], [.229, .224, .225])])(image).unsqueeze(0).to(device)
    else:
        x = torch.randn(1, 3, args.size, args.size, device=device)

    class Forward(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, value):
            if is_pet:
                out = self.network(value, test=True)
                return out['pred_logits'], out['pred_points']
            return self.network(value)

    wrapper = Forward(model).eval()

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        # Explicit FP32: this is not AMP, TensorRT, or end-to-end I/O timing.
        with torch.autocast(device_type=device.type, enabled=False):
            for _ in range(args.warmup):
                wrapper(x)
            sync()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
            times = []
            for _ in range(args.iterations):
                sync()
                start = time.perf_counter()
                wrapper(x)
                sync()
                times.append((time.perf_counter() - start) * 1000)
    result = dict(model='PET' if is_pet else 'P2RLoss-student', checkpoint=str(args.checkpoint),
                  params=sum(p.numel() for p in model.parameters()),
                  trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  batch_size=1, input_size=[args.size, args.size], precision='FP32',
                  warmup=args.warmup, iterations=args.iterations,
                  latency_ms=statistics.mean(times), median_latency_ms=statistics.median(times),
                  fps=1000 / statistics.mean(times), device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  python_version=platform.python_version(), image=args.image,
                  timing='synchronized wall time of forward only; excludes I/O, transfer, normalization, matching',
                  input_dependent_forward=is_pet, gflops=None, counted_gflops=None)
    if device.type == 'cuda':
        result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
                      extra_peak_allocated_mib=(torch.cuda.max_memory_allocated(device) - baseline) / 1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2)
    # Persist timing even if the optional FLOPs tracer fails.
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + '\n')
    if args.flops:
        try:
            from fvcore.nn import FlopCountAnalysis
            with torch.inference_mode():
                profile = FlopCountAnalysis(wrapper, x)
                profile.unsupported_ops_warnings(False)
                counted = profile.total() / 1e9
                unsupported = dict(profile.unsupported_ops())
            result.update(counted_gflops=counted, unsupported_ops=unsupported,
                          gflops=counted if not unsupported else None,
                          flop_convention='fvcore: one multiply-add counts as one operation; traced path only')
        except Exception as error:
            result['flops_error'] = str(error)
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

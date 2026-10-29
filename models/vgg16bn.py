# -*- coding: utf-8 -*-

import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


if __name__ == '__main__':
    from utils import UpSample_P2P, SimpleDecoder
else:
    from .utils import UpSample_P2P, SimpleDecoder

class VGG16_BN(nn.Module):
    def __init__(self, config):
        super().__init__()
        vgg = models.vgg16_bn(weights=None)
        weight_path = Path(config.PRETRAINED) if config is not None else Path(__file__).resolve().parents[1] / 'pretrained' / 'vgg16_bn-6c64b313.pth'
        if not weight_path.is_file():
            raise FileNotFoundError(f"VGG16-BN weights not found: {weight_path}. Set --opts MODEL.PRETRAINED /path/to/weights.pth")
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        vgg.load_state_dict(state_dict)
        print(f"✅ 成功加载本地 VGG16-BN 预训练权重: {weight_path}")
        
        features = list(vgg.features.children())
        lids = [0, 33, 43]
        self.encoders = nn.ModuleList(nn.Sequential(*features[a:b]) for a, b in zip(lids[:-1], lids[1:]))
        self.num_channels = [512, 512]
        self.num_stage = len(self.num_channels)
        self.fuse_layer = UpSample_P2P(self.num_channels, ouc=256, bn=False, relu=False)
        
        self.decoders = SimpleDecoder(
            in_channel = self.fuse_layer.fuse_channel, 
            fea_channel = self.fuse_layer.fuse_channel, 
            up_scale = 2, 
            out_channel = 2
        )
        # self.register_buffer("up_scale", torch.tensor(100))
    
    def forward(self, image, need_fp=False):
        fea2 = self.encoding(image)
        if need_fp:
            fea2 = F.dropout2d(fea2, p=0.5)
        denmap = self.decoding(fea2)

        return denmap
    
    def encoding(self, x):
        feas = []
        for module in self.encoders:
            feas.append(x := module(x))
        feas = feas[-self.num_stage:]
        fea = self.fuse_layer(feas)
        
        return fea

    def decoding(self, fea2):
        denmap = self.decoders(fea2)
        if denmap.size(1) > 1:
            den1, den2 = denmap[:, :1], denmap[:, 1:2]
            den = den2 - den1
        else:
            den = denmap
        # raise
        return den
    
if __name__ == '__main__':
    model = VGG16_BN(None).cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    y = model(x)
    print(x.shape, y.shape)

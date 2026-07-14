#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def chanbonier_loss(network_output, gt, eps=1e-3):
    return torch.mean(torch.sqrt((network_output - gt) ** 2 + eps ** 2))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

class VGGPerceptualLoss(nn.Module):
    """
    VGG16 Perceptual Loss

    Args:
        layer_ids: VGG feature layer indices.
        weights: Weight for each selected layer.
        criterion: "l1" or "l2".
        resize: Whether to resize inputs before feeding into VGG.
        resize_size: Target spatial size.
    """
    def __init__(
        self,
        layer_ids=(3, 8, 15, 22),
        weights=(1.0, 1.0, 1.0, 1.0),
        criterion="l1",
        resize=True,
        resize_size=(224, 224),
    ):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.layer_ids = set(layer_ids)
        self.weights = weights
        self.resize = resize
        self.resize_size = resize_size
        if criterion == "l1":
            self.criterion = nn.L1Loss()
        elif criterion == "l2":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("criterion must be 'l1' or 'l2'")
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 3, H, W), values in [0, 1]
            target: (B, 3, H, W), values in [0, 1]
        """
        if self.resize:
            pred = F.interpolate(
                pred,
                size=self.resize_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            target = F.interpolate(
                target,
                size=self.resize_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        pred = self.normalize(pred)
        target = self.normalize(target)
        loss = 0.0
        weight_idx = 0
        for i, layer in enumerate(self.vgg):
            pred = layer(pred)
            target = layer(target)
            if i in self.layer_ids:
                loss += self.weights[weight_idx] * self.criterion(pred, target)
                weight_idx += 1
        return loss
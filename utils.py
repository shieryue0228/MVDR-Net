import os

import cv2
import torch
import random
from math import log10
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


# recommend
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal(m.weight.data, mode='fan_out')
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0
    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        # ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis=2)
    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


from pyiqa.archs.musiq_arch import MUSIQ

class MusiqWrapper(torch.nn.Module):
    def __init__(self, weight_path=None, device='cuda'):
        super().__init__()
        # 初始化 MUSIQ 模型
        self.musiq = MUSIQ()

        # 如果没有给路径，就尝试自动定位
        if weight_path is None:
            # 默认缓存目录
            home = os.path.expanduser("~")
            default_path = os.path.join(home, ".cache/torch/hub/pyiqa/musiq_ava_ckpt-e8d3f067.pth")
            weight_path = default_path

        assert os.path.exists(weight_path), f"权重文件不存在: {weight_path}"


        # 加载预训练权重
        ckpt = torch.load(weight_path, map_location=device)
        self.musiq.load_state_dict(ckpt, strict=True)

        self.musiq.to(device)
        self.musiq.eval()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.Tensor, shape [N, 3, H, W], range [0,1]
        Returns:
            scores: numpy array, shape [N]
        """
        x = x.clamp(0, 1)
        with torch.no_grad():
            scores = self.musiq(x)   # [N, 1]
        return scores.squeeze(-1).cpu().numpy()


import torch

def rain_layer_to_mask(rain_layer):
    """
    将 [B, 1, H, W] 的雨层图转换为 [B, 1, H, W] 的雨掩码。
    Args:
        rain_layer (torch.Tensor): 形状 [B, 1, H, W]，单通道格式的雨层图。
    Returns:
        rain_mask (torch.Tensor): 形状 [B, 1, H, W]，值在 [0, 1]，0 表示干净，1 表示雨。
    """
    # 1. 检查输入形状
    assert rain_layer.dim() == 4 and rain_layer.size(1) == 1, "rain_layer must have shape [B, 1, H, W]"

    # 2. 直接使用单通道雨层
    rain_mask = rain_layer.clone()  # [B, 1, H, W]

    # 3. 归一化到 [0, 1] 使用 sigmoid 方法
    # 假设 rain_layer 值可能为未激活强度，使用 sigmoid 映射到 [0, 1]
    rain_mask = torch.sigmoid(rain_mask)

    # 4. 可选：阈值处理（视需求调整）
    # 或：rain_mask = (rain_mask > 0.5).float()  # 二值化，视数据集调整阈值
    # 或：rain_mask = torch.clamp(rain_mask / rain_mask.max(), 0, 1)  # 如果需要基于最大值归一化

    return rain_mask
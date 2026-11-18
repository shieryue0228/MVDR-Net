import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import Conv2d
import numpy as np
from torch.nn.functional import l1_loss

from loss.contrast import Vgg19

class PerpetualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)


# Charbonnier loss
class CharLoss(nn.Module):
    def __init__(self):
        super(CharLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        diff = torch.add(pred, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss

class TVLoss(torch.nn.Module):
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class LocalContrastLoss(nn.Module): 
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.l1 = nn.L1Loss(reduction='none').cuda()
        self.vgg = Vgg19().cuda()
        self.weights = [1/32, 1/16, 1/8, 1/4, 1]
        self.patch_layers = [0, 1, 2]
        self.mask_layers = [0, 1, 2]
    def extract_patches(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "H, W 必须能整除 patch_size"
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        return x.contiguous().view(B, C * self.patch_size * self.patch_size, -1).permute(0, 2, 1)  # [B, patches, feat_dim]
    def forward(self, pred, pseudo, input_img):
        rain_mask = torch.abs(input_img - pseudo).mean(dim=1, keepdim=True)
        rain_mask = rain_mask / (rain_mask.max() + 1e-7)
        pred_feats   = self.vgg(pred)
        pseudo_feats = self.vgg(pseudo)
        input_feats  = self.vgg(input_img)
        total_loss = 0
        for i in range(len(pred_feats)):
            p_f  = pred_feats[i]
            ps_f = pseudo_feats[i]
            in_f = input_feats[i]
            if i in self.patch_layers:
                p_patch  = self.extract_patches(p_f)
                ps_patch = self.extract_patches(ps_f)
                in_patch = self.extract_patches(in_f)
                d_ap = self.l1(p_patch, ps_patch.detach()).mean(dim=2)
                d_an = self.l1(p_patch, in_patch.detach()).mean(dim=2)
                if i in self.mask_layers:
                    mask_resized = F.interpolate(
                        rain_mask, size=p_f.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                    mask_patch = self.extract_patches(mask_resized)
                    mean_mask = mask_patch.mean(dim=2)
                    scale = (1.0 + mean_mask).detach()
                    d_an = d_an * scale
                contrast = d_ap / (d_an + 1e-7)
                contrast = contrast.mean(dim=1)
            else: #  
                d_ap = self.l1(p_f, ps_f.detach()).mean(dim=[1,2,3])
                d_an = self.l1(p_f, in_f.detach()).mean(dim=[1,2,3])
                contrast = d_ap / (d_an + 1e-7)
            total_loss += self.weights[i] * contrast.mean()
        return total_loss

import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False, negative_mode='hazy', lowpass_pool=8, lowpass_weight=1.0):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.negative_mode = negative_mode
        self.lowpass_pool = lowpass_pool
        self.lowpass_weight = lowpass_weight
        if self.negative_mode not in ('hazy', 'hazy_lowpass'):
            raise ValueError('Unsupported CR negative_mode: {}'.format(self.negative_mode))
        if self.lowpass_pool <= 0:
            raise ValueError('lowpass_pool must be positive')
        if self.lowpass_weight < 0:
            raise ValueError('lowpass_weight must be non-negative')

    def lowpass(self, x):
        low = F.avg_pool2d(
            x,
            kernel_size=self.lowpass_pool,
            stride=self.lowpass_pool,
            ceil_mode=True
        )
        return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        lowpass_vgg = None
        if self.negative_mode == 'hazy_lowpass':
            lowpass_n = self.lowpass(n).detach()
            lowpass_vgg = self.vgg(lowpass_n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
                if lowpass_vgg is not None and self.lowpass_weight > 0:
                    d_an_lowpass = self.l1(a_vgg[i], lowpass_vgg[i].detach())
                    contrastive_lowpass = d_ap / (d_an_lowpass + 1e-7)
                    contrastive = (
                        contrastive + self.lowpass_weight * contrastive_lowpass
                    ) / (1.0 + self.lowpass_weight)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss

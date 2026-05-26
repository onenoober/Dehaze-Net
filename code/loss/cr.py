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


class CRPlusV2Loss(nn.Module):
    def __init__(
        self,
        negative_modes='hazy,output_lowpass,under_dehazed_mix',
        start_negative_modes='hazy,under_dehazed_mix',
        curriculum_steps=20000,
        lowpass_pool=8,
        frequency_weight=0.1,
        lowfreq_weight=0.1,
        under_dehazed_mix=0.5,
        ratio_cap=2.0
    ):
        super(CRPlusV2Loss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.negative_modes = self.parse_modes(negative_modes)
        self.start_negative_modes = self.parse_modes(start_negative_modes)
        self.curriculum_steps = curriculum_steps
        self.lowpass_pool = lowpass_pool
        self.frequency_weight = frequency_weight
        self.lowfreq_weight = lowfreq_weight
        self.under_dehazed_mix = under_dehazed_mix
        self.ratio_cap = ratio_cap
        supported = ('hazy', 'hazy_lowpass', 'output_lowpass', 'under_dehazed_mix')
        for mode in self.negative_modes + self.start_negative_modes:
            if mode not in supported:
                raise ValueError('Unsupported CRPlus-v2 negative mode: {}'.format(mode))
        if self.lowpass_pool <= 0:
            raise ValueError('crplus_v2_lowpass_pool must be positive')
        if self.frequency_weight < 0:
            raise ValueError('crplus_v2_frequency_weight must be non-negative')
        if self.lowfreq_weight < 0:
            raise ValueError('crplus_v2_lowfreq_weight must be non-negative')
        if self.ratio_cap <= 0:
            raise ValueError('crplus_v2_ratio_cap must be positive')
        if self.curriculum_steps < 0:
            raise ValueError('crplus_v2_curriculum_steps must be non-negative')

    def parse_modes(self, value):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [item.strip() for item in value.split(',') if item.strip()]

    def lowpass_image(self, x):
        low = F.avg_pool2d(
            x,
            kernel_size=self.lowpass_pool,
            stride=self.lowpass_pool,
            ceil_mode=True
        )
        return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)

    def lowfreq_distance(self, a, b):
        low_a = F.avg_pool2d(
            a,
            kernel_size=self.lowpass_pool,
            stride=self.lowpass_pool,
            ceil_mode=True
        )
        low_b = F.avg_pool2d(
            b,
            kernel_size=self.lowpass_pool,
            stride=self.lowpass_pool,
            ceil_mode=True
        )
        return self.l1(low_a, low_b)

    def frequency_distance(self, a, b):
        amp_a = torch.log1p(torch.abs(torch.fft.rfft2(a, norm='ortho')))
        amp_b = torch.log1p(torch.abs(torch.fft.rfft2(b, norm='ortho')))
        return self.l1(amp_a, amp_b)

    def vgg_distance(self, a_vgg, b_vgg):
        loss = 0
        for i in range(len(a_vgg)):
            loss += self.weights[i] * self.l1(a_vgg[i], b_vgg[i].detach())
        return loss

    def combined_distance(self, a, b, a_vgg=None, b_vgg=None):
        if a_vgg is None:
            a_vgg = self.vgg(a)
        if b_vgg is None:
            b_vgg = self.vgg(b)
        distance = self.vgg_distance(a_vgg, b_vgg)
        if self.frequency_weight > 0:
            distance = distance + self.frequency_weight * self.frequency_distance(a, b)
        if self.lowfreq_weight > 0:
            distance = distance + self.lowfreq_weight * self.lowfreq_distance(a, b)
        return distance

    def make_negatives(self, out, hazy):
        out_detached = out.detach()
        mix = self.under_dehazed_mix
        return {
            'hazy': hazy.detach(),
            'hazy_lowpass': self.lowpass_image(hazy.detach()),
            'output_lowpass': self.lowpass_image(out_detached),
            'under_dehazed_mix': (mix * out_detached + (1.0 - mix) * hazy.detach()).clamp(0, 1)
        }

    def active_modes(self, step):
        if self.curriculum_steps > 0 and step < self.curriculum_steps:
            return self.start_negative_modes
        return self.negative_modes

    def forward(self, out, target, hazy, step=0):
        out_vgg = self.vgg(out)
        target_detached = target.detach()
        target_vgg = self.vgg(target_detached)
        d_pos = self.combined_distance(out, target_detached, out_vgg, target_vgg)
        losses = []
        negatives = self.make_negatives(out, hazy)
        for mode in self.active_modes(step):
            negative = negatives[mode]
            negative_vgg = self.vgg(negative)
            d_neg = self.combined_distance(out, negative, out_vgg, negative_vgg)
            ratio = d_pos / (d_neg + 1e-7)
            losses.append(ratio.clamp(max=self.ratio_cap))
        return torch.stack(losses).mean()

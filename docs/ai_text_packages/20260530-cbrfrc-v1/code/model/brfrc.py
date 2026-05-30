import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def lowpass(x, pool_size):
    if pool_size <= 0:
        raise ValueError('pool_size must be positive')
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def tensor_stats(x):
    detached = x.detach()
    return {
        'mean': detached.mean(),
        'std': detached.std(unbiased=False),
        'min': detached.min(),
        'max': detached.max(),
    }


class BaselineRelativeFrequencyResidualCorrector(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=16,
        wavelet_levels=2,
        gate_init=-4.0,
        max_residual=0.08,
        hf_scale=0.1,
        use_haze_prior=True,
        preserve_highfreq=True,
        pyramid_type='laplacian',
        lf_pool=8,
        mid_pool=4,
        hf_gate_init=-5.0,
        max_color_residual=0.04,
        max_hf_residual=0.03,
    ):
        super(BaselineRelativeFrequencyResidualCorrector, self).__init__()
        if hidden_channels <= 0:
            raise ValueError('hidden_channels must be positive')
        if lf_pool <= 0 or mid_pool <= 0:
            raise ValueError('lf_pool and mid_pool must be positive')
        if pyramid_type not in ('laplacian', 'haar'):
            raise ValueError('Unsupported pyramid_type: {}'.format(pyramid_type))
        self.in_channels = in_channels
        self.wavelet_levels = wavelet_levels
        self.gate_init = gate_init
        self.max_residual = max_residual
        self.hf_scale = hf_scale
        self.use_haze_prior = use_haze_prior
        self.preserve_highfreq = preserve_highfreq
        self.pyramid_type = pyramid_type
        self.lf_pool = lf_pool
        self.mid_pool = mid_pool
        self.hf_gate_init = hf_gate_init
        self.max_color_residual = max_color_residual
        self.max_hf_residual = max_hf_residual
        self.last_diag = None

        input_channels = in_channels * 11
        if use_haze_prior:
            input_channels += 2

        self.shared = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )
        self.raw_lf_head = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.raw_color_head = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.raw_hf_head = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.gate_lf_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.gate_color_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.gate_hf_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for head in (self.raw_lf_head, self.raw_color_head, self.raw_hf_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head, bias in (
            (self.gate_lf_head, self.gate_init),
            (self.gate_color_head, self.gate_init),
            (self.gate_hf_head, self.hf_gate_init),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, float(bias))

    def haze_cues(self, low):
        dark = torch.min(low, dim=1, keepdim=True)[0]
        luma = low[:, 0:1] * 0.299 + low[:, 1:2] * 0.587 + low[:, 2:3] * 0.114
        dark = dark - dark.mean(dim=(2, 3), keepdim=True)
        luma = luma - luma.mean(dim=(2, 3), keepdim=True)
        return torch.cat([dark, luma], dim=1)

    def pyramid(self, x):
        if self.pyramid_type != 'laplacian':
            raise NotImplementedError('CBRFRC-v1 implements laplacian pyramid only.')
        lp_mid = lowpass(x, self.mid_pool)
        lp_lf = lowpass(x, self.lf_pool)
        hf = x - lp_mid
        return lp_mid, lp_lf, hf

    def forward(self, hazy, j0, feats=None, target=None):
        del feats
        lp_i_mid, lp_i_lf, hf_i = self.pyramid(hazy)
        lp_j0_mid, lp_j0_lf, hf_j0 = self.pyramid(j0)
        inputs = [
            hazy,
            j0,
            hazy - j0,
            lp_i_mid,
            lp_j0_mid,
            lp_i_mid - lp_j0_mid,
            lp_i_lf,
            lp_j0_lf,
            lp_i_lf - lp_j0_lf,
            hf_i,
            hf_j0,
        ]
        if self.use_haze_prior:
            inputs.append(self.haze_cues(lp_i_lf))

        feat = self.shared(torch.cat(inputs, dim=1))
        raw_lf = self.raw_lf_head(feat)
        raw_color = self.raw_color_head(feat)
        raw_hf = self.raw_hf_head(feat)
        gate_lf = torch.sigmoid(self.gate_lf_head(feat))
        gate_color = torch.sigmoid(self.gate_color_head(feat))
        gate_hf = torch.sigmoid(self.gate_hf_head(feat))

        c_lf = gate_lf * self.max_residual * torch.tanh(raw_lf)
        c_color = gate_color * self.max_color_residual * torch.tanh(raw_color)
        hf_extra_scale = self.hf_scale if self.preserve_highfreq else 1.0
        c_hf = hf_extra_scale * gate_hf * self.max_hf_residual * torch.tanh(raw_hf)
        out = torch.clamp(j0 + c_lf + c_color + c_hf, 0.0, 1.0)

        target_lf = None
        if target is not None:
            target_lf = lowpass(target, self.lf_pool) - lp_j0_lf.detach()

        diag = {
            'gate_lf': tensor_stats(gate_lf),
            'gate_color': tensor_stats(gate_color),
            'gate_hf': tensor_stats(gate_hf),
            'c_lf_norm': c_lf.detach().reshape(c_lf.shape[0], -1).norm(dim=1).mean(),
            'c_color_norm': c_color.detach().reshape(c_color.shape[0], -1).norm(dim=1).mean(),
            'c_hf_norm': c_hf.detach().reshape(c_hf.shape[0], -1).norm(dim=1).mean(),
        }
        self.last_diag = diag

        return {
            'out': out,
            'j0': j0,
            'c_lf': c_lf,
            'c_color': c_color,
            'c_hf': c_hf,
            'gate_lf': gate_lf,
            'gate_color': gate_color,
            'gate_hf': gate_hf,
            'raw_lf': raw_lf,
            'raw_color': raw_color,
            'raw_hf': raw_hf,
            'target_lf': target_lf,
            'diag': diag,
        }


class DEANetCBRFRC(nn.Module):
    def __init__(
        self,
        baseline,
        corrector,
        freeze_baseline=True,
        use_baseline_detach=True,
    ):
        super(DEANetCBRFRC, self).__init__()
        self.baseline = baseline
        self.corrector = corrector
        self.freeze_baseline = freeze_baseline
        self.use_baseline_detach = use_baseline_detach
        if freeze_baseline:
            self.freeze_baseline_parameters()

    def freeze_baseline_parameters(self):
        self.baseline.eval()
        for param in self.baseline.parameters():
            param.requires_grad = False

    def _baseline_forward(self, hazy):
        if hasattr(self.baseline, 'forward_with_features'):
            out, feats = self.baseline.forward_with_features(hazy)
        else:
            out = self.baseline(hazy)
            feats = None
        return out.clamp(0.0, 1.0), feats

    def forward(self, hazy, target=None, return_aux=False):
        if self.freeze_baseline:
            self.baseline.eval()
            with torch.no_grad():
                j0, feats = self._baseline_forward(hazy)
        else:
            j0, feats = self._baseline_forward(hazy)
        corrector_j0 = j0.detach() if self.use_baseline_detach else j0
        out_dict = self.corrector(hazy, corrector_j0, feats=feats, target=target)
        if return_aux:
            return out_dict
        return out_dict['out']

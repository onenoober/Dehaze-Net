import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DEABlockTrain, DEBlockTrain, CGAFusion


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class LowFrequencyPrior(nn.Module):
    def __init__(
        self,
        out_channels,
        adapter_channels=8,
        pool_size=8,
        gate_init=0.0,
        residual_center=False,
        train_dropout=0.0,
        gate_max=0.0,
        conditional_mask=False,
        mask_hidden_channels=8,
        mask_init_bias=2.0,
        haze_aware_mask=False,
        haze_mask_strength=1.0,
        residual_calibration=False,
        calib_hidden_channels=8,
        calib_alpha_max=1.0,
        residual_selector=False,
        selector_hidden_channels=8,
        selector_init_bias=2.0
    ):
        super(LowFrequencyPrior, self).__init__()
        if pool_size <= 0:
            raise ValueError('pool_size must be positive')
        if train_dropout < 0 or train_dropout >= 1:
            raise ValueError('train_dropout must be in [0, 1)')
        if gate_max < 0:
            raise ValueError('gate_max must be non-negative')
        if mask_hidden_channels <= 0:
            raise ValueError('mask_hidden_channels must be positive')
        if haze_mask_strength < 0:
            raise ValueError('haze_mask_strength must be non-negative')
        if calib_hidden_channels <= 0:
            raise ValueError('calib_hidden_channels must be positive')
        if calib_alpha_max < 0:
            raise ValueError('calib_alpha_max must be non-negative')
        if selector_hidden_channels <= 0:
            raise ValueError('selector_hidden_channels must be positive')
        if residual_selector and not residual_calibration:
            raise ValueError('residual_selector requires residual_calibration')
        self.pool_size = pool_size
        self.residual_center = residual_center
        self.train_dropout = train_dropout
        self.gate_max = gate_max
        self.conditional_mask = conditional_mask
        self.haze_aware_mask = haze_aware_mask
        self.haze_mask_strength = haze_mask_strength
        self.residual_calibration = residual_calibration
        self.calib_alpha_max = calib_alpha_max
        self.residual_selector = residual_selector
        self.last_mask_stats = None
        self.last_alpha_stats = None
        self.last_selector_stats = None
        self.adapter = nn.Sequential(
            nn.Conv2d(3, adapter_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(adapter_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        if residual_calibration:
            self.calib_low_encoder = nn.Sequential(
                nn.Conv2d(3, calib_hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True)
            )
            self.calib_target_hint = nn.Conv2d(out_channels, calib_hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
            self.calib_shared = nn.Sequential(
                nn.Conv2d(calib_hidden_channels * 2, calib_hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True)
            )
            self.calib_direction_head = nn.Conv2d(calib_hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            self.calib_alpha_head = nn.Conv2d(calib_hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
            nn.init.zeros_(self.calib_alpha_head.weight)
            nn.init.zeros_(self.calib_alpha_head.bias)
            if residual_selector:
                self.selector_low_encoder = nn.Sequential(
                    nn.Conv2d(3, selector_hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(True)
                )
                self.selector_target_hint = nn.Conv2d(out_channels, selector_hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
                self.selector_head = nn.Sequential(
                    nn.Conv2d(selector_hidden_channels * 2, selector_hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(selector_hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
                )
                nn.init.zeros_(self.selector_head[-1].weight)
                nn.init.constant_(self.selector_head[-1].bias, float(selector_init_bias))
        if conditional_mask:
            mask_input_channels = 3
            if haze_aware_mask:
                mask_input_channels += 2
            self.mask_low_encoder = nn.Sequential(
                nn.Conv2d(mask_input_channels, mask_hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True)
            )
            self.mask_target_hint = nn.Conv2d(out_channels, mask_hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
            self.mask_head = nn.Sequential(
                nn.Conv2d(mask_hidden_channels * 2, mask_hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(mask_hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
            )
            nn.init.zeros_(self.mask_head[-1].weight)
            nn.init.constant_(self.mask_head[-1].bias, float(mask_init_bias))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def haze_cues(self, low):
        dark = torch.min(low, dim=1, keepdim=True)[0]
        luma = low[:, 0:1] * 0.299 + low[:, 1:2] * 0.587 + low[:, 2:3] * 0.114
        dark_centered = dark - dark.mean(dim=(2, 3), keepdim=True)
        luma_centered = luma - luma.mean(dim=(2, 3), keepdim=True)
        return torch.cat([dark_centered, luma_centered], dim=1) * self.haze_mask_strength

    def forward(self, hazy, target):
        low = F.avg_pool2d(
            hazy,
            kernel_size=self.pool_size,
            stride=self.pool_size,
            ceil_mode=True
        )
        low = F.interpolate(low, size=target.shape[-2:], mode='bilinear', align_corners=False)
        prior = self.adapter(low)
        if self.residual_calibration:
            lf_prior = prior
            calib_low = self.calib_low_encoder(low)
            target_hint = self.calib_target_hint(target.detach())
            calib_feat = self.calib_shared(torch.cat([calib_low, target_hint], dim=1))
            direction = torch.tanh(self.calib_direction_head(calib_feat))
            alpha = torch.sigmoid(self.calib_alpha_head(calib_feat)) * self.calib_alpha_max
            with torch.no_grad():
                detached_alpha = alpha.detach()
                self.last_alpha_stats = {
                    'mean': detached_alpha.mean(),
                    'std': detached_alpha.std(unbiased=False),
                    'min': detached_alpha.min(),
                    'max': detached_alpha.max()
                }
            calib_prior = direction * alpha
            if self.residual_selector:
                selector_low = self.selector_low_encoder(low)
                selector_hint = self.selector_target_hint(target.detach())
                selector = torch.sigmoid(self.selector_head(torch.cat([selector_low, selector_hint], dim=1)))
                with torch.no_grad():
                    detached_selector = selector.detach()
                    self.last_selector_stats = {
                        'mean': detached_selector.mean(),
                        'std': detached_selector.std(unbiased=False),
                        'min': detached_selector.min(),
                        'max': detached_selector.max()
                    }
                prior = selector * lf_prior + (1.0 - selector) * calib_prior
            else:
                self.last_selector_stats = None
                prior = calib_prior
        else:
            self.last_alpha_stats = None
            self.last_selector_stats = None
        if self.conditional_mask:
            mask_input = low
            if self.haze_aware_mask:
                mask_input = torch.cat([low, self.haze_cues(low)], dim=1)
            mask_low = self.mask_low_encoder(mask_input)
            target_hint = self.mask_target_hint(target.detach())
            mask = torch.sigmoid(self.mask_head(torch.cat([mask_low, target_hint], dim=1)))
            with torch.no_grad():
                detached_mask = mask.detach()
                self.last_mask_stats = {
                    'mean': detached_mask.mean(),
                    'std': detached_mask.std(unbiased=False),
                    'min': detached_mask.min(),
                    'max': detached_mask.max()
                }
            prior = prior * mask
        else:
            self.last_mask_stats = None
        if self.residual_center:
            prior = prior - prior.mean(dim=(2, 3), keepdim=True)
        if self.training and self.train_dropout > 0:
            keep_prob = 1.0 - self.train_dropout
            mask = torch.empty(
                prior.shape[0], 1, 1, 1,
                dtype=prior.dtype,
                device=prior.device
            ).bernoulli_(keep_prob).div_(keep_prob)
            prior = prior * mask
        gate = self.gate
        if self.gate_max > 0:
            gate = gate.clamp(min=-self.gate_max, max=self.gate_max)
        return target + gate * prior


class DEANet(nn.Module):
    def __init__(
        self,
        base_dim=32,
        use_lf_prior=False,
        lf_prior_channels=8,
        lf_prior_pool=8,
        lf_prior_gate_init=0.0,
        lf_prior_residual_center=False,
        lf_prior_train_dropout=0.0,
        lf_prior_gate_max=0.0,
        lf_prior_injection='pre_mix',
        lf_conditional_mask=False,
        lf_mask_hidden_channels=8,
        lf_mask_init_bias=2.0,
        lf_haze_aware_mask=False,
        lf_haze_mask_strength=1.0,
        lf_residual_calibration=False,
        lf_calib_hidden_channels=8,
        lf_calib_alpha_max=1.0,
        lf_residual_selector=False,
        lf_selector_hidden_channels=8,
        lf_selector_init_bias=2.0
    ):
        super(DEANet, self).__init__()
        if lf_prior_injection not in ('pre_mix', 'post_mix'):
            raise ValueError('Unsupported lf_prior_injection: {}'.format(lf_prior_injection))
        self.use_lf_prior = use_lf_prior
        self.lf_prior_injection = lf_prior_injection
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1
        self.down_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)
        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)
        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block2 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block3 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block4 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block5 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block6 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block7 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block8 = DEABlockTrain(default_conv, base_dim * 4, 3)
        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self.mix1 = CGAFusion(base_dim * 4, reduction=8)
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)
        if use_lf_prior:
            self.lf_prior = LowFrequencyPrior(
                out_channels=base_dim * 4,
                adapter_channels=lf_prior_channels,
                pool_size=lf_prior_pool,
                gate_init=lf_prior_gate_init,
                residual_center=lf_prior_residual_center,
                train_dropout=lf_prior_train_dropout,
                gate_max=lf_prior_gate_max,
                conditional_mask=lf_conditional_mask,
                mask_hidden_channels=lf_mask_hidden_channels,
                mask_init_bias=lf_mask_init_bias,
                haze_aware_mask=lf_haze_aware_mask,
                haze_mask_strength=lf_haze_mask_strength,
                residual_calibration=lf_residual_calibration,
                calib_hidden_channels=lf_calib_hidden_channels,
                calib_alpha_max=lf_calib_alpha_max,
                residual_selector=lf_residual_selector,
                selector_hidden_channels=lf_selector_hidden_channels,
                selector_init_bias=lf_selector_init_bias
            )
        else:
            self.lf_prior = None

    def forward(self, x):
        hazy = x
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        if self.lf_prior is not None and self.lf_prior_injection == 'pre_mix':
            x8 = self.lf_prior(hazy, x8)
        x_level3_mix = self.mix1(x_down3, x8)
        if self.lf_prior is not None and self.lf_prior_injection == 'post_mix':
            x_level3_mix = self.lf_prior(hazy, x_level3_mix)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DEABlockTrain, DEBlockTrain, CGAFusion


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class LowFrequencyPrior(nn.Module):
    def __init__(self, out_channels, adapter_channels=8, pool_size=8, gate_init=0.0):
        super(LowFrequencyPrior, self).__init__()
        if pool_size <= 0:
            raise ValueError('pool_size must be positive')
        self.pool_size = pool_size
        self.adapter = nn.Sequential(
            nn.Conv2d(3, adapter_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(adapter_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hazy, target):
        low = F.avg_pool2d(
            hazy,
            kernel_size=self.pool_size,
            stride=self.pool_size,
            ceil_mode=True
        )
        low = F.interpolate(low, size=target.shape[-2:], mode='bilinear', align_corners=False)
        prior = self.adapter(low)
        return target + self.gate * prior


class DEANet(nn.Module):
    def __init__(self, base_dim=32, use_lf_prior=False, lf_prior_channels=8, lf_prior_pool=8, lf_prior_gate_init=0.0):
        super(DEANet, self).__init__()
        self.use_lf_prior = use_lf_prior
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
                gate_init=lf_prior_gate_init
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
        if self.lf_prior is not None:
            x8 = self.lf_prior(hazy, x8)
        x_level3_mix = self.mix1(x_down3, x8)

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

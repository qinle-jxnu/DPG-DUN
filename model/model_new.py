import math

import cv2
import numpy as np
import torch
from pytorch_wavelets import DWTForward
from timm.layers import trunc_normal_
from torch import nn
import torch.nn.functional as F
from torch.nn import init

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2)
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        x = x.transpose(1, 2).view(b, c, h, w)
        return x
class SpatialChannelGate(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)
class FFN(nn.Module):
    def __init__(self, dim, hidden=4):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, dim * hidden, kernel_size=1)
        self.pos = nn.Conv2d(dim * hidden, dim * hidden, kernel_size=3, padding=1, groups=dim * hidden)
        self.fc2 = nn.Conv2d(dim * hidden, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x
class MSSR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.local3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.local5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.local7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.mix = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.gate = SpatialChannelGate(dim)
        self.ffn = FFN(dim)
        self.alpha = nn.Parameter(5e-2 * torch.ones((dim, 1, 1)))
        self.beta = nn.Parameter(5e-2 * torch.ones((dim, 1, 1)))

    def forward(self, x):
        residual = self.norm(x)
        residual = torch.cat([self.local3(residual), self.local5(residual), self.local7(residual)], dim=1)
        residual = self.mix(residual)
        residual = self.gate(residual)
        x = x + self.alpha * residual
        x = x + self.beta * self.ffn(x)
        return x
class CSHG(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU()
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU()
        )

        self.up = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.GELU()
        )

        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

        self.conv_out = nn.Conv2d(
            dim * 2, dim, kernel_size=3, padding=1
        )

        # Haar 小波分解
        self.dwt = DWTForward(
            J=1,
            wave="haar",
            mode="zero"
            # mode="symmetric"
        )

        # 3个高频分量：[LH, HL, HH]
        self.high_conv1 = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False
            ),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

        self.high_conv2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False
            ),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

        # 初始值为0，训练初期不影响原始特征
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))

    def high_frequency_gate(self, x, high_conv):
        """
        x: 上一阶段特征 [B, C, H, W]

        返回：
        gate [B, C, H/2, W/2]
        """
        _, yh = self.dwt(x)

        # yh[0]: [B, C, 3, H/2, W/2]
        high = yh[0].flatten(1, 2)

        return torch.sigmoid(high_conv(high))

    def forward(self, x, x_as=None):
        """
        x:
            当前阶段特征 [B, C, 96, 96]

        x_as:
            上一阶段返回的特征：
            x_as[0] = [B, C, 96, 96]
            x_as[1] = [B, C, 48, 48]
        """

        # 96×96
        x_down0 = x

        # 96×96 -> 48×48
        x_down1 = self.down1(x_down0)

        if x_as is not None:
            # 上一阶段96×96特征经过DWT，得到48×48高频门控
            gate1 = self.high_frequency_gate(
                x_as[0], self.high_conv1
            )

            # 残差门控，避免 sigmoid 直接把特征整体压小
            x_down1 = x_down1 + self.alpha1 * x_down1 * gate1

        # 48×48 -> 24×24
        x_down2 = self.down2(x_down1)

        if x_as is not None:
            # 上一阶段48×48特征经过DWT，得到24×24高频门控
            gate2 = self.high_frequency_gate(
                x_as[1], self.high_conv2
            )

            x_down2 = x_down2 + self.alpha2 * x_down2 * gate2

        # 24×24 -> 48×48
        x_up1 = self.up(x_down2)
        x_up1 = self.mix(x_up1 + x_down1)

        # 48×48 -> 96×96
        x_up2 = self.up(x_up1)
        x_up2 = self.mix(x_up2 + x_down0)

        x_out = x_up2 + self.res(x_up2)

        # 输入输出都是 [B, C, 96, 96]
        output = self.conv_out(
            torch.cat([x, x_out], dim=1)
        )

        # 给下一阶段传递96×96和48×48特征
        return output, [x_up2, x_up1]

def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class IterationModule(nn.Module):
    def __init__(self,dim,block_size,layer):
        super(IterationModule, self).__init__()
        self.dim = dim
        self.block_size = block_size
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.AConv1 =nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim,dim,kernel_size=3, padding=1,groups=dim)
        )
        self.conv1 = nn.Conv2d(dim, 1, 1)
        self.cat = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.post = nn.Conv2d(dim,1,3,1,1)
        self.CSHG = CSHG(dim)
        self.MSSR = MSSR(dim)

    def forward(self,xk,z,r,x_pre,Phi):
        r = self.alpha*r
        xk = self.AConv1(xk)
        xk = self.cat(torch.cat([xk,z],dim=1))
        m,x_pre = self.CSHG(xk,x_pre)
        m = self.MSSR(m)
        z = m
        x = m
        x = self.conv1(x)
        n = x - self.alpha * PhiTPhi_fun(x, Phi)
        x = r + n
        return x,z,x_pre

class Model(nn.Module):
    def __init__(self,ratio,depth=10,dim=32,block_size=32):
        super(Model,self).__init__()
        self.dim = dim
        self.ratio = ratio
        self.block_size = block_size
        self.depth = depth
        self.measurement = int(1024 * self.ratio)
        self.alpha = nn.Parameter(torch.Tensor([1.0]))
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=3, padding=1,bias=True)
        self.iters = nn.ModuleList()
        for _ in range(self.depth):
            self.iters.append(IterationModule(self.dim,block_size,_))
        self.apply(self._init_weights)
        self.DGM_Conv = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1,bias=True),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=7, padding=3),
            nn.GELU()
        )


    def forward(self,x):
        x = self.res1(x)+x
        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)
        x_init = F.conv_transpose2d(y, Phi, stride=self.block_size, padding=0, bias=None)
        x_pre = None
        r = self.alpha * x_init
        xk = x_init
        x = self.DGM_Conv(xk)
        z = x
        x = self.conv2(x)
        n = x - self.alpha * PhiTPhi_fun(x, Phi)
        x = r + n
        for i in range(self.depth):
            x, z,x_pre = self.iters[i](x, z,r,x_pre,Phi)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(18,1,96,96).to(device)
    model = Model(ratio=0.1, depth=1,dim=16 ,block_size=32).to(device)
    y = model(x)
    print(y.shape)
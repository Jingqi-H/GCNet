import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
        # self.bn = SynchronizedBatchNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, e=None):
        if e is not None:
            if x.shape[2] != e.shape[2]:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, e], 1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = F.relu(self.conv1(x), inplace=True)
        # print('conv1:', x.shape)
        x = F.relu(self.conv2(x), inplace=True)
        # print('conv2:', x.shape)
        return x

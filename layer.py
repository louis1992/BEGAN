from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

# Encoder from BEGAN.
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.num_channels = opt.nc
        self.h = opt.h
        self.b_size = opt.b_size
        self.scale_size = opt.scale_size
        
        self.layer0 = nn.Conv2d(3, self.num_channels, 3, 1, 1)
        self.layer1 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.layer2 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.down1 = nn.Conv2d(self.num_channels, self.num_channels, 1, 1 , 0)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.layer3 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.layer4 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.down2 = nn.Conv2d(self.num_channels, 2*self.num_channels, 1, 1, 0)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.layer5 = nn.Conv2d(2*self.num_channels, 2*self.num_channels, 3, 1, 1)
        self.layer6 = nn.Conv2d(2*self.num_channels, 2*self.num_channels, 3 ,1 ,1)
        self.down3 = nn.Conv2d(2*self.num_channels, 3*self.num_channels, 1, 1, 0)
        self.pool3 = nn.AvgPool2d(2, 2)

        if self.scale_size == 64:
            self.layer7 = nn.Conv2d(3*self.num_channels, 3*self.num_channels, 3, 1, 1)
            self.layer8 = nn.Conv2d(3*self.num_channels, 3*self.num_channels, 3, 1, 1)
            self.layer9 = nn.Linear(8*8*3*self.num_channels, 64)

        elif self.scale_size == 128:
            self.layer7 = nn.Conv2d(3*self.num_channels, 3*self.num_channels, 3, 1, 1)
            self.layer8 = nn.Conv2d(3*self.num_channels, 3*self.num_channels, 3, 1, 1)
            self.down4 = nn.Conv2d(3*self.num_channels, 4*self.num_channels, 1, 1, 0)
            self.pool4 = nn.AvgPool2d(2, 2)

            self.layer9 = nn.Conv2d(4*self.num_channels, 4*self.num_channels, 3, 1, 1)
            self.layer10 = nn.Conv2d(4*self.num_channels, 4*self.num_channels, 3, 1, 1)
            self.layer11 = nn.Linear(8*8*4*self.num_channels, self.h)

    def forward(self, x):
        x = F.elu(self.layer0(x), True)
        x = F.elu(self.layer1(x), True)
        x = F.elu(self.layer2(x), True)
        x = self.down1(x)
        x = self.pool1(x)

        x = F.elu(self.layer3(x), True)
        x = F.elu(self.layer4(x), True)
        x = self.down2(x)
        x = self.pool2(x)

        x = F.elu(self.layer5(x), True)
        x = F.elu(self.layer6(x), True)
        x = self.down3(x)
        x = self.pool3(x)

        if self.scale_size == 64:
            x = F.elu(self.layer7(x), True)
            x = F.elu(self.layer8(x), True)
            x = x.view(self.b_size, 8*8*3*self.num_channels)
            x = self.layer9(x)

        elif self.scale_size == 128:
            x = F.elu(self.layer7(x), True)
            x = F.elu(self.layer8(x), True)
            x = self.down4(x)
            x = self.pool4(x)
            x = F.elu(self.layer9(x), True)
            x = F.elu(self.layer10(x), True)
            x = x.view(self.b_size, 8*8*4*self.num_channels)
            x = F.elu(self.layer11(x), True)

        return x

# Decoder from BEGAN.
class Decoder(nn.Module):
    def __init__(self, opt, disc=False):
        super(Decoder, self).__init__()
        self.num_channels = opt.nc
        self.b_size = opt.b_size
        self.h = opt.h
        self.disc = disc
        self.t_act = opt.tanh
        self.scale_size = opt.scale_size

        self.layer0 = nn.Linear(self.h, 8*8*self.num_channels)
        self.layer1 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.layer2 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.layer3 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.layer4 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.layer5 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.layer6 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.layer7 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.layer8 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        if self.scale_size == 128:
            self.up4 = nn.UpsamplingNearest2d(scale_factor=2)

            self.layer9 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
            self.layer10 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)

        self.layer11 = nn.Conv2d(self.num_channels, 3, 3, 1, 1)

    def forward(self, x):
        x = self.layer0(x)
        x = x.view(self.b_size, self.num_channels, 8 ,8)

        x = F.elu(self.layer1(x), True)
        x = F.elu(self.layer2(x), True)
        x = self.up1(x)

        x = F.elu(self.layer3(x), True)
        x = F.elu(self.layer4(x), True)
        x = self.up2(x)

        x = F.elu(self.layer5(x), True)
        x = F.elu(self.layer6(x), True)
        x = self.up3(x)

        x = F.elu(self.layer7(x), True)
        x = F.elu(self.layer8(x), True)
        if self.scale_size == 128:
            x = self.up4(x)

            x = F.elu(self.layer9(x), True)
            x = F.elu(self.layer10(x), True)

        x = self.layer11(x)

        x = F.tanh(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        self.enc = Encoder(nc)
        self.dec = Decoder(nc, True)

    def forward(self, x):
        x = self.dec(self.enc(x))
        return x

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)

class L1_loss(_Loss):
    pass

# ResNet.
'''
class ResNet(nn.Module):
    # num_blocks = layers.
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer1 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer1 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d()
'''
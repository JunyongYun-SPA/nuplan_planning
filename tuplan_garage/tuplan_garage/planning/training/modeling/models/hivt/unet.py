import os
import numpy as np

import torch
import torch.nn as nn
from tuplan_garage.planning.training.modeling.models.hivt.unet_parts import *
from tuplan_garage.planning.training.modeling.models.hivt.embedding  import SingleInputEmbedding


class UNet(nn.Module):
    def __init__(self, n_channels=28, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 16))

    def forward(self, x):
        x1 = self.inc(x) #torch.Size([32, 64, 200, 200])
        x2 = self.down1(x1) #torch.Size([32, 128, 100, 100])
        x3 = self.down2(x2) #torch.Size([32, 256, 50, 50])
        x4 = self.down3(x3) #torch.Size([32, 512, 25, 25])
        x5 = self.down4(x4) #torch.Size([32, 1024, 12, 12])
        x = self.up1(x5, x4) #torch.Size([32, 512, 25, 25])
        x = self.up2(x, x3) #torch.Size([32, 256, 50, 50])
        x = self.up3(x, x2) #torch.Size([32, 128, 100, 100])
        x = self.up4(x, x1) #torch.Size([32, 64, 200, 200])
        logits = self.outc(x) #torch.Size([32, 1, 200, 200])
        return logits

class UNet_v2(nn.Module):
    def __init__(self, n_channels=28, n_classes=1, bilinear=False):
        super(UNet_v2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc_map = (DoubleConv(1, 64))
        self.down1_map = (Down(64, 128))
        self.down2_map = (Down(128, 256))
        self.down3_map = (Down(256, 512))
        self.down4_map = (Down(512, 1024 // factor))
        
        self.inc_hist = (DoubleConv(11, 64))
        self.down1_hist = (Down(64, 128))
        self.down2_hist = (Down(128, 256))
        self.down3_hist = (Down(256, 512))
        self.down4_hist = (Down(512, 1024 // factor))
        
        self.inc_future = (DoubleConv(16, 64))
        self.down1_future = (Down(64, 128))
        self.down2_future = (Down(128, 256))
        self.down3_future = (Down(256, 512))
        self.down4_future = (Down(512, 1024 // factor))
        
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 16))
        
        self.time_encoder = SingleInputEmbedding(in_channel=1, out_channel=64)

    def forward(self, x):
        
        x_map, x_hist, x_future = x[:, 0:1, :, :], x[:, 1:12, :, :], x[:, 12:, :, :]
        
        x1_map = self.inc_map(x_map) #torch.Size([32, 64, 200, 200])
        x2_map = self.down1_map(x1_map) #torch.Size([32, 128, 100, 100])
        x3_map = self.down2_map(x2_map) #torch.Size([32, 256, 50, 50])
        x4_map = self.down3_map(x3_map) #torch.Size([32, 512, 25, 25])
        x5_map = self.down4_map(x4_map) #torch.Size([32, 1024, 12, 12])
        
        x1_hist = self.inc_hist(x_hist) #torch.Size([32, 64, 200, 200])
        x2_hist = self.down1_hist(x1_hist) #torch.Size([32, 128, 100, 100])
        x3_hist = self.down2_hist(x2_hist) #torch.Size([32, 256, 50, 50])
        x4_hist = self.down3_hist(x3_hist) #torch.Size([32, 512, 25, 25])
        x5_hist = self.down4_hist(x4_hist) #torch.Size([32, 1024, 12, 12])
        
        x1_future = self.inc_future(x_future) #torch.Size([32, 64, 200, 200])
        x2_future = self.down1_future(x1_future) #torch.Size([32, 128, 100, 100])
        x3_future = self.down2_future(x2_future) #torch.Size([32, 256, 50, 50])
        x4_future = self.down3_future(x3_future) #torch.Size([32, 512, 25, 25])
        x5_future = self.down4_future(x4_future) #torch.Size([32, 1024, 12, 12])
        
        x1_env = x1_future + x1_map + x1_hist
        x2_env = x2_future + x2_map + x2_hist
        x3_env = x3_future + x3_map + x3_hist
        x4_env = x4_future + x4_map + x4_hist
        x5_env = x5_future + x5_map + x5_hist
        
        x = self.up1(x5_env, x4_env) #torch.Size([32, 512, 25, 25])
        x = self.up2(x, x3_env) #torch.Size([32, 256, 50, 50])
        x = self.up3(x, x2_env) #torch.Size([32, 128, 100, 100])
        x = self.up4(x, x1_env) #torch.Size([32, 64, 200, 200])
        
        logits = self.outc(x) #torch.Size([32, 16, 200, 200])
        return logits
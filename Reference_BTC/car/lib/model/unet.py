import torch
import torch.nn as nn


def double_conv(in_ch, out_ch):
    conv_op = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
    return conv_op
    
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()
        self.conv1 = double_conv(in_ch, 8)
        self.conv2 = double_conv(8, 16)
        self.conv3 = double_conv(16, 32)
        self.conv4 = double_conv(32, 64)
        
        self.conv5 = double_conv(96, 32)
        self.conv6 = double_conv(48, 16)
        self.conv7 = double_conv(24, 8)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        
        self.upsample1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)
        
        self.conv0 = nn.Conv2d(in_channels=8, out_channels=out_ch, kernel_size=1)
        
    
    def forward(self, x):
        #Encoder
        down1 = self.conv1(x)
        pool1 = self.pooling(down1)
        down2 = self.conv2(pool1)
        pool2 = self.pooling(down2)
        down3 = self.conv3(pool2)
        pool3 = self.pooling(down3)
        down4 = self.conv4(pool3)
        
        #Decoder
        upsample1 = self.upsample1(down4)
        cat1 = torch.cat([down3, upsample1], dim=1)
        up1 = self.conv5(cat1)
        upsample2 = self.upsample2(up1)
        cat2 = torch.cat([down2, upsample2], dim=1)
        up2 = self.conv6(cat2)
        upsample3 = self.upsample3(up2)
        cat3 = torch.cat([down1, upsample3], dim=1)
        up3 = self.conv7(cat3)
        
        outputs = self.conv0(up3)
        
        return outputs


    def warmup(self, imgsz=(1, 3, 80, 160), device):
        # Warmup model by running inference once
        im = torch.empty(*imgsz, dtype=torch.float, device=device)  # input
        for _ in range(3):  #
            self.forward(im)  # warmup


class torc
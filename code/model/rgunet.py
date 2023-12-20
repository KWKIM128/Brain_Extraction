import torch
import torch.nn as nn
import math
from torchsummary import summary
 
class Ghost_Module(nn.Module):
    def __init__(self, in_c, out_c, ratio=2, stride=1):
        super(Ghost_Module, self).__init__()
        self.oup = out_c
        init_channels = math.ceil(out_c / ratio)
        new_channels = init_channels * (ratio - 1)
 
        self.conv = nn.Conv2d(in_c, init_channels, kernel_size=3, 
                              stride=1, padding=3 // 2, bias=False)
        self.depthconv =  nn.Conv2d(init_channels, new_channels,
                                    kernel_size=3, stride = 1, 
                                    padding=3 // 2, 
                                    groups=init_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(init_channels)
        self.bn2 = nn.BatchNorm2d(new_channels)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn1(x1)
        x1 = self.leakyrelu(x1)
        x2 = self.depthconv(x1)
        x2 = self.bn2(x2)
        x2 = self.leakyrelu(x2)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
    
class Residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(Residual_block, self).__init__()
 
        self.ghost1 = Ghost_Module(in_c, out_c)
        self.match_channels = nn.Conv2d(in_c, out_c, kernel_size=1)
 
    def forward(self, x):
        x1 = self.ghost1(x)
        inputs = self.match_channels(x)
        x2 = x1 + inputs
        return x2
    
class Encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
 
        self.res = Residual_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
 
    def forward(self, inputs):
        x = self.res(inputs)
        p = self.pool(x)
 
        return x, p
class Decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
 
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.res = Residual_block(out_c + out_c, out_c)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(inplace=True)
 
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        #x = self.bn(x)
        #x = self.relu(x)
        return x
 
class RGUNet(nn.Module):
    def __init__(self):
        super().__init__()
 
        """ Encoder """
        self.e1 = Encoder_block(3, 32)
        self.e2 = Encoder_block(32, 64)
        self.e3 = Encoder_block(64, 128)
 
        """ Bottleneck """
        self.b = Residual_block(128, 256)
 
        """ Decoder """
        self.d1 = Decoder_block(256, 128)
        self.d2 = Decoder_block(128, 64)
        self.d3 = Decoder_block(64, 32)
 
        """ Classifier """
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()  # Add a sigmoid activation for binary segmentation
 
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
 
        """ Bottleneck """
        b = self.b(p3)
 
        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
 
        outputs = self.outputs(d3)
 
        return outputs
 
    
if __name__ == '__main__':
 
    model = RGUNet()
    summary(model, (3, 240, 240))

import torch.nn as nn
import torch.nn.functional as F

def space_to_channel(x, p=2):
    """ [B,C,H,W] -> [B,p²C,H/p,W/p] """
    B, C, H, W = x.shape
    x = x.view(B, C, H//p, p, W//p, p)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    return x.view(B, p*p*C, H//p, W//p)

def channel_to_space(x, p=2):
    """ [B,p²C,H/p,W/p] -> [B,C,H,W] appromiximate the PixelShuffle"""
    B, Cp2, H, W = x.shape
    C = Cp2 // (p*p)
    x = x.view(B, p, p, C, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    return x.view(B, C, H*p, W*p)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=2):
        super().__init__()
        self.p = p
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_ch*(p**2), out_ch, 1)

    def forward(self, x):
        main = F.avg_pool2d(x, self.p)
        main = self.conv(main)
        
        residual = space_to_channel(x, self.p)
        residual = self.res_conv(residual)
        
        return main + residual 

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=2):
        super().__init__()
        self.p = p
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch*(p**2), 1),
            nn.PixelShuffle(p)
        )

    def forward(self, x):

        main = F.interpolate(x, scale_factor=self.p, mode='nearest')
        main = self.conv(main)
        
        residual = self.res_conv(x)
        
        return main + residual  
    
class DCAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        #  encoder (128x128x3 -> 16x16x128)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x64
            DownBlock(64, 128, p=2),    # 32x32x128
            DownBlock(128, 256, p=2),   # 16x16x256
            nn.Conv2d(256, 128, 1),     # 16x16x128
        )
        
        # 解码器 (16x16x128 -> 128x128x3)
        self.decoder = nn.Sequential(
            UpBlock(128, 256, p=2),     # 32x32x256
            UpBlock(256, 128, p=2),     # 64x64x128
            UpBlock(128, 64, p=2),      # 128x128x64
            nn.Conv2d(64, 3, 3, padding=1), # 128x128x3
            nn.Sigmoid() 
        )
    def encode(self, x):
        return self.encoder(x)  
    def decode(self, z):
        return self.decoder(z)  
    def forward(self, x):
        return self.decode(self.encode(x))
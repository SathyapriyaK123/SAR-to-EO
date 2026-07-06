import torch
import torch.nn as nn

def init_weights(m):
    """
    Initializes weights according to the Pix2Pix paper:
    Normal distribution with mean 0.0 and std 0.02.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class UNetBlock(nn.Module):
    """
    Helper block for U-Net architecture.
    """
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        self.down = down
        
        if down:
            self.model = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    """
    U-Net Generator (256x256 input size).
    Standard architecture from the Pix2Pix paper.
    """
    def __init__(self, in_channels=1, out_channels=3, features=64):
        super(Generator, self).__init__()
        
        # Encoder (Downsampling)
        self.e1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        self.e2 = UNetBlock(features, features * 2, down=True, use_bn=True)             # 128 -> 64
        self.e3 = UNetBlock(features * 2, features * 4, down=True, use_bn=True)         # 64 -> 32
        self.e4 = UNetBlock(features * 4, features * 8, down=True, use_bn=True)         # 32 -> 16
        self.e5 = UNetBlock(features * 8, features * 8, down=True, use_bn=True)         # 16 -> 8
        self.e6 = UNetBlock(features * 8, features * 8, down=True, use_bn=True)         # 8 -> 4
        self.e7 = UNetBlock(features * 8, features * 8, down=True, use_bn=True)         # 4 -> 2
        
        # Bottleneck (e8)
        self.e8 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),  # 2 -> 1
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Upsampling) with skip connections
        self.d1 = UNetBlock(features * 8, features * 8, down=False, use_bn=True, use_dropout=True)  # 1 -> 2
        self.d2 = UNetBlock(features * 16, features * 8, down=False, use_bn=True, use_dropout=True) # 2 -> 4
        self.d3 = UNetBlock(features * 16, features * 8, down=False, use_bn=True, use_dropout=True) # 4 -> 8
        self.d4 = UNetBlock(features * 16, features * 8, down=False, use_bn=True, use_dropout=False)# 8 -> 16
        self.d5 = UNetBlock(features * 16, features * 4, down=False, use_bn=True, use_dropout=False)# 16 -> 32
        self.d6 = UNetBlock(features * 8, features * 2, down=False, use_bn=True, use_dropout=False) # 32 -> 64
        self.d7 = UNetBlock(features * 4, features, down=False, use_bn=True, use_dropout=False)     # 64 -> 128
        
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),      # 128 -> 256
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forwards
        e1_out = self.e1(x)
        e2_out = self.e2(e1_out)
        e3_out = self.e3(e2_out)
        e4_out = self.e4(e3_out)
        e5_out = self.e5(e4_out)
        e6_out = self.e6(e5_out)
        e7_out = self.e7(e6_out)
        e8_out = self.e8(e7_out)
        
        # Decoder forwards with skip connections (concatenating along channel dimension)
        d1_out = self.d1(e8_out)
        d1_cat = torch.cat([d1_out, e7_out], dim=1)
        
        d2_out = self.d2(d1_cat)
        d2_cat = torch.cat([d2_out, e6_out], dim=1)
        
        d3_out = self.d3(d2_cat)
        d3_cat = torch.cat([d3_out, e5_out], dim=1)
        
        d4_out = self.d4(d3_cat)
        d4_cat = torch.cat([d4_out, e4_out], dim=1)
        
        d5_out = self.d5(d4_cat)
        d5_cat = torch.cat([d5_out, e3_out], dim=1)
        
        d6_out = self.d6(d5_cat)
        d6_cat = torch.cat([d6_out, e2_out], dim=1)
        
        d7_out = self.d7(d6_cat)
        d7_cat = torch.cat([d7_out, e1_out], dim=1)
        
        d8_out = self.d8(d7_cat)
        return d8_out


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field).
    Evaluates whether overlapping 70x70 patches of the image are real or fake.
    """
    def __init__(self, in_channels=4, features=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1: No BatchNorm
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4 (stride=1, padding=1)
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5 (stride=1, padding=1, output single-channel logit patch)
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x, y):
        # Concatenate condition x (SAR) and image y (optical) along channel dimension
        concat_input = torch.cat([x, y], dim=1)
        return self.model(concat_input)


if __name__ == "__main__":
    # Test architectures
    gen = Generator(in_channels=1, out_channels=3)
    disc = Discriminator(in_channels=4)
    
    # Initialize
    gen.apply(init_weights)
    disc.apply(init_weights)
    
    # Create random batch of size 2 (1 channel SAR, 256x256)
    x = torch.randn(2, 1, 256, 256)
    y_fake = gen(x)
    print("Generator output shape:", y_fake.shape) # Should be [2, 3, 256, 256]
    
    y_real = torch.randn(2, 3, 256, 256)
    pred_fake = disc(x, y_fake)
    pred_real = disc(x, y_real)
    print("Discriminator output shape:", pred_real.shape) # Should be [2, 1, 30, 30]

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# LayerNorm
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# --------------------------------------------------------------------------
# Parallel Channel Attention
class Channel(nn.Module):
    def __init__(self, channels):
        super(Channel, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(3)
        self.mlp = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, groups=channels)
                for _ in range(16)
            ]
        )

    def forward(self, x):
        avg_out = self.maxpool(x)
        output = [nn.Sigmoid()(nn.LeakyReLU()(mlp(avg_out))) for mlp in self.mlp]
        add = nn.LeakyReLU()(sum(output))
        out = nn.LeakyReLU()(add * x)

        return out


# --------------------------------------------------------------------------
# Multi Branch Gate Spatial Convolution
# class Spatial(nn.Module):
# 	def __init__(self, channels):
# 		super(Spatial, self).__init__()

# 		self.project_in = nn.Conv2d(channels, channels*3, kernel_size=1)
# 		self.dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, groups=channels*3)
# 		self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

# 	def forward(self, x):

# 		x = self.project_in(x)
# 		x1, x2, x3 = self.dwconv(x).chunk(3, dim=1)
# 		a = nn.LeakyReLU()(x1 * x2)
# 		b = nn.LeakyReLU()(a * x3)
# 		out = self.project_out(b)
# 		out = nn.LeakyReLU()(out)
# 		return out


class ConvBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        xx = self.block(x)
        out = x + xx

        return out


# --------------------------------------------------------------------------
# MLP
class MLP(nn.Module):
    def __init__(self, channels):
        super(MLP, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.body(x)


# --------------------------------------------------------------------------
# Efficient Transformer Block
class Transformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.spatial_blocks = ConvBlock(channels)
        self.channel_blocks = Channel(channels)

        self.mlp = MLP(channels)

    def forward(self, img):
        x = self.norm1(img)

        x_1 = self.spatial_blocks(x)
        x_2 = nn.LeakyReLU()(x_1)
        x_3 = self.channel_blocks(x_2)
        x_4 = nn.LeakyReLU()(x_3)
        y = x_4 + img

        y_1 = self.norm2(y)

        y_2 = self.mlp(y_1)

        out = y_2 + y

        return out


# --------------------------------------------------------------------------
# Combination CNN and Transforemr
# class Restorationblock(nn.Module):
# 	def __init__(self, channels):
# 		super().__init__()

# 		self.ctblock = nn.Sequential(
# 				MLP(channels),
# 				Transformer(channels),
# 				MLP(channels),
# 				Transformer(channels),
# 				MLP(channels),
# 				Transformer(channels)
# 				)

# 	def forward(self, x):

# 		out = self.ctblock(x) + x

# 		return out


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# @ARCH_REGISTRY.register()
class test(nn.Module):
    def __init__(self, channel=64, layer_block=[2, 3, 4, 5]):
        super().__init__()
        self.input_projection = nn.Conv2d(3, channel, kernel_size=1, stride=1)

        ## level 1
        self.encoder1 = nn.Sequential(
            *[Transformer(channel) for i in range(layer_block[0])]
        )  # Restoration_Block(channel)
        self.down1_2 = Downsample(channel)
        ## level 2
        self.encoder2 = nn.Sequential(
            *[Transformer(channel * 2) for i in range(layer_block[1])]
        )  # Restoration_Block(channel*2)
        self.down2_3 = Downsample(channel * 2)
        ## level 3
        self.encoder3 = nn.Sequential(
            *[Transformer(channel * 4) for i in range(layer_block[2])]
        )  # Restoration_Block(channel*4)
        self.down3_Bottom = Downsample(channel * 4)
        ## Bottom
        self.Bottom = nn.Sequential(
            *[Transformer(channel * 8) for i in range(layer_block[3])]
        )  # Bottom_Block(channel*8)
        self.upBottom_3 = Upsample(channel * 8)
        ## level 3
        self.decoder3 = nn.Sequential(
            *[Transformer(channel * 8) for i in range(layer_block[2])]
        )  # Restoration_Block(channel*8)
        self.reduce3 = nn.Conv2d(channel * 8, channel * 4, kernel_size=1, stride=1)
        self.up3_2 = Upsample(channel * 4)
        ## level 2
        self.decoder2 = nn.Sequential(
            *[Transformer(channel * 4) for i in range(layer_block[1])]
        )  # Restoration_Block(channel*4)
        self.reduce2 = nn.Conv2d(channel * 4, channel * 2, kernel_size=1, stride=1)
        self.up2_1 = Upsample(channel * 2)
        ## level 1
        self.decoder1 = nn.Sequential(
            *[Transformer(channel * 2) for i in range(layer_block[0])]
        )  # Restoration_Block(channel*2)
        self.reduce1 = nn.Conv2d(channel * 2, channel * 1, kernel_size=1, stride=1)

        self.output_projection = nn.Conv2d(channel, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.input_projection(x)
        ## level 1
        x2 = self.encoder1(x1)
        x3 = self.down1_2(x2)
        ## level 2
        x4 = self.encoder2(x3)
        x5 = self.down2_3(x4)
        ## level 3
        x6 = self.encoder3(x5)
        x7 = self.down3_Bottom(x6)
        ## Bottom
        x8 = self.Bottom(x7)
        x9 = self.upBottom_3(x8)
        ## level 3
        x9 = torch.cat([x9, x6], 1)
        x10 = self.decoder3(x9)
        x10 = self.reduce3(x10)
        x11 = self.up3_2(x10)
        ## level 2
        x11 = torch.cat([x11, x3], 1)
        x12 = self.decoder2(x11)
        x12 = self.reduce2(x12)
        x13 = self.up2_1(x12)
        ## level 1
        x13 = torch.cat([x13, x2], 1)
        x14 = self.decoder1(x13)
        x14 = self.reduce1(x14)

        residual = x1 + x14

        out = self.output_projection(residual)

        return out

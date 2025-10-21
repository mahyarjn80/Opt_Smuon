from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .base import Architecture


# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding='same', bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNetModule(nn.Module):
    def __init__(self, block1_width=64, block2_width=256, block3_width=256, 
                 whiten_kernel_size=2, output_dim=10):
        super().__init__()
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, block1_width),
            ConvGroup(block1_width, block2_width),
            ConvGroup(block2_width, block3_width),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(block3_width, output_dim, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


@dataclass
class CifarNet(Architecture):
    """CifarNet architecture from airbench.
    
    A high-performance CNN designed for CIFAR-10 with:
    - Whitening preprocessing layer
    - Three ConvGroup blocks with increasing channels
    - Specialized BatchNorm with low momentum
    - Mixed precision (half/float)
    """
    block1_width: int = 64
    block2_width: int = 256
    block3_width: int = 256
    whiten_kernel_size: int = 2

    def create(self, input_shape, output_dim):
        """Create a CifarNet model.
        
        Args:
            input_shape: Expected to be (3, 32, 32) for CIFAR
            output_dim: Number of output classes (typically 10 for CIFAR-10)
        
        Returns:
            CifarNetModule instance
        """
        return CifarNetModule(
            block1_width=self.block1_width,
            block2_width=self.block2_width,
            block3_width=self.block3_width,
            whiten_kernel_size=self.whiten_kernel_size,
            output_dim=output_dim
        )


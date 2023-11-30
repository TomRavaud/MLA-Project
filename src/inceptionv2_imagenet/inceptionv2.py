"""
An implementation of the InceptionV2 model (GoogLeNet + Batch Normalization).
"""

# Import libraries
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchinfo


class InceptionBN(nn.Module):
    """
    Inception block for InceptionV2.
    (https://d2l.ai/chapter_convolutional-modern/googlenet.html)
    
    The Inception block introduced in the GoogleNet paper has been modified
    to include batch normalization layers (as well as the other network
    layers), leading to Inception-V2 (or Inception-BN) model (Batch
    Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift, Ioffe et al., 2015, https://arxiv.org/abs/1502.03167).
    """
    def __init__(self,
                 c1: int,
                 c2: int,
                 c3: int,
                 c4: int,
                 factor: float=1,
                 **kwargs: dict):
        """Constructor of the class. The number of output channels for each
        branch is commonly tuned to make the model more efficient.

        Args:
            c1 (int): Number of output channels for branch 1.
            c2 (int): Number of output channels for branch 2.
            c3 (int): Number of output channels for branch 3.
            c4 (int): Number of output channels for branch 4.
            factor (float, optional): Factor to modulate the number of output
            channels of the inception blocks. Defaults to 1.
        """
        super(InceptionBN, self).__init__(**kwargs)
        
        # Branch 1
        # The "lazy" modules are used to avoid the initialization of the
        # parameters until the first forward pass.
        self.b1 = nn.Sequential(nn.LazyConv2d(int(factor*c1),
                                              kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        
        # Branch 2
        self.b2 = nn.Sequential(nn.LazyConv2d(int(factor*c2[0]),
                                              kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(),
                                nn.LazyConv2d(int(factor*c2[1]),
                                              kernel_size=3,
                                              padding=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        
        # Branch 3
        self.b3 = nn.Sequential(nn.LazyConv2d(int(factor*c3[0]),
                                              kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(),
                                nn.LazyConv2d(int(factor*c3[1]),
                                              kernel_size=5,
                                              padding=2),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        
        # Branch 4
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=1,
                                             padding=1),
                                nn.LazyConv2d(int(factor*c4),
                                              kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Compute the output of each branch.
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        
        # Concatenate the outputs along the channel dimension.
        return torch.cat((o1, o2, o3, o4), dim=1)


class GoogleNetBN(nn.Module):
    """
    GoogleNet model.
    (https://d2l.ai/chapter_convolutional-modern/googlenet.html)
    The architecture has been simplified with respect to the original paper.
    In particular, the numerous tricks for stabilizing training through
    intermediate loss functions, applied to multiple layers of the network,
    are no longer used due to the availability of improved training algorithms.
    
    The GoogleNet blocks have been modified to include batch normalization
    layers, leading to Inception-V2 (or Inception-BN) model (Batch
    Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift, Ioffe et al., 2015, https://arxiv.org/abs/1502.03167).
    """
    def __init__(self, nb_classes: int=10, inception_factor: float=1) -> None:
        """Constructor of the class.

        Args:
            num_classes (int, optional): Number of classes of the dataset.
            Defaults to 10.
            inception_factor (float, optional): Factor to modulate the number
            of output channels of the inception blocks. Defaults to 1.
        """
        super(GoogleNetBN, self).__init__()
        
        # Assemble the blocks in order to build the network.
        self.net = nn.Sequential(self.b1(),
                                 self.b2(),
                                 self.b3(inception_factor=inception_factor),
                                 self.b4(inception_factor=inception_factor),
                                 self.b5(inception_factor=inception_factor),
                                 nn.LazyLinear(nb_classes),
                                 )
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)
    
    
    def b1(self) -> nn.Sequential:
        """First module of the network.

        Returns:
            nn.Sequential: First module of the network.
        """
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    
    def b2(self) -> nn.Sequential:
        """Second module of the network.

        Returns:
            nn.Sequential: Second module of the network.
        """
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b3(self, inception_factor: float) -> nn.Sequential:
        """Third module of the network. It connects two complete Inception
        blocks in series, followed by a max-pooling layer that reduces the
        height and width of the feature map.
        
        Args:
            inception_factor (float): Factor to modulate the number of output
            channels of the inception blocks.

        Returns:
            nn.Sequential: Third module of the network.
        """
        return nn.Sequential(
            InceptionBN(64, (96, 128), (16, 32), 32, inception_factor),
            InceptionBN(128, (128, 192), (32, 96), 64, inception_factor),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b4(self, inception_factor: float) -> nn.Sequential:
        """Fourth module of the network. It consists of five Inception blocks
        in series, followed by a max-pooling layer.
        
        Args:
            inception_factor (float): Factor to modulate the number of output
            channels of the inception blocks.

        Returns:
            nn.Sequential: Fourth module of the network.
        """
        return nn.Sequential(
            InceptionBN(192, (96, 208), (16, 48), 64, inception_factor),
            InceptionBN(160, (112, 224), (24, 64), 64, inception_factor),
            InceptionBN(128, (128, 256), (24, 64), 64, inception_factor),
            InceptionBN(112, (144, 288), (32, 64), 64, inception_factor),
            InceptionBN(256, (160, 320), (32, 128), 128, inception_factor),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
    def b5(self, inception_factor: float) -> nn.Sequential:
        """Fifth module of the network. It consists of two Inception blocks
        in series, followed by a global average pooling layer to make the
        height and width of the feature map equal to 1. Finally, the output
        is turned into a two-dimensional array to feed the fully connected
        layer.
        
        Args:
            inception_factor (float): Factor to modulate the number of output
            channels of the inception blocks.

        Returns:
            nn.Sequential: Fifth module of the network.
        """
        return nn.Sequential(
            InceptionBN(256, (160, 320), (32, 128), 128, inception_factor),
            InceptionBN(384, (192, 384), (48, 128), 128, inception_factor),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            )


if __name__ == "__main__":
    
    # Create a random tensor of shape (1, 3, 224, 224).
    x = torch.randn(1, 3, 224, 224)
    
    # Create the model.
    model = GoogleNetBN(inception_factor=1)
    
    # Print the model summary
    # (layers, output shape, number of parameters, memory usage, ...)
    torchinfo.summary(model, input_size=(1, 3, 224, 224), device='cpu')
    
    # Perform the forward pass.
    out = model(x)
    
    # Print the output shape.
    print(out.shape)

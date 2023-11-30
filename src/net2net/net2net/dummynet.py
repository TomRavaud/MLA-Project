"""
Dummy networks used to test the Net2Net class
"""

# Import libraries
import torch
import torch.nn as nn


class DummyNet(nn.Module):
    """
    A dummy network only used to test the Net2Net class
    """
    def __init__(self):
        """Constructor of the class

        Args:
            input_size (int): Size of the input
        """        
        super(DummyNet, self).__init__()
        
        # Define the components of the network
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network

        Args:
            x (torch.Tensor): Input of the network

        Returns:
            torch.Tensor: Output of the network
        """        
        # Apply the network to the input
        # Convolutions
        out = self.layer1(x)
        out = self.layer2(out)
        
        return torch.mean(out)


class DummyNetBN(nn.Module):
    """
    A dummy network only used to test the Net2Net class
    """
    def __init__(self):
        """Constructor of the class

        Args:
            input_size (int): Size of the input
        """        
        super(DummyNetBN, self).__init__()
        
        # Define the components of the network
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network

        Args:
            x (torch.Tensor): Input of the network

        Returns:
            torch.Tensor: Output of the network
        """        
        # Apply the network to the input
        # Convolutions
        out = self.layer1(x)
        out = self.layer2(out)
        
        return torch.mean(out)


class DummyNetConcat(nn.Module):
    """
    A dummy network only used to test the Net2Net class
    """
    def __init__(self):
        """Constructor of the class

        Args:
            input_size (int): Size of the input
        """        
        super(DummyNetConcat, self).__init__()
        
        # Define the components of the network
        # First two layers in parallel (same input and their outputs are
        # concatenated)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU()
            )
        
        # Third and fourth layers in parallel (same input and their outputs are
        # concatenated)
        self.layer3 = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
            )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(5, 2, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU()
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network

        Args:
            x (torch.Tensor): Input of the network

        Returns:
            torch.Tensor: Output of the network
        """        
        # Apply the network to the input
        
        # First two layers in parallel
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        
        # Concatenate the outputs
        concat = torch.cat((out1, out2), dim=1)
        
        # Third and fourth layers in parallel
        out3 = self.layer3(concat)
        out4 = self.layer4(concat)
        
        # Concatenate the outputs and compute the mean
        out = torch.cat((out3, out4), dim=1)
        
        return torch.mean(out)


class DummyNetFC(nn.Module):
    """
    A dummy network only used to test the Net2Net class
    """
    def __init__(self):
        """Constructor of the class

        Args:
            input_size (int): Size of the input
        """        
        super(DummyNetFC, self).__init__()
        
        # Define the components of the network
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            )
        self.layer2 = nn.Sequential(
            nn.Linear(2, 1),
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network

        Args:
            x (torch.Tensor): Input of the network

        Returns:
            torch.Tensor: Output of the network
        """        
        # Apply the network to the input
        # Convolutions
        out = self.layer1(x)
        out = self.layer2(out)
        
        return torch.mean(out)


if __name__ == "__main__":
    
    # Create a random input (1 channel, 32x32 image)
    dummy_input = torch.randn((1, 1, 32, 32))
    
    # Create the network
    model = DummyNetFC()
    
    # Apply the network
    output = model(dummy_input)
    
    # Print the output
    print(output.shape)

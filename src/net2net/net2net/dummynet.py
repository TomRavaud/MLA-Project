"""
Dummy network used to test the Net2Net class
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
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=2, stride=1, padding=0),
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
        
        return out


if __name__ == "__main__":
    
    # Create a random input (1 channel, 32x32 image)
    dummy_input = torch.randn((1, 1, 32, 32))
    
    # Create the network
    model = DummyNet()
    
    # Apply the network
    output = model(dummy_input)
    
    # Print the output
    print(output.shape)

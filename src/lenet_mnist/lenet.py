import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    LeNet model
    (Gradient-based learning applied to document recognition, LeCun et al.,
    1998, http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
    where sigmoid activations are replaced by ReLU activations, batch
    normalization is applied and max pooling is used instead of average
    pooling.
    """
    def __init__(self, nb_classes: int):
        """Constructor of the class

        Args:
            input_size (int): Size of the input
        """        
        super(LeNet, self).__init__()
        
        # Define the components of the network
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nb_classes)
        self.relu = nn.ReLU()
    
    
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
        
        # Flatten the output
        out = out.reshape(out.size(0), -1)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


class LeNetWithoutBN(nn.Module):
    """
    LeNet model
    (Gradient-based learning applied to document recognition, LeCun et al.,
    1998, http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
    where sigmoid activations are replaced by ReLU activations, and max pooling
    is used instead of average pooling.
    """
    def __init__(self, nb_classes: int):
        """Constructor of the class

        Args:
            input_size (int): Size of the input
        """        
        super(LeNetWithoutBN, self).__init__()
        
        # Define the components of the network
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nb_classes)
        self.relu = nn.ReLU()
    
    
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
        
        # Flatten the output
        out = out.reshape(out.size(0), -1)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


# This snippet is not executed when this file is imported
if __name__ == "__main__":
    
    # Create a random input (1 channel, 32x32 image)
    dummy_input = torch.randn((1, 1, 32, 32))
    
    # Create the network
    model = LeNet(nb_classes=10)
    
    # Apply the network
    output = model(dummy_input)
    
    # Print the output shape
    print(output.shape)

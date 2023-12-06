"""
An implementation of the Net2DeeperNet algorithm to deepen a network of convolutional layer
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


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
            # nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(16),
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



def add_module(model: nn.Module,
                   target_module_name: str):
    # Get the original convolutional layer from the model
    #original_layer = getattr(model, target_module_name)

    # Split the original convolutional layer
    # Divide the module name into its components
    module_path = target_module_name.split('.')
    print(module_path)
    current_module = getattr(model, module_path[0])

    
    # If the module is at the top level
    if len(module_path) == 1:
        for name, module in model.named_modules():
            setattr()
        # Replace the original layer in the model
        new_module = nn.Conv2d(in_channels=current_module.in_channels,
                            out_channels=current_module.out_channels,
                            kernel_size=current_module.kernel_size,
                            stride=current_module.stride,
                            padding=current_module.padding)
        setattr(model, str(int(target_module_name) + 1), new_module)



        # Remove the original layer from the model
        #delattr(model, target_module_name)

    else:
        # Get the current module
        current_module_name = module_path[0]
        current_module = getattr(model, current_module_name)
        
        # Recursively replace the module
        add_module(current_module, '.'.join(module_path[1:]))


class Net2Net:
    
    def __init__(self, teacher_network: nn.Module):
        """Constructor of the class

        Args:
            model (nn.Module): A pre-trained model to be used as teacher
        """
        # Initialize the student network with the teacher network
        self.student_network = copy.deepcopy(teacher_network)

    
    def net2deeper(self,
                  target_layer: str):
        """Deepen a layer of a neural network

        Args:
            target_layer (str): Layer to be deepen
        """


        with torch.no_grad():
            # Create a copy of the student network
            new_student_network = copy.deepcopy(self.student_network)

            for name, module in self.student_network.named_modules():

                if name == target_layer :
                    # Assuming you want to double the number of output channels
                    """
                    new_module = nn.Conv2d(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding)"""
                    add_module(new_student_network,
                                   name)
                    print(new_student_network)

            # Get the parameters of the target layer

            self.student_network = new_student_network

            

if __name__ == '__main__':

    # TEST 
    # Create a model
    model2 = LeNetWithoutBN(nb_classes=10)
    
    # Instantiate a Net2Net object from a (pre-trained) model
    net2net = Net2Net(teacher_network=model2)
    
    # Set a layer to be deepen
    target_layer = "layer1.0"
    
    # Depen a layer of the network
    net2net.net2deeper(target_layer=target_layer)
    
    # Create a random input
    x = torch.randn(1, 1, 32, 32)
    
    # Compute the output of the teacher network
    y_teacher = model2(x)
    
    # Compute the output of the student network
    y_student = net2net.student_network(x)
    #print(model2)
    #print(net2net.student_network)
    
    # The outputs should be the same
    print("TEST2 (LeNetWithoutBN): The outputs of the teacher and student networks should be the same.")
    print("Teacher output: ", y_teacher)
    print("Student output: ", y_student, "\n")

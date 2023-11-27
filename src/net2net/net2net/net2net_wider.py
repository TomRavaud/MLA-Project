"""
An implementation of the Net2WiderNet algorithm to widen a convolutional layer
followed by an other convolutional layer in a neural network.
"""

# Import libraries
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy

# Import custom modules
from dummynet import DummyNet


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


def replace_module(model: nn.Module,
                   target_module_name: str,
                   new_module: nn.Module):
    """Recursively replace a module in a model
    
    Args:
        model (nn.Module): The model in which the module is to be replaced
        target_module_name (str): The name of the module to be replaced
        new_module (nn.Module): The new module
    """
    # Divide the module name into its components
    module_path = target_module_name.split('.')
    
    # If the module is at the top level
    if len(module_path) == 1:
        # Replace the module
        setattr(model, target_module_name, new_module)
    else:
        # Get the current module
        current_module_name = module_path[0]
        current_module = getattr(model, current_module_name)
        
        # Recursively replace the module
        replace_module(current_module, '.'.join(module_path[1:]), new_module)


class Net2Net:
    
    def __init__(self, teacher_network: nn.Module):
        """Constructor of the class

        Args:
            model (nn.Module): A pre-trained model to be used as teacher
        """
        # Initialize the student network with the teacher network
        self.student_network = copy.deepcopy(teacher_network)

    
    def net2wider(self,
                  target_layer: str,
                  next_layer: str,
                  width: int,
                  sigma: float = 0.01):
        """Widen a layer of a neural network

        Args:
            target_layer (str): Layer to be widened
            next_layer (str): Next layer in the network
            width (int): New width of the layer
            sigma (float, optional): Standard deviation of the noise added to
            the weights and biases of the supplementary filters. Defaults to
            0.01.
        """
        
        # Wrap the computation in a no_grad() block to prevent PyTorch from
        # building the computational graph
        with torch.no_grad():
            
            # Get the weights and biases of the target layer in the teacher
            # network (the student network is a copy of the teacher network)
            teacher_w1 =\
                self.student_network.state_dict()[target_layer + ".weight"]
            teacher_b1 =\
                self.student_network.state_dict()[target_layer + ".bias"]

            # Get the weights and biases of the next layer in the teacher
            # network (the student network is a copy of the teacher network)
            teacher_w2 =\
                self.student_network.state_dict()[next_layer + '.weight']
            teacher_b2 =\
                self.student_network.state_dict()[next_layer + '.bias']
 
 
            # Get the number of filters of the target layer
            nb_filters_teacher = teacher_w1.shape[0]

            # Apply the random mapping function
            # Unchanged indices
            unchanged_indices = np.arange(nb_filters_teacher)
            # Random indices for supplementary filters
            random_indices = np.random.randint(nb_filters_teacher,
                                               size=width-nb_filters_teacher)
            # Concatenate the unchanged and random indices
            indices = np.concatenate((unchanged_indices, random_indices))

            # Compute the replication factor of each filter (the number of
            # times a same filter is used)
            replication_factor = np.bincount(indices)

            # Initialize the weights and biases of the target layer and the
            # next layer in the student network
            student_w1 = torch.zeros((width,
                                      teacher_w1.shape[1],
                                      teacher_w1.shape[2],
                                      teacher_w1.shape[3]))
            student_b1 = torch.zeros(width)
            student_w2 = torch.zeros((teacher_w2.shape[0],
                                      width,
                                      teacher_w2.shape[2],
                                      teacher_w2.shape[3]))

            # Copy the weights and biases of the target layer and the next
            # layer of the teacher network to the student network
            student_w1[:nb_filters_teacher, :, :, :] = teacher_w1
            student_b1[:nb_filters_teacher] = teacher_b1
            student_w2[:, :nb_filters_teacher, :, :] =\
                teacher_w2 / replication_factor[unchanged_indices][None,
                                                                   :,
                                                                   None,
                                                                   None]

            # Add the weights and biases of the supplementary filters to the
            # student network, with a small amount of noise to break symmetry
            student_w1[nb_filters_teacher:, :, :, :] =\
                teacher_w1[random_indices, :, :, :] +\
                    torch.randn((width-nb_filters_teacher,
                                 teacher_w1.shape[1],
                                 teacher_w1.shape[2],
                                 teacher_w1.shape[3])) * sigma

            student_b1[nb_filters_teacher:] = teacher_b1[random_indices] +\
                    torch.randn(width-nb_filters_teacher) * sigma

            student_w2[:, nb_filters_teacher:, :, :] =\
                teacher_w2[:, random_indices, :, :] /\
                replication_factor[random_indices][None, :, None, None]
            

            # Create a copy of the student network
            new_student_network = copy.deepcopy(self.student_network)
            
            # Replace the target layer and the next layer in the student network
            # by the new layers with the supplementary filters
            for name, module in self.student_network.named_modules():

                if name == target_layer:

                    replace_module(new_student_network,
                                   name,
                                   nn.Conv2d(module.in_channels,
                                             width,
                                             kernel_size=module.kernel_size,
                                             stride=module.stride,
                                             padding=module.padding))

                elif name == next_layer:

                    replace_module(new_student_network,
                                   name,
                                   nn.Conv2d(width,
                                             module.out_channels,
                                             kernel_size=module.kernel_size,
                                             stride=module.stride,
                                             padding=module.padding))

 
            # Update the structure of the student network
            self.student_network = new_student_network
            
            
            # Put the weights and biases of the target layer and the next layer
            # of the student network in a dictionary
            # (the biases of the next layer are not modified)
            modified_state_dict = {
                target_layer + ".weight": student_w1,
                target_layer + ".bias": student_b1,
                next_layer + ".weight": student_w2,
                next_layer + ".bias": teacher_b2,
            }
            
            # Replace the weights and biases of the target layer and the next
            # layer in the student network by the new weights and biases
            # (load only the modified weights and biases)
            self.student_network.load_state_dict(modified_state_dict,
                                                 strict=False)


if __name__ == '__main__':

    # TEST 1
    # Create a model
    model1 = DummyNet()
    
    # Create an input
    x = torch.tensor([[1, 2, 0],
                       [0, 1, 0],
                       [1, 1, 1]], dtype=torch.float32)
    x = x[None, None, :, :]
    
    # Set the weights and biases of the network
    w1 = torch.tensor([[[[1, 0], [1, 1]]],
                        [[[0, 0], [2, 1]]]], dtype=torch.float32)
    b1 = torch.tensor([0, 0], dtype=torch.float32)
    w2 = torch.tensor([[[[0, 0],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]]], dtype=torch.float32)
    b2 = torch.tensor([0], dtype=torch.float32)
        
    # Put the weights and biases in a dictionary
    modified_state_dict = {
        "layer1.0" + ".weight": w1,
        "layer1.0" + ".bias": b1,
        "layer2.0" + ".weight": w2,
        "layer2.0" + ".bias": b2,
    }

    # Load only the modified weights and biases
    model1.load_state_dict(modified_state_dict, strict=False)
        
    # Instantiate a Net2Net object from a (pre-trained) model
    net2net = Net2Net(teacher_network=model1)
    
    # Set a layer to be widened
    target_layer = "layer1.0"
    # Set the next layer
    next_layer = "layer2.0"
    # Set the new width of the layer
    new_width = 3
    
    # Widen a layer of the network
    net2net.net2wider(target_layer=target_layer,
                      next_layer=next_layer,
                      width=new_width,
                      sigma=0.)
    
    # Compute the output of the teacher network
    y_teacher = model1(x)
    
    # Compute the output of the student network
    y_student = net2net.student_network(x)
    
    # The outputs should be the same
    print("TEST1 (DummyNet): The outputs of the teacher and student networks should be the same.")
    print("Teacher output: ", y_teacher)
    print("Student output: ", y_student, "\n")
    
    
    # TEST 2
    # Create a model
    model2 = LeNetWithoutBN(nb_classes=10)
    
    # Instantiate a Net2Net object from a (pre-trained) model
    net2net = Net2Net(teacher_network=model2)
    
    # Set a layer to be widened
    target_layer = "layer1.0"
    # Set the next layer
    next_layer = "layer2.0"
    # Set the new width of the layer
    new_width = 10
    
    # Widen a layer of the network
    net2net.net2wider(target_layer=target_layer,
                      next_layer=next_layer,
                      width=new_width,
                      sigma=0.)
    
    # Create a random input
    x = torch.randn(1, 1, 32, 32)
    
    # Compute the output of the teacher network
    y_teacher = model2(x)
    
    # Compute the output of the student network
    y_student = net2net.student_network(x)
    
    # The outputs should be the same
    print("TEST2 (LeNetWithoutBN): The outputs of the teacher and student networks should be the same.")
    print("Teacher output: ", y_teacher)
    print("Student output: ", y_student, "\n")

"""
An implementation of the Net2DeeperNet algorithm to deepen a network of convolutional layer
"""
# Import libraries
import torch
import torch.nn as nn
import numpy as np
import copy
import utils
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


# Import custom modules and packages
from models.lenet import LeNet
import params.lenet_mnist
import params.inceptionv2_cifar
import params.inceptionv2_imagenet


def add_modules(model: nn.Module,
                target_module_name: str,
                new_conv: nn.Module,
                new_bn: nn.Module):
    """Add new modules to a neural network. The new modules are added after a
    a convolutional layer (target_module_name), followed by a batch
    normalization layer and a ReLU activation. The new modules are composed of
    a convolutional layer (new_conv), followed by a batch normalization layer
    (new_bn) and a ReLU activation.

    Args:
        model (nn.Module): The neural network to which the new modules are
        added
        target_module_name (str): The name of the convolutional layer after
        which the new modules are added (in reality, the modules are added
        after the batch normalization layer and the ReLU activation following
        the convolutional layer)
        new_conv (nn.Module): The convolutional layer to add to the network
        new_bn (nn.Module): The batch normalization layer to add to the network
    """
    
    # Divide the module name into its components
    module_path = target_module_name.split('.')
    
    # Create a list of the modules and their names starting from the
    # previous convolutional layer
    if len(module_path) == 1:
        
        # Create a list to store the name and the module of the modules after
        # the previous convolutional layer
        list_after_module_name = []
        
        # Go through the modules of the network
        for name, module in model.named_modules():
            
            # Check if the current module is after the previous convolutional
            # layer
            if name == module_path[0] or len(list_after_module_name)!=0:
                
                list_after_module_name.append((name, module))
                
        # Insert the new modules (ie {convolutional layer + batch normalization
        # layer + ReLU activation}) to the network after the previous
        # convolutional layer (we assume that this layer is followed by a batch
        # normalization layer and a ReLU activation, it is why we add 3, 4 and
        # 5 to the module name)
        setattr(model, str(int(target_module_name) + 3), new_conv)
        setattr(model, str(int(target_module_name) + 4), new_bn)
        setattr(model, str(int(target_module_name) + 5), nn.ReLU())
        
        # If there were other modules after the three new modules we added,
        # add them back to the network with an updated name
        if len(list_after_module_name) > 3:
            
            # Go through the modules after the three new modules we added
            for i in range(1, len(list_after_module_name) - 2):
                
                # Add the module back to the network with an updated name
                setattr(model,
                        str(int(list_after_module_name[-i][0])+3),
                        list_after_module_name[-i][1])
        
    else:
        # Get the current module
        current_module_name = module_path[0]
        current_module = getattr(model, current_module_name)
        
        # Recursively call the function to add the new modules to the network
        add_modules(current_module,
                    '.'.join(module_path[1:]),
                    new_conv, new_bn)


# Create a network model_1_epoch, a copy of a network model and do 1 epoch of training on this model
def train_1_epoch(model, dataset_used):

    #make a copy of the model
    model_1_epoch = copy.deepcopy(model)

    if dataset_used == "CIFAR10":
        learning_params = params.inceptionv2_cifar.LEARNING
    
            # Compose several transforms together to be applied to data
        # (Note that transforms are not applied yet)
        transform = transforms.Compose([
            # Modify the size of the images
            transforms.Resize(params.inceptionv2_cifar.IMAGE_SHAPE),

            # Convert a PIL Image or numpy.ndarray to tensor
            transforms.ToTensor(),

            # Normalize a tensor image with pre-computed mean and standard
            # deviation (based on the data used to train the model(s))
            # (be careful, it only works on torch.*Tensor)
            transforms.Normalize(**params.inceptionv2_cifar.NORMALIZE_PARAMS),
        ])


        # Load the train dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root=params.inceptionv2_cifar.PATH,
            train=True,
            transform=transform,
            download=True,
        )

        # Load the test dataset
        test_dataset=torchvision.datasets.CIFAR10(
            root=params.inceptionv2_cifar.PATH,
            train=False,
            transform=transform,
            download=True,
        )


    elif dataset_used == "MNIST":
        learning_params = params.lenet_mnist.LEARNING
    
            # Compose several transforms together to be applied to data
        # (Note that transforms are not applied yet)
        transform = transforms.Compose([
            # Modify the size of the images
            transforms.Resize(params.lenet_mnist.IMAGE_SHAPE),

            # Convert a PIL Image or numpy.ndarray to tensor
            transforms.ToTensor(),

            # Normalize a tensor image with pre-computed mean and standard
            # deviation (based on the data used to train the model(s))
            # (be careful, it only works on torch.*Tensor)
            transforms.Normalize(**params.lenet_mnist.NORMALIZE_PARAMS),
        ])


        # Load the train dataset
        train_dataset = torchvision.datasets.MNIST(
            root=params.lenet_mnist.PATH,
            train=True,
            transform=transform,
            download=True,
        )

        # Load the test dataset
        test_dataset=torchvision.datasets.MNIST(
            root=params.lenet_mnist.PATH,
            train=False,
            transform=transform,
            download=True,
        )

    elif dataset_used == "IMAGENET":
        learning_params = params.inceptionv2_imagenet.LEARNING
    
            # Compose several transforms together to be applied to data
        # (Note that transforms are not applied yet)
        transform = transforms.Compose([
            # Modify the size of the images
            transforms.Resize(params.inceptionv2_imagenet.IMAGE_SHAPE),

            # Convert a PIL Image or numpy.ndarray to tensor
            transforms.ToTensor(),

            # Normalize a tensor image with pre-computed mean and standard
            # deviation (based on the data used to train the model(s))
            # (be careful, it only works on torch.*Tensor)
            transforms.Normalize(**params.inceptionv2_imagenet.NORMALIZE_PARAMS),
        ])


        # Load the train dataset
        train_dataset = torchvision.datasets.IMAGENET(
            root=params.inceptionv2_imagenet.PATH,
            train=True,
            transform=transform,
            download=True,
        )

        # Load the test dataset
        test_dataset=torchvision.datasets.IMAGENET(
            root=params.inceptionv2_imagenet.PATH,
            train=False,
            transform=transform,
            download=True,
        )

    else :
        print("dataset unknown")
        

    # As CIFAR-10 does not provide a validation dataset, we will split the
    # train dataset into a train and a validation dataset

    # Start by loading the train dataset, with the same transform as the
    # test dataset


    # Set the train dataset size as a percentage of the original train dataset
    train_size = (len(train_dataset) - len(test_dataset))/len(train_dataset)

    # Splits train data indices into train and validation data indices
    train_indices, _ = train_test_split(range(len(train_dataset)),
                                                train_size=train_size)

    # Extract the corresponding subsets of the train dataset
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)



    # Combine a dataset and a sampler, and provide an iterable over the dataset
    # (setting shuffle argument to True calls a RandomSampler, and avoids to
    # have to create a Sampler object)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = learning_params['batch_size'],
        shuffle = True,
        num_workers=10,  # Asynchronous data loading and augmentation
        pin_memory=True,  # Increase the transferring speed to the GPU
    )


    # Use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Move the model to the device
    model.to(device)

    # Define the loss function (combines nn.LogSoftmax() and nn.NLLLoss())
    criterion = torch.nn.CrossEntropyLoss()

    # Set the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_params["learning_rate"],
        weight_decay=learning_params["weight_decay"])

    
    # train one epoch of the model
    utils.train.train(model_1_epoch,
                            device,
                            train_loader,
                            optimizer,
                            criterion,
                            0)
    return model_1_epoch



class Net2Net:
    
    def __init__(self, teacher_network: nn.Module):
        """Constructor of the class

        Args:
            model (nn.Module): A pre-trained model to be used as teacher
        """
        # Initialize the student network with the teacher network
        self.student_network = copy.deepcopy(teacher_network)

        # Initialize batch norms of the deepen network with 
    def set_deepen_batchnorm(model,model_1_epoch):
        """
        Args:
            model (nn.Module): The neural network with the batch norm to update
            added_batchnorm: dictionnary of added batch norm
        """
    
        with torch.no_grad():
            # create a dictionnary of modifications
            modified_state_dict = {}

            # Search the batch norm modules in the network
            for name, layer in model_1_epoch.named_children():
                
                if isinstance(layer, nn.BatchNorm2d):
                    if name in params.lenet_mnist.deeper_batchnorm :
                        # Add running mean and running std in the dictionnary of modifications
                        modified_state_dict[name + '.bias'] = layer.running_mean
                        modified_state_dict[name + '.weight'] = layer.running_var

                if isinstance(layer, nn.Sequential):
                    for name_inside, layer_inside in layer.named_children():
                        if isinstance(layer_inside, nn.BatchNorm2d):
                            # Create the name of the current layer
                            name_inside = ".".join([name,name_inside])
                            if name_inside in params.lenet_mnist.deeper_batchnorm :
                                # Add running mean and running std in the dictionnary of modifications
                                #modifier gamma, beta : weights, bias
                                modified_state_dict[name_inside + '.bias'] = layer_inside.running_mean #bias
                                modified_state_dict[name_inside + '.weight'] = layer_inside.running_var #weight
            
            # load the modification in the model
            model.load_state_dict(modified_state_dict,strict=False)
        return

    def net2deeper(self,
                   deeper_operations: dict):
        """Deepen a layer of a neural network

        Args:
            deeper_operations (dict): A dictionary containing the operations
            to perform to deepen the network.
        """  
                # Go through the list of deepening operations
        for key in deeper_operations.keys():
        
            # Get the parameters of the deeper operation
            target_conv_layers = deeper_operations[key]["target_conv_layers"]
            dataset_used = deeper_operations[key]["dataset"]
    
            # Deepen a layer of the network
            self.net2deeper_operation(target_conv_layers,dataset_used)

        return
    


    def net2deeper_operation(self, previous_conv: str, dataset: str):
        """Deepen a layer of a neural network by adding a convolutional layer
        followed by a batch normalization layer and a ReLU activation.

        Args:
            previous_conv (str): The name of the convolutional layer after
            which the new modules are added (in reality, the modules are added
            after the batch normalization layer and the ReLU activation
            following the convolutional layer)

        Raises:
            NotImplementedError: If the stride of the convolutional layer is
            different from 1
        """
        
        # Go through the modules of the network
        for name, module in self.student_network.named_modules():
            
            # Check if the current module is the one to be copied
            if name == previous_conv:
                
                # Create a new module, ie copy the current convolutional layer
                # (the number of input channels must match the number of output
                # channels of the previous convolutional layer !)
                # (with some padding to keep the size of the feature maps 
                # unchanged, assuming the stride is 1)
                if module.stride == (1, 1):
                    new_conv = nn.Conv2d(module.out_channels,
                                         module.out_channels,
                                         kernel_size=module.kernel_size,
                                         stride=1,
                                         padding=((module.kernel_size[0]-1)//2,
                                                  (module.kernel_size[1]-1)//2))
                else:
                    raise NotImplementedError(
                        "Convolutional layers with stride different from 1"
                        "are not implemented yet.")
                
                # Create a new batch normalization layer
                new_bn = nn.BatchNorm2d(module.out_channels)
                
                break
            
        # Initialize the weights and bias of the new convolutional layer to be
        # an identity function and a zero vector, respectively
        new_weights = np.zeros_like(new_conv.weight.data)
        
        # Put a 1 in the center of each convolutional kernel
        for i in range(new_weights.shape[0]):
            for j in range(new_weights.shape[1]):
                new_weights[i,
                            j,
                            new_weights.shape[2]//2,
                            new_weights.shape[3]//2] = 1
        
        # Initialize the bias to zero
        new_bias = np.zeros_like(new_conv.bias.data)
        
        # Load the weights and bias of the new convolutional layer
        new_conv.weight.data = torch.from_numpy(new_weights)
        new_conv.bias.data = torch.from_numpy(new_bias)
        
        # TODO: Initialize the weights and bias of the new batch normalization
        print("The weights and bias of the new batch normalization layer are"
              "not initialized yet. To be implemented.")
        
        # Add the new modules (ie {convolutional layer + batch normalization
        # layer + ReLU activation}) to the network after the previous
        # convolutional layer
        add_modules(self.student_network, previous_conv, new_conv, new_bn)

    
        model_1_epoch = train_1_epoch(self.student_network,dataset_used)

        self.set_deepen_batchnorm(model,model_1_epoch)


        

if __name__ == '__main__':

    # TEST 
    # Create a model
    model = LeNet(nb_classes=10)
    
    # Instantiate a Net2Net object from a (pre-trained) model
    net2net = Net2Net(teacher_network=model)
    
    
    # Set the convolutional layer to copy in order to deepen the network
    # (we assume that this layer is followed by a batch normalization layer
    # and a ReLU activation)
    previous_conv = "layer1.0"
    
    # Depen a layer of the network
    net2net.net2deeper_operation(previous_conv=previous_conv)
    
    # Create a random input
    x = torch.randn(1, 1, 32, 32)
    
    # Compute the output of the teacher network
    y_teacher = model(x)
    
    # Compute the output of the student network
    y_student = net2net.student_network(x)
    
    # The outputs should be the same
    print("TEST (LeNet): The outputs of the teacher and student networks"
          "should be the same.")
    print("Teacher output: ", y_teacher)
    print("Student output: ", y_student, "\n")

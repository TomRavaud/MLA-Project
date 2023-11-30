"""
An implementation of the Net2WiderNet algorithm to widen convolutional layers
followed by convolutional of fully-connected layers in a neural network.
"""

# Import libraries
import torch
import numpy as np
import torch.nn as nn
import copy


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
        
        # Define the dictionary that will contain the modified weights and
        # biases of the student network
        self.modified_state_dict = {}

    
    def net2wider(self,
                  target_conv_layers: list,
                  next_layers: list,
                  width: list,
                  batch_norm_layers: list = [None],
                  sigma: float = 0.01):
        """Widen convolutional layers of a neural networks

        Args:
            target_conv_layers (list): Convolutional layers to be widened (if
            they are numerous, they must appear in the order in which they
            are concatenated in the network)
            
            next_layers (list): Next layers in the network (the layers for
            which the change in the number of filters of the target layer will
            have an impact on the number of parameters)
            
            width (list): New width of the layer
            
            batch_norm_layers (list, optional): Batch normalization layers that
            follow the target layers. If None, no batch normalization layer is
            used. Defaults to [None].
            
            sigma (float, optional): Standard deviation of the noise added to
            the weights and biases of the supplementary filters. Defaults to
            0.01.
        """
        
        # Get the number of output filters of each target layer
        nb_filters = [
            self.student_network.state_dict()
                [target_conv_layer + ".weight"].shape[0]
                    for target_conv_layer in target_conv_layers]
        
        
        # Go through the target convolutional layers which are to be widened
        for i in range(len(target_conv_layers)):
            
            # Widen the target layer and get the indices given by the random
            # mapping function
            indices = self._net2wider_target_conv(
                target_conv_layer=target_conv_layers[i],
                width=width[i],
                sigma=sigma)
        
            # Check if there is a batch normalization layer between the target
            # layer and the next layer in the teacher network (the layer must
            # be indicated by the user)
            if batch_norm_layers[i]: 
                self._net2wider_bn(batch_norm_layer=batch_norm_layers[i],
                                   width=width[i],
                                   indices=indices)
        
        
            # Check is the next layer is a convolutional layer or a fully
            # connected layer
            for name, module in self.student_network.named_modules():

                # Check if the name of the module is the same as the name of the
                # next layer
                if name not in next_layers:
                    continue
                
                # Check if the next layer is a convolutional layer
                if isinstance(module, nn.Conv2d):
                    self._net2wider_next_conv(
                        next_conv_layer=name,
                        width=width[i],
                        indices=indices,
                        index_input=i,
                        nb_filters=nb_filters)

                # Check if the next layer is a fully connected layer
                elif isinstance(module, nn.Linear):
                    self._net2wider_next_fc(
                        next_fc_layer=name,
                        width=width[i],
                        indices=indices,
                        index_input=i,
                        nb_filters=nb_filters)

                else:
                    raise ValueError(
                        f"Instance of {type(module)} is not supported. "
                        "The next layer must be a convolutional layer "
                        "or a fully connected layer.")

            
                # Replace the weights and biases of the target layer and the
                # next layer in the student network by the new weights and
                # biases (load only the modified weights and biases)
                self.student_network.load_state_dict(self.modified_state_dict,
                                                     strict=False)

                # Re-initialize the dictionary of modified weights and biases
                self.modified_state_dict = {}
            
            
            # Update the number of filters of the current target layer
            nb_filters[i] = width[i]
    

    def _net2wider_next_fc(self,
                           next_fc_layer: str,
                           width: int,
                           indices: np.ndarray,
                           index_input: int,
                           nb_filters: list):
        """Widen a fully connected layer of a neural network which takes as
        input the output of convolutional layers (outputs of the convolutional
        layers are concatenated, averaged and flattened before being fed to
        the fully connected layer)

        Args:
            next_fc_layer (str): Name of the next fully connected layer
            width (int): New width of the widened target layer
            indices (np.ndarray): Indices given by the random mapping function
            index_input (int): Index of the target layer in the list of the
            convolutional layers to be concatenated
            nb_filters (list): Number of filters of each convolutional layer
        """
        
        # Wrap the computation in a no_grad() block to prevent PyTorch from
        # building the computational graph
        with torch.no_grad():
            
            # Get the weights and biases of the next layer in the teacher
            # network (the student network is a copy of the teacher network)
            teacher_w2 =\
                self.student_network.state_dict()[next_fc_layer + '.weight']
            teacher_b2 =\
                self.student_network.state_dict()[next_fc_layer + '.bias']

            # Set the number of filters after the concatenation of the filters
            # (take the new width of the target layer into account)
            nb_filters_concat =\
                np.sum(nb_filters) + width - nb_filters[index_input]
            
            # Initialize the weights of the next layer in the student network
            # (take concatenation into account)
            student_w2 = torch.zeros((teacher_w2.shape[0],
                                      nb_filters_concat))
            
            # Compute the replication factor of each filter (the number of
            # times a same filter is used)
            replication_factor = np.bincount(indices)
            
            
            # Get the index before which the weights and biases of the units
            # in the student network are unchanged
            index_unchanged_before =\
                np.cumsum(nb_filters)[index_input-1] if index_input > 0 else 0
            
            # Get the index after which the weights and biases of the units
            # in the student network are unchanged
            index_unchanged_after = index_unchanged_before + width
            
            # Copy the weights and biases of the next layer of the teacher
            # network to the student network (unchanged units)
            student_w2[:, :index_unchanged_before] =\
                teacher_w2[:, :index_unchanged_before]

            # Copy the weights and biases corresponding to the target layer
            # of the teacher network to the student network (repeated units)
            student_w2[:, index_unchanged_before:
                          index_unchanged_before+nb_filters[index_input]] =\
                teacher_w2[:, index_unchanged_before:
                              index_unchanged_before+nb_filters[index_input]] /\
                    replication_factor[indices[:nb_filters[index_input]]][None, :]
                
            # Add the weights of the supplementary units to the student network
            # (repeated units)
            student_w2[:, index_unchanged_before+nb_filters[index_input]:
                          index_unchanged_after] =\
                teacher_w2[:, index_unchanged_before:index_unchanged_after]\
                          [:, indices[nb_filters[index_input]:]] /\
                    replication_factor[indices[nb_filters[index_input]:]][None, :]
            
            # Copy the weights and biases of the next layer of the teacher
            # network to the student network (unchanged units)
            student_w2[:, index_unchanged_after:] =\
                teacher_w2[:, index_unchanged_before+nb_filters[index_input]:]
                
            
            # Replace the next layer in the student network by the new layer
            # with the supplementary units
            for name, module in self.student_network.named_modules():
                if name == next_fc_layer:
                    replace_module(self.student_network,
                                   name,
                                   nn.Linear(nb_filters_concat,
                                             module.out_features))
                    break
        
            # Put the weights and biases of the next layer of the student
            # network in the dictionary of modified weights and biases
            # (the biases of the next layer are not modified)
            self.modified_state_dict[next_fc_layer + ".weight"] = student_w2
            self.modified_state_dict[next_fc_layer + ".bias"] = teacher_b2
            

    def _net2wider_next_conv(self,
                             next_conv_layer: str,
                             width: int,
                             indices: np.ndarray,
                             index_input: int,
                             nb_filters: list):
        """Widen a convolutional layer of a neural network which takes as
        input the output of convolutional layers (outputs of the convolutional
        layers are concatenated)

        Args:
            next_conv_layer (str): Name of the next convolutional layer
            width (int): New width of the widened target layer
            indices (np.ndarray): Indices given by the random mapping function
            index_input (int): Index of the target layer in the list of the
            convolutional layers to be concatenated
            nb_filters (list): Number of filters of each convolutional layer
        """
        
        # Wrap the computation in a no_grad() block to prevent PyTorch from
        # building the computational graph
        with torch.no_grad():
            
            # Get the weights and biases of the next layer in the teacher
            # network (the student network is a copy of the teacher network)
            teacher_w2 =\
                self.student_network.state_dict()[next_conv_layer + '.weight']
            teacher_b2 =\
                self.student_network.state_dict()[next_conv_layer + '.bias']
            
            # Set the number of filters after the concatenation of the filters
            # (take the new width of the target layer into account)
            nb_filters_concat =\
                np.sum(nb_filters) + width - nb_filters[index_input]
            
            # Initialize the weights of the next layer in the student network
            # (take concatenation into account)
            student_w2 = torch.zeros((teacher_w2.shape[0],
                                      nb_filters_concat,
                                      teacher_w2.shape[2],
                                      teacher_w2.shape[3]))
            
            
            # Compute the replication factor of each filter (the number of
            # times a same filter is used)
            replication_factor = np.bincount(indices)
            
            
            # Get the index before which the weights and biases of the filters
            # in the student network are unchanged
            index_unchanged_before =\
                np.cumsum(nb_filters)[index_input-1] if index_input > 0 else 0
            
            # Get the index after which the weights and biases of the filters
            # in the student network are unchanged
            index_unchanged_after = index_unchanged_before + width
            
            # Copy the weights and biases of the next layer of the teacher
            # network to the student network (unchanged units)
            student_w2[:, :index_unchanged_before, :, :] =\
                teacher_w2[:, :index_unchanged_before, :, :]

            # Copy the weights and biases corresponding to the target layer
            # of the teacher network to the student network (repeated units)
            student_w2[:, index_unchanged_before:
                          index_unchanged_before+nb_filters[index_input], :, :] =\
                teacher_w2[:, index_unchanged_before:
                              index_unchanged_before+nb_filters[index_input], :, :] /\
                    replication_factor[indices[:nb_filters[index_input]]][None,
                                                                          :,
                                                                          None,
                                                                          None]
                
            # Add the weights of the supplementary units to the student network
            # (repeated units)
            student_w2[:, index_unchanged_before+nb_filters[index_input]:
                          index_unchanged_after, :, :] =\
                    teacher_w2[:, index_unchanged_before:index_unchanged_after, :, :]\
                              [:, indices[nb_filters[index_input]:]] /\
                        replication_factor[indices[nb_filters[index_input]:]][None,
                                                                              :,
                                                                              None,
                                                                              None]
            
            # Copy the weights and biases of the next layer of the teacher
            # network to the student network (unchanged units)
            student_w2[:, index_unchanged_after:, :, :] =\
                teacher_w2[:,
                           index_unchanged_before+nb_filters[index_input]:,
                           :, :]
            
            
            # Replace the next layer in the student network by the new layer
            # with the supplementary units
            for name, module in self.student_network.named_modules():
                if name == next_conv_layer:
                    replace_module(self.student_network,
                                      name,
                                      nn.Conv2d(nb_filters_concat,
                                                module.out_channels,
                                                kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding))
                    break
        
            # Put the weights and biases of the next layer of the student
            # network in the dictionary of modified weights and biases
            # (the biases of the next layer are not modified)
            self.modified_state_dict[next_conv_layer + ".weight"] = student_w2
            self.modified_state_dict[next_conv_layer + ".bias"] = teacher_b2


    def _net2wider_target_conv(self,
                               target_conv_layer: str,
                               width: int,
                               sigma: float) -> np.ndarray:
        """Widen a convolutional layer of a neural network

        Args:
            target_conv_layer (str): Name of the target convolutional layer
            width (int): New width of the widened target layer
            sigma (float): Standard deviation of the noise added to the weights

        Returns:
            np.ndarray: Indices given by the random mapping function
        """
        
        # Get the weights and biases of the target layer in the teacher
        # network (the student network is a copy of the teacher network)
        teacher_w1 =\
            self.student_network.state_dict()[target_conv_layer + ".weight"]
        teacher_b1 =\
            self.student_network.state_dict()[target_conv_layer + ".bias"]
        
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
        
        # Initialize the weights and biases of the target layer in the student
        # network
        student_w1 = torch.zeros((width,
                                  teacher_w1.shape[1],
                                  teacher_w1.shape[2],
                                  teacher_w1.shape[3]))
        student_b1 = torch.zeros(width)
        
        # Copy the weights and biases of the target layer and the next
        # layer of the teacher network to the student network
        student_w1[:nb_filters_teacher, :, :, :] = teacher_w1
        student_b1[:nb_filters_teacher] = teacher_b1
        
        # Copy the weights and biases of the target layer and the next
        # layer of the teacher network to the student network
        student_w1[:nb_filters_teacher, :, :, :] = teacher_w1
        student_b1[:nb_filters_teacher] = teacher_b1
        
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
        
        # Replace the target layer and the next layer in the student network
        # by the new layers with the supplementary filters
        for name, module in self.student_network.named_modules():
            if name == target_conv_layer:
                replace_module(self.student_network,
                               name,
                               nn.Conv2d(module.in_channels,
                                         width,
                                         kernel_size=module.kernel_size,
                                         stride=module.stride,
                                         padding=module.padding))
                break
        
        
        # Put the weights and biases of the target layer of the student
        # network in the dictionary of modified weights and biases
        self.modified_state_dict[target_conv_layer + ".weight"] = student_w1
        self.modified_state_dict[target_conv_layer + ".bias"] = student_b1 
        
        return indices


    def _net2wider_bn(self,
                      batch_norm_layer: str,
                      width: int,
                      indices: np.ndarray) -> None:
        """Widen a batch normalization layer of a neural network

        Args:
            batch_norm_layer (str): Name of the batch normalization layer
            width (int): New width of the widened target layer
            indices (np.ndarray): Indices given by the random mapping function
        """
        
        # Get the weights (scale parameters) and biases (shift parameters)
        # of the batch normalization layer in the teacher network
        teacher_gamma =\
            self.student_network.state_dict()[batch_norm_layer + ".weight"]
        teacher_beta =\
            self.student_network.state_dict()[batch_norm_layer + ".bias"]
        
        # Initialize the weights and biases of the BN layer in the student
        # network (there is one scale parameter and one shift parameter per
        # output filter)
        student_gamma = torch.zeros(width)
        student_beta = torch.zeros(width)
        
        # Get the number of output filters of the target layer
        nb_filters_teacher = teacher_gamma.shape[0]

        # Copy the weights and biases of the BN layer of the teacher
        # network to the student network
        student_gamma[:nb_filters_teacher] = teacher_gamma
        student_beta[:nb_filters_teacher] = teacher_beta

        # Add the weights and biases of the supplementary filters to the
        # student network
        student_gamma[nb_filters_teacher:] =\
            teacher_gamma[indices[nb_filters_teacher:]]
        student_beta[nb_filters_teacher:] =\
            teacher_beta[indices[nb_filters_teacher:]]
        
        # Replace the target layer and the next layer in the student network
        # by the new layers with the supplementary filters
        for name, _ in self.student_network.named_modules():
            if name == batch_norm_layer:
                replace_module(self.student_network,
                               name,
                               nn.BatchNorm2d(width))
                break
    
        # Put the weights and biases of the batch normalization layer of the
        # student network the dictionary of modified weights and biases
        self.modified_state_dict[batch_norm_layer + ".weight"] = student_gamma
        self.modified_state_dict[batch_norm_layer + ".bias"] = student_beta



if __name__ == '__main__':
    
    # Import custom modules
    from dummynet import DummyNet, DummyNetBN, DummyNetConcat, DummyNetFC
    
    
    def unit_test(model: nn.Module,
                  target_conv_layers: list,
                  next_layers: list,
                  width: list,
                  batch_norm_layers: list) -> None:
        """Test the Net2WiderNet algorithm on a dummy network

        Args:
            model (nn.Module): A dummy network
            target_conv_layers (list): Layers to be widened
            next_layers (list): Layers following the target layers
            width (list): New width of the target layers
            batch_norm_layers (list): Batch normalization layers that follow
            the target layers
        """
        # Instantiate a Net2Net object from a (pre-trained) model
        net2net = Net2Net(teacher_network=model)

        # Widen the network
        net2net.net2wider(target_conv_layers=target_conv_layers,
                          next_layers=next_layers,
                          width=width,
                          batch_norm_layers=batch_norm_layers,
                          sigma=0.)

        # Create a dummy input
        x = torch.randn((1, 1, 32, 32))

        # Compute the output of the teacher network
        y_teacher = model(x)

        # Compute the output of the student network
        y_student = net2net.student_network(x)

        # The outputs should be the same
        print(f"Test ({model.__class__.__name__}): The outputs of the teacher"
              " and student networks should be the same.")
        print("Teacher output: ", y_teacher)
        print("Student output: ", y_student, "\n")
        
    
    
    #######################################################
    ## Test the Net2WiderNet algorithm on dummy networks ##
    #######################################################

    #----------------------------------------------------#
    # 1. DummyNet - Two consecutive convolutional layers #
    #----------------------------------------------------#
    
    # Create a model
    model1 = DummyNet()
    
    # Set a layer to be widened
    target_conv_layers = ["layer1.0"]
    # Set the next layer
    next_layers = ["layer2.0"]
    # Set the new width of the layer
    new_width = [3]
    # Set the batch normalization layer that follows the target layer
    batch_norm_layers = [None]
    
    # Test the Net2WiderNet algorithm
    unit_test(model=model1,
              target_conv_layers=target_conv_layers,
              next_layers=next_layers,
              width=new_width,
              batch_norm_layers=batch_norm_layers)


    #----------------------------------------------------------------------#
    # 2. DummyNetBN - Two consecutive convolutional layers with batch norm #
    #----------------------------------------------------------------------#
    
    # Create a model
    model2 = DummyNetBN()
    
    # Set a layer to be widened
    target_conv_layers = ["layer1.0"]
    # Set the next layer
    next_layers = ["layer2.0"]
    # Set the new width of the layer
    new_width = [3]
    # Set the batch normalization layer that follows the target layer
    batch_norm_layers = ["layer1.1"]
    
    # Test the Net2WiderNet algorithm
    unit_test(model=model2,
              target_conv_layers=target_conv_layers,
              next_layers=next_layers,
              width=new_width,
              batch_norm_layers=batch_norm_layers)
    
    
    #-----------------------------------------------------------------#
    # 3. DummyNetConcat - The outputs of two convolutional layers are #
    # concatenated and fed to two convolutional layers                #
    #-----------------------------------------------------------------#
    
    # Create a model
    model3 = DummyNetConcat()
    
    # Set the layers to be widened
    target_conv_layers = ["layer1.0", "layer2.0"]
    # Set the next layers
    next_layers = ["layer3.0", "layer4.0"]
    # Set the new width of the layers
    new_width = [6, 8]
    # Set the batch normalization layers that follow the target layers
    batch_norm_layers = ["layer1.1", "layer2.1"]
    
    # Test the Net2WiderNet algorithm
    unit_test(model=model3,
              target_conv_layers=target_conv_layers,
              next_layers=next_layers,
              width=new_width,
              batch_norm_layers=batch_norm_layers)
    
    
    #----------------------------------------------------------------------#
    # 4. DummyNetFC - The output of a convolutional layer is flattened and #
    # fed to a fully connected layer                                       #
    #----------------------------------------------------------------------#
    
    # Create a model
    model4 = DummyNetFC()
    
    # Set the layer to be widened
    target_conv_layers = ["layer1.0"]
    # Set the next layers
    next_layers = ["layer2.0"]
    # Set the new width of the layers
    new_width = [7]
    # Set the batch normalization layers that follow the target layers
    batch_norm_layers = ["layer1.1"]
    
    # Test the Net2WiderNet algorithm
    unit_test(model=model4,
              target_conv_layers=target_conv_layers,
              next_layers=next_layers,
              width=new_width,
              batch_norm_layers=batch_norm_layers)

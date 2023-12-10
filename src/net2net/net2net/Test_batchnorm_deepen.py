# Import libraries
import torch
import torch.nn as nn
import copy


# Import custom modules and packages
import params.lenet_mnist
import utils.log
import utils.train
import utils.test
import utils.validate
from models.lenet import LeNet


LEARNING_PARAMS = params.lenet_mnist.LEARNING

# Create a network model_1_epoch, a copy of a network model and do 1 epoch of training on this model
def train_1_epoch(model,device,train_loader,criterion,LEARNING_PARAMS):

    #make a copy of the model
    model_1_epoch = copy.deepcopy(model)
    
    # create the optimizer
    optimizer = torch.optim.Adam(
        model_1_epoch.parameters(),
        lr=LEARNING_PARAMS["learning_rate"],
        weight_decay=LEARNING_PARAMS["weight_decay"])
    
    # train one epoch of the model
    utils.train.train(model_1_epoch,
                            device,
                            train_loader,
                            optimizer,
                            criterion,
                            0)
    return model_1_epoch


# Initialize batch norms of the deepen network with 
def set_deepen_batchnorm(model,device,train_loader,criterion,LEARNING_PARAMS):
    """
    Args:
        model (nn.Module): The neural network with the batch norm to update
        model_1_epoch (nn.Module): The same model but with one epoch of training
    """
    model_1_epoch = train_1_epoch(model,device,train_loader,criterion,LEARNING_PARAMS)
    with torch.no_grad():
        # create a dictionnary of modifications
        modified_state_dict = {}

        # Search the batch norm modules in the network
        for name, layer in model_1_epoch.named_children():
            
            if isinstance(layer, nn.BatchNorm2d):
                # Add running mean and running std in the dictionnary of modifications
                modified_state_dict[name + '.running_mean'] = layer.running_mean
                modified_state_dict[name + '.running_var'] = layer.running_var

            if isinstance(layer, nn.Sequential):
                for name_inside, layer_inside in layer.named_children():
                    if isinstance(layer_inside, nn.BatchNorm2d):

                        # Create the name of the current layer
                        name_inside = ".".join([name,name_inside])

                        # Add running mean and running std in the dictionnary of modifications
                        modified_state_dict[name_inside + '.running_mean'] = layer_inside.running_mean
                        modified_state_dict[name_inside + '.running_var'] = layer_inside.running_var
        
        # load the modification in the model
        model.load_state_dict(modified_state_dict,strict=False)
    return
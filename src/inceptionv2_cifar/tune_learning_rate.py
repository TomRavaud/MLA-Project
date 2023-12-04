"""
Tune the learning rate using Optuna
"""

# Import packages
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to be able to save
                       # figures when running the script on a remote server
import optuna
import warnings
import os
import numpy as np


# Import custom modules and packages
import params.inceptionv2_cifar
import utils.log
import utils.train
import utils.test
import utils.validate
from inceptionv2 import GoogleNetBN


def objective(learning_params: dict) -> float:
    """Objective function

    Args:
        learning_params (dict): Learning parameters for the training

    Returns:
        float: The validation accuracy
    """
    
    # Compose several transforms together to be applied to data
    # (Note that transforms are not applied yet)
    transform = transforms.Compose([
        # Modify the size of the images
        transforms.Resize(params.inceptionv2_cifar.IMAGE_SHAPE),

        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),

        # Normalize a tensor image with pre-computed mean and standard deviation
        # (based on the data used to train the model(s))
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

    # As CIFAR-10 does not provide a validation dataset, we will split the train
    # dataset into a train and a validation dataset

    # Start by loading the train dataset, with the same transform as the
    # test dataset
    val_dataset = torchvision.datasets.CIFAR10(
        root=params.inceptionv2_cifar.PATH,
        train=True,
        transform=transform,
    )

    # Set the train dataset size as a percentage of the original train dataset
    train_size = (len(train_dataset) - len(test_dataset))/len(train_dataset)

    # Splits train data indices into train and validation data indices
    train_indices, val_indices = train_test_split(range(len(train_dataset)),
                                                  train_size=train_size)

    # Extract the corresponding subsets of the train dataset
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)


    # Combine a dataset and a sampler, and provide an iterable over the dataset
    # (setting shuffle argument to True calls a RandomSampler, and avoids to
    # have to create a Sampler object)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = learning_params['batch_size'],
        shuffle = True,
        num_workers=20,  # Asynchronous data loading and augmentation
        pin_memory=True,  # Increase the transferring speed to the GPU
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = learning_params['batch_size'],
        shuffle = False,  # SequentialSampler
        num_workers=20,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=learning_params["batch_size"],
        shuffle=True,
        num_workers=20,
        pin_memory=True,
    )


    # Use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Instantiate a model
    model = GoogleNetBN(
        nb_classes=params.inceptionv2_cifar.NB_CLASSES,
        inception_factor=params.inceptionv2_cifar.INCEPTION_FACTOR)\
            .to(device=device)

    # Define the loss function (combines nn.LogSoftmax() and nn.NLLLoss())
    criterion = torch.nn.CrossEntropyLoss()

    # Set the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_params["learning_rate"],
        weight_decay=learning_params["weight_decay"])


    # Create tensors to store the loss and accuracy values
    loss_values = torch.zeros(2, learning_params["nb_epochs"])
    accuracy_values = torch.zeros(2, learning_params["nb_epochs"])


    # Initialize the best validation accuracy
    best_val_accuracy = 0
    
    # Initialize the associated best epoch
    best_epoch = 0
    
    
    # Loop over the epochs
    for epoch in range(learning_params["nb_epochs"]):
        
        # Training
        train_loss, train_accuracy = utils.train.train(model,
                                                       device,
                                                       train_loader,
                                                       optimizer,
                                                       criterion,
                                                       epoch)

        # Validation
        val_loss, val_accuracy = utils.validate.validate(model,
                                                         device,
                                                         val_loader,
                                                         criterion,
                                                         epoch) 

        # Store the computed losses
        loss_values[0, epoch] = train_loss
        loss_values[1, epoch] = val_loss
        # Store the computed accuracies
        accuracy_values[0, epoch] = train_accuracy
        accuracy_values[1, epoch] = val_accuracy
        
        
        # Early stopping based on validation accuracy: stop the training if
        # the accuracy has not improved for the last PATIENCE epochs
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
        
        elif epoch - best_epoch >= params.inceptionv2_cifar.PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break


    # Test the model (for logging purposes)
    _, test_accuracy = utils.test.test(model,
                                       device,
                                       test_loader,
                                       criterion)

    # Get the learning parameters table
    params_table = utils.log.parameters_table(
        dataset=params.inceptionv2_cifar.DATASET_NAME,
        learning_params=learning_params)

    # Set the path to the results directory
    results_directory = params.inceptionv2_cifar.PATH +\
        f"/logs/{params.inceptionv2_cifar.STUDY_NAME}/" +\
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Generate the log directory
    utils.log.generate_log(
        results_directory=results_directory,
        test_accuracy=test_accuracy,
        parameters_table=params_table,
        model=model,
        accuracy_values=accuracy_values)
    
    return best_val_accuracy


def objective_wrapper(trial: optuna.Trial) -> float:
    """Wrapper of the objective function to use Optuna

    Args:
        trial (optuna.Trial): A trial object

    Returns:
        float: The validation accuracy averaged over multiple runs
    """
    
    print("\nTrial", trial.number)
    
    # Load learning parameters
    LEARNING_PARAMS = params.inceptionv2_cifar.LEARNING
    
    # Select the learning rate uniformly within the range
    # (use a logarithmic scale)
    lr = trial.suggest_float("lr",
                             *params.inceptionv2_cifar.LR_RANGE,
                             log=True)
    # Update the learning parameters
    LEARNING_PARAMS.update({"learning_rate": lr})
    
    # Run the objective function multiple times to get a more accurate
    # estimate of the objective function
    intermediate_accuracies = []
    
    for i in range(params.inceptionv2_cifar.NB_SEEDS):
        
        # Run the objective function
        intermediate_accuracies.append(objective(LEARNING_PARAMS))
    
    # Aggregate the intermediate accuracies
    accuracy = np.mean(intermediate_accuracies)
    
    return accuracy


# Ignore warnings
warnings.filterwarnings("ignore")


# Make the logs directory if it does not exist
if not os.path.exists(params.inceptionv2_cifar.PATH +\
                      f"/logs/{params.inceptionv2_cifar.STUDY_NAME}"):
    os.makedirs(params.inceptionv2_cifar.PATH +\
                f"/logs/{params.inceptionv2_cifar.STUDY_NAME}")


# Load the study if it already exists (otherwise create a new one)
try:
    study = optuna.load_study(
        storage=params.inceptionv2_cifar.STUDY_DB_PATH,
        study_name=params.inceptionv2_cifar.STUDY_NAME,
        )
except:
    study = optuna.create_study(
        direction="maximize",  # Maximize the accuracy
        sampler=optuna.samplers.TPESampler(),
        storage=params.inceptionv2_cifar.STUDY_DB_PATH,
        study_name=params.inceptionv2_cifar.STUDY_NAME,
        )

study.optimize(objective_wrapper,
               n_trials=params.inceptionv2_cifar.NB_TRIALS)

from tabulate import tabulate
import os
import torch
from typing import List, Any
import matplotlib.pyplot as plt
import numpy as np


def parameters_table(dataset: str,
                     learning_params: dict) -> List[List[Any]]:
    """Generate a table containing the parameters used to train the network
    
    Args:
        dataset (str): The name of the dataset
        learning_params (dict): The parameters used to train the network

    Returns:
        table: The table of parameters
    """
    # Generate the description of the training parameters
    data = [
        [
            "Dataset",
            "Batch size",
            "Nb epochs",
            "Learning rate",
            "Weight decay",
            "Momentum",
        ],
        [
            dataset,
            learning_params["batch_size"],
            learning_params["nb_epochs"],
            learning_params["learning_rate"],
            learning_params["weight_decay"],
            learning_params["momentum"],
        ],
    ]
    
    # Generate the table
    table = tabulate(data,
                     headers="firstrow",
                     tablefmt="fancy_grid",
                     maxcolwidths=20,
                     numalign="center",)
    
    return table


def generate_log(results_directory: str,
                 test_accuracy: float,
                 parameters_table: List[List[Any]],
                 model: torch.nn.Module,
                 accuracy_values: torch.Tensor) -> None:
    """Create a directory to store the results of the training and save the
    results in it

    Args:
        results_directory (str): Path to the directory where the results will
        be stored
        test_accuracy (float): Test accuracy
        parameters_table (table): Table of learning parameters
        model (nn.Module): The network
        accuracy_values (Tensor): Accuracy values
    """    
    # Create the directory
    os.mkdir(results_directory)
    
    # Open a text file
    test_results_file = open(results_directory + "/test_results.txt", "w")
    # Write the test accuracy in it
    test_results_file.write(f"Test accuracy: {test_accuracy}")
    # Close the file
    test_results_file.close()
    
    # Open a text file
    parameters_file = open(results_directory + "/parameters_table.txt", "w")
    # Write the table of learning parameters in it
    parameters_file.write(parameters_table)
    # Close the file
    parameters_file.close()
    
    # Open a text file
    network_file = open(results_directory + "/network.txt", "w")
    # Write the network in it
    print(model, file=network_file)
    # Close the file
    network_file.close()
    
    # Create and save the accuracy curve
    train_accuracies = accuracy_values[0]
    val_accuracies = accuracy_values[1]
    
    plt.figure()

    plt.plot(train_accuracies, "b", label="train accuracy")
    plt.plot(val_accuracies, "r", label="validation accuracy")

    plt.legend()
    plt.xlabel("Epoch")
    
    plt.savefig(results_directory + "/accuracy_curve.png")
    
    # Save the training and validation accuracies
    np.save(results_directory + "/train_accuracies.npy", train_accuracies)
    np.save(results_directory + "/val_accuracies.npy", val_accuracies)
    
    # Save the model parameters
    torch.save(model.state_dict(),
               results_directory + "/" + "network.params")


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Test the functions
    print(parameters_table())

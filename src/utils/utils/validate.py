"""
Function to validate the model for one epoch
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def validate(model: torch.nn.Module,
             device: str,
             val_loader: torch.utils.data.DataLoader,
             criterion: nn.Module,
             epoch: int) -> tuple:
    """Validate the model for one epoch

    Args:
        model (Model): The model to validate
        device (string): The device to use (cpu or cuda)
        val_loader (Dataloader): The validation data loader
        criterion (Loss): The classification loss to use
        epoch (int): The current epoch
        
    Returns:
        double, double: The validation loss, the validation accuracy
    """
    # Initialize the validation loss and accuracy
    val_loss = 0.
    val_correct = 0
    
    # Configure the model for testing
    # (turn off dropout layers, batchnorm layers, etc)
    model.eval()
    
    # Add a progress bar
    val_loader_pbar = tqdm(val_loader, unit="batch")
    
    # Turn off gradients computation (the backward computational graph is
    # built during the forward pass and weights are updated during the backward
    # pass, here we avoid building the graph)
    with torch.no_grad():
        
        # Loop over the validation batches
        for images, labels in val_loader_pbar:

            # Print the epoch and validation mode
            val_loader_pbar.set_description(f"Epoch {epoch} [val]")

            # Move images and labels to GPU (if available)
            images = images.to(device)
            labels = labels.to(device)
            
            # Perform forward pass (only, no backpropagation)
            predicted_labels = model(images)

            # Compute loss
            loss = criterion(predicted_labels,
                             labels)

            # Print the batch loss next to the progress bar
            val_loader_pbar.set_postfix(batch_loss=loss.item())

            # Accumulate batch loss to average over the epoch
            val_loss += loss.item()
            
            # Get the number of correct predictions
            val_correct += torch.sum(
                torch.argmax(predicted_labels, dim=1) == labels
                ).item()
            

    # Compute the losses and accuracies
    val_loss /= len(val_loader)
    val_accuracy = 100*val_correct/len(val_loader.dataset)
    
    return val_loss, val_accuracy

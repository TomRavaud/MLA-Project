"""
Function to test the model on the test set
"""

import torch
import torch.nn as nn
import numpy as np


def test(model: nn.Module,
         device: str,
         test_loader: torch.utils.data.DataLoader,
         criterion: nn.Module,) -> tuple:
    """Test the model on the test set

    Args:
        model (nn.Module): The model to test
        device (str): The device to use for the computations
        test_loader (torch.utils.data.DataLoader): The dataloader for the
        test set
        criterion (nn.Module): The loss function

    Returns:
        double, double: The average loss, the accuracy
    """
    # Testing
    test_loss = 0.
    test_correct = 0

    # Configure the model for testing
    model.eval()


    with torch.no_grad():
        
        # Loop over the testing batches
        for images, labels in test_loader:
            
            # Move images and labels to GPU (if available)
            images = images.to(device)
            labels = labels.to(device)

            # Perform forward pass
            predicted_labels = model(images)

            # Compute loss
            loss = criterion(predicted_labels,
                             labels)
            
            # Accumulate batch loss to average of the entire testing set
            test_loss += loss.item()

            # Get the number of correct predictions
            test_correct +=\
                torch.sum(
                    torch.argmax(
                        predicted_labels, dim=1) == labels
                ).item()


    # Compute the loss and accuracy
    test_loss /= len(test_loader)
    test_accuracy = 100*test_correct/len(test_loader.dataset)
    
    return test_loss, test_accuracy
